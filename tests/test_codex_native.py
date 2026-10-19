"""Native tests use only trusted fixtures and fixed synthetic subprocesses."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from rethlas_core import codex_native_provider as native
from rethlas_core.online_round import NativeRound
from rethlas_core.private_fs import canonical_json, digest
from rethlas_core.provider import ProviderCancelled, ProviderConfigurationError, ProviderIndeterminate, ProviderResponseError, ProviderTimeout
from rethlas_core.research import ResearchPipeline
from rethlas_core.retrieval import RetrievalService
from rethlas_core.validity import ValidityError, ValidityConflict, ValidityRecoveryRequired
from test_project_service import ProjectFixtureMixin
from test_project_mcp import connected
from rethlas_core.project_mcp import build_mcp_app
import asyncio


INSTALLATION = {"entry": "/synthetic/official-codex", "entry_sha256": "a" * 64,
    "compatibility_version": "0.163.0", "compatibility": "trusted synthetic host; no CLI launch"}
MESSAGES = [{"role": "system", "content": "Return candidate JSON."}, {"role": "user", "content": "Synthetic question."}]
GENERATION = {"proof_markdown": "SYNTHETIC candidate only.", "dependency_revision_ids": []}
REVIEW = {"verdict": "correct", "verification_report": {"summary": "Synthetic review, not a proof.",
    "critical_errors": [], "gaps": []}, "repair_hints": ""}
MARKER = "SYNTHETIC_RAW_SECRET_NOT_FOR_LOGS_4829"


def fixed_transport(argv, stdin, *, cwd, cancel_event=None):
    """Trusted test-only transport injection, absent from production config."""
    data = GENERATION if "proof_markdown" in Path(argv[argv.index("--output-schema") + 1]).read_text() else REVIEW
    final = argv[argv.index("-o") + 1]
    code = "import os,sys; os.write(1,b'{\"type\":\"tool.event\"}\\n'); os.write(2,sys.argv[2].encode()); open(sys.argv[1],'wb').write(sys.stdin.buffer.read())"
    return native._run([sys.executable, "-I", "-B", "-c", code, final, MARKER], canonical_json(data), cwd=cwd, cancel_event=cancel_event)


class NativeProviderTests(unittest.TestCase):
    def setUp(self):
        self.install = patch.object(native, "_installation", return_value=deepcopy(INSTALLATION))
        self.install.start()
        self.addCleanup(self.install.stop)

    def envelope(self, selection=None, purpose="generation"):
        provider = native.CodexNativeProvider(selection={"type": "codex_cli_native", **(selection or {})})
        envelope = provider.prepare(MESSAGES, purpose=purpose, service_id="synthetic", request_key="b" * 64, run_id="synthetic-run")
        self.addCleanup(provider.cleanup, envelope)
        return provider, envelope

    def test_four_independent_model_effort_combinations_and_stable_binding(self):
        for selection in ({}, {"model": "synthetic-model"}, {"reasoning_effort": "high"},
                          {"model": "synthetic-model", "reasoning_effort": "high"}):
            with self.subTest(selection=selection):
                provider, envelope = self.envelope(selection)
                again = native.CodexNativeProvider(selection={"type": "codex_cli_native", **selection})
                self.assertEqual(provider.binding(), again.binding())
                self.assertNotIn(envelope["cwd"], canonical_json(provider.binding()).decode())
                self.assertEqual("-m" in envelope["argv"], "model" in selection)
                self.assertEqual(any(s.startswith("model_reasoning_effort=") for s in envelope["argv"]), "reasoning_effort" in selection)
                self.assertIsNone(provider.binding()["resolved"]["model"])
                self.assertNotIn("-a", envelope["argv"])
                self.assertNotIn("--ignore-user-config", envelope["argv"])
                self.assertNotIn("--resume", envelope["argv"])
                self.assertEqual(list(Path(envelope["cwd"]).iterdir()), [])
                self.assertIn("features.multi_agent=false", envelope["argv"])

    def test_tool_events_allowed_and_raw_streams_not_returned(self):
        provider, envelope = self.envelope()
        real = native._run
        with patch.object(native, "_run", side_effect=lambda argv, stdin, **kw: self._fixed(real, argv, stdin, **kw)):
            value = provider.execute(envelope)
        self.assertEqual(value["content"], canonical_json(GENERATION).decode())
        self.assertNotIn(MARKER, repr(value))
        self.assertIsNone(value["model"])
        self.assertIsNone(value["reported_model"])

    def test_production_reader_observes_both_markers_without_export(self):
        markers = {"stdout": MARKER + "_STDOUT", "stderr": MARKER + "_STDERR"}
        observed = {name: bytearray() for name in markers}
        descriptors = {}
        real_spawn, real_read = native.subprocess.Popen, native.os.read
        def spawn(*args, **kwargs):
            process = real_spawn(*args, **kwargs)
            descriptors.update({process.stdout.fileno(): "stdout", process.stderr.fileno(): "stderr"})
            return process
        def read(fd, size):
            chunk = real_read(fd, size)
            if fd in descriptors:
                observed[descriptors[fd]].extend(chunk)
            return chunk
        stdout, stderr, logged = io.StringIO(), io.StringIO(), io.StringIO()
        handler = logging.StreamHandler(logged)
        logger = logging.getLogger()
        logger.addHandler(handler)
        try:
            with tempfile.TemporaryDirectory() as cwd, redirect_stdout(stdout), redirect_stderr(stderr), \
                    patch.object(native.subprocess, "Popen", side_effect=spawn), patch.object(native.os, "read", side_effect=read):
                code = "import os,sys; sys.stdin.buffer.read(); os.write(1,sys.argv[1].encode()); os.write(2,sys.argv[2].encode())"
                result = native._run([sys.executable, "-I", "-B", "-c", code, *markers.values()], b"{}", cwd=cwd)
        finally:
            logger.removeHandler(handler)
            handler.close()
        self.assertEqual(result, {"stream_bytes": sum(len(value.encode()) for value in markers.values()), "exit_code": 0})
        for name, marker in markers.items():
            self.assertEqual(bytes(observed[name]), marker.encode())
            self.assertNotIn(marker, repr(result) + stdout.getvalue() + stderr.getvalue() + logged.getvalue())
        self.assertEqual((stdout.getvalue(), stderr.getvalue(), logged.getvalue()), ("", "", ""))

    @staticmethod
    def _fixed(real, argv, stdin, **kwargs):
        final = argv[argv.index("-o") + 1]
        code = "import os,sys; sys.stdin.buffer.read(); os.write(1,b'{\"type\":\"tool.event\"}\\n'); os.write(2,sys.argv[2].encode()); open(sys.argv[1],'wb').write(sys.argv[3].encode())"
        return real([sys.executable, "-I", "-B", "-c", code, final, MARKER, canonical_json(GENERATION).decode()], stdin, **kwargs)

    def test_binding_rejects_extra_selection_keys_even_with_recomputed_digest(self):
        value = native.CodexNativeProvider(selection={"type": "codex_cli_native"}).binding()
        value["selection"]["type"] = "codex_cli_native"
        value["configuration_sha256"] = digest({k: v for k, v in value.items() if k != "configuration_sha256"})
        with self.assertRaises(ProviderConfigurationError):
            native.validate_binding(value)
        value["selection"] = None
        value["configuration_sha256"] = digest({k: v for k, v in value.items() if k != "configuration_sha256"})
        with self.assertRaises(ProviderConfigurationError):
            native.validate_binding(value)

    def test_stdin_limit_and_no_arbitrary_production_transport(self):
        with self.assertRaises(ProviderConfigurationError):
            native.CodexNativeProvider(selection={"type": "codex_cli_native", "executable": "/tmp/not-allowed"})
        with self.assertRaises(TypeError):
            native.CodexNativeProvider(selection={"type": "codex_cli_native"}, transport=lambda: None)
        provider = native.CodexNativeProvider(selection={"type": "codex_cli_native"})
        with self.assertRaises(ProviderConfigurationError):
            provider.prepare([{"role": "user", "content": "x" * 16384}], purpose="generation", service_id="s", request_key="k", run_id="r")

    def test_final_regular_stable_single_link_and_exact_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "final"
            path.write_bytes(b"a" * 16384)
            self.assertEqual(len(native._final(path)), 16384)
            path.write_bytes(b"a" * 16385)
            with self.assertRaises(ProviderResponseError):
                native._final(path)
            path.write_bytes(b"{}")
            os.link(path, Path(tmp) / "hardlink")
            with self.assertRaises(ProviderResponseError):
                native._final(path)
            path.unlink()
            path.symlink_to(Path(tmp) / "hardlink")
            with self.assertRaises(OSError):
                native._final(path)

    def test_stream_limit_never_reads_one_extra_byte_and_sanitizes(self):
        received = []
        real = os.read
        def bounded(fd, size):
            chunk = real(fd, size)
            received.append(len(chunk))
            return chunk
        with tempfile.TemporaryDirectory() as cwd, patch.object(native.os, "read", side_effect=bounded):
            with self.assertRaises(ProviderResponseError) as caught:
                native._run([sys.executable, "-I", "-c", "import os; os.write(1,b'x'*1048577)"], b"{}", cwd=cwd)
        # Popen may read its internal error pipe too; successful exec returns EOF.
        self.assertEqual(sum(received), 1048576)
        self.assertNotIn(MARKER, str(caught.exception))

    def test_cleanup_is_bounded_and_cwd_tamper_needs_new_approval(self):
        provider, envelope = self.envelope()
        Path(envelope["cwd"], "planted.txt").write_text("synthetic")
        with self.assertRaises(ProviderConfigurationError):
            provider.validate_execution(envelope)
        Path(envelope["cwd"], "planted.txt").unlink()
        changed = deepcopy(envelope)
        changed["argv"][0] = "/tmp/not-official"
        with self.assertRaises(ProviderConfigurationError):
            provider.validate_execution(changed)
        with patch.object(native, "_cleanup_tree", side_effect=ProviderIndeterminate("synthetic cleanup failure")):
            with self.assertRaises(ProviderIndeterminate):
                provider.cleanup(envelope)
        provider.cleanup(envelope)
        self.assertFalse(Path(envelope["cwd"]).parent.exists())

    def test_cleanup_helper_timeout_is_bounded_and_failure_retains_tree(self):
        provider, envelope = self.envelope()
        real = native.subprocess.Popen
        def slow_cleanup(argv, **kwargs):
            return real([sys.executable, "-I", "-c", "import time; time.sleep(30)"], **kwargs)
        start = time.monotonic()
        with patch.dict(native.LIMITS, cleanup_seconds=.4), patch.object(native.subprocess, "Popen", side_effect=slow_cleanup):
            with self.assertRaises(ProviderIndeterminate):
                provider.cleanup(envelope)
        self.assertLess(time.monotonic() - start, .8)
        self.assertTrue(Path(envelope["cwd"]).parent.exists())
        provider.cleanup(envelope)

    def test_final_read_does_not_probe_past_limit_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "final"
            path.write_bytes(b"x" * 16384)
            real = os.read
            sizes = []
            def tracked(fd, size):
                sizes.append(size)
                return real(fd, size)
            with patch.object(native.os, "read", side_effect=tracked):
                self.assertEqual(len(native._final(path)), 16384)
            self.assertEqual(sizes, [16384])
            def mutate(fd, size):
                raw = real(fd, size)
                with path.open("ab") as handle:
                    handle.write(b"y")
                return raw
            with patch.object(native.os, "read", side_effect=mutate):
                with self.assertRaises(ProviderResponseError):
                    native._final(path)
            path.write_bytes(b"\xff")
            with self.assertRaises(ProviderResponseError):
                native._final(path)

    def test_term_ignoring_child_is_killed_and_closed_pipes_still_cancel(self):
        with tempfile.TemporaryDirectory() as cwd:
            script = "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); os.close(0); os.close(1); os.close(2); time.sleep(30)"
            cancel = threading.Event()
            timer = threading.Timer(.15, cancel.set)
            timer.start()
            started = time.monotonic()
            try:
                with self.assertRaises(ProviderCancelled):
                    native._run([sys.executable, "-I", "-c", script], b"{}", cwd=cwd, cancel_event=cancel)
            finally:
                timer.join()
            self.assertGreaterEqual(time.monotonic() - started, 2)
            self.assertLess(time.monotonic() - started, 6)

    def test_successful_leader_descendant_is_terminated_before_return(self):
        with tempfile.TemporaryDirectory() as cwd:
            pid_file = Path(cwd) / "child.pid"
            script = ("import subprocess,sys; sys.stdin.buffer.read(); "
                "p=subprocess.Popen([sys.executable,'-I','-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)'],"
                "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
                "open(sys.argv[1],'w').write(str(p.pid))")
            native._run([sys.executable, "-I", "-c", script, str(pid_file)], b"{}", cwd=cwd)
            pid = int(pid_file.read_text())
            state = Path("/proc") / str(pid) / "stat"
            if state.exists():
                self.assertEqual(state.read_text().split()[2], "Z", "owned descendant must not remain running")

    def test_exit_timeout_cancel_and_owned_descendant_cleanup(self):
        with tempfile.TemporaryDirectory() as cwd:
            with self.assertRaises(ProviderIndeterminate) as caught:
                native._run([sys.executable, "-I", "-c", "import sys; sys.stderr.write('" + MARKER + "'); sys.exit(7)"], b"{}", cwd=cwd)
            self.assertNotIn(MARKER, str(caught.exception))
            with patch.dict(native.LIMITS, seconds=.1):
                with self.assertRaises(ProviderTimeout):
                    native._run([sys.executable, "-I", "-c", "import time; time.sleep(30)"], b"{}", cwd=cwd)
            cancel = threading.Event()
            timer = threading.Timer(.1, cancel.set)
            timer.start()
            try:
                with self.assertRaises(ProviderCancelled):
                    native._run([sys.executable, "-I", "-c", "import time; time.sleep(30)"], b"{}", cwd=cwd, cancel_event=cancel)
            finally:
                timer.join()


class NativeResearchTests(ProjectFixtureMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.runner = self.service.issue(role="research_runner", ttl_seconds=3600)
        self.install = patch.object(native, "_installation", return_value=deepcopy(INSTALLATION))
        self.install.start()
        self.addCleanup(self.install.stop)
        self.calls = []
        def transport(argv, stdin, **kwargs):
            self.calls.append({"argv": list(argv), "stdin": stdin, "cwd": kwargs["cwd"]})
            value = GENERATION if len(self.calls) == 1 else REVIEW
            Path(argv[argv.index("-o") + 1]).write_bytes(canonical_json(value))
            return {"stream_bytes": 1, "exit_code": 0}
        self.transport = patch.object(native, "_run", side_effect=transport)
        self.transport.start()
        self.addCleanup(self.transport.stop)
        self.provider = native.CodexNativeProvider(selection={"type": "codex_cli_native"})
        self.pipeline = ResearchPipeline(self.service, RetrievalService(self.service),
            generation_provider=self.provider, review_provider=native.CodexNativeProvider(selection={"type": "codex_cli_native", "reasoning_effort": "high"}))
        self.round = NativeRound(self.service)
        self.addCleanup(self._cleanup_pending)

    def _cleanup_pending(self):
        try:
            value = self.pipeline.status(self.runner, request_id="native-one")
            envelope = value.get("approval_required", {})
            if envelope.get("schema") == "rethlas.codex-native-execution":
                self.provider.cleanup(envelope)
        except Exception:
            pass

    def run_one(self):
        return self.pipeline.run(self.runner, request_id="native-one", question="Synthetic: show 1 + 1 = 2.", question_only=True)

    def approve(self, value):
        self.round.approve(value["approval_required"], materials="public_or_synthetic")
        return self.pipeline.resume(self.runner, request_id="native-one")

    def ready_generation(self):
        first = self.run_one()
        self.assertEqual(first["historical"]["stage"], "awaiting_native_policy_approval")
        self.assertEqual(self.calls, [])
        generation = self.approve(first)
        self.assertEqual(generation["historical"]["stage"], "awaiting_generation_approval")
        self.assertEqual(self.calls, [])
        return generation

    def test_three_approvals_two_starts_fresh_review_history_and_no_resend(self):
        generation = self.ready_generation()
        review = self.approve(generation)
        self.assertEqual(review["historical"]["stage"], "awaiting_review_approval")
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(Path(self.calls[0]["cwd"]).parent.exists())
        self.assertIn("SYNTHETIC candidate only.", review["approval_required"]["stdin"])
        self.assertNotIn(self.calls[0]["cwd"], review["approval_required"]["stdin"])
        self.assertNotEqual(self.calls[0]["cwd"], review["approval_required"]["cwd"])
        final = self.approve(review)
        self.assertEqual(final["request_status"], "accepted")
        self.assertEqual(len(self.calls), 2)
        self.assertFalse(final["historical"]["mathematical_correctness_proved"])
        self.assertIsNone(final["historical"]["review_model"])
        self.assertEqual(self.run_one()["historical"], final["historical"])
        self.assertEqual(len(self.calls), 2)
        with self.service._operator() as (_, state):
            counts = {k: v for k, v in state["counts"].items() if k.startswith("native.")}
        self.assertEqual(sorted(counts.values()), [1, 1, 2])
        self.assertEqual(self.pipeline.history(self.runner)["count"], 1)

    def test_spawn_failure_charged_started_and_unknown_not_resent(self):
        generation = self.ready_generation()
        with patch.object(native, "_run", side_effect=OSError(MARKER)):
            with self.assertRaises(ProviderIndeterminate) as caught:
                self.approve(generation)
        self.assertNotIn(MARKER, str(caught.exception))
        status = self.pipeline.status(self.runner, request_id="native-one")
        self.assertEqual(status["request_status"], "started")
        with self.assertRaises(ValidityRecoveryRequired):
            self.pipeline.resume(self.runner, request_id="native-one")
        with self.service._operator() as (_, state):
            self.assertEqual(sum(n for k, n in state["counts"].items() if k.startswith("native.")), 2)

    def test_invalid_json_retained_before_parse_and_never_resent(self):
        generation = self.ready_generation()
        def malformed(argv, stdin, **kwargs):
            self.calls.append(argv)
            Path(argv[argv.index("-o") + 1]).write_bytes(("not-json " + MARKER).encode())
            return {"exit_code": 0, "stream_bytes": 0}
        with patch.object(native, "_run", side_effect=malformed):
            with self.assertRaises(ValidityError) as caught:
                self.approve(generation)
        self.assertNotIn(MARKER, str(caught.exception))
        self.assertEqual(self.pipeline.status(self.runner, request_id="native-one")["historical"]["stage"], "generation_received")
        with self.assertRaises(ValidityError):
            self.pipeline.resume(self.runner, request_id="native-one")
        self.assertEqual(len(self.calls), 1)

    def test_approval_tamper_cross_purpose_and_no_approval_zero_spawn(self):
        generation = self.ready_generation()
        altered = deepcopy(generation["approval_required"])
        altered["purpose"] = "review"
        with self.assertRaises(ValidityConflict):
            self.round.approve(altered, materials="public_or_synthetic")
        again = self.pipeline.resume(self.runner, request_id="native-one")
        self.assertEqual(again["historical"]["stage"], "awaiting_generation_approval")
        self.assertEqual(self.calls, [])
        with self.assertRaises(ValidityError):
            self.round.approve({"version": "online-text-approval-3"}, materials="public_or_synthetic")

    def test_usage_caps_remain_same_after_two_starts(self):
        generation = self.ready_generation()
        review = self.approve(generation)
        self.approve(review)
        key = generation["approval_required"]["request_key"]
        with self.service._operator() as (_, state):
            self.assertEqual(state["counts"]["native." + key], 2)
            frozen = deepcopy(state["counts"])
        with self.assertRaises(ValidityConflict):
            self.round.approve(review["approval_required"], materials="public_or_synthetic")
        self.pipeline.resume(self.runner, request_id="native-one")
        with self.service._operator() as (_, state):
            self.assertEqual(state["counts"], frozen)
        self.assertEqual(len(self.calls), 2)

    def test_native_mcp_history_allowed_but_start_and_self_approval_unavailable(self):
        self.run_one()
        credentials = self.base / "native-credentials"
        credentials.mkdir(mode=0o700)
        config = {"version": "research-client-4", "root": str(self.service.root), "embedding": {"mode": "lexical"},
            "indexer": None, "generation_provider": {"type": "codex_cli_native"}, "review_provider": {"type": "codex_cli_native"}}
        for name, selected in (("reader", self.reader), ("writer", self.writer), ("runner", self.runner)):
            path = credentials / (name + ".token")
            path.write_text(selected["token"])
            path.chmod(0o600)
            config[name] = {"run_id": selected["run_id"], "token_file": str(path)}
        config_path = self.base / "native-client.json"
        config_path.write_bytes(canonical_json(config))
        config_path.chmod(0o600)
        async def scenario():
            async with connected(build_mcp_app(config_path), self.base / "native-mcp.stderr.log") as client:
                names = {item.name for item in (await client.list_tools()).tools}
                self.assertFalse(any("approv" in name or "native" in name for name in names))
                result = await client.call_tool("research_run", {"request_id": "mcp-not-authorized",
                    "question": "Synthetic question", "question_only": True})
                value = result.structuredContent or json.loads(result.content[0].text)
                self.assertEqual(value["status"], "rejected")
                history = await client.call_tool("research_history", {})
                value = history.structuredContent or json.loads(history.content[0].text)
                self.assertEqual(value["count"], 1)
        asyncio.run(scenario())
        self.assertEqual(self.calls, [])

    def test_cleanup_failure_keeps_started_fence_before_review(self):
        generation = self.ready_generation()
        envelope = generation["approval_required"]
        with patch.object(native, "_cleanup_tree", side_effect=ProviderIndeterminate("synthetic cleanup failure")):
            with self.assertRaises(ProviderIndeterminate):
                self.approve(generation)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.pipeline.status(self.runner, request_id="native-one")["request_status"], "started")
        with self.assertRaises(ValidityRecoveryRequired):
            self.pipeline.resume(self.runner, request_id="native-one")
        self.provider.cleanup(envelope)

    def test_missing_schema_field_is_retained_without_resend(self):
        generation = self.ready_generation()
        def incomplete(argv, stdin, **kwargs):
            self.calls.append(argv)
            Path(argv[argv.index("-o") + 1]).write_bytes(b'{"proof_markdown":"Synthetic missing dependencies"}')
            return {"exit_code": 0, "stream_bytes": 0}
        with patch.object(native, "_run", side_effect=incomplete):
            with self.assertRaises(ValidityError):
                self.approve(generation)
        self.assertEqual(self.pipeline.status(self.runner, request_id="native-one")["historical"]["stage"], "generation_received")
        with self.assertRaises(ValidityError):
            self.pipeline.resume(self.runner, request_id="native-one")
        self.assertEqual(len(self.calls), 1)

    def test_not_enough_capability_time_and_withdrawal_zero_start(self):
        generation = self.ready_generation()
        envelope = generation["approval_required"]
        self.round.approve(envelope, materials="public_or_synthetic")
        self.runner = self.service.issue(role="research_runner", ttl_seconds=310)
        with self.assertRaises(ValidityError):
            self.pipeline.resume(self.runner, request_id="native-one")
        self.runner = self.service.issue(role="research_runner", ttl_seconds=3600)
        self.assertEqual(self.calls, [])
        self.call("withdraw", self.withdrawer, {"version": "project-1", "request_id": "native-withdraw",
            "targets": [generation["historical"].get("proof", self.pipeline._load(envelope["request_key"], "session",
                generation["historical"]["session_sha256"])["roots"]["claim"])], "reason": "synthetic withdrawn goal"})
        with self.assertRaises(ValidityError):
            self.pipeline.resume(self.runner, request_id="native-one")
        self.assertEqual(self.calls, [])

    def test_started_journal_failure_counts_attempt_without_spawn(self):
        generation = self.ready_generation()
        with patch.object(self.service, "_checkpoint", side_effect=lambda phase: (_ for _ in ()).throw(OSError("synthetic"))
                          if phase == "after_research_started" else None):
            with self.assertRaises(Exception):
                self.approve(generation)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.pipeline.status(self.runner, request_id="native-one")["request_status"], "started")
        with self.assertRaises(ValidityRecoveryRequired):
            self.pipeline.resume(self.runner, request_id="native-one")
        self.provider.cleanup(generation["approval_required"])

    def test_atomic_start_race_one_attempt_one_started_change(self):
        generation = self.ready_generation()
        envelope = generation["approval_required"]
        self.round.approve(envelope, materials="public_or_synthetic")
        key = envelope["request_key"]
        progress = generation["historical"]
        session = self.pipeline._load(key, "session", progress["session_sha256"])
        with self.service._operator() as (_, state):
            providers = self.service._operation(state["requests"][key])["request"]["provider"]
        policy = self.round.policy(key, providers)
        barrier = threading.Barrier(2)
        outcomes = []
        def start():
            barrier.wait()
            try:
                self.round.start(self.runner, key, progress, envelope, policy, session["selected"], session["context"])
                outcomes.append("started")
            except (ValidityConflict, ValidityRecoveryRequired):
                outcomes.append("blocked")
        workers = [threading.Thread(target=start) for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=10)
        self.assertEqual(sorted(outcomes), ["blocked", "started"])
        with self.service._operator() as (_, state):
            self.assertEqual(state["counts"]["native." + key], 1)
            self.assertEqual(state["counts"]["native." + key + ".generation"], 1)
            self.assertEqual(state["requests"][key]["status"], "started")
        self.provider.cleanup(envelope)
