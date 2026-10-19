"""Native text start entry exercised in trusted hosts; never real Codex."""
from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import pty
import selectors
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from rethlas_core import codex_native_provider as native
from rethlas_core.private_fs import canonical_json, strict_json_loads
from rethlas_core.project_client import configured_text, read_configuration
from rethlas_core.research_cli import main, parser
from rethlas_core.research_start import start_project
from test_codex_native import INSTALLATION, GENERATION, REVIEW, MARKER


ROOT = Path(__file__).resolve().parents[1]
# Injection exists only inside this trusted test host, never as a production
# CLI/config/environment option. The actual wrapper/module runs below it.
HOST = r'''
from pathlib import Path
import json,runpy,sys
from unittest.mock import patch
from rethlas_core import codex_native_provider as native
base=Path(sys.argv[1]); kind=sys.argv[2]; target=sys.argv[3]
real=native._run
installation={"entry":"/synthetic/official-codex", "entry_sha256":"a"*64,
    "compatibility_version":"0.163.0", "compatibility":"trusted synthetic host; no CLI launch"}
def fixed(argv,stdin,**kwargs):
    counter=base/'calls.json'
    calls=json.loads(counter.read_bytes()) if counter.exists() else []
    purpose='generation' if 'proof_markdown' in Path(argv[argv.index('--output-schema')+1]).read_text() else 'review'
    calls.append({'purpose':purpose,'stdin':stdin.decode(),'cwd':kwargs['cwd'],'argv':argv})
    counter.write_text(json.dumps(calls)); counter.chmod(0o600)
    result=({'proof_markdown':'SYNTHETIC candidate only.','dependency_revision_ids':[]} if purpose=='generation' else
        {'verdict':'correct','verification_report':{'summary':'SYNTHETIC review, not proof.','critical_errors':[],'gaps':[]},'repair_hints':''})
    code="import os,sys; sys.stdin.buffer.read(); os.write(1,b'{\\\"type\\\":\\\"tool.event\\\"}\\n'); os.write(2,b'SYNTHETIC_RAW_SECRET_NOT_FOR_LOGS_4829'); open(sys.argv[1],'wb').write(sys.argv[2].encode())"
    return real([sys.executable,'-I','-B','-c',code,argv[argv.index('-o')+1],json.dumps(result)],stdin,**kwargs)
sys.argv=[target,*sys.argv[4:]]
with patch.object(native,'_installation',return_value=installation),patch.object(native,'_run',side_effect=fixed):
    if kind=='wrapper': runpy.run_path(target,run_name='__main__')
    else: runpy.run_module(target,run_name='__main__')
'''


class ResearchStartTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="native-start-test-")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / "research"
        self.question = self.base / "question.md"
        self.question.write_text("Synthetic: prove 1 + 1 = 2.")
        self.question.chmod(0o600)
        self.install = patch.object(native, "_installation", return_value=deepcopy(INSTALLATION))
        self.install.start()
        self.addCleanup(self.install.stop)
        self.calls = []
        def fixed(argv, stdin, **kwargs):
            self.calls.append({"argv": argv, "stdin": stdin, "cwd": kwargs["cwd"]})
            value = GENERATION if len(self.calls) == 1 else REVIEW
            Path(argv[argv.index("-o") + 1]).write_bytes(canonical_json(value))
            return {"stream_bytes": 0, "exit_code": 0}
        self.transport = patch.object(native, "_run", side_effect=fixed)
        self.transport.start()
        self.addCleanup(self.transport.stop)
        self.addCleanup(self._pending_cleanup)

    def _pending_cleanup(self):
        from rethlas_core.codex_native_provider import CodexNativeProvider
        if not (self.root / "project/research").exists():
            return
        for path in (self.root / "project/research").glob("*.json"):
            try:
                obj = strict_json_loads(path.read_bytes())
                value = obj.get("value", {})
                if type(value) is dict and value.get("schema") == "rethlas.codex-native-execution":
                    CodexNativeProvider.cleanup(value)
            except Exception:
                pass

    def invoke(self, *arguments, approvals=None):
        output = io.StringIO()
        args = parser().parse_args(["start", str(self.root), *map(str, arguments)])
        with redirect_stdout(output), patch("rethlas_core.research_start._confirm", side_effect=approvals or [False]):
            result = start_project(args)
        self.assertNotIn(MARKER, output.getvalue())
        return result

    def test_new_default_refusal_permissions_history_and_explicit_continue(self):
        first = self.invoke("--question-file", self.question)
        self.assertEqual(first["status"], "native_approval_pending")
        self.assertEqual(self.calls, [])
        config = read_configuration(self.root / "client.json")
        self.assertEqual(config["version"], "research-client-4")
        self.assertEqual(config["generation_provider"], {"type": "codex_cli_native"})
        self.assertEqual(config["review_provider"], {"type": "codex_cli_native"})
        for path in self.root.rglob("*"):
            self.assertEqual(path.stat().st_mode & 0o777, 0o700 if path.is_dir() else 0o600)
        history = self.invoke()
        self.assertEqual(history["status"], "native_history_only")
        self.assertEqual(self.calls, [])
        result = self.invoke("--continue", approvals=[True, True, True])
        self.assertEqual(result["request_status"], "accepted")
        self.assertEqual(len(self.calls), 2)
        self.invoke()
        self.assertEqual(len(self.calls), 2)

    def test_model_effort_not_inherited_between_purposes_and_reopen_changes_reject(self):
        self.invoke("--question-file", self.question, "--generation-model", "synthetic-model", "--review-effort", "high")
        config = read_configuration(self.root / "client.json")
        self.assertEqual(config["generation_provider"], {"type": "codex_cli_native", "model": "synthetic-model"})
        self.assertEqual(config["review_provider"], {"type": "codex_cli_native", "reasoning_effort": "high"})
        from rethlas_core.validity import ValidityError
        with self.assertRaises(ValidityError):
            self.invoke("--generation-model", "changed")
        with self.assertRaises(ValidityError):
            configured_text(None, config, "generation")
        self.assertEqual(self.calls, [])

    def test_three_default_rejections_keep_actual_review_pending(self):
        self.invoke("--question-file", self.question)
        generation = self.invoke("--continue", approvals=[True, False])
        self.assertEqual(generation["research"]["historical"]["stage"], "awaiting_generation_approval")
        self.assertEqual(len(self.calls), 0)
        review = self.invoke("--continue", approvals=[True, False])
        self.assertEqual(review["research"]["historical"]["stage"], "awaiting_review_approval")
        self.assertEqual(len(self.calls), 1)
        self.assertIn("SYNTHETIC candidate only.", review["research"]["approval_required"]["stdin"])
        self.invoke()
        self.assertEqual(len(self.calls), 1)

    def test_unknown_partial_directory_and_repository_rejected_unchanged(self):
        self.root.mkdir(mode=0o700)
        marker = self.root / "unknown.txt"
        marker.write_bytes(b"do not change")
        with self.assertRaises(FileNotFoundError):
            self.invoke("--question-file", self.question)
        self.assertEqual(list(self.root.iterdir()), [marker])
        self.assertEqual(marker.read_bytes(), b"do not change")
        args = parser().parse_args(["start", str(ROOT / "forbidden-native"), "--question-file", str(self.question)])
        with self.assertRaises(ValueError):
            start_project(args)
        self.assertFalse((ROOT / "forbidden-native").exists())

    def test_new_public_round_reuses_project_run_and_keeps_old_attempts(self):
        self.invoke("--question-file", self.question, approvals=[True, True, True])
        original = strict_json_loads((self.root / "start.json").read_bytes())
        original_config = (self.root / "client.json").read_bytes()
        next_question = self.base / "next.md"
        next_question.write_text("Synthetic next question: prove 2 + 2 = 4.")
        next_question.chmod(0o600)
        pending = self.invoke("--new-round", "--question-file", next_question)
        self.assertEqual(pending["status"], "native_approval_pending")
        self.assertNotEqual(pending["request_id"], original["request_id"])
        self.assertEqual((self.root / "client.json").read_bytes(), original_config)
        self.assertEqual(len(self.calls), 2)
        history = self.invoke()
        self.assertEqual(history["history"]["count"], 2)
        from rethlas_core.project_service import ProjectService
        from rethlas_core.private_fs import digest
        service = ProjectService(self.root / "project")
        old_key = digest({"service_id": service.config["service_id"], "request_id": original["request_id"]})
        with service._operator() as (_, state):
            self.assertEqual(state["counts"]["native." + old_key], 2)
        from rethlas_core.validity import ValidityError
        with self.assertRaises(ValidityError):
            self.invoke("--new-round", "--question-file", next_question)
        self.assertEqual(len(self.calls), 2)

    def test_explicit_stage_refresh_requires_new_exact_approval_and_keeps_request(self):
        self.invoke("--question-file", self.question)
        first = self.invoke("--continue", approvals=[True, False])
        old = first["research"]["approval_required"]
        new = self.invoke("--continue", "--refresh-stage")
        envelope = new["research"]["approval_required"]
        self.assertEqual(envelope["request_key"], old["request_key"])
        self.assertNotEqual(envelope["cwd"], old["cwd"])
        self.assertEqual(envelope["stdin"], old["stdin"])
        self.assertFalse(Path(old["cwd"]).parent.exists())
        self.assertEqual(self.calls, [])
        self.invoke("--continue", approvals=[True, True])
        self.assertEqual(len(self.calls), 2)

    def test_non_tty_confirm_and_eof_always_refuse(self):
        from rethlas_core.research_start import _confirm
        with redirect_stdout(io.StringIO()), patch("sys.stdin.isatty", return_value=False), patch("builtins.input") as read:
            self.assertFalse(_confirm({"synthetic": True}, "synthetic"))
            read.assert_not_called()
        with redirect_stdout(io.StringIO()), patch("sys.stdin.isatty", return_value=True), patch("sys.stdout.isatty", return_value=True), patch("builtins.input", side_effect=EOFError):
            self.assertFalse(_confirm({"synthetic": True}, "synthetic"))

    def test_stage_renewal_required_is_explicit_and_no_provider_constructed(self):
        self.invoke("--question-file", self.question)
        before = (self.root / "request.json").read_bytes()
        with patch("rethlas_core.research_start._remaining", return_value=310), patch.object(native, "_installation", side_effect=AssertionError("must not inspect provider")):
            value = self.invoke("--continue")
        self.assertEqual(value["status"], "renewal_required")
        self.assertEqual((self.root / "request.json").read_bytes(), before)
        self.assertEqual(self.calls, [])

    def test_renew_explicit_same_run_request_and_spent_budget(self):
        first = self.invoke("--question-file", self.question)
        self.invoke("--continue", approvals=[True, True, False])
        before = (self.root / "request.json").read_bytes()
        config = read_configuration(self.root / "client.json")
        runner = Path(config["runner"]["token_file"])
        old_token = runner.read_bytes()
        self.invoke("--renew", approvals=[False])
        self.assertEqual(runner.read_bytes(), old_token)
        self.invoke("--renew", approvals=[True])
        self.assertNotEqual(runner.read_bytes(), old_token)
        self.assertEqual((self.root / "request.json").read_bytes(), before)
        self.assertEqual(read_configuration(self.root / "client.json"), config)
        self.invoke("--continue", approvals=[True])
        self.assertEqual(len(self.calls), 2)

    def _host(self, kind, arguments, tty=False):
        target = str(ROOT / "tools/research.py") if kind == "wrapper" else "rethlas_core.research_cli"
        command = [sys.executable, "-I", "-B", "-c", HOST, str(self.base), kind, target, "start", str(self.root), *map(str, arguments)]
        environment = {"PATH": "/nonexistent", "LANG": "C.UTF-8", "HOME": str(self.base / "empty-home"),
                       "CODEX_HOME": str(self.base / "empty-codex")}
        if not tty:
            result = subprocess.run(command, cwd="/", stdin=subprocess.DEVNULL, capture_output=True, timeout=30, env=environment)
            self.assertEqual(result.returncode, 0, result.stdout.decode() + result.stderr.decode())
            self.assertEqual(result.stderr, b"")
            self.assertNotIn(MARKER.encode(), result.stdout)
            return result.stdout
        master, slave = pty.openpty()
        process = subprocess.Popen(command, cwd="/", stdin=slave, stdout=slave, stderr=slave, start_new_session=True, env=environment)
        os.close(slave)
        output = bytearray()
        confirmations = 0
        deadline = time.monotonic() + 30
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(master, selectors.EVENT_READ)
                while time.monotonic() < deadline:
                    for _, _ in selector.select(.05):
                        try:
                            chunk = os.read(master, 65536)
                        except OSError:
                            chunk = b""
                        if not chunk:
                            break
                        output.extend(chunk)
                        wanted = output.count(b"Approve exactly this? Type yes:")
                        while confirmations < wanted:
                            os.write(master, b"yes\n")
                            confirmations += 1
                    if process.poll() is not None:
                        break
                else:
                    self.fail("synthetic native PTY timed out")
            process.wait(timeout=3)
            self.assertEqual(process.returncode, 0, output.decode())
            self.assertEqual(confirmations, 3)
            self.assertNotIn(MARKER.encode(), output)
            return bytes(output)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, 9)
                process.wait(timeout=3)
            os.close(master)

    def test_real_wrapper_non_tty_refuses_and_module_reopen_never_sends(self):
        output = self._host("wrapper", ["--question-file", self.question])
        self.assertIn(b"native_approval_pending", output)
        self.assertFalse((self.base / "calls.json").exists())
        reopened = self._host("module", [])
        self.assertIn(b"native_history_only", reopened)
        self.assertFalse((self.base / "calls.json").exists())

    def test_real_wrapper_pty_three_confirmations_two_synthetic_children_and_reopen(self):
        output = self._host("wrapper", ["--question-file", self.question], tty=True)
        self.assertIn(b"accepted", output)
        calls = json.loads((self.base / "calls.json").read_bytes())
        self.assertEqual([c["purpose"] for c in calls], ["generation", "review"])
        self.assertNotEqual(calls[0]["cwd"], calls[1]["cwd"])
        self.assertTrue(all(not Path(c["cwd"]).parent.exists() for c in calls))
        self.assertIn("SYNTHETIC candidate only.", calls[1]["stdin"])
        self._host("module", [])
        self.assertEqual(json.loads((self.base / "calls.json").read_bytes()), calls)
