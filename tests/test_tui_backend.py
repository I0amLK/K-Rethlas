"""Local-only dual-model/native-login acceptance; no real model requests."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from rethlas_core.authorization import AuthorizationError
from rethlas_core.codex_provider import (
    CodexCLIProvider, CodexUnavailable, effort_choices, inspect_codex, native_profile,
)
from rethlas_core.online_round import OnlineRound
from rethlas_core.private_fs import canonical_json, digest, strict_json_loads
from rethlas_core.project_client import configured_text, read_configuration
from rethlas_core.project_service import ProjectService
from rethlas_core.provider import ProviderCancelled, ProviderConfigurationError, ProviderResponseError, TextProvider
from rethlas_core.research import ResearchPipeline
from rethlas_core.retrieval import RetrievalService
from rethlas_core.validity import ValidityConflict, ValidityError, ValidityRecoveryRequired
from test_project_service import ProjectFixtureMixin


def response(content, model):
    return canonical_json({"id": "dual-synthetic", "object": "chat.completion", "created": 1,
        "model": model, "choices": [{"index": 0, "message": {"role": "assistant", "content": canonical_json(content).decode()},
                                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}})


def review_result():
    return {"verdict": "wrong", "verification_report": {"summary": "Synthetic review only.",
        "critical_errors": [], "gaps": [{"location": "draft", "issue": "Synthetic omitted argument."}]},
        "repair_hints": "Supply the argument."}


class DualModelTests(ProjectFixtureMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.calls = []
        self.declared = ["assumption:0"]
        self.generation = self.provider("candidate-model", "generation")
        self.review_provider = self.provider("review-model", "review")
        self.pipeline = ResearchPipeline(self.service, RetrievalService(self.service),
            generation_provider=self.generation, review_provider=self.review_provider)
        self.question = "Prove the synthetic compactness conclusion."

    def provider(self, model, purpose):
        def send(endpoint, payload, headers, **kwargs):
            messages = strict_json_loads(payload)["messages"]
            self.calls.append((purpose, model, messages))
            content = ({"proof_markdown": "SYNTHETIC candidate, not a proof certificate.",
                        "dependency_revision_ids": self.declared} if purpose == "generation" else review_result())
            return response(content, model)
        return TextProvider(mode="mock", endpoint="https://example.invalid/v1/chat/completions", model=model,
                            admit=lambda *args: None, transport=send)

    def run_research(self, request_id="dual-one", **kwargs):
        return self.pipeline.run(self.runner, request_id=request_id, question=self.question,
                                 selected_refs=[self.ref("proof:0")], **kwargs)

    def test_two_models_dispatch_separately_and_report_both(self):
        value = self.run_research()
        self.assertEqual([row[:2] for row in self.calls], [("generation", "candidate-model"), ("review", "review-model")])
        historical = value["historical"]
        self.assertEqual((historical["generation_model"], historical["review_model"]), ("candidate-model", "review-model"))
        self.assertFalse(historical["mathematical_correctness_proved"])
        self.assertNotIn("guidance", canonical_json(self.calls[1][2]).decode())
        self.assertNotEqual(self.calls[0][2][0], self.calls[1][2][0])
        with self.assertRaises(ValidityConflict):
            other = ResearchPipeline(self.service, RetrievalService(self.service),
                generation_provider=self.provider("different-model", "generation"), review_provider=self.review_provider)
            other.run(self.runner, request_id="dual-one", question=self.question, selected_refs=[self.ref("proof:0")])
        self.assertEqual(len(self.calls), 2)

    def test_standalone_review_binds_only_review_provider(self):
        parent = self.run_research()
        self.calls.clear()
        pipeline = ResearchPipeline(self.service, RetrievalService(self.service), review_provider=self.review_provider)
        value = pipeline.review(self.runner, request_id="only-review", proof_ref=parent["historical"]["proof"], statement=self.question)
        self.assertEqual([row[0] for row in self.calls], ["review"])
        self.assertEqual(set(value["historical"]["providers"]), {"review"})
        self.assertIsNone(value["historical"]["generation_model"])

    def test_citation_roots_are_disclosed_but_never_premises(self):
        context = self.service.prepare_context(self.reader["token"], [self.ref("assumption:0")],
            run_id=self.reader["run_id"], reference_refs=[self.ref("claim:0")])
        self.assertIn("assumption:0", {row["revision_id"] for row in context["overlay"]["support"]})
        self.assertIn("claim:0", {row["revision_id"] for row in context["overlay"]["citation_only"]})
        self.declared = ["claim:0"]
        with self.assertRaises(ValidityError):
            self.pipeline.run(self.runner, request_id="no-promotion", question=self.question,
                selected_refs=[self.ref("assumption:0")], reference_refs=[self.ref("claim:0")])
        self.assertEqual(len(self.calls), 1)

    def test_question_only_empty_project_with_citations_and_no_circular_premise(self):
        service = ProjectService.initialize(self.base / "empty", owner="owner")
        runner = service.issue(role="research_runner")
        self.declared = []
        pipeline = ResearchPipeline(service, RetrievalService(service), generation_provider=self.generation,
                                    review_provider=self.review_provider)
        value = pipeline.run(runner, request_id="question-only", question="Prove 1 + 1 = 2.", selected_refs=[], question_only=True)
        self.assertEqual(value["request_status"], "accepted")
        sent = strict_json_loads(self.calls[0][2][1]["content"])
        self.assertEqual(sent["authorized_dependency_revision_ids"], [])
        self.assertEqual(len(sent["goal_only_revision_ids"]), 1)
        context = service.get_revision(runner["token"], value["historical"]["proof"], run_id=runner["run_id"])
        proof = value["historical"]["proof"]["revision_id"]
        self.assertFalse(any(edge["type"] == "depends_on" and edge["source"]["revision_id"] == proof for edge in context["dto"]["relations"]))
        self.calls.clear()
        self.declared = []
        cited = self.pipeline.run(self.runner, request_id="only-question-cited", question=self.question,
            selected_refs=[], question_only=True, reference_refs=[self.ref("source:0")])
        sent = strict_json_loads(self.calls[0][2][1]["content"])
        self.assertEqual(next(r for r in sent["context"]["revisions"] if r["revision_id"] == "source:0")["use"], "historical_citation_only")
        self.withdraw("source:0")
        later = self.pipeline.status(self.runner, request_id="only-question-cited")
        self.assertEqual(later["historical"], cited["historical"])
        proof = later["current"]["current"][cited["historical"]["proof"]["revision_id"]]
        report = later["current"]["current"][cited["historical"]["review"]["revision_id"]]
        self.assertFalse(proof["math_blocked"])
        self.assertTrue(report["review_input_obsolete"])

    def test_question_only_revision_keeps_prior_goal_out_of_premises(self):
        self.declared = []
        parent = self.pipeline.run(self.runner, request_id="only-parent", question=self.question,
            selected_refs=[], question_only=True, reference_refs=[self.ref("source:0")])
        value = self.pipeline.revise(self.runner, request_id="only-revise", parent_request_id="only-parent",
            instructions="Explicitly address the synthetic missing argument.")
        self.assertEqual(value["request_status"], "accepted")
        sent = strict_json_loads(self.calls[2][2][1]["content"])
        self.assertEqual(sent["authorized_dependency_revision_ids"], [])
        context = self.service.get_revision(self.reader["token"], value["historical"]["proof"], run_id=self.reader["run_id"])
        proof = value["historical"]["proof"]["revision_id"]
        self.assertFalse(any(edge["source"]["revision_id"] == proof and edge["type"] == "depends_on"
                             and edge["target"]["revision_id"] in sent["goal_only_revision_ids"]
                             for edge in context["dto"]["relations"]))
        self.assertNotEqual(parent["historical"]["proof"], value["historical"]["proof"])

    def test_question_goal_cannot_be_declared_as_proved_premise(self):
        original = self.generation.complete
        def circular(*args, **kwargs):
            value = original(*args, **kwargs)
            sent = strict_json_loads(args[0][1]["content"])
            value["content"] = canonical_json({"proof_markdown": "Circular synthetic output.",
                "dependency_revision_ids": sent["goal_only_revision_ids"]}).decode()
            return value
        with patch.object(self.generation, "complete", side_effect=circular), self.assertRaises(ValidityError):
            self.pipeline.run(self.runner, request_id="circular", question=self.question, selected_refs=[], question_only=True)
        self.assertEqual(len(self.calls), 1)

    def test_no_support_is_not_silently_question_only(self):
        for kwargs in ({"selected_refs": [], "question_only": False},
                       {"selected_refs": [self.ref("assumption:0")], "question_only": True},
                       {"selected_refs": [], "question_only": "yes"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidityError):
                self.pipeline.run(self.runner, request_id="invalid-mode-" + str(len(str(kwargs))), question=self.question, **kwargs)
        self.assertEqual(self.calls, [])

    def test_history_authority_limits_and_restart_projection(self):
        original = self.run_research()
        offline = ResearchPipeline(ProjectService(self.service.root), RetrievalService(self.service), None)
        history = offline.history(self.runner)
        self.assertEqual(history["count"], 1)
        self.assertEqual(history["items"][0]["request_id"], "dual-one")
        self.assertEqual(offline.status(self.runner, request_id="dual-one")["historical"], original["historical"])
        for limit in (True, 0, 65, "10"):
            with self.subTest(limit=limit), self.assertRaises(ValidityError):
                offline.history(self.runner, limit=limit)
        with self.assertRaises(AuthorizationError):
            offline.history(self.reader)

    def test_resume_retained_prompt_requires_exact_destination_and_stays_unsent_until_approved(self):
        approved = OnlineRound.initialize(self.service, allowed_sha256=["0" * 64], materials="public_or_synthetic")
        for provider in (self.generation, self.review_provider):
            provider.mode = "live"
            provider.admit = lambda messages, purpose, p=provider: approved.admit_text(messages, purpose, provider=p.binding())
        original_generation, original_review = self.generation.complete, self.review_provider.complete
        def live_result(provider, original):
            def complete(messages, *, purpose, before_send=None, **kwargs):
                provider.admit(messages, purpose)
                if before_send:
                    before_send()
                self.calls.append((purpose, provider.model, messages))
                content = ({"proof_markdown": "Synthetic approved candidate.", "dependency_revision_ids": ["assumption:0"]}
                           if purpose == "generation" else review_result())
                return {"content": canonical_json(content).decode(), "model": provider.model, "mode": "live",
                        "usage": {}, "response_sha256": digest(content)}
            return complete
        with patch.object(self.generation, "complete", side_effect=live_result(self.generation, original_generation)), \
             patch.object(self.review_provider, "complete", side_effect=live_result(self.review_provider, original_review)):
            first = self.run_research("await-exact")
            prompt = first["approval_required"]
            self.assertEqual(prompt["provider"]["model"], "candidate-model")
            self.assertEqual(self.calls, [])
            OnlineRound.approve_text(self.service, messages=prompt["messages"], purpose=prompt["purpose"], materials="public_or_synthetic")
            second = self.pipeline.resume(self.runner, request_id="await-exact")
            self.assertEqual(second["historical"]["stage"], "awaiting_generation_approval")
            self.assertEqual(self.calls, [])
            OnlineRound.approve_text(self.service, messages=prompt["messages"], purpose=prompt["purpose"],
                                     materials="public_or_synthetic", provider=prompt["provider"])
            second = self.pipeline.resume(self.runner, request_id="await-exact")
            self.assertEqual(second["historical"]["stage"], "awaiting_review_approval")
            prompt = second["approval_required"]
            OnlineRound.approve_text(self.service, messages=prompt["messages"], purpose=prompt["purpose"],
                                     materials="public_or_synthetic", provider=prompt["provider"])
            self.assertEqual(self.pipeline.resume(self.runner, request_id="await-exact")["request_status"], "accepted")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(approved.usage()["used"]["text_requests"], 2)

    def test_started_resume_never_resends_and_provider_changes_are_bound(self):
        from rethlas_core.provider import ProviderTimeout
        with patch.object(self.generation, "complete", side_effect=ProviderTimeout("synthetic")):
            with self.assertRaises(ProviderTimeout):
                self.run_research("started")
        with self.assertRaises(ValidityRecoveryRequired):
            self.pipeline.resume(self.runner, request_id="started")
        self.assertEqual(self.calls, [])


class NativeCodexTests(unittest.TestCase):
    def test_live_adapter_constructs_without_api_and_never_dispatches(self):
        admission = Mock()
        provider = CodexCLIProvider(model="gpt-5.4", admit=admission)
        self.assertFalse(provider.preflight()["run_enabled"])
        with patch("rethlas_core.codex_provider.subprocess.Popen", side_effect=AssertionError("no real model")):
            with self.assertRaises(CodexUnavailable):
                provider.complete([{"role": "user", "content": "Synthetic question"}])
        admission.assert_not_called()

    def test_inspection_is_redacted_and_login_only_when_explicit(self):
        calls = []
        def capture(argv, **kwargs):
            calls.append(argv[1:])
            if argv[1:] == ["--version"]:
                return 0, b"codex-cli 0.163.0\n", b""
            if argv[1:] == ["exec", "--help"]:
                return 0, b"--ignore-user-config --ephemeral --output-schema --json --model --sandbox", b""
            return 0, b"", b"Logged in using an API key: private-value-that-must-never-escape\n"
        with patch("rethlas_core.codex_provider.shutil.which", return_value="/fake/codex"), \
             patch("rethlas_core.codex_provider._capture", side_effect=capture):
            first = inspect_codex()
            self.assertEqual(first["login_status"], "not_checked")
            self.assertNotIn(["login", "status"], calls)
            second = inspect_codex(check_login=True)
        self.assertEqual(second["login_status"], "logged_in")
        self.assertNotIn("private-value", canonical_json(second).decode())
        self.assertFalse(second["run_enabled"])

    def test_v2_native_configuration_needs_no_api_key_or_online_round(self):
        with tempfile.TemporaryDirectory(prefix="codex-client-test-") as directory:
            root = Path(directory)
            service = ProjectService.initialize(root / "project", owner="owner")
            config = {"version": "research-client-2", "root": str(service.root), "reader": None,
                "writer": None, "indexer": None, "runner": None, "embedding": {"mode": "lexical"},
                "generation_provider": {"type": "codex_cli", "model": "gpt-5.4"},
                "review_provider": {"type": "codex_cli", "model": "review-model"}}
            path = root / "client.json"
            path.write_bytes(canonical_json(config))
            self.assertEqual(read_configuration(path), config)
            for purpose in ("generation", "review"):
                provider = configured_text(service, config, purpose)
                self.assertEqual(provider.model, config[purpose + "_provider"]["model"])
                self.assertFalse(provider.preflight()["run_enabled"])
                self.assertNotIn("key_file", config[purpose + "_provider"])

    def test_synthetic_cli_fresh_argv_stdin_schema_and_no_raw_tools(self):
        records = []
        with tempfile.TemporaryDirectory(prefix="codex-fake-host-") as directory:
            script = Path(directory) / "fake_cli.py"
            script.write_text("import json,sys\nrequest=json.load(sys.stdin)\n"
                "print(json.dumps({'type':'thread.started','thread_id':'synthetic-fresh'}))\n"
                "print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':json.dumps({'proof_markdown':'synthetic','dependency_revision_ids':[]})}}))\n"
                "print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10,'output_tokens':20}}))\n", encoding="utf-8")
            def transport(argv, stdin, *, cwd, **kwargs):
                records.append((list(argv), strict_json_loads(stdin), cwd, sorted(Path(cwd).iterdir())))
                return subprocess.run([sys.executable, "-I", "-B", str(script), *argv[1:]], input=stdin,
                    cwd=cwd, capture_output=True, check=True, timeout=kwargs["timeout"]).stdout
            provider = CodexCLIProvider(model="candidate-model", mode="mock", admit=lambda *args: None, transport=transport)
            for purpose in ("generation", "review"):
                value = provider.complete([{"role": "user", "content": "SYNTHETIC private prompt not in argv"}], purpose=purpose)
                self.assertEqual(value["mode"], "mock")
                self.assertIsNone(value["reported_model"])
        self.assertEqual(len(records), 2)
        self.assertNotEqual(records[0][2], records[1][2])
        for argv, messages, _, files in records:
            self.assertNotIn("resume", argv)
            self.assertIn("--ignore-user-config", argv)
            self.assertIn("--sandbox", argv)
            self.assertIn("read-only", argv)
            self.assertNotIn("SYNTHETIC private prompt", " ".join(argv))
            self.assertEqual(len(files), 1)
            self.assertEqual(messages[0]["role"], "user")

    def test_cli_events_reject_tool_unknown_partial_and_duplicates(self):
        invalid = [b'{"type":"item.completed","item":{"type":"command_execution","text":"echo bad"}}\n',
                   b'{"type":"error","message":"private"}\n', b'{}\n',
                   b'{"type":"item.completed","item":{"type":"agent_message","text":"draft"}}\n']
        for raw in invalid:
            provider = CodexCLIProvider(model="m", mode="mock", admit=lambda *args: None,
                                       transport=lambda *args, raw=raw, **kwargs: raw)
            with self.subTest(raw=raw), self.assertRaises(ProviderResponseError):
                provider.complete([{"role": "user", "content": "synthetic"}])

    def test_mock_cancel_and_injection_are_explicit(self):
        with self.assertRaises(ProviderConfigurationError):
            CodexCLIProvider(model="m", mode="live", admit=lambda *args: None, transport=lambda *args: b"")
        event = threading.Event()
        event.set()
        transport = Mock()
        provider = CodexCLIProvider(model="m", mode="mock", admit=lambda *args: None, transport=transport)
        with self.assertRaises(ProviderCancelled):
            provider.complete([{"role": "user", "content": "synthetic"}], cancel_event=event)
        transport.assert_not_called()


class NativeSelectionTests(unittest.TestCase):
    def selections(self):
        return [{"type": "codex_cli"}, {"type": "codex_cli", "model": "synthetic-native"},
                {"type": "codex_cli", "reasoning_effort": "high"},
                {"type": "codex_cli", "model": "synthetic-native", "reasoning_effort": "high"}]

    def test_four_intents_detach_and_remain_unresolved_without_cli_or_admission(self):
        for selection in self.selections():
            with self.subTest(selection=selection), \
                 patch("rethlas_core.codex_provider.inspect_codex", side_effect=AssertionError("no automatic CLI")), \
                 patch("rethlas_core.codex_provider.subprocess.Popen", side_effect=AssertionError("no execution")), \
                 patch("rethlas_core.codex_provider._capture", side_effect=AssertionError("no capture")), \
                 patch("rethlas_core.codex_provider.shutil.which", side_effect=AssertionError("no lookup")):
                admission = Mock()
                source = deepcopy(selection)
                provider = CodexCLIProvider(selection=source, mode="live", admit=admission)
                source["model"] = "mutated-after-construction"
                self.assertEqual(provider.selection, selection)
                preflight = provider.preflight()
                self.assertEqual(preflight["selection"], selection)
                self.assertEqual(preflight["resolution_status"], "blocked")
                for name in ("resolved", "resolved_model", "resolved_reasoning_effort"):
                    self.assertIsNone(preflight[name])
                self.assertFalse(preflight["run_enabled"])
                with self.assertRaises(CodexUnavailable):
                    provider.binding()
                with self.assertRaises(CodexUnavailable):
                    provider.complete([{"role": "user", "content": "Synthetic unresolved question."}])
                admission.assert_not_called()

    def test_profile_rejects_unknown_null_placeholder_and_invalid_types(self):
        invalid = [None, [], "codex_cli", True, {}, {"type": "other"},
                   {"type": "codex_cli", "extra": True}]
        for field, values in (("model", (None, "", "default", "unknown", "auto", "bad model", True, [])),
                              ("reasoning_effort", (None, "", "default", "unknown", "minimal", True, []))):
            invalid.extend({"type": "codex_cli", field: value} for value in values)
        for profile in invalid:
            with self.subTest(profile=profile), self.assertRaises(ProviderConfigurationError):
                native_profile(profile)
        original = {"type": "codex_cli", "reasoning_effort": "ultra"}
        clean = native_profile(original)
        original["reasoning_effort"] = "low"
        self.assertEqual(clean["reasoning_effort"], "ultra")

    def test_effort_choices_are_default_plus_static_vocabulary_not_certification(self):
        choices = effort_choices()
        self.assertEqual([value for _, value in choices], ["", "low", "medium", "high", "xhigh", "max", "ultra"])
        self.assertIn("尚未确认", choices[0][0])
        for label, _ in choices[1:]:
            self.assertIn("兼容未确认", label)

    def test_synthetic_frozen_effective_settings_match_transport_not_raw_report(self):
        raw = (canonical_json({"type": "item.completed", "item": {"type": "agent_message", "text": "synthetic"}})
               + b"\n" + canonical_json({"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 2}}) + b"\n")
        for selection in self.selections():
            records = []
            def transport(argv, stdin, *, effective_settings, **kwargs):
                records.append((list(argv), strict_json_loads(stdin), deepcopy(effective_settings)))
                # Mutation by the synthetic host must not alter the provider.
                effective_settings["resolved"]["model"] = "host-mutated-copy"
                return raw
            resolved = {"model": "synthetic-native", "reasoning_effort": "high",
                        "provider": {"id": "synthetic-host", "destination": "https://example.invalid/v1"}}
            provider = CodexCLIProvider(selection=selection, resolved=resolved, mode="mock",
                                        admit=lambda *args: None, transport=transport)
            frozen = provider.binding()
            resolved["provider"]["id"] = "mutated-input"
            value = provider.complete([{"role": "user", "content": "SYNTHETIC secret stdin, never argv"}])
            self.assertEqual(provider.binding(), frozen)
            self.assertEqual(records[0][2], frozen)
            self.assertEqual(frozen["selection"], {k: v for k, v in selection.items() if k != "type"})
            self.assertEqual(provider.preflight()["resolution_status"], "synthetic_test_host_only")
            self.assertEqual((value["model"], value["resolved_model"], value["resolved_reasoning_effort"]),
                             ("synthetic-native", "synthetic-native", "high"))
            self.assertEqual(value["requested_model"], selection.get("model"))
            self.assertEqual(value["requested_reasoning_effort"], selection.get("reasoning_effort"))
            self.assertIsNone(value["reported_model"])
            self.assertIsNone(value["reported_reasoning_effort"])
            argv = records[0][0]
            self.assertNotIn("SYNTHETIC secret", " ".join(argv))
            self.assertNotIn("--reasoning-effort", argv)
            self.assertNotIn("high", argv)
            self.assertNotIn("resume", argv)

    def test_synthetic_setting_mutation_at_admission_never_reaches_transport(self):
        for field, value in (("model", "changed-model"), ("reasoning_effort", "max"),
                             ("endpoint", "https://other.invalid/v1"), ("timeout", 31)):
            transport = Mock()
            provider = None
            def admit(messages, purpose):
                setattr(provider, field, value)
            provider = CodexCLIProvider(selection={"type": "codex_cli"}, mode="mock", admit=admit, transport=transport,
                resolved={"model": "synthetic-native", "reasoning_effort": "high",
                          "provider": {"id": "synthetic-host", "destination": "https://example.invalid/v1"}})
            with self.subTest(field=field), self.assertRaises(CodexUnavailable):
                provider.complete([{"role": "user", "content": "Synthetic request"}])
            transport.assert_not_called()

    def test_synthetic_injection_cannot_open_live_or_guess_missing_defaults(self):
        resolved = {"model": "synthetic-native", "reasoning_effort": "high",
                    "provider": {"id": "synthetic-host", "destination": "https://example.invalid"}}
        with self.assertRaises(ProviderConfigurationError):
            CodexCLIProvider(selection={"type": "codex_cli"}, resolved=resolved, mode="live", admit=Mock())
        with self.assertRaises(ProviderConfigurationError):
            CodexCLIProvider(selection={"type": "codex_cli"}, mode="mock", admit=Mock(), transport=Mock())
        with self.assertRaises((ProviderConfigurationError, ValidityError)):
            CodexCLIProvider(selection={"type": "codex_cli", "reasoning_effort": "low"}, resolved=resolved,
                             mode="mock", admit=Mock(), transport=Mock())

    def test_legacy_model_binding_preserves_exact_four_fields_and_hash(self):
        provider = CodexCLIProvider(model="legacy-model", mode="live", admit=Mock())
        expected = {"provider_type": "codex_cli", "mode": "live", "model": "legacy-model",
            "configuration_sha256": digest({"provider_type": "codex_cli", "mode": "live", "model": "legacy-model",
                "endpoint": "codex://native-login", "timeout": 30, "response_bytes": provider.max_response_bytes,
                "compatibility_version": "0.163.0"})}
        self.assertEqual(provider.binding(), expected)

    def test_v3_four_intents_read_save_without_changing_legacy_v2_bytes(self):
        with tempfile.TemporaryDirectory(prefix="native-selection-config-") as directory:
            root = Path(directory)
            service = ProjectService.initialize(root / "project", owner="synthetic-owner")
            config = {"version": "research-client-2", "root": str(service.root), "reader": None,
                "writer": None, "indexer": None, "runner": None, "embedding": {"mode": "lexical"},
                "generation_provider": {"type": "codex_cli", "model": "legacy-generation"},
                "review_provider": {"type": "codex_cli", "model": "legacy-review"}}
            path = root / "client.json"
            legacy = canonical_json(config)
            path.write_bytes(legacy)
            self.assertEqual(read_configuration(path), config)
            self.assertEqual(path.read_bytes(), legacy)
            for index, generation in enumerate(self.selections()):
                review = self.selections()[(index + 1) % 4]
                versioned = {**config, "version": "research-client-3", "generation_provider": generation,
                             "review_provider": review}
                path.write_bytes(canonical_json(versioned))
                self.assertEqual(read_configuration(path), versioned)
                for purpose, expected in (("generation", generation), ("review", review)):
                    provider = configured_text(service, versioned, purpose)
                    self.assertEqual(provider.selection, expected)
                    self.assertFalse(provider.preflight()["run_enabled"])


class ModelAliasTests(unittest.TestCase):
    def test_alias_is_explicit_and_requested_reported_are_separate(self):
        content = {"synthetic": True}
        raw = response(content, "reported-snapshot")
        def provider(aliases=None):
            return TextProvider(mode="mock", model="requested-model", endpoint="https://example.invalid/v1/chat/completions",
                admit=lambda *args: None, transport=lambda *args, **kwargs: raw, allowed_response_models=aliases)
        with self.assertRaises(ProviderResponseError):
            provider().complete([{"role": "user", "content": "public synthetic"}])
        value = provider(["reported-snapshot"]).complete([{"role": "user", "content": "public synthetic"}])
        self.assertEqual((value["requested_model"], value["reported_model"]), ("requested-model", "reported-snapshot"))
        self.assertNotEqual(provider().binding(), provider(["reported-snapshot"]).binding())


if __name__ == "__main__":
    unittest.main()
