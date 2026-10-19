"""Synthetic offline contracts. Written separately from authorized execution."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import unittest

from rethlas_core.codex_binding import (
    CLI_VERSION, OBSERVED_EFFORTS, make_codex_binding, validate_codex_binding,
)
from rethlas_core.online_round import OnlineRound
from rethlas_core.private_fs import canonical_json, digest, strict_json_loads
from rethlas_core.research import ResearchPipeline
from rethlas_core.retrieval import RetrievalService
from rethlas_core.validity import ValidityConflict, ValidityError, ValidityRecoveryRequired
from test_project_service import ProjectFixtureMixin


def binding(*, mode="mock", selection=None, effort="high", provider="synthetic"):
    return make_codex_binding(mode=mode, selection={} if selection is None else selection,
        resolved={"model": "synthetic-model", "reasoning_effort": effort,
                  "provider": {"id": provider, "destination": "https://example.invalid/v1"}},
        cli_version=CLI_VERSION, execution_sha256="a" * 64)


def reseal(value):
    value["configuration_sha256"] = digest({k: v for k, v in value.items() if k != "configuration_sha256"})
    return value


class CodexBindingTests(unittest.TestCase):
    def test_default_selection_preserves_absence_and_detaches_all_nested_facts(self):
        source = binding()
        validated = validate_codex_binding(source)
        self.assertEqual(validated["selection"], {})
        self.assertNotIn("model", validated)
        self.assertEqual(source["configuration_sha256"], digest({k: v for k, v in source.items() if k != "configuration_sha256"}))
        source["resolved"]["provider"]["id"] = "changed"
        self.assertEqual(validated["resolved"]["provider"]["id"], "synthetic")
        selection = {"model": "synthetic-model", "reasoning_effort": "high"}
        made = binding(selection=selection)
        selection["model"] = "changed"
        self.assertEqual(made["selection"]["model"], "synthetic-model")

    def test_observed_efforts_are_only_static_vocabulary(self):
        self.assertEqual(OBSERVED_EFFORTS, ("low", "medium", "high", "xhigh", "max", "ultra"))
        for effort in OBSERVED_EFFORTS:
            self.assertEqual(binding(effort=effort)["resolved"]["reasoning_effort"], effort)
        for effort in (None, "", "unknown", "default", "auto", "minimal", True, []):
            with self.subTest(effort=effort), self.assertRaises(ValidityError):
                binding(effort=effort)

    def test_rejects_unknown_missing_extra_null_and_mismatched_resolution(self):
        original = binding()
        bad = []
        for name in original:
            value = deepcopy(original)
            value.pop(name)
            bad.append(value)
        for name, content in (("version", "codex-provider-binding-2"), ("cli_version", "0.164.0"),
                              ("mode", "default"), ("provider_type", "openai_chat"),
                              ("selection", None), ("resolved", None)):
            value = deepcopy(original)
            value[name] = content
            bad.append(reseal(value))
        value = deepcopy(original)
        value["extra"] = True
        bad.append(reseal(value))
        for section in ("selection", "resolved"):
            value = deepcopy(original)
            value[section]["extra"] = "unknown"
            bad.append(reseal(value))
        for name in ("model", "reasoning_effort", "provider"):
            value = deepcopy(original)
            value["resolved"].pop(name)
            bad.append(reseal(value))
        for model in (None, "", "default", "unknown", "auto", "non ASCII 模型", "../model", True):
            value = deepcopy(original)
            value["resolved"]["model"] = model
            bad.append(reseal(value))
        for name in ("model", "reasoning_effort"):
            value = deepcopy(original)
            value["selection"][name] = None
            bad.append(reseal(value))
        value = deepcopy(original)
        value["selection"] = {"model": "another-model"}
        bad.append(reseal(value))
        value = deepcopy(original)
        value["selection"] = {"reasoning_effort": "max"}
        bad.append(reseal(value))
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValidityError):
                validate_codex_binding(value)

    def test_hash_tampering_effort_and_provider_changes_are_detected(self):
        for path, replacement in ((("configuration_sha256",), "0" * 64),
                                  (("execution_sha256",), "b" * 64),
                                  (("resolved", "reasoning_effort"), "max"),
                                  (("resolved", "provider", "id"), "another"),
                                  (("resolved", "provider", "destination"), "https://other.invalid")):
            value = binding()
            target = value
            for part in path[:-1]:
                target = target[part]
            target[path[-1]] = replacement
            with self.subTest(path=path), self.assertRaises(ValidityError):
                validate_codex_binding(value)
        for value in ("A" * 64, "0" * 63, "", None):
            invalid = binding()
            invalid["execution_sha256"] = value
            with self.assertRaises(ValidityError):
                validate_codex_binding(reseal(invalid))

    def test_provider_identity_and_destination_are_explicit_and_safe(self):
        for identity in (None, "", "default", "unknown", "non ascii 身份", "a/b", "a" * 129):
            value = binding()
            value["resolved"]["provider"]["id"] = identity
            with self.subTest(identity=identity), self.assertRaises(ValidityError):
                validate_codex_binding(reseal(value))
        for destination in (None, "", "http://example.invalid", "https://user:secret@example.invalid",
                            "https://example.invalid?", "https://example.invalid#", "https://example.invalid:444",
                            "https://localhost", "https://127.0.0.1", "https://[::1]", "https://10.0.0.1",
                            "https://example.invalid/../secret", "https://example.invalid/%2e%2e/secret",
                            "https://example.invalid/\\secret", "https://example.invalid/\nsecret"):
            value = binding()
            value["resolved"]["provider"]["destination"] = destination
            with self.subTest(destination=destination), self.assertRaises(ValidityError):
                validate_codex_binding(reseal(value))
        value = binding()
        value["resolved"]["provider"]["destination"] = "https://example.invalid:443/v1"
        self.assertEqual(validate_codex_binding(reseal(value)), value)


class CodexApprovalTests(ProjectFixtureMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.round = OnlineRound.initialize(self.service,
            allowed_sha256=[hashlib.sha256(b"unrelated public source").hexdigest()], materials="public_or_synthetic")
        self.messages = [{"role": "user", "content": "Synthetic exact prompt."}]

    def test_legacy_approval_bytes_hashes_and_versions_remain_exact(self):
        request = {"purpose": "review", "messages": self.messages}
        expected = {"version": "online-text-approval-1", "service_id": self.service.config["service_id"],
                    "round_sha256": digest(self.round.config), "materials": "public_or_synthetic", **request}
        legacy = {"provider_type": "codex_cli", "mode": "live", "model": "legacy-model",
                  "configuration_sha256": "0" * 64}
        for provider in (None, legacy):
            value = deepcopy(expected)
            if provider is not None:
                value.update(version="online-text-approval-2", provider=provider)
            self.assertEqual(self.round._approval_value(request, provider), value)
            OnlineRound.approve_text(self.service, messages=self.messages, purpose="review",
                                     materials="public_or_synthetic", provider=provider)
            with self.service._operator():
                raw = self.service.parts["research"].read_bytes(digest(value) + ".json", max_bytes=65536)
            self.assertEqual(raw, canonical_json(value))
            self.assertTrue(self.round.text_approved(self.messages, "review", provider=provider))
        invalid = {**legacy, "reasoning_effort": "high"}
        with self.assertRaises(ValidityError):
            self.round.text_approved(self.messages, "review", provider=invalid)

    def test_versioned_approval_binds_all_resolution_facts_without_allowlist_fallback(self):
        provider = binding(mode="live")
        self.assertFalse(self.round.text_approved(self.messages, "review", provider=provider))
        OnlineRound.approve_text(self.service, messages=self.messages, purpose="review",
                                 materials="public_or_synthetic", provider=provider)
        self.assertEqual(self.round._approval_value({"purpose": "review", "messages": self.messages}, provider)["version"],
                         "online-text-approval-3")
        self.assertTrue(self.round.text_approved(self.messages, "review", provider=provider))
        for changed in (binding(mode="live", effort="max"), binding(mode="live", provider="another"),
                        binding(mode="live", selection={"model": "synthetic-model"})):
            self.assertFalse(self.round.text_approved(self.messages, "review", provider=changed))
        self.assertFalse(self.round.text_approved(self.messages, "generation", provider=provider))
        self.assertEqual(self.round.usage()["used"], {})
        unknown = {**provider, "version": "unknown"}
        with self.assertRaises(ValidityError):
            self.round.text_approved(self.messages, "review", provider=unknown)


class SyntheticNativeProvider:
    mode = "mock"
    endpoint = "https://example.invalid/v1"

    def __init__(self):
        self.facts = binding()
        self.calls = 0
        self.before_hook = None
        self.after_hook = None
        self.response_hook = None
        self.blocked = False

    def binding(self):
        # Deliberately return shared mutable state: the pipeline must detach it.
        return self.facts

    def preflight(self):
        return {"run_enabled": not self.blocked}

    def complete(self, messages, *, purpose, cancel_event=None, before_send=None):
        if self.before_hook:
            self.before_hook()
        if before_send:
            before_send()
        self.calls += 1
        content = ({"proof_markdown": "Synthetic candidate.", "dependency_revision_ids": ["assumption:0"]}
            if purpose == "generation" else {"verdict": "wrong", "verification_report": {"summary": "Synthetic gap.",
                "critical_errors": [], "gaps": [{"location": "draft", "issue": "Synthetic missing argument."}]},
                "repair_hints": "Supply the argument."})
        resolved, selection = self.facts["resolved"], self.facts["selection"]
        result = {"content": canonical_json(content).decode(), "mode": "mock", "model": resolved["model"],
            "requested_model": selection.get("model"), "requested_reasoning_effort": selection.get("reasoning_effort"),
            "resolved_model": resolved["model"], "resolved_reasoning_effort": resolved["reasoning_effort"],
            "reported_model": None, "reported_reasoning_effort": None, "usage": {}, "response_sha256": digest(content)}
        if self.after_hook:
            self.after_hook()
        if self.response_hook:
            self.response_hook(result)
        return result


class CodexResearchTests(ProjectFixtureMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.generator, self.reviewer = SyntheticNativeProvider(), SyntheticNativeProvider()
        self.pipeline = ResearchPipeline(self.service, RetrievalService(self.service),
            generation_provider=self.generator, review_provider=self.reviewer)

    def run_native(self, request_id="native-research"):
        return self.pipeline.run(self.runner, request_id=request_id, question="Synthetic question.",
                                 selected_refs=[self.ref("proof:0")])

    def test_native_facts_are_retained_in_operation_report_and_result_without_rewriting_replay(self):
        first = self.run_native()
        result = first["historical"]
        self.assertEqual(result["providers"]["generation"], self.generator.facts)
        self.assertEqual(result["providers"]["review"]["selection"], {})
        self.assertEqual(result["review_model"], "synthetic-model")
        with self.service._operator() as (_, state):
            retained = next(item for item in state["requests"].values() if item["request_id"] == "native-research")
            operation = self.service._operation(retained)
        self.assertEqual(operation["request"]["provider"], result["providers"])
        context = self.call("get_revision", self.reader, result["review"])
        revision = next(item for item in context["dto"]["revisions"] if item["revision_id"] == result["review"]["revision_id"])
        self.assertEqual(strict_json_loads(revision["body"])["provider"], result["providers"])
        facts = {purpose: {"selection": {}, "resolved": deepcopy(provider.facts["resolved"]),
                          "reported_model": None, "reported_reasoning_effort": None}
                 for purpose, provider in (("generation", self.generator), ("review", self.reviewer))}
        self.assertEqual(result["native_settings"], facts)
        self.assertEqual(strict_json_loads(revision["body"])["native_settings"], facts)
        self.assertEqual(self.run_native()["historical"], result)
        self.assertEqual((self.generator.calls, self.reviewer.calls), (1, 1))
        self.generator.facts = binding(effort="max")
        with self.assertRaises(ValidityConflict):
            self.run_native()
        self.assertEqual(self.pipeline.status(self.runner, request_id="native-research")["historical"], result)

    def test_binding_is_detached_and_drift_is_rejected_before_approval(self):
        frozen = self.pipeline._provider_binding("generation")
        self.generator.facts["resolved"]["reasoning_effort"] = "max"
        reseal(self.generator.facts)
        self.assertEqual(frozen["resolved"]["reasoning_effort"], "high")
        with self.assertRaises(ValidityConflict):
            self.pipeline._approval(None, None, {}, [], "generation", None, frozen)
        self.assertEqual(self.generator.calls, 0)

    def test_presend_drift_blocks_and_started_request_is_never_resent(self):
        original = deepcopy(self.generator.facts)
        self.generator.before_hook = lambda: setattr(self.generator, "facts", binding(effort="max"))
        with self.assertRaises(ValidityConflict):
            self.run_native()
        self.assertEqual(self.generator.calls, 0)
        self.generator.facts = original
        self.generator.before_hook = None
        with self.assertRaises(ValidityRecoveryRequired):
            self.run_native()
        self.assertEqual(self.generator.calls, 0)

    def test_postsend_drift_prevents_response_retention(self):
        self.generator.after_hook = lambda: setattr(self.generator, "facts", binding(provider="another"))
        with self.assertRaises(ValidityConflict):
            self.run_native()
        result = self.pipeline.status(self.runner, request_id="native-research")
        self.assertEqual(result["request_status"], "started")
        self.assertNotIn("generation_sha256", result["historical"])
        self.assertEqual(self.generator.calls, 1)

    def test_response_effective_and_reported_facts_must_match_binding(self):
        for name, value in (("model", "wrong"), ("resolved_model", "wrong"),
                            ("resolved_reasoning_effort", "max"), ("requested_model", "synthetic-model"),
                            ("requested_reasoning_effort", "high"), ("reported_model", "synthetic-model"),
                            ("reported_reasoning_effort", "high")):
            self.generator.response_hook = lambda result, name=name, value=value: result.update({name: value})
            request_id = "mismatch-" + name
            with self.subTest(name=name), self.assertRaises(ValidityConflict):
                self.run_native(request_id)
            state = self.pipeline.status(self.runner, request_id=request_id)
            self.assertEqual(state["request_status"], "started")
            self.assertNotIn("generation_sha256", state["historical"])
            with self.assertRaises(ValidityRecoveryRequired):
                self.run_native(request_id)

    def test_missing_response_resolution_fact_is_not_invented(self):
        self.generator.response_hook = lambda result: result.pop("resolved_reasoning_effort")
        with self.assertRaises(ValidityConflict):
            self.run_native()
        self.assertEqual(self.generator.calls, 1)
        with self.assertRaises(ValidityRecoveryRequired):
            self.run_native()
        self.assertEqual(self.generator.calls, 1)

    def test_generation_drift_during_review_is_rejected_before_report(self):
        self.reviewer.after_hook = lambda: setattr(self.generator, "facts", binding(effort="max"))
        with self.assertRaises(ValidityConflict):
            self.run_native()
        result = self.pipeline.status(self.runner, request_id="native-research")["historical"]
        self.assertNotIn("report_sha256", result)
        self.assertEqual((self.generator.calls, self.reviewer.calls), (1, 1))

    def test_unknown_default_preflight_blocks_before_permanent_request(self):
        self.generator.blocked = True
        with self.assertRaises(Exception) as blocked:
            self.run_native()
        self.assertEqual(type(blocked.exception).__name__, "CodexUnavailable")
        self.assertEqual(str(blocked.exception), "configured provider is blocked before model dispatch")
        self.assertEqual(self.pipeline.history(self.runner)["total"], 0)
        self.assertEqual(self.generator.calls, 0)
