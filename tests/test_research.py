from __future__ import annotations

from copy import deepcopy
import unittest
from unittest import mock

from rethlas_core.private_fs import canonical_json, digest, strict_json_loads
from rethlas_core.provider import TextProvider
from rethlas_core.research import ResearchPipeline
from rethlas_core.retrieval import RetrievalService
from rethlas_core.online_round import OnlineRound
from rethlas_core.validity import ValidityError, ValidityCapacity, ValidityConflict, ValidityRecoveryRequired
from test_project_service import ProjectFixtureMixin


MODEL = "xiaoliu/gpt-6.1-sol:max"


def completion(content):
    return canonical_json({"id": "mock-completion", "object": "chat.completion", "created": 1, "model": MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}})


class ResearchTests(ProjectFixtureMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.calls = []
        def transport(endpoint, data, headers, **kwargs):
            payload = strict_json_loads(data)
            self.calls.append(payload["messages"])
            if len(self.calls) % 2:
                return completion(canonical_json({"proof_markdown": "SYNTHETIC draft using the supplied compactness assumption.",
                    "dependency_revision_ids": ["assumption:0"]}).decode())
            return completion(canonical_json({"verdict": "wrong", "verification_report": {"summary": "Synthetic incomplete argument.", "critical_errors": [],
                "gaps": [{"location": "draft", "issue": "Synthetic draft omits the compactness argument."}]},
                "repair_hints": "Supply the missing argument."}).decode())
        self.provider = TextProvider(mode="mock", endpoint="https://example.invalid/v1/chat/completions", model=MODEL,
                                     admit=lambda messages, purpose: None, transport=transport)
        self.pipeline = ResearchPipeline(self.service, RetrievalService(self.service), self.provider)
        self.question = "证明一个合成紧致连续性结论。"

    def run_research(self, request_id="research-one"):
        return self.pipeline.run(self.runner, request_id=request_id, question=self.question,
                                 selected_refs=[self.ref("proof:0")])

    def test_full_mock_pipeline_retains_candidate_and_fresh_review(self):
        value = self.run_research()
        self.assertEqual(value["request_status"], "accepted")
        self.assertEqual(value["historical"]["mode"], "mock")
        self.assertEqual(value["historical"]["verdict"], "wrong")
        self.assertFalse(value["historical"]["mathematical_correctness_proved"])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.calls[1]), 2)
        self.assertNotIn("GENERATOR_PRIVATE_CONFIDENCE", canonical_json(self.calls[1]).decode())
        self.assertNotEqual(self.calls[0][0], self.calls[1][0])
        ref = value["historical"]["proof"]
        context = self.call("get_revision", self.reader, ref)
        draft = next(r for r in context["dto"]["revisions"] if r["revision_id"] == ref["revision_id"])
        self.assertEqual(draft["assertion_state"], "candidate")
        self.assertEqual(draft["validity"], "needs_review")
        from rethlas_core.project_service import ProjectService
        restarted = ResearchPipeline(ProjectService(self.service.root), RetrievalService(self.service), self.provider)
        self.assertEqual(restarted.status(self.runner, request_id="research-one")["historical"], value["historical"])

    def test_withdrawal_after_research_changes_current_not_historical(self):
        value = self.run_research()
        self.withdraw("assumption:0")
        later = self.pipeline.status(self.runner, request_id="research-one")
        self.assertEqual(value["historical"], later["historical"])
        proof_id = value["historical"]["proof"]["revision_id"]
        self.assertTrue(later["current"]["current"][proof_id]["math_blocked"])

    def test_exact_replay_never_automatically_sends_again(self):
        first = self.run_research()
        second = self.run_research()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(first["historical"], second["historical"])
        with self.assertRaises(ValidityConflict):
            self.pipeline.run(self.runner, request_id="research-one", question="changed", selected_refs=[self.ref("proof:0")])

    def test_lost_external_response_remains_started_without_automatic_resend(self):
        from rethlas_core.provider import ProviderTimeout
        with mock.patch.object(self.provider, "complete", side_effect=ProviderTimeout("synthetic timeout")):
            with self.assertRaises(ProviderTimeout):
                self.run_research()
        with self.assertRaises(ValidityRecoveryRequired):
            self.run_research()
        self.assertEqual(len(self.calls), 0)

    def test_retained_invalid_result_is_rejected_without_another_request(self):
        original = self.provider.complete
        def invalid(*args, **kwargs):
            value = original(*args, **kwargs)
            value["content"] = "{}"
            return value
        with mock.patch.object(self.provider, "complete", side_effect=invalid):
            with self.assertRaises(ValidityError):
                self.run_research()
        with self.assertRaises(ValidityError):
            self.run_research()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.pipeline.status(self.runner, request_id="research-one")["historical"]["stage"], "generation_received")

    def test_explicit_revision_retains_parent_and_adds_distinct_candidate(self):
        parent = self.run_research()
        revised = self.pipeline.revise(self.runner, request_id="revision-one", parent_request_id="research-one",
                                       instructions="Explicitly address the missing compactness argument.")
        self.assertEqual(len(self.calls), 4)
        self.assertNotEqual(parent["historical"]["proof"], revised["historical"]["proof"])
        self.assertEqual(parent["historical"], self.pipeline.status(self.runner, request_id="research-one")["historical"])
        self.assertIn("Supply the missing argument.", canonical_json(self.calls[2]).decode())
        self.assertNotIn("guidance", canonical_json(self.calls[3]).decode())
        with self.assertRaises(ValidityConflict):
            self.pipeline.revise(self.runner, request_id="revision-one", parent_request_id="research-one", instructions="changed")
        self.assertEqual(len(self.calls), 4)

    def test_explicit_review_sends_only_one_fresh_request(self):
        parent = self.run_research()
        from rethlas_core.research import REVIEW_INSTRUCTION
        result = {"content": canonical_json({"verdict": "wrong", "verification_report": {"summary": "Synthetic incomplete argument.", "critical_errors": [],
            "gaps": [{"location": "draft", "issue": "Synthetic gap."}]}, "repair_hints": "Complete it."}).decode(),
            "model": MODEL, "mode": "mock", "usage": {}, "response_sha256": digest("fresh synthetic review")}
        with mock.patch.object(self.provider, "complete", return_value=result) as send:
            value = self.pipeline.review(self.runner, request_id="fresh-review", proof_ref=parent["historical"]["proof"],
                                         statement=self.question)
        send.assert_called_once()
        self.assertEqual(send.call_args.kwargs["purpose"], "review")
        self.assertEqual(send.call_args.args[0][0]["content"], REVIEW_INSTRUCTION)
        self.assertEqual(value["historical"]["proof"], parent["historical"]["proof"])
        self.assertNotEqual(value["historical"]["review"], parent["historical"]["review"])

    def test_live_mode_waits_for_exact_operator_prompt_approval(self):
        import hashlib
        approved = OnlineRound.initialize(self.service,
            allowed_sha256=[hashlib.sha256(b"unrelated public material").hexdigest()], materials="public_or_synthetic")
        self.provider.mode = "live"
        self.provider.admit = approved.admit_text
        def unsent_grant(messages, *, purpose, cancel_event=None, before_send=None):
            approved.admit_text(messages, purpose)
            if before_send is not None:
                before_send()
            self.calls.append(messages)
            content = (canonical_json({"proof_markdown": "Synthetic compactness draft.",
                        "dependency_revision_ids": ["assumption:0"]}).decode() if purpose == "generation" else
                       canonical_json({"verdict": "wrong", "verification_report": {"summary": "Synthetic incomplete argument.", "critical_errors": [],
                            "gaps": [{"location": "draft", "issue": "Synthetic argument is incomplete."}]},
                            "repair_hints": "Complete the argument."}).decode())
            return {"content": content, "model": MODEL, "mode": "live", "usage": {}, "response_sha256": digest(content)}
        # No network/credential is used: the trusted transport boundary is replaced.
        with mock.patch.object(self.provider, "complete", side_effect=unsent_grant):
            first = self.run_research()
            self.assertEqual(first["historical"]["stage"], "awaiting_generation_approval")
            self.assertEqual(self.calls, [])
            OnlineRound.approve_text(self.service, messages=first["approval_required"]["messages"],
                                     purpose="generation", materials="public_or_synthetic")
            second = self.run_research()
            self.assertEqual(second["historical"]["stage"], "awaiting_review_approval")
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(second["historical"]["proof"]["revision_id"].split(".")[0], "proof")
            OnlineRound.approve_text(self.service, messages=second["approval_required"]["messages"],
                                     purpose="review", materials="public_or_synthetic")
            final = self.run_research()
            self.assertEqual(final["request_status"], "accepted")
            self.assertEqual(len(self.calls), 2)
            self.assertEqual(approved.usage()["used"]["text_requests"], 2)
            self.run_research()
            self.assertEqual(len(self.calls), 2)

    def test_citation_only_dependency_cannot_be_promoted_by_model(self):
        original = self.provider.complete
        def cited_as_premise(*args, **kwargs):
            response = original(*args, **kwargs)
            response["content"] = canonical_json({"proof_markdown": "Synthetic draft.",
                                                   "dependency_revision_ids": ["source:0"]}).decode()
            return response
        with mock.patch.object(self.provider, "complete", side_effect=cited_as_premise):
            with self.assertRaises(ValidityError):
                self.run_research()
        self.assertEqual(len(self.calls), 1)
        with self.assertRaises(ValidityError):
            self.run_research()
        self.assertEqual(len(self.calls), 1)

    def test_correct_llm_report_does_not_promote_candidate_mathematical_state(self):
        original = self.provider.complete
        def correct(*args, **kwargs):
            response = original(*args, **kwargs)
            if kwargs["purpose"] == "review":
                response["content"] = canonical_json({"verdict": "correct", "verification_report": {
                    "summary": "Synthetic zero-finding report, not a proof certificate.", "critical_errors": [], "gaps": []},
                    "repair_hints": ""}).decode()
            return response
        with mock.patch.object(self.provider, "complete", side_effect=correct):
            value = self.run_research()
        self.assertEqual(value["historical"]["verdict"], "correct")
        self.assertFalse(value["historical"]["mathematical_correctness_proved"])
        for name in ("proof", "review"):
            context = self.call("get_revision", self.reader, value["historical"][name])
            revision = next(r for r in context["dto"]["revisions"] if r["revision_id"] == value["historical"][name]["revision_id"])
            self.assertEqual((revision["assertion_state"], revision["validity"]), ("candidate", "needs_review"))

    def test_same_stage_compare_and_swap_prevents_second_dispatch(self):
        key, binding = self.pipeline._begin(self.runner, {"request_id": "race", "kind": "run",
            "question": self.question, "selected_refs": [self.ref("proof:0")]}, None)
        self.pipeline._progress(self.runner, key, {"stage": "generation_started"}, previous={}, status="started")
        with self.assertRaises(ValidityConflict):
            self.pipeline._progress(self.runner, key, {"stage": "generation_started"}, previous={}, status="started")
        self.assertEqual(self.calls, [])

    def test_review_citation_withdrawal_obsoletes_report_not_proof(self):
        value = self.run_research()
        self.withdraw("source:0")
        later = self.pipeline.status(self.runner, request_id="research-one")
        proof = later["current"]["current"][value["historical"]["proof"]["revision_id"]]
        report = later["current"]["current"][value["historical"]["review"]["revision_id"]]
        self.assertFalse(proof["math_blocked"])
        self.assertTrue(report["review_input_obsolete"])
        with self.assertRaises(ValidityError):
            self.call("prepare_context", self.reader, [value["historical"]["review"]])

    def test_unrelated_commit_during_model_request_does_not_invalidate_exact_input(self):
        from rethlas_core.research_records import candidate
        original = self.provider.complete
        def completed(*args, **kwargs):
            value = original(*args, **kwargs)
            if len(self.calls) == 1:
                entity, revision = candidate("unrelated", "unrelated.1", "Reference", "Unrelated source", "Unrelated public mathematics.")
                self.pipeline._append(self.runner, "unrelated-during-generation", {"entities": [entity], "revisions": [revision],
                    "relations": [], "evidence": []})
            return value
        with mock.patch.object(self.provider, "complete", side_effect=completed):
            value = self.run_research()
        self.assertEqual(value["request_status"], "accepted")
        self.assertEqual(len(self.calls), 2)

    def test_withdrawal_during_provider_does_not_accept_stale_candidate(self):
        original = self.provider.complete
        def complete(*args, **kwargs):
            value = original(*args, **kwargs)
            self.withdraw("assumption:0")
            return value
        with mock.patch.object(self.provider, "complete", side_effect=complete):
            with self.assertRaises(ValidityError):
                self.run_research()
        self.assertEqual(len(self.calls), 1)
        with self.service._operator() as (_, state):
            binding = next(b for b in state["requests"].values() if b["request_id"] == "research-one")
            self.assertEqual(binding["status"], "started")


class OnlineRoundTests(ProjectFixtureMixin, unittest.TestCase):
    def test_explicit_allowlist_and_request_cap_are_durable(self):
        import hashlib
        text = "公开合成材料"
        allowed = [hashlib.sha256(text.encode()).hexdigest()]
        approved = OnlineRound.initialize(self.service, allowed_sha256=allowed, materials="public_or_synthetic")
        with self.assertRaises(ValidityError):
            approved.admit_embedding(["private-not-approved"], "documents")
        for _ in range(10):
            approved.admit_embedding([text], "documents")
        with self.assertRaises(ValidityCapacity):
            OnlineRound(self.service).admit_embedding([text], "queries")
        self.assertEqual(approved.usage()["used"]["embedding_requests"], 10)
        with self.assertRaises(ValidityConflict):
            OnlineRound.initialize(self.service, allowed_sha256=["0" * 64], materials="public_or_synthetic")

    def test_exact_staged_prompt_approval_never_resets_cap(self):
        import hashlib
        approved = OnlineRound.initialize(self.service,
            allowed_sha256=[hashlib.sha256(b"public source").hexdigest()], materials="public_or_synthetic")
        messages = [{"role": "user", "content": "A public synthetic review prompt created after generation."}]
        self.assertFalse(approved.text_approved(messages, "review"))
        result = OnlineRound.approve_text(self.service, messages=messages, purpose="review", materials="public_or_synthetic")
        self.assertTrue(result["limits_unchanged"])
        self.assertTrue(approved.text_approved(messages, "review"))
        self.assertFalse(approved.text_approved(messages, "generation"))
        self.assertFalse(approved.text_approved([{"role": "user", "content": messages[0]["content"] + "x"}], "review"))
        for _ in range(4):
            approved.admit_text(messages, "review")
        OnlineRound.approve_text(self.service, messages=messages, purpose="review", materials="public_or_synthetic")
        with self.assertRaises(ValidityCapacity):
            approved.admit_text(messages, "review")
        self.assertEqual(approved.usage()["used"]["text_requests"], 4)

    def test_text_budget_is_separate_and_failed_admissions_never_refunded(self):
        import hashlib
        text = "public synthetic prompt"
        approved = OnlineRound.initialize(self.service,
            allowed_sha256=[hashlib.sha256(text.encode()).hexdigest()], materials="public_or_synthetic")
        for _ in range(4):
            approved.admit_text([{"role": "user", "content": text}], "generation")
        with self.assertRaises(ValidityCapacity):
            approved.admit_text([{"role": "user", "content": text}], "review")
        self.assertEqual(approved.usage()["used"]["text_requests"], 4)
        self.assertNotIn("embedding_requests", approved.usage()["used"])
