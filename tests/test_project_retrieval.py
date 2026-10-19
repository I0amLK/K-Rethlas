"""Authorized project retrieval integration; synthetic/cache replay, no live API."""
from __future__ import annotations

from copy import deepcopy
import unittest
from unittest import mock

from rethlas_core.authorization import AuthorizationError
from rethlas_core.context_views import exact_ref
from rethlas_core.embedding import DIMENSIONS, MODEL, EmbeddingCacheMiss, EmbeddingProvider, cache_key
from rethlas_core.managed_memory import ManagedMemory
from rethlas_core.private_fs import canonical_json
from rethlas_core.project_service import ProjectService
from rethlas_core.research_records import candidate
from rethlas_core.retrieval import RetrievalService
from rethlas_core.retrieval_index import RetrievalIndexError
from rethlas_core.validity import ValidityError, ValidityConflict
from test_project_service import ProjectFixtureMixin


def synthetic_vector():
    # Deterministic test protocol vector, never presented as a real BGE embedding.
    return [1.0] + [0.0] * (DIMENSIONS - 1)


class SyntheticProvider:
    mode = "bge_cache_only"
    model = MODEL
    dimensions = DIMENSIONS

    def __init__(self, hook=None):
        self.calls = []
        self.hook = hook

    def embed(self, texts, *, kind="documents", cancel_event=None):
        self.calls.append((list(texts), kind))
        if self.hook is not None:
            self.hook(kind)
        return [synthetic_vector() for _ in texts]


class ProjectRetrievalTests(ProjectFixtureMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.retrieval = RetrievalService(self.service)

    def build(self, retrieval=None):
        value = self.retrieval if retrieval is None else retrieval
        return value.build_index(self.writer["token"], run_id=self.writer["run_id"])

    def search(self, query, retrieval=None, **kwargs):
        value = self.retrieval if retrieval is None else retrieval
        return value.search(self.reader["token"], query, run_id=self.reader["run_id"], **kwargs)

    def append_candidate(self, label, body, kind="Claim"):
        entity, revision = candidate("new." + label, "new." + label + ".0", kind, label, body)
        head = self.call("status", self.reader)["binding"]["graph"]["head"]
        request = {"version": 1, "request_id": "append." + label, "expected_head": head,
                   "entities": [entity], "revisions": [revision], "relations": [], "evidence": []}
        self.call("commit", self.writer, request)
        return exact_ref(revision)

    def frozen_source_bytes(self):
        return {str(path.relative_to(self.service.root / "memory")): path.read_bytes()
                for directory in ("bodies", "graphs") for path in (self.service.root / "memory" / directory).iterdir()
                if path.is_file()}

    def test_lexical_search_returns_exact_body_complete_dependencies_and_current_state(self):
        built = self.build()
        self.assertEqual(built["mode"], "lexical")
        self.assertFalse(built["knowledge_authority"])
        result = self.search("合成证明草稿", filters={"types": ["ProofArtifact"]})
        self.assertEqual(result["retrieval_mode"], "lexical")
        self.assertEqual(result["count"], 1)
        hit = result["results"][0]
        self.assertEqual(hit["ref"], self.ref("proof:0"))
        revisions = {item["revision_id"]: item for item in hit["context"]["dto"]["revisions"]}
        self.assertTrue({"proof:0", "claim:0", "definition:0", "assumption:0", "source:0"} <= set(revisions))
        for key in revisions:
            original = next(item for item in self.graph["revisions"] if item["revision_id"] == key)
            self.assertEqual(revisions[key]["body"], original["body"])
            self.assertEqual(revisions[key]["body_sha256"], original["body_sha256"])
        self.assertTrue(hit["context"]["overlay"]["input_eligible"])
        self.assertEqual(hit["context"]["assertion_state"], "candidate")
        self.assertFalse(result["mathematical_correctness_proved"])
        self.assertNotIn("PRIVATE_METADATA_ACTOR", canonical_json(result).decode())

    def test_whole_project_visibility_is_explicit_not_problem_row_acl(self):
        managed = ManagedMemory(self.service, self.retrieval, reader=self.reader, writer=self.writer, indexer=self.writer)
        first = managed.append("数学/连续性", "proof_steps", {"text": "唯一共同词 条件甲"}, request_id="record.a")
        second = managed.append("数学/一致连续性", "proof_steps", {"text": "唯一共同词 条件乙"}, request_id="record.b")
        self.build()
        unfiltered = self.search("唯一共同词")
        ids = {hit["ref"]["revision_id"] for hit in unfiltered["results"]}
        self.assertTrue({first["ref"]["revision_id"], second["ref"]["revision_id"]} <= ids)
        filtered = managed.search("数学/连续性", "唯一共同词", channels=["proof_steps"])
        self.assertEqual([hit["ref"] for hit in filtered["results"]], [first["ref"]])
        self.assertEqual(filtered["results"][0]["metadata"]["problem_id"], "数学/连续性")

    def test_events_only_reference_is_searchable_and_not_promoted(self):
        managed = ManagedMemory(self.service, self.retrieval, reader=self.reader, writer=self.writer, indexer=self.writer)
        appended = managed.append("分析/来源", "events", {"source": "synthetic public source", "text": "阿尔法唯一来源"},
                                  request_id="event.only")
        self.build()
        result = managed.search("分析/来源", "阿尔法唯一来源", channels=["events"])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["ref"], appended["ref"])
        self.assertEqual(result["results"][0]["type"], "Reference")
        self.assertEqual(result["results"][0]["verification_kind"], "unknown")

    def test_support_withdrawal_excludes_dependent_proof_without_touching_index_or_history(self):
        self.build()
        index_before = (self.service.root / "index/current.json").read_bytes()
        source_before = self.frozen_source_bytes()
        self.assertEqual(self.search("证明草稿", filters={"types": ["ProofArtifact"]})["count"], 1)
        self.withdraw("assumption:0")
        result = self.search("证明草稿", filters={"types": ["ProofArtifact"]})
        self.assertEqual(result["count"], 0)
        self.assertEqual((self.service.root / "index/current.json").read_bytes(), index_before)
        self.assertEqual(self.frozen_source_bytes(), source_before)
        self.assertTrue(self.call("status", self.reader, refs=[self.ref("proof:0")])["current"]["proof:0"]["math_blocked"])

    def test_citation_only_withdrawal_keeps_claim_and_historical_reference_bytes(self):
        self.build()
        self.withdraw("source:0")
        result = self.search("合成命题成立", filters={"types": ["Claim"]})
        self.assertEqual(result["count"], 1)
        context = result["results"][0]["context"]
        self.assertTrue(context["overlay"]["input_eligible"])
        reference = next(item for item in context["dto"]["revisions"] if item["revision_id"] == "source:0")
        self.assertEqual(reference["body"], next(item["body"] for item in self.graph["revisions"] if item["revision_id"] == "source:0"))
        self.assertEqual(reference["use"], "historical_citation_only")
        citation = next(item for item in context["overlay"]["citation_only"] if item["revision_id"] == "source:0")
        self.assertTrue(citation["withdrawn"])
        self.assertTrue(citation["not_authorized_as_premise"])

    def test_directly_withdrawn_root_is_not_recommended(self):
        self.build()
        self.withdraw("claim:0")
        self.assertEqual(self.search("合成命题成立", filters={"types": ["Claim"]})["count"], 0)
        with self.assertRaises(ValidityError):
            self.call("get_revision", self.reader, self.ref("claim:0"))

    def test_cross_project_token_is_rejected_before_index_or_provider_access(self):
        other = ProjectService.initialize(self.base / "other", owner="owner")
        provider = SyntheticProvider()
        retrieval = RetrievalService(other, provider=provider)
        with mock.patch.object(retrieval, "_read", side_effect=AssertionError("must authorize before index")):
            with self.assertRaises(AuthorizationError):
                retrieval.search(self.reader["token"], "连续", run_id=self.reader["run_id"])
        with mock.patch.object(other, "_store", side_effect=AssertionError("must authorize before graph")):
            with self.assertRaises(AuthorizationError):
                retrieval.build_index(self.writer["token"], run_id=self.writer["run_id"])
        self.assertEqual(provider.calls, [])

    def test_reader_cannot_build_and_revoked_reader_cannot_query(self):
        with self.assertRaises(AuthorizationError):
            self.retrieval.build_index(self.reader["token"], run_id=self.reader["run_id"])
        self.build()
        claims = self.service.authority.authenticate(self.reader["token"], action="project.search")
        self.service.authority.revoke(claims["token_id"])
        with mock.patch.object(self.retrieval, "_read", side_effect=AssertionError("authorize first")):
            with self.assertRaises(AuthorizationError):
                self.search("连续")

    def test_stale_graph_index_requires_explicit_rebuild(self):
        built = self.build()
        # Use a genuinely unique lexical token: Chinese character/bigram
        # retrieval intentionally also matches common characters in other text.
        query = "zzunindexedretrievaltoken"
        reference = self.append_candidate("after-index", query)
        with self.assertRaises(RetrievalIndexError):
            self.search(query)
        rebuilt = self.build()
        self.assertNotEqual(built["binding"]["graph"]["head"], rebuilt["binding"]["graph"]["head"])
        result = self.search(query)
        self.assertEqual([hit["ref"] for hit in result["results"]], [reference])

    def test_deletion_and_rebuild_preserve_graph_and_authorized_results(self):
        self.build()
        original = self.search("合成证明草稿", filters={"types": ["ProofArtifact"]})
        source_before = self.frozen_source_bytes()
        status_before = self.call("status", self.reader)
        (self.service.root / "index/current.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.search("合成证明草稿")
        self.build()
        rebuilt = self.search("合成证明草稿", filters={"types": ["ProofArtifact"]})
        self.assertEqual([hit["ref"] for hit in rebuilt["results"]], [hit["ref"] for hit in original["results"]])
        self.assertEqual(self.call("status", self.reader)["binding"], status_before["binding"])
        self.assertEqual(self.frozen_source_bytes(), source_before)

    def test_corrupt_or_wrong_project_index_is_not_a_knowledge_authority(self):
        self.build()
        path = self.service.root / "index/current.json"
        original = path.read_bytes()
        path.write_bytes(b'{"malformed":true}')
        path.chmod(0o600)
        with self.assertRaises(RetrievalIndexError):
            self.search("连续")
        path.write_bytes(original)
        other = ProjectService.initialize(self.base / "other", owner="owner")
        other_reader = other.issue(role="project_reader")
        other.parts["index"].write_bytes("current.json", original)
        with self.assertRaises(RetrievalIndexError):
            RetrievalService(other).search(other_reader["token"], "连续", run_id=other_reader["run_id"])

    def test_cache_miss_build_explicitly_uses_lexical_or_refuses(self):
        cache = self.base / "empty-cache"
        cache.mkdir(mode=0o700)
        provider = EmbeddingProvider(cache_dir=cache)
        retrieval = RetrievalService(self.service, provider=provider)
        with mock.patch("rethlas_core.embedding._http", side_effect=AssertionError("external request forbidden")):
            result = self.build(retrieval)
            self.assertEqual(result["mode"], "lexical")
            self.assertTrue(result["cache_miss_lexical_fallback"])
            self.assertEqual(self.search("连续", retrieval)["retrieval_mode"], "lexical")
            strict = RetrievalService(self.service, provider=provider, lexical_fallback=False)
            with self.assertRaises(EmbeddingCacheMiss):
                self.build(strict)

    def test_exact_cache_adapter_enters_application_index_path_without_external_call(self):
        # Cache rows below are protocol fixtures, not claimed real BGE quality.
        cache = self.base / "synthetic-cache"
        cache.mkdir(mode=0o700)
        for revision in self.graph["revisions"]:
            text = revision["body"]
            (cache / (cache_key(text) + ".json")).write_bytes(canonical_json({"model": MODEL,
                "input_sha256": cache_key(text), "text": text, "vector": synthetic_vector()}))
        query = "固定协议查询"
        (cache / (cache_key(query) + ".json")).write_bytes(canonical_json({"model": MODEL,
            "input_sha256": cache_key(query), "text": query, "vector": synthetic_vector()}))
        retrieval = RetrievalService(self.service, provider=EmbeddingProvider(cache_dir=cache))
        with mock.patch("rethlas_core.embedding._http", side_effect=AssertionError("external request forbidden")):
            built = self.build(retrieval)
            self.assertEqual(built["mode"], "bge_cache_only")
            result = self.search(query, retrieval, filters={"types": ["Claim"]})
        self.assertEqual(result["retrieval_mode"], "bge_cache_only")
        self.assertIsNone(result["fallback_reason"])
        self.assertEqual([hit["ref"] for hit in result["results"]], [self.ref("claim:0")])

    def test_query_cache_miss_is_labeled_lexical_not_bge(self):
        class QueryMiss(SyntheticProvider):
            def embed(self, texts, *, kind="documents", cancel_event=None):
                if kind == "queries":
                    raise EmbeddingCacheMiss("synthetic missing query")
                return super().embed(texts, kind=kind, cancel_event=cancel_event)
        retrieval = RetrievalService(self.service, provider=QueryMiss())
        self.build(retrieval)
        result = self.search("合成命题成立", retrieval, filters={"types": ["Claim"]})
        self.assertEqual(result["retrieval_mode"], "lexical")
        self.assertEqual(result["fallback_reason"], "cache_miss")
        self.assertEqual(result["count"], 1)
        strict = RetrievalService(self.service, provider=QueryMiss(), lexical_fallback=False)
        with self.assertRaises(EmbeddingCacheMiss):
            self.search("连续", strict)

    def test_withdrawal_during_embedding_is_checked_before_delivering_results(self):
        provider = SyntheticProvider()
        retrieval = RetrievalService(self.service, provider=provider)
        self.build(retrieval)
        provider.hook = lambda kind: self.withdraw("assumption:0") if kind == "queries" else None
        result = self.search("synthetic query", retrieval, filters={"types": ["ProofArtifact"]})
        self.assertEqual(result["count"], 0)
        self.assertEqual(len(provider.calls), 2)

    def test_citation_withdrawal_during_embedding_returns_fresh_overlay_without_math_block(self):
        provider = SyntheticProvider()
        retrieval = RetrievalService(self.service, provider=provider)
        self.build(retrieval)
        provider.hook = lambda kind: self.withdraw("source:0") if kind == "queries" else None
        result = self.search("synthetic query", retrieval, filters={"types": ["Claim"]})
        self.assertEqual(result["count"], 1)
        overlay = result["results"][0]["context"]["overlay"]
        self.assertTrue(overlay["input_eligible"])
        self.assertTrue(next(item for item in overlay["citation_only"] if item["revision_id"] == "source:0")["withdrawn"])

    def test_revocation_or_graph_change_during_embedding_cannot_deliver_old_result(self):
        provider = SyntheticProvider()
        retrieval = RetrievalService(self.service, provider=provider)
        self.build(retrieval)
        claims = self.service.authority.authenticate(self.reader["token"], action="project.search")
        provider.hook = lambda kind: self.service.authority.revoke(claims["token_id"]) if kind == "queries" else None
        with self.assertRaises(AuthorizationError):
            self.search("synthetic query", retrieval)

    def test_graph_change_during_query_or_build_refuses_stale_projection(self):
        provider = SyntheticProvider()
        retrieval = RetrievalService(self.service, provider=provider)
        self.build(retrieval)
        provider.hook = lambda kind: self.append_candidate("during-query", "fresh candidate") if kind == "queries" else None
        with self.assertRaises(ValidityConflict):
            self.search("synthetic query", retrieval)
        provider.hook = lambda kind: self.append_candidate("during-build", "other candidate") if kind == "documents" else None
        with self.assertRaises(ValidityConflict):
            self.build(retrieval)

    def test_search_does_not_consume_fixed_use_slots(self):
        self.build()
        with self.service._operator() as (_, state):
            old_uses, old_sequence = deepcopy(state["uses"]), state["sequence"]
        for _ in range(10):
            self.search("合成命题成立", filters={"types": ["Claim"]})
        with self.service._operator() as (_, state):
            self.assertEqual(state["uses"], old_uses)
            self.assertEqual(state["sequence"], old_sequence)

    def test_invalid_queries_limits_and_filter_shapes_fail_explicitly(self):
        self.build()
        for query in ("", " ", "x" * 8193):
            with self.subTest(query_length=len(query)), self.assertRaises(ValidityError):
                self.search(query)
        for limit in (True, 0, 21, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValidityError):
                self.search("连续", top_k=limit)
        for filters in ({"unknown": True}, {"types": "Claim"}, {"channels": "events"},
                        {"types": ["Unknown"]}, {"problem_id": 1}, {"channels": ["Unknown"]}):
            with self.subTest(filters=filters), self.assertRaises(ValidityError):
                self.search("连续", filters=filters)

    def test_untrusted_managed_marker_is_not_a_crashing_or_trusted_metadata_envelope(self):
        reference = self.append_candidate("bad-envelope", '{"version":"managed-record-1","text":"zzmalformedmarkertoken"}')
        self.build()
        result = self.search("zzmalformedmarkertoken")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["ref"], reference)
        self.assertEqual(result["results"][0]["metadata"], {})

    def test_uncommitted_revision_or_stale_body_hash_cannot_be_read_as_current(self):
        self.build()
        unknown = {**self.ref("claim:0"), "revision_id": "unknown:0"}
        wrong = {**self.ref("claim:0"), "body_sha256": "0" * 64}
        for ref in (unknown, wrong):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                self.call("get_revision", self.reader, ref)


if __name__ == "__main__":
    unittest.main()
