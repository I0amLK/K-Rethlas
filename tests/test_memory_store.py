"""Synthetic, local-only MemoryStore contract tests.

These tests do not invoke the CLI, launch an application, or use network/model
services. Recovery's process targets import only the helpers explicitly named
from this module, never this module's TestCase classes. Execution is intentionally
left to a separately authorized test run.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest


from rethlas_core.authorization import AuthorizationError
from rethlas_core.memory_store import (
    DEFAULT_LIMITS, MemoryCancelled, MemoryCapacity, MemoryConflict,
    MemoryCorruption, MemoryRecoveryRequired, MemoryStore, MemoryStoreError,
)


ACTIONS = ["memory.read", "memory.commit", "memory.maintain"]
COLLECTION_IDS = {
    "entities": "entity_id", "revisions": "revision_id",
    "relations": "relation_id", "evidence": "evidence_id",
}
RUN_ID = "fixture-run"


def encoded(value):
    """Independent protocol encoder, not the store's implementation helper."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_private(path, data):
    """Create synthetic damage/orphans without making a public-mode fixture."""
    path = Path(path)
    path.write_bytes(data)
    path.chmod(0o600)


def provenance(actor="fixture-author"):
    return {"source": "synthetic-local-fixture", "actor": actor,
            "recorded_at": "2026-01-01T00:00:00Z"}


def request_fixture(head, *, label="one", request_id=None, body=None):
    """One record in every collection; a citation self-edge is not a premise."""
    body = body if body is not None else "# 合成命题\r\nα / β\n\nExact body.\n"
    entity_id, revision_id = "entity/" + label, "revision/" + label
    evidence_id = "evidence/" + label
    endpoint = {"entity_id": entity_id, "revision_id": revision_id}
    return {
        "version": 1, "request_id": request_id or "request/" + label,
        "expected_head": head,
        "entities": [{"entity_id": entity_id, "type": "Claim", "name": "同名命题",
                      "provenance": provenance()}],
        "revisions": [{"entity_id": entity_id, "revision_id": revision_id,
                       "body": body, "body_sha256": sha(body.encode("utf-8")),
                       "provenance": provenance(), "assertion_state": "candidate",
                       "validity": "needs_review", "context_revision_ids": [],
                       "evidence_ids": [evidence_id]}],
        "relations": [{"relation_id": "relation/" + label, "type": "cites",
                       "source": dict(endpoint), "target": dict(endpoint),
                       "provenance": provenance(), "assertion_state": "candidate",
                       "context_revision_ids": [], "evidence_ids": [evidence_id]}],
        "evidence": [{"evidence_id": evidence_id,
                      "locator": "fixture://not-dereferenced/" + label,
                      "sha256": sha(b"synthetic evidence, not a fetched resource"),
                      "provenance": provenance()}],
    }


def graph_from_requests(*requests):
    return {"schema_version": "1.0", **{
        name: [deepcopy(record) for request in requests for record in request[name]]
        for name in COLLECTION_IDS}}


def receipt_head(receipt):
    fields = {
        "format_version": "m2a-1", "store_id": receipt["binding"]["store_id"],
        "generation": receipt["generation"], "graph_sha256": receipt["graph_sha256"],
        "receipt_key": receipt["request_key"], "receipt_sha256": sha(encoded(receipt)),
        "parent_head": receipt["parent_manifest"]["head"],
    }
    return sha(encoded(fields))


class MutableClock:
    def __init__(self, value=None):
        self.value = time.time() if value is None else value

    def __call__(self):
        return self.value


class MemoryFixtureMixin:
    """No TestCase base: safe to import into a second discoverable test module."""

    def setUp(self):
        super().setUp()
        self.temporary = tempfile.TemporaryDirectory(prefix="memory-store-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.clock = MutableClock()
        self.store, self.token = self.make_store("primary")
        self.root = self.store.root
        self.initial_head = self.head()

    def make_store(self, name, *, limits=None, actions=None, owner="owner", domain="fixture"):
        store = MemoryStore.initialize(self.base / name, owner=owner, domain=domain, limits=limits)
        store.clock = self.clock
        token = store.issue(run_id=RUN_ID, actions=ACTIONS if actions is None else actions,
                            ttl_seconds=300)
        return store, token

    def head(self, store=None):
        return read_json((store or self.store).root / "manifest.json")["head"]

    def status(self, store=None, token=None):
        return (store or self.store).status(token or self.token, run_id=RUN_ID)

    def export(self, store=None, token=None, **kwargs):
        return (store or self.store).export(token or self.token, run_id=RUN_ID, **kwargs)

    def commit(self, request, store=None, token=None, **kwargs):
        return (store or self.store).commit(token or self.token, request, run_id=RUN_ID, **kwargs)

    def authoritative_files(self, root=None):
        """Projection is intentionally excluded; all other bytes are compared."""
        root = root or self.root
        return {str(path.relative_to(root)): path.read_bytes()
                for path in root.rglob("*") if path.is_file()
                and path.relative_to(root).parts[0] != "projection"}

    def assert_unchanged_head(self, expected=None, store=None):
        self.assertEqual(self.head(store), expected or self.initial_head)

    def assert_rejected_without_head_change(self, request, exception=MemoryStoreError,
                                           *, store=None, token=None, **kwargs):
        before = self.head(store)
        with self.assertRaises(exception):
            self.commit(request, store=store, token=token, **kwargs)
        self.assertEqual(self.head(store), before)


class MemoryStoreContractTests(MemoryFixtureMixin, unittest.TestCase):
    def test_initialize_partial_limits_and_empty_status(self):
        store, token = self.make_store("partial", limits={"entries": 7})
        expected = {**DEFAULT_LIMITS, "entries": 7}
        self.assertEqual(store.limits, expected)
        status = self.status(store, token)
        self.assertEqual(status["generation"], 0)
        self.assertEqual(status["counts"], dict.fromkeys(COLLECTION_IDS, 0))
        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["pending"]["status"], "empty")
        self.assertFalse(status["projection_authoritative"])
        self.assertFalse(status["mathematically_reviewed"])
        self.assertEqual(self.export(store, token)["graph"], graph_from_requests())
        with self.assertRaises(MemoryConflict):
            MemoryStore.initialize(store.root, owner="owner", domain="fixture")

    def test_absent_or_incomplete_store_never_initializes_or_repairs_on_open(self):
        missing = self.base / "absent-store"
        with self.assertRaises((OSError, ValueError)):
            MemoryStore(missing)
        self.assertFalse(missing.exists())
        store, token = self.make_store("incomplete-store")
        (store.root / "manifest.json").unlink()
        before = self.authoritative_files(store.root)
        with self.assertRaises(MemoryStoreError):
            MemoryStore(store.root)
        self.assertEqual(self.authoritative_files(store.root), before)
        self.assertFalse((store.root / "manifest.json").exists())

    def test_initialize_rejects_bad_scope_and_limits(self):
        cases = [
            {"owner": "../owner", "domain": "fixture"},
            {"owner": "owner", "domain": "other"},
            {"owner": "owner", "domain": "fixture", "limits": {"unknown": 1}},
            {"owner": "owner", "domain": "fixture", "limits": {"entries": True}},
            {"owner": "owner", "domain": "fixture", "limits": {"entries": 1.0}},
            {"owner": "owner", "domain": "fixture", "limits": {"history": 0}},
            {"owner": "owner", "domain": "fixture", "limits": {"journal_bytes": 32768}},
        ]
        for index, kwargs in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(MemoryStoreError):
                MemoryStore.initialize(self.base / ("invalid-" + str(index)), **kwargs)

    def test_unicode_slash_ids_are_opaque_and_markdown_is_hash_addressed(self):
        request = request_fixture(self.initial_head, label="../数学/α\\β",
                                  request_id="../请求/唯一")
        original = deepcopy(request)
        result = self.commit(request)
        self.assertFalse(result["replayed"])
        self.assertEqual(result["head"], receipt_head(result["receipt"]))
        self.assertEqual(self.export()["graph"], graph_from_requests(original))
        self.assertFalse(self.export()["audit_only"])
        self.assertEqual(self.status()["counts"], dict.fromkeys(COLLECTION_IDS, 1))
        revision = original["revisions"][0]
        body_path = self.root / "bodies" / (revision["body_sha256"] + ".md")
        self.assertEqual(body_path.read_bytes(), revision["body"].encode("utf-8"))
        stored_path = self.root / "graphs" / (result["receipt"]["graph_sha256"] + ".json")
        stored = read_json(stored_path)
        self.assertNotIn("body", stored["revisions"][0])
        self.assertEqual(stored_path.read_bytes(), encoded(stored))
        self.assertNotIn(revision["body"].encode("utf-8"), stored_path.read_bytes())
        for folder, extension in (("bodies", "md"), ("graphs", "json"), ("receipts", "json")):
            for path in (self.root / folder).iterdir():
                self.assertTrue(path.is_file())
                self.assertRegex(path.name, r"^[0-9a-f]{64}\." + extension + r"$")
        self.assertEqual({path.name for path in self.root.iterdir() if path.is_dir()},
                         {"authority", "bodies", "graphs", "receipts", "journal",
                          "quarantine", "projection"})
        request["revisions"][0]["body"] = "caller mutation after commit"
        request["entities"][0]["provenance"]["actor"] = "caller mutation"
        self.assertEqual(self.export()["graph"], graph_from_requests(original))

    def test_all_four_immutable_ids_compare_complete_canonical_record(self):
        seed = request_fixture(self.initial_head)
        committed = self.commit(seed)
        # Every persisted record field participates, including nested provenance,
        # endpoints, exact body text, and non-identity metadata.
        mutations = {
            "entities": {"entity_id": "changed-id", "type": "Problem", "name": "changed",
                         "provenance": provenance("changed")},
            "revisions": {"entity_id": "changed-entity", "body": "changed body",
                          "body_sha256": "0" * 64, "provenance": provenance("changed"),
                          "assertion_state": "retracted", "validity": "usable",
                          "context_revision_ids": ["missing"], "evidence_ids": [],
                          "review_kind": "unknown", "review_bindings": [],
                          "research_progress": "completed", "historical_status": "solved"},
            "relations": {"type": "depends_on", "source": {"entity_id": "x", "revision_id": "y"},
                          "target": {"entity_id": "x", "revision_id": "y"},
                          "provenance": provenance("changed"), "assertion_state": "retracted",
                          "context_revision_ids": ["missing"], "evidence_ids": []},
            "evidence": {"locator": "changed", "sha256": "0" * 64,
                         "provenance": provenance("changed")},
        }
        for collection, fields in mutations.items():
            for field, value in fields.items():
                # Identity stays fixed, except entity_id of a revision is metadata.
                if field == COLLECTION_IDS[collection]:
                    continue
                with self.subTest(collection=collection, field=field):
                    request = request_fixture(committed["head"], label="new",
                                              request_id="mutation/" + collection + "/" + field)
                    changed = deepcopy(seed[collection][0])
                    changed[field] = deepcopy(value)
                    if collection == "revisions" and field == "body":
                        changed["body_sha256"] = sha(value.encode("utf-8"))
                    request[collection].append(changed)
                    self.assert_rejected_without_head_change(request, MemoryConflict)
        for collection, identity in COLLECTION_IDS.items():
            for field in seed[collection][0]:
                if field == identity:
                    continue
                with self.subTest(collection=collection, omitted_field=field):
                    request = request_fixture(committed["head"], label="omitted-field",
                                              request_id="omit/" + collection + "/" + field)
                    partial_record = deepcopy(seed[collection][0])
                    del partial_record[field]
                    request[collection].append(partial_record)
                    self.assert_rejected_without_head_change(request, MemoryConflict)
        self.assertEqual(self.export()["graph"], graph_from_requests(seed))
        # Object key order is not an immutable-record change.
        request = request_fixture(committed["head"], label="canonical-order")
        for collection in COLLECTION_IDS:
            request[collection].append(dict(reversed(list(seed[collection][0].items()))))
        self.commit(request)
        self.assertEqual(self.status()["counts"], dict.fromkeys(COLLECTION_IDS, 2))

    def test_incremental_omission_cannot_delete_any_existing_record(self):
        seed = request_fixture(self.initial_head)
        first = self.commit(seed)
        delta = request_fixture(first["head"], label="delta")
        second = self.commit(delta)
        self.assertEqual(self.export()["graph"], graph_from_requests(seed, delta))
        empty = {"version": 1, "request_id": "attempt-delete-by-omission",
                 "expected_head": second["head"], **{key: [] for key in COLLECTION_IDS}}
        self.assert_rejected_without_head_change(empty, MemoryConflict)
        deletion = deepcopy(empty)
        deletion["request_id"] = "explicit-delete-unsupported"
        deletion["delete"] = {"entities": [seed["entities"][0]["entity_id"]]}
        self.assert_rejected_without_head_change(deletion)
        self.assertEqual(self.export()["graph"], graph_from_requests(seed, delta))

    def test_array_order_is_part_of_full_record_immutability(self):
        request = request_fixture(self.initial_head, label="ordered-record")
        second_evidence = deepcopy(request["evidence"][0])
        second_evidence["evidence_id"] = "evidence/ordered-second"
        request["evidence"].append(second_evidence)
        request["revisions"][0]["evidence_ids"].append(second_evidence["evidence_id"])
        request["relations"][0]["evidence_ids"].append(second_evidence["evidence_id"])
        committed = self.commit(request)
        for collection in ("revisions", "relations"):
            changed = request_fixture(committed["head"], label="new-order",
                                      request_id="reorder/" + collection)
            reordered = deepcopy(request[collection][0])
            reordered["evidence_ids"].reverse()
            changed[collection].append(reordered)
            with self.subTest(collection=collection):
                self.assert_rejected_without_head_change(changed, MemoryConflict)
        self.assertEqual(self.export()["graph"], graph_from_requests(request))

    def test_same_body_and_display_name_preserve_independent_identities(self):
        first = request_fixture(self.initial_head, label="first", body="identical\n")
        result = self.commit(first)
        second = request_fixture(result["head"], label="second", body="identical\n")
        self.commit(second)
        graph = self.export()["graph"]
        for collection, identity in COLLECTION_IDS.items():
            self.assertEqual(len(graph[collection]), 2)
            self.assertEqual(len({record[identity] for record in graph[collection]}), 2)
        self.assertEqual(len(list((self.root / "bodies").glob("*.md"))), 1)
        self.assertEqual(graph["revisions"][0]["body_sha256"], graph["revisions"][1]["body_sha256"])

    def test_no_caller_assertion_can_promote_review_or_retract(self):
        for collection, field, value in (
                ("revisions", "assertion_state", "reviewed"),
                ("revisions", "assertion_state", "retracted"),
                ("revisions", "validity", "usable"),
                ("revisions", "validity", "retracted"),
                ("revisions", "validity", "unavailable"),
                ("relations", "assertion_state", "reviewed"),
                ("relations", "assertion_state", "retracted")):
            with self.subTest(collection=collection, field=field, value=value):
                request = request_fixture(self.initial_head)
                request[collection][0][field] = value
                self.assert_rejected_without_head_change(request)
        solved = request_fixture(self.initial_head, label="legacy-solved")
        solved["entities"][0]["type"] = "ResearchArtifact"
        solved["revisions"][0].update(research_progress="completed", historical_status="solved")
        self.commit(solved)
        revision = self.export()["graph"]["revisions"][0]
        self.assertEqual(revision["historical_status"], "solved")
        self.assertEqual(revision["assertion_state"], "candidate")
        self.assertEqual(revision["validity"], "needs_review")
        self.assertFalse(self.status()["mathematically_reviewed"])

    def test_self_authored_complete_review_graph_still_cannot_confer_trust(self):
        request = request_fixture(self.initial_head, label="self-reviewed")
        claim = request["revisions"][0]
        claim.update(assertion_state="reviewed", validity="usable")
        request["entities"].append({"entity_id": "review/entity", "type": "Review",
                                    "name": "Caller claims human review", "provenance": provenance()})
        body = "Synthetic self-authored review, not a trusted promotion.\n"
        review = {"entity_id": "review/entity", "revision_id": "review/revision",
                  "body": body, "body_sha256": sha(body.encode("utf-8")),
                  "provenance": provenance(), "assertion_state": "reviewed", "validity": "usable",
                  "context_revision_ids": [], "evidence_ids": list(claim["evidence_ids"]),
                  "review_kind": "human_review"}
        request["revisions"].append(review)
        relation = request["relations"][0]
        relation.update(type="reviews", assertion_state="reviewed",
                        source={"entity_id": review["entity_id"], "revision_id": review["revision_id"]})
        snapshots = {item["revision_id"]: {key: deepcopy(value) for key, value in item.items()
                                           if key != "review_bindings"}
                     for item in request["revisions"]}
        binding_payload = {"relation": relation, "revisions": snapshots,
                           "evidence": {item["evidence_id"]: item for item in request["evidence"]}}
        review["review_bindings"] = [{"relation_id": relation["relation_id"],
                                      "sha256": sha(encoded(binding_payload))}]
        # Unlike incomplete 'reviewed' strings, this is structurally covered by a
        # full exact Review/evidence/digest graph; storage must still reject it.
        self.assert_rejected_without_head_change(request, MemoryConflict)
        self.assertEqual(list((self.root / "receipts").iterdir()), [])

    def test_bad_hash_dangling_references_endpoint_pairs_and_dependency_cycle(self):
        cases = []
        for field, value in (("body_sha256", "0" * 64), ("entity_id", "absent-entity"),
                             ("context_revision_ids", ["absent-context"]),
                             ("evidence_ids", ["absent-evidence"])):
            request = request_fixture(self.initial_head)
            request["revisions"][0][field] = value
            cases.append((field, request))
        changed_line_endings = request_fixture(self.initial_head)
        changed_line_endings["revisions"][0]["body"] = changed_line_endings["revisions"][0]["body"].replace("\r\n", "\n")
        cases.append(("exact-utf8-not-normalized", changed_line_endings))
        endpoint = request_fixture(self.initial_head)
        endpoint["relations"][0]["target"]["entity_id"] = "different-entity"
        cases.append(("endpoint-pair", endpoint))
        context = request_fixture(self.initial_head)
        context["revisions"][0]["context_revision_ids"] = [context["revisions"][0]["revision_id"]]
        cases.append(("context-not-definition", context))
        cycle = request_fixture(self.initial_head)
        cycle["relations"][0]["type"] = "depends_on"
        cases.append(("dependency-self-cycle", cycle))
        for label, request in cases:
            with self.subTest(case=label):
                self.assert_rejected_without_head_change(request)
        self.assertEqual(list((self.root / "receipts").iterdir()), [])

    def test_duplicate_incremental_ids_in_each_of_four_collections_are_rejected(self):
        for collection in COLLECTION_IDS:
            with self.subTest(collection=collection):
                request = request_fixture(self.initial_head)
                request[collection].append(deepcopy(request[collection][0]))
                self.assert_rejected_without_head_change(request)
        request = request_fixture(self.initial_head)
        request["revisions"][0]["revision_id"] = request["entities"][0]["entity_id"]
        request["relations"][0]["source"]["revision_id"] = request["revisions"][0]["revision_id"]
        request["relations"][0]["target"]["revision_id"] = request["revisions"][0]["revision_id"]
        self.assert_rejected_without_head_change(request)

    def test_request_closed_schema_required_fields_and_strict_integer_version(self):
        seed = request_fixture(self.initial_head)
        for key in seed:
            with self.subTest(missing=key):
                request = deepcopy(seed)
                del request[key]
                self.assert_rejected_without_head_change(request)
        for key, value in (("body", "outside"), ("reviewed", True), ("operation", "delete"),
                           ("version", True), ("version", 1.0), ("version", 2),
                           ("request_id", "  "), ("expected_head", "not-a-hash"),
                           ("entities", {})):
            with self.subTest(field=key, value=value):
                request = deepcopy(seed)
                request[key] = value
                self.assert_rejected_without_head_change(request)
        for collection in COLLECTION_IDS:
            request = deepcopy(seed)
            request[collection][0]["unrecognized"] = "not permitted"
            with self.subTest(record_extra=collection):
                self.assert_rejected_without_head_change(request)

    def test_non_strict_json_values_are_rejected_before_any_head_or_receipt_write(self):
        class DictSubclass(dict):
            pass

        for value in (float("nan"), float("inf"), float("-inf"), b"bytes", (1, 2),
                      {1: "coerced-key"}, {"nested": object()}, DictSubclass(x=1), "\ud800"):
            with self.subTest(value_type=type(value).__name__):
                request = request_fixture(self.initial_head)
                request["untrusted"] = value
                self.assert_rejected_without_head_change(request, (ValueError, TypeError))
        self.assertEqual(list((self.root / "receipts").iterdir()), [])

    def test_exact_replay_precedes_stale_cas_even_after_head_advances(self):
        request = request_fixture(self.initial_head)
        original = self.commit(request)
        next_request = request_fixture(original["head"], label="advanced")
        advanced = self.commit(next_request)
        before = self.authoritative_files()
        replay = self.commit(deepcopy(request))
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["head"], original["head"])
        self.assertEqual(replay["receipt"], original["receipt"])
        self.assertEqual(self.head(), advanced["head"])
        self.assertEqual(self.authoritative_files(), before)
        fresh_same_scope = self.store.issue(run_id=RUN_ID, actions=ACTIONS)
        self.assertTrue(self.commit(request, token=fresh_same_scope)["replayed"])

    def test_bound_request_id_rejects_other_payload_base_run_and_epoch(self):
        request = request_fixture(self.initial_head)
        original = self.commit(request)
        changed = deepcopy(request)
        changed["entities"][0]["name"] = "different payload"
        self.assert_rejected_without_head_change(changed, MemoryConflict)
        changed = deepcopy(request)
        changed["expected_head"] = original["head"]
        self.assert_rejected_without_head_change(changed, MemoryConflict)
        other_run = "other-run"
        other_token = self.store.issue(run_id=other_run, actions=ACTIONS)
        with self.assertRaises(MemoryConflict):
            self.store.commit(other_token, request, run_id=other_run)
        self.store.authority.bump_epoch(RUN_ID)
        new_epoch = self.store.issue(run_id=RUN_ID, actions=ACTIONS)
        self.assert_rejected_without_head_change(request, MemoryConflict, token=new_epoch)
        self.assertEqual(self.head(), original["head"])

    def test_stale_new_request_cas_rejects_without_preparing_receipt(self):
        first = self.commit(request_fixture(self.initial_head))
        before = {path.name for path in (self.root / "receipts").iterdir()}
        stale = request_fixture(self.initial_head, label="stale")
        self.assert_rejected_without_head_change(stale, MemoryConflict)
        self.assertEqual({path.name for path in (self.root / "receipts").iterdir()}, before)
        stale["expected_head"] = first["head"]
        stale["request_id"] = "new-request-after-conflict"
        self.commit(stale)
        self.assertEqual(self.status()["generation"], 2)

    def test_foreign_store_and_legacy_role_tokens_cannot_read_or_write(self):
        foreign, foreign_token = self.make_store("foreign")
        request = request_fixture(self.initial_head)
        self.assert_rejected_without_head_change(request, AuthorizationError, token=foreign_token)
        with self.assertRaises(AuthorizationError):
            self.store.status(foreign_token, run_id=RUN_ID)
        self.assertEqual(self.status(foreign, foreign_token)["generation"], 0)
        for role, action in (("generation", "status"), ("verification", "status"),
                             ("publisher", "status")):
            token = self.store.authority.issue(owner="owner", run_id=RUN_ID, domain="fixture",
                                               role=role, actions=[action], now=self.clock())
            with self.subTest(role=role):
                self.assert_rejected_without_head_change(request, AuthorizationError, token=token)
                with self.assertRaises(AuthorizationError):
                    self.store.status(token, run_id=RUN_ID)

    def test_read_only_capability_cannot_commit_recover_or_rebuild_projection(self):
        token = self.store.issue(run_id=RUN_ID, actions=["memory.read"])
        self.assertEqual(self.store.status(token, run_id=RUN_ID)["generation"], 0)
        self.assertEqual(self.store.export(token, run_id=RUN_ID)["generation"], 0)
        request = request_fixture(self.initial_head)
        self.assert_rejected_without_head_change(request, AuthorizationError, token=token)
        for operation in (lambda: self.store.recover(token, run_id=RUN_ID),
                          lambda: self.store.recover(token, run_id=RUN_ID, apply=True),
                          lambda: self.store.rebuild_projection(token, run_id=RUN_ID)):
            with self.assertRaises(AuthorizationError):
                operation()
        self.assert_unchanged_head()

    def test_commit_and_maintain_actions_do_not_grant_read_or_each_other(self):
        commit_token = self.store.issue(run_id=RUN_ID, actions=["memory.commit"])
        maintain_token = self.store.issue(run_id=RUN_ID, actions=["memory.maintain"])
        with self.assertRaises(AuthorizationError):
            self.store.status(commit_token, run_id=RUN_ID)
        with self.assertRaises(AuthorizationError):
            self.store.recover(commit_token, run_id=RUN_ID)
        with self.assertRaises(AuthorizationError):
            self.store.export(maintain_token, run_id=RUN_ID)
        self.assert_rejected_without_head_change(request_fixture(self.initial_head), AuthorizationError,
                                                token=maintain_token)
        self.assertEqual(self.store.recover(maintain_token, run_id=RUN_ID)["status"], "inspected")

    def test_wrong_owner_domain_run_scope_and_tampered_token_are_rejected(self):
        request = request_fixture(self.initial_head)
        for suffix, owner, domain in (("owner", "different-owner", "fixture"),
                                      ("domain", "owner", "production")):
            run_id = "scope-" + suffix
            self.store.authority.register_run(owner, run_id, domain)
            token = self.store.authority.issue(owner=owner, run_id=run_id, domain=domain,
                                               role="memory", actions=ACTIONS, now=self.clock())
            with self.subTest(scope=suffix), self.assertRaises(AuthorizationError):
                self.store.commit(token, request, run_id=run_id)
            self.assert_unchanged_head()
        with self.assertRaises(AuthorizationError):
            self.store.commit(self.token, request, run_id="wrong-run")
        payload, signature = self.token.split(".")
        forged = payload + "." + ("A" if signature[0] != "A" else "B") + signature[1:]
        self.assert_rejected_without_head_change(request, AuthorizationError, token=forged)
        self.assert_rejected_without_head_change(request, AuthorizationError, token="not-a-token")

    def test_expired_not_yet_valid_and_invalid_clock_do_not_advance_head(self):
        request = request_fixture(self.initial_head)
        issued = self.clock.value
        for moment in (issued - 1, issued + 300, float("nan"), float("inf"), True):
            with self.subTest(moment=moment):
                self.clock.value = moment
                self.assert_rejected_without_head_change(request, AuthorizationError)
        self.clock.value = issued
        self.assertEqual(self.status()["generation"], 0)

    def test_revocation_close_and_old_epoch_reject_before_head_advance(self):
        for operation in ("revoke", "close", "epoch"):
            store, token = self.make_store("lifecycle-" + operation)
            request = request_fixture(self.head(store))
            if operation == "revoke":
                claims = store.authority.authenticate(token, action="memory.read", now=self.clock())
                store.authority.revoke(claims["token_id"])
            elif operation == "close":
                store.authority.close_run(RUN_ID)
            else:
                store.authority.bump_epoch(RUN_ID)
            with self.subTest(operation=operation):
                self.assert_rejected_without_head_change(request, AuthorizationError,
                                                        store=store, token=token)
                with self.assertRaises(AuthorizationError):
                    store.status(token, run_id=RUN_ID)

    def test_preexisting_cancel_does_not_prepare_or_advance(self):
        cancelled = threading.Event()
        cancelled.set()
        self.assert_rejected_without_head_change(request_fixture(self.initial_head), MemoryCancelled,
                                                cancel_event=cancelled)
        self.assertEqual(list((self.root / "receipts").iterdir()), [])
        self.assertEqual((self.root / "journal/pending.log").read_bytes(), b"")

    def test_history_export_is_audit_only_and_rejects_orphan_graphs_and_receipts(self):
        seed = request_fixture(self.initial_head)
        first = self.commit(seed)
        second = self.commit(request_fixture(first["head"], label="new"))
        audit = self.export(head=first["head"])
        self.assertTrue(audit["audit_only"])
        self.assertEqual(audit["graph"], graph_from_requests(seed))
        self.assertEqual(audit["generation"], 1)
        self.assertTrue(self.export(head=self.initial_head)["audit_only"])
        self.assertFalse(self.export(head=second["head"])["audit_only"])
        orphan_graph = b'{"not":"committed"}'
        orphan_hash = sha(orphan_graph)
        write_private(self.root / "graphs" / (orphan_hash + ".json"), orphan_graph)
        for uncommitted_identifier in (orphan_hash, first["receipt"]["request_key"], "0" * 64):
            with self.subTest(head=uncommitted_identifier), self.assertRaises(MemoryConflict):
                self.export(head=uncommitted_identifier)
        self.assertEqual(self.head(), second["head"])

    def test_projection_forgery_deletion_and_rebuild_never_change_authority(self):
        request = request_fixture(self.initial_head)
        committed = self.commit(request)
        before = self.authoritative_files()
        forged = {"head": "0" * 64, "body": "forged", "validity": "usable", "revisions": []}
        projection = self.root / "projection/ids.json"
        write_private(projection, encoded(forged))
        self.assertEqual(self.export()["graph"], graph_from_requests(request))
        self.assertEqual(self.status()["head"], committed["head"])
        projection.unlink()
        self.assertEqual(self.export()["graph"], graph_from_requests(request))
        rebuilt = self.store.rebuild_projection(self.token, run_id=RUN_ID)
        self.assertFalse(rebuilt["authoritative"])
        self.assertEqual(rebuilt["kind"], "ids-only-not-retrieval")
        actual = read_json(projection)
        self.assertEqual(actual["head"], committed["head"])
        self.assertEqual(actual["revisions"], [{key: request["revisions"][0][key]
                                              for key in ("entity_id", "revision_id", "body_sha256")}])
        self.assertNotIn("body", actual["revisions"][0])
        self.assertEqual(self.authoritative_files(), before)
        self.assertEqual(sha(projection.read_bytes()), rebuilt["projection_sha256"])

    def test_issue_rejects_invalid_actions_ttl_and_run_without_persisting_tokens(self):
        before = self.authoritative_files()
        cases = [
            {"run_id": "new-run", "actions": []},
            {"run_id": "new-run", "actions": ["memory.read", "memory.read"]},
            {"run_id": "new-run", "actions": ["publish"]},
            {"run_id": "new-run", "actions": ["memory.read"], "ttl_seconds": 0},
            {"run_id": "new-run", "actions": ["memory.read"], "ttl_seconds": True},
            {"run_id": "new-run", "actions": ["memory.read"], "ttl_seconds": 3601},
            {"run_id": "new-run", "actions": ["memory.read"], "ttl_seconds": float("nan")},
            {"run_id": "../run", "actions": ["memory.read"]},
        ]
        for index, kwargs in enumerate(cases):
            with self.subTest(case=index), self.assertRaises((AuthorizationError, MemoryStoreError)):
                self.store.issue(**kwargs)
            self.assertEqual(self.authoritative_files(), before)
        self.assert_unchanged_head()

    def test_capacity_request_and_body_use_exact_utf8_bytes(self):
        small_request, token = self.make_store("request-cap", limits={"request_bytes": 256})
        request = request_fixture(self.head(small_request))
        self.assertGreater(len(encoded(request)), 256)
        self.assert_rejected_without_head_change(request, MemoryCapacity, store=small_request, token=token)
        small_body, token = self.make_store("body-cap", limits={"body_bytes": 3})
        request = request_fixture(self.head(small_body), body="数学")
        self.assertEqual(len(request["revisions"][0]["body"]), 2)
        self.assert_rejected_without_head_change(request, MemoryCapacity, store=small_body, token=token)
        self.assertEqual(list((small_body.root / "receipts").iterdir()), [])

    def test_graph_exchange_and_entry_capacity_do_not_advance_head(self):
        for name, limits in (("graph", {"graph_bytes": 512}),
                             ("exchange", {"exchange_bytes": 512})):
            store, token = self.make_store(name + "-cap", limits=limits)
            with self.subTest(limit=name):
                self.assert_rejected_without_head_change(request_fixture(self.head(store)), MemoryCapacity,
                                                        store=store, token=token)
        store, token = self.make_store("entry-cap", limits={"entries": 1})
        first = self.commit(request_fixture(self.head(store)), store=store, token=token)
        self.assert_rejected_without_head_change(request_fixture(first["head"], label="overflow"),
                                                MemoryCapacity, store=store, token=token)
        self.assertEqual(self.status(store, token)["counts"], dict.fromkeys(COLLECTION_IDS, 1))

    def test_hydration_budget_is_checked_before_committing_a_reader_unloadable_graph(self):
        draft = request_fixture(self.initial_head, body="x")
        exchange = graph_from_requests(draft)
        exchange_size = len(encoded(exchange))
        stored = {"format_version": "m2a-graph-1", "math_schema_version": "1.0",
                  **{name: exchange[name] for name in ("entities", "relations", "evidence")},
                  "revisions": [{key: value for key, value in item.items() if key != "body"}
                                for item in exchange["revisions"]]}
        hydration_size = len(encoded(stored)) + 1
        self.assertGreater(hydration_size, exchange_size)
        store, token = self.make_store("hydration-too-small", limits={"exchange_bytes": exchange_size})
        draft["expected_head"] = self.head(store)
        self.assert_rejected_without_head_change(draft, MemoryCapacity, store=store, token=token)
        self.assertEqual(self.status(store, token)["generation"], 0)
        self.assertEqual(list((store.root / "receipts").iterdir()), [])
        enough, token = self.make_store("hydration-exact", limits={"exchange_bytes": hydration_size})
        draft["expected_head"] = self.head(enough)
        self.commit(draft, store=enough, token=token)
        self.assertEqual(self.status(enough, token)["generation"], 1)
        self.assertEqual(self.export(enough, token)["graph"], exchange)

    def test_each_collection_entry_budget_rejects_excess_before_preparing(self):
        for collection, identity in COLLECTION_IDS.items():
            with self.subTest(collection=collection):
                store, token = self.make_store("entries-" + collection, limits={"entries": 1})
                request = request_fixture(self.head(store))
                extra = deepcopy(request[collection][0])
                extra[identity] += "/extra"
                request[collection].append(extra)
                self.assert_rejected_without_head_change(request, MemoryCapacity, store=store, token=token)
                self.assertEqual(list((store.root / "receipts").iterdir()), [])

    def test_history_and_receipt_caps_preserve_existing_receipt_and_replay(self):
        for name in ("history", "receipts"):
            store, token = self.make_store(name + "-cap", limits={name: 1})
            request = request_fixture(self.head(store))
            first = self.commit(request, store=store, token=token)
            original_receipts = {path.name: path.read_bytes() for path in (store.root / "receipts").iterdir()}
            with self.subTest(limit=name):
                self.assert_rejected_without_head_change(request_fixture(first["head"], label="overflow"),
                                                        MemoryCapacity, store=store, token=token)
                replay = self.commit(request, store=store, token=token)
                self.assertTrue(replay["replayed"])
                self.assertEqual(replay["receipt"], first["receipt"])
                self.assertEqual({path.name: path.read_bytes() for path in (store.root / "receipts").iterdir()},
                                 original_receipts)

    def test_orphan_and_temporary_files_count_toward_bytes_without_automatic_gc(self):
        before = self.status()["bytes"]
        orphan = self.root / "bodies" / ("e" * 64 + ".md")
        temporary = self.root / "graphs" / (".tmp-" + "a" * 32)
        write_private(orphan, b"orphan" * 37)
        write_private(temporary, b"unfinished" * 29)
        self.assertEqual(self.status()["bytes"] - before,
                         len(orphan.read_bytes()) + len(temporary.read_bytes()))
        self.store.recover(self.token, run_id=RUN_ID, apply=True)
        self.store.rebuild_projection(self.token, run_id=RUN_ID)
        self.assertTrue(orphan.exists())
        self.assertTrue(temporary.exists())
        self.assert_unchanged_head()

    def test_store_byte_capacity_includes_orphans_and_does_not_collect_them(self):
        store, token = self.make_store("store-byte-cap", limits={"store_bytes": 65536})
        usage = self.status(store, token)["bytes"]
        orphan = store.root / "bodies" / ("d" * 64 + ".md")
        write_private(orphan, b"x" * (65536 - usage + 1))
        with self.assertRaises(MemoryCapacity):
            store.status(token, run_id=RUN_ID)
        self.assert_rejected_without_head_change(request_fixture(self.head(store)), MemoryCapacity,
                                                store=store, token=token)
        self.assertTrue(orphan.exists())

    def test_file_capacity_boundary_counts_directories_orphans_and_temps(self):
        store, token = self.make_store("file-cap", limits={"files": 32})
        # The documented file budget includes directory entries, not just data.
        count = sum(1 for _ in store.root.rglob("*"))
        for index in range(32 - count):
            write_private(store.root / "graphs" / (".tmp-" + format(index, "032x")), b"")
        self.assertEqual(sum(1 for _ in store.root.rglob("*")), 32)
        # Reserving temp+final at the exact boundary must be a capacity rejection,
        # not a filesystem traversal error or an implicit garbage collection.
        self.assert_rejected_without_head_change(request_fixture(self.head(store)), MemoryCapacity,
                                                store=store, token=token)
        self.assertEqual(sum(1 for _ in store.root.rglob("*")), 32)

    def test_store_configuration_changes_fail_closed(self):
        original = (self.root / "store.json").read_bytes()
        config = read_json(self.root / "store.json")
        config["owner"] = "someone-else"
        write_private(self.root / "store.json", encoded(config))
        for operation in (lambda: self.status(), lambda: self.export(),
                          lambda: self.store.recover(self.token, run_id=RUN_ID, apply=True)):
            with self.assertRaises(MemoryCorruption):
                operation()
        self.assert_unchanged_head()
        self.assertNotEqual((self.root / "store.json").read_bytes(), original)

    def test_recovery_apply_requires_real_boolean(self):
        for value in (0, 1, "true", None, []):
            with self.subTest(value=value), self.assertRaises(MemoryStoreError):
                self.store.recover(self.token, run_id=RUN_ID, apply=value)
        self.assert_unchanged_head()


if __name__ == "__main__":
    unittest.main()
