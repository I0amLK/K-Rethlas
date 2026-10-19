"""Synthetic T08 structure/authority tests; no provider or mathematical verdict."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from rethlas_core.authorization import Authority, AuthorizationError, ROLE_ACTIONS
from rethlas_core.context_views import (
    POLICY, VERSION, ContextError, ContextConflict, ContextCapacity,
    ContextRecoveryRequired, ContextIndeterminate, active_map, capture_graph,
    check_branch_request, exact_ref, join_graph, minimal_view, protocol,
)
from rethlas_core.context_workspace import ContextWorkspace, _branch_dir
from rethlas_core.memory_store import MemoryStore, MemoryConflict, MemoryCancelled
from rethlas_core.private_fs import canonical_json, digest

from test_memory_store import MutableClock, provenance, encoded, sha, read_json, write_private


def mathematical_fixture():
    types = {"problem": "Problem", "definition": "DefinitionContext", "assumption": "Assumption",
             "source": "Reference", "claim": "Claim", "proof": "ProofArtifact"}
    bodies = {"problem": "证明一个合成连续性结论。", "definition": "在指定度量空间中定义连续；不是一致连续。",
              "assumption": "假设定义域紧致。", "source": "合成公开参考正文，没有网络定位要求。",
              "claim": "在已声明条件下，合成命题成立（待审）。", "proof": "使用显式定义、假设和参考的合成证明草稿。"}
    graph = {"schema_version": "1.0", "entities": [], "revisions": [], "relations": [], "evidence": []}
    for name, kind in types.items():
        graph["entities"].append({"entity_id": name, "type": kind, "name": "同名连续",
                                  "provenance": provenance("PRIVATE_METADATA_ACTOR")})
        graph["revisions"].append({"entity_id": name, "revision_id": name + ":0", "body": bodies[name],
            "body_sha256": sha(bodies[name].encode()), "provenance": provenance("PRIVATE_METADATA_ACTOR"),
            "assertion_state": "candidate", "validity": "needs_review", "context_revision_ids": [],
            "evidence_ids": ["evidence"]})
    graph["evidence"] = [{"evidence_id": "evidence", "locator": "PRIVATE_LOCATOR_HISTORY",
                           "sha256": sha(b"synthetic"), "provenance": provenance()}]
    graph["relations"] = [edge("proof:0", "claim:0", "attempts_to_prove"),
                          edge("proof:0", "assumption:0", "assumes"),
                          edge("claim:0", "definition:0", "in_context"),
                          edge("claim:0", "source:0", "cites")]
    return graph


def edge(source, target, kind, *, name=None):
    return {"relation_id": name or source + "/" + kind + "/" + target, "type": kind,
            "source": {"entity_id": source.split(":")[0], "revision_id": source},
            "target": {"entity_id": target.split(":")[0], "revision_id": target},
            "provenance": provenance("PRIVATE_RELATION_HISTORY"), "assertion_state": "candidate",
            "context_revision_ids": [], "evidence_ids": []}


def active(graph):
    result = {}
    for revision in graph["revisions"]:
        result[revision["entity_id"]] = revision["revision_id"]
    return [{"entity_id": key, "revision_id": value} for key, value in sorted(result.items())]


def manifest_for(graph, *, workspace_id="1" * 64, snapshot_sha256="2" * 64, join_sha256="3" * 64,
                 claim="claim:0", proof="proof:0"):
    by_id = {r["revision_id"]: r for r in graph["revisions"]}
    roots = {"problem": exact_ref(by_id["problem:0"]), "claim": exact_ref(by_id[claim]), "proof": exact_ref(by_id[proof])}
    from rethlas_core.context_views import closure
    selected, _ = closure(graph, {r["revision_id"] for r in roots.values()}, proof_pair=(proof, claim))
    dependencies = selected - {r["revision_id"] for r in roots.values()}
    return {"version": VERSION, "request_id": "proof-view", "workspace_id": workspace_id,
            "snapshot_sha256": snapshot_sha256, "join_sha256": join_sha256, "policy_version": POLICY,
            "roots": roots, "dependencies": [exact_ref(by_id[k]) for k in sorted(dependencies)],
            "references": [exact_ref(by_id["source:0"])]}


class ContextFixtureMixin:
    def setUp(self):
        super().setUp()
        self.temp = tempfile.TemporaryDirectory(prefix="context-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = MemoryStore.initialize(self.base / "source", owner="owner", domain="fixture")
        self.source_token = self.source.issue(run_id="source-run", actions=list(ROLE_ACTIONS["memory"]))
        initial = self.source.status(self.source_token, run_id="source-run")
        self.graph = mathematical_fixture()
        q = {"version": 1, "request_id": "source-seed", "expected_head": initial["head"],
             **{k: self.graph[k] for k in ("entities", "revisions", "relations", "evidence")}}
        committed = self.source.commit(self.source_token, q, run_id="source-run")
        self.spec = {"version": VERSION, "request_id": "capture", "owner": "owner", "domain": "fixture",
                     "source": {"store_id": self.source.config["store_id"], "config_sha256": self.source.config_sha256,
                                "head": committed["head"]}, "public": [exact_ref(r) for r in self.graph["revisions"]],
                     "base_active": active(self.graph), "branches": ["left", "right"]}
        self.workspace = ContextWorkspace.initialize(self.base / "workspace", self.source, self.source_token,
                                                     run_id="source-run", spec=self.spec)
        self.grants = {b: self.workspace.issue(kind="branch", branch_id=b, actions=list(ROLE_ACTIONS["context_branch"]))
                       for b in self.spec["branches"]}

    def branch(self, name="left"):
        return self.grants[name]

    def read_branch(self, name="left", **kwargs):
        grant = self.branch(name)
        return self.workspace.branch_read(grant["token"], run_id=grant["run_id"], **kwargs)

    def branch_request(self, name="left", *, old_dependency=False):
        snapshot = self.read_branch(name)
        revisions = []
        for entity in ("claim", "proof"):
            item = deepcopy(next(r for r in self.graph["revisions"] if r["entity_id"] == entity))
            item.update(revision_id=entity + ":" + name, body="合成分支 " + name + " " + entity)
            item["body_sha256"] = sha(item["body"].encode())
            revisions.append(item)
        edges = [edge("proof:" + name, "claim:" + name, "attempts_to_prove"),
                 edge("proof:" + name, "assumption:0", "assumes"),
                 edge("claim:" + name, "definition:0", "in_context"),
                 edge("claim:" + name, "source:0", "cites")]
        if old_dependency:
            edges.append(edge("proof:" + name, "claim:0", "depends_on"))
        return {"version": 1, "request_id": "branch-" + name, "expected_head": snapshot["head"],
                "entities": [], "revisions": revisions, "relations": edges, "evidence": []}

    def commit_branch(self, name, request):
        grant = self.branch(name)
        return self.workspace.branch_commit(grant["token"], request, run_id=grant["run_id"])

    def seal_branch(self, name):
        snapshot = self.read_branch(name)
        grant = self.branch(name)
        request = {"version": VERSION, "request_id": "seal-" + name, "expected_head": snapshot["head"],
                   "active": active(snapshot["graph"])}
        return self.workspace.seal(grant["token"], request, run_id=grant["run_id"])

    def join(self, *, conflict=False, old_dependency=False):
        if conflict:
            for name in ("left", "right"):
                self.commit_branch(name, self.branch_request(name, old_dependency=old_dependency and name == "left"))
        for name in ("left", "right"):
            self.seal_branch(name)
        grant = self.workspace.issue(kind="join", actions=list(ROLE_ACTIONS["context_join"]))
        choices = active(self.graph)
        if conflict:
            for choice in choices:
                if choice["entity_id"] in ("claim", "proof"):
                    choice["revision_id"] = choice["entity_id"] + ":left"
        historical = [exact_ref(next(r for r in self.graph["revisions"] if r["revision_id"] == "claim:0"))] if old_dependency else []
        request = {"version": VERSION, "request_id": "join", "snapshot_sha256": self.workspace.config["snapshot_sha256"],
                   "selection": {"choices": choices, "historical": historical}}
        result = self.workspace.join_prepare(grant["token"], request, run_id=grant["run_id"])
        bundle = self.workspace.join_read(grant["token"], run_id=grant["run_id"])["bundle"]
        return grant, request, result, bundle

    def freeze_view(self, *, conflict=False):
        grant, request, result, bundle = self.join(conflict=conflict)
        manifest = manifest_for(bundle["graph"], workspace_id=self.workspace.config["workspace_id"],
            snapshot_sha256=self.workspace.config["snapshot_sha256"], join_sha256=result["join_sha256"],
            claim="claim:left" if conflict else "claim:0", proof="proof:left" if conflict else "proof:0")
        view = self.workspace.build_view(manifest)
        verifier = self.workspace.issue(kind="view", actions=list(ROLE_ACTIONS["context_verifier"]))
        return verifier, manifest, view


class PureContextTests(unittest.TestCase):
    def setUp(self):
        self.graph = mathematical_fixture()
        self.manifest = manifest_for(self.graph)

    def test_exact_bodies_and_required_assumptions_delivered_without_metadata(self):
        dto = minimal_view(self.graph, self.manifest)
        self.assertEqual({r["revision_id"] for r in dto["revisions"]}, {r["revision_id"] for r in self.graph["revisions"]})
        self.assertNotIn("PRIVATE_", canonical_json(dto).decode())
        self.assertIn("假设定义域紧致。", canonical_json(dto).decode())
        self.assertEqual(dto["manifest_sha256"], digest(self.manifest))

    def test_missing_or_extra_dependency_rejects_whole_view(self):
        for dependencies in ([], self.manifest["dependencies"] + [self.manifest["roots"]["problem"]]):
            bad = deepcopy(self.manifest)
            bad["dependencies"] = dependencies
            with self.subTest(dependencies=dependencies), self.assertRaises(ContextError):
                minimal_view(self.graph, bad)

    def test_reference_requires_explicit_exact_approval(self):
        self.manifest["references"] = []
        with self.assertRaises(ContextError):
            minimal_view(self.graph, self.manifest)

    def test_legacy_reference_wrapper_rejects_even_with_approval(self):
        reference = next(r for r in self.graph["revisions"] if r["entity_id"] == "source")
        reference["provenance"]["source"] = "legacy-jsonl://snapshot/" + "0" * 64
        with self.assertRaisesRegex(ContextError, "legacy history"):
            minimal_view(self.graph, self.manifest)

    def test_legacy_revision_namespace_is_rejected_without_provenance_marker(self):
        reference = next(r for r in self.graph["revisions"] if r["entity_id"] == "source")
        reference["revision_id"] = "legacy:revision:" + "b" * 64
        for relation in self.graph["relations"]:
            if relation["target"]["entity_id"] == "source":
                relation["target"]["revision_id"] = reference["revision_id"]
        for name in ("dependencies", "references"):
            self.manifest[name] = [exact_ref(reference) if r["entity_id"] == "source" else r for r in self.manifest[name]]
        with self.assertRaisesRegex(ContextError, "legacy history"):
            minimal_view(self.graph, self.manifest)

    def test_public_allowlist_preserves_outgoing_edges_instead_of_hiding_history(self):
        graph = deepcopy(self.graph)
        graph["entities"].append({"entity_id": "history", "type": "ResearchArtifact", "name": "private",
                                   "provenance": provenance()})
        body = "PRIVATE_HISTORY_MARKER"
        graph["revisions"].append({"entity_id": "history", "revision_id": "history:0", "body": body,
            "body_sha256": sha(body.encode()), "provenance": provenance(), "assertion_state": "candidate",
            "validity": "needs_review", "context_revision_ids": [], "evidence_ids": [], "research_progress": "completed"})
        graph["relations"].append(edge("claim:0", "history:0", "derived_from"))
        spec = {"version": VERSION, "request_id": "capture", "owner": "owner", "domain": "fixture",
                "source": {"store_id": "a" * 64, "config_sha256": "b" * 64, "head": "c" * 64},
                "public": [exact_ref(r) for r in self.graph["revisions"]], "base_active": active(self.graph), "branches": ["one"]}
        with self.assertRaisesRegex(ContextError, "public allowlist"):
            capture_graph(graph, spec)
        spec["public"].append(exact_ref(graph["revisions"][-1]))
        spec["base_active"].append({"entity_id": "history", "revision_id": "history:0"})
        captured = capture_graph(graph, spec)
        self.assertTrue(any(r["type"] == "derived_from" for r in captured["relations"]))
        with self.assertRaises(ContextError):
            minimal_view(captured, self.manifest)

    def test_research_history_dependency_rejects(self):
        entity = next(e for e in self.graph["entities"] if e["entity_id"] == "source")
        entity["type"] = "ResearchArtifact"
        next(r for r in self.graph["revisions"] if r["entity_id"] == "source")["research_progress"] = "completed"
        with self.assertRaises(ContextError):
            minimal_view(self.graph, self.manifest)

    def test_unsupported_outgoing_edge_cannot_be_silently_filtered(self):
        self.graph["relations"].append(edge("claim:0", "source:0", "derived_from"))
        with self.assertRaisesRegex(ContextError, "unsupported"):
            minimal_view(self.graph, self.manifest)

    def test_relation_context_is_a_required_dependency(self):
        self.graph["relations"] = [r for r in self.graph["relations"] if r["type"] != "in_context"]
        self.graph["relations"][0]["context_revision_ids"] = ["definition:0"]
        dto = minimal_view(self.graph, self.manifest)
        self.assertIn("definition:0", {r["revision_id"] for r in dto["revisions"]})

    def test_citation_and_context_cycles_terminate_without_dropping_records(self):
        self.graph["relations"].append(edge("source:0", "claim:0", "cites"))
        next(r for r in self.graph["revisions"] if r["revision_id"] == "definition:0")["context_revision_ids"] = ["definition:0"]
        self.assertEqual(len(minimal_view(self.graph, self.manifest)["revisions"]), 6)

    def test_wrong_root_pair_hash_or_type_rejects(self):
        for key, value in (("body_sha256", "f" * 64), ("entity_id", "definition"), ("revision_id", "definition:0")):
            bad = deepcopy(self.manifest)
            bad["roots"]["claim"][key] = value
            with self.subTest(key=key), self.assertRaises(ContextError):
                minimal_view(self.graph, bad)

    def test_proof_must_attempt_exact_claim(self):
        self.graph["relations"] = [r for r in self.graph["relations"] if r["type"] != "attempts_to_prove"]
        with self.assertRaises(ContextError):
            minimal_view(self.graph, self.manifest)

    def test_unknown_manifest_dto_fields_reject(self):
        for key in ("history", "confidence", "command", "path"):
            with self.subTest(key=key), self.assertRaises(ContextError):
                minimal_view(self.graph, {**self.manifest, key: "not allowed"})
        dto = minimal_view(self.graph, self.manifest)
        with self.assertRaises(ContextError):
            protocol({**dto, "history": "private"}, "dto")

    def test_immutable_ids_collide_in_each_collection(self):
        from rethlas_core.context_views import union_graph
        for collection, field in (("entities", "name"), ("revisions", "body"), ("relations", "evidence_ids"), ("evidence", "locator")):
            other = deepcopy(self.graph)
            if field == "evidence_ids":
                other[collection][0][field] = ["evidence"]
            else:
                other[collection][0][field] += " changed"
            if collection == "revisions":
                other[collection][0]["body_sha256"] = sha(other[collection][0]["body"].encode())
            with self.subTest(collection=collection), self.assertRaises(ContextConflict):
                union_graph([self.graph, other])

    def test_equal_revision_with_disjoint_dependency_edges_cannot_be_unioned(self):
        from rethlas_core.context_views import union_graph
        left, right = deepcopy(self.graph), deepcopy(self.graph)
        left["relations"].append(edge("claim:0", "assumption:0", "depends_on", name="left-extra"))
        right["relations"].append(edge("claim:0", "source:0", "depends_on", name="right-extra"))
        with self.assertRaisesRegex(ContextConflict, "outgoing structure"):
            union_graph([left, right])

    def test_equal_records_deduplicate_and_names_do_not_merge_identity(self):
        from rethlas_core.context_views import union_graph
        combined = union_graph([self.graph, deepcopy(self.graph)])
        self.assertEqual(len(combined["entities"]), 6)
        self.assertEqual(len({e["name"] for e in combined["entities"]}), 1)
        self.assertEqual(len({e["entity_id"] for e in combined["entities"]}), 6)

    def test_retroactive_edges_reject_but_exact_edge_replay_is_allowed(self):
        request = {"version": 1, "request_id": "q", "expected_head": "1" * 64,
                   "entities": [], "revisions": [], "relations": [deepcopy(self.graph["relations"][0])], "evidence": []}
        check_branch_request(self.graph, request)
        request["relations"][0]["relation_id"] = "new-edge-against-old-proof"
        with self.assertRaises(ContextConflict):
            check_branch_request(self.graph, request)

    def test_graph_size_and_strict_counter_types(self):
        from rethlas_core.context_views import copied
        with self.assertRaises(ContextCapacity):
            copied({"value": "x" * 100}, 10)
        with self.assertRaises(ContextError):
            protocol({"version": VERSION, "workspace_id": "0" * 64, "counts": {"read.view": 1.0}}, "state")

    def test_distinct_definition_and_assumption_keep_exact_semantics(self):
        graph = deepcopy(self.graph)
        for entity_id, new_id, text in (("definition", "definition:uniform", "一致连续：一个delta对所有点同时适用。"),
                                       ("assumption", "assumption:unbounded", "定义域不要求紧致。")):
            revision = deepcopy(next(r for r in graph["revisions"] if r["entity_id"] == entity_id))
            revision.update(revision_id=new_id, body=text, body_sha256=sha(text.encode()))
            graph["revisions"].append(revision)
        # Names and entity types cannot authorize replacing an exact dependency.
        dto = minimal_view(graph, self.manifest)
        ids = {r["revision_id"] for r in dto["revisions"]}
        self.assertIn("definition:0", ids)
        self.assertNotIn("definition:uniform", ids)
        bad = deepcopy(self.manifest)
        bad["dependencies"] = [exact_ref(next(r for r in graph["revisions"] if r["revision_id"] == "definition:uniform"))
                               if r["revision_id"] == "definition:0" else r for r in bad["dependencies"]]
        with self.assertRaises(ContextError):
            minimal_view(graph, bad)

    def test_closure_depth_and_revision_count_fail_instead_of_truncate(self):
        from rethlas_core.context_views import closure
        graph = deepcopy(self.graph)
        source = next(r for r in graph["revisions"] if r["entity_id"] == "source")
        for index in range(66):
            item = deepcopy(source)
            item["revision_id"] = "source:chain-%d" % index
            graph["revisions"].append(item)
            if index:
                graph["relations"].append(edge("source:chain-%d" % (index - 1), item["revision_id"], "cites"))
        with self.assertRaises(ContextCapacity):
            closure(graph, {"source:chain-0"})
        graph["relations"] = []
        for index in range(200):
            item = deepcopy(source)
            item["revision_id"] = "source:extra-%d" % index
            graph["revisions"].append(item)
        with self.assertRaises(ContextCapacity):
            closure(graph, {r["revision_id"] for r in graph["revisions"]})

    def test_same_entity_conflict_never_chooses_by_order(self):
        other = deepcopy(self.graph)
        old = next(r for r in other["revisions"] if r["entity_id"] == "claim")
        new = {**old, "revision_id": "claim:other", "body": "不同候选", "body_sha256": sha("不同候选".encode())}
        other["revisions"].append(new)
        with self.assertRaises(ContextConflict):
            join_graph(self.graph, [self.graph, other], [active(self.graph), active(other)],
                       {"choices": [], "historical": []})
        choices = active(self.graph)
        left = join_graph(self.graph, [self.graph, other], [active(self.graph), active(other)], {"choices": choices, "historical": []})
        right = join_graph(self.graph, [other, self.graph], [active(other), active(self.graph)], {"choices": choices, "historical": []})
        self.assertEqual(left, right)
        self.assertNotIn("claim:other", {r["revision_id"] for r in left["revisions"]})

    def test_policy_schema_and_role_matrix_have_identical_actions(self):
        schema = json.loads(files("rethlas_core").joinpath("schemas", "policy.schema.json").read_bytes())
        props = schema["properties"]["rules"]["items"]["properties"]
        self.assertEqual(set(props["role"]["enum"]), set(ROLE_ACTIONS))
        self.assertEqual(set(props["action"]["enum"]), set().union(*ROLE_ACTIONS.values()))


class WorkspaceContextTests(ContextFixtureMixin, unittest.TestCase):
    def test_two_branches_share_baseline_but_not_private_writes(self):
        before = self.read_branch("right")
        request = self.branch_request("left")
        result = self.commit_branch("left", request)
        self.assertNotEqual(self.read_branch("left")["head"], before["head"])
        self.assertEqual(self.read_branch("right"), before)
        self.assertEqual(self.commit_branch("left", request)["head"], result["head"])
        self.assertTrue(self.commit_branch("left", request)["replayed"])

    def test_source_advance_does_not_change_frozen_snapshot(self):
        from test_memory_store import request_fixture
        snapshot = (self.workspace.root / "snapshot.json").read_bytes()
        head = self.source.status(self.source_token, run_id="source-run")["head"]
        self.source.commit(self.source_token, request_fixture(head, label="later"), run_id="source-run")
        self.assertEqual((self.workspace.root / "snapshot.json").read_bytes(), snapshot)
        self.assertNotIn("revision/later", {r["revision_id"] for r in self.read_branch()["graph"]["revisions"]})

    def test_cross_branch_wrong_role_or_run_reject_before_hydration(self):
        left, right = self.branch("left"), self.branch("right")
        with mock.patch.object(self.workspace, "_store", side_effect=AssertionError("hydrated")):
            with self.assertRaises(AuthorizationError):
                self.workspace.branch_read(left["token"], run_id=right["run_id"])
            with self.assertRaises(AuthorizationError):
                self.workspace.join_read(left["token"], run_id=left["run_id"])
            with self.assertRaises(AuthorizationError):
                self.workspace.branch_read(self.source_token, run_id="source-run")

    def test_expired_revoked_and_old_epoch_reject_before_hydration(self):
        grant = self.branch()
        claims = self.workspace.authority.check(grant["token"], owner="owner", run_id=grant["run_id"],
                                                 domain="fixture", action="context.branch.read")
        self.workspace.clock = lambda: claims["expires_at"]
        with mock.patch.object(self.workspace, "_store", side_effect=AssertionError("hydrated")), self.assertRaises(AuthorizationError):
            self.read_branch()
        import time
        self.workspace.clock = time.time
        self.workspace.authority.revoke(claims["token_id"])
        with self.assertRaises(AuthorizationError):
            self.read_branch()
        replacement = self.workspace.issue(kind="branch", branch_id="left", actions=["context.branch.read"])
        self.workspace.authority.bump_epoch(replacement["run_id"])
        with self.assertRaises(AuthorizationError):
            self.workspace.branch_read(replacement["token"], run_id=replacement["run_id"])

    def test_branch_read_rechecks_expiry_before_return(self):
        grant = self.branch()
        claims = self.workspace.authority.authenticate(grant["token"], action="context.branch.read")
        clock = MutableClock(claims["issued_at"] + 1)
        self.workspace.clock = clock
        def checkpoint(stage):
            if stage == "before_read_return":
                clock.value = claims["expires_at"]
        with mock.patch.object(self.workspace, "_checkpoint", side_effect=checkpoint), self.assertRaises(AuthorizationError):
            self.read_branch()

    def test_outer_expiry_stops_inner_commit_at_manifest_guard(self):
        request = self.branch_request()
        grant = self.branch()
        claims = self.workspace.authority.authenticate(grant["token"], action="context.branch.commit")
        clock = MutableClock(claims["expires_at"] - 2)
        self.workspace.clock = clock
        branch = self.workspace.branches["left"]
        store = self.workspace._store(branch)
        initial = read_json(store.root / "manifest.json")
        def checkpoint(stage):
            if stage == "before_manifest":
                clock.value += 3
        with mock.patch.object(store, "_checkpoint", side_effect=checkpoint), mock.patch.object(self.workspace, "_store", return_value=store):
            with self.assertRaises(AuthorizationError):
                self.commit_branch("left", request)
        self.assertEqual(read_json(store.root / "manifest.json"), initial)

    def test_seal_closes_run_and_denies_exact_commit_replay(self):
        request = self.branch_request()
        self.commit_branch("left", request)
        result = self.seal_branch("left")
        self.assertEqual(result["status"], "sealed")
        with self.assertRaises(AuthorizationError):
            self.commit_branch("left", request)
        with self.assertRaises(AuthorizationError):
            self.read_branch("left")
        with self.assertRaises(AuthorizationError):
            self.workspace.issue(kind="branch", branch_id="left", actions=["context.branch.read"])
        self.assertEqual(self.workspace.recover_branch("left")["status"], "sealed")

    def test_join_barrier_denies_before_any_branch_hydration(self):
        self.seal_branch("left")
        with mock.patch.object(self.workspace, "_store", side_effect=AssertionError("hydrated")), self.assertRaises(ContextRecoveryRequired):
            self.workspace.issue(kind="join", actions=["context.join.read"])

    def test_conflict_join_requires_complete_explicit_choices(self):
        grant, request, result, bundle = self.join(conflict=True)
        chosen = {x["entity_id"]: x["revision_id"] for x in request["selection"]["choices"]}
        self.assertEqual(chosen["claim"], "claim:left")
        self.assertNotIn("claim:right", {r["revision_id"] for r in bundle["graph"]["revisions"]})
        before = (self.workspace.root / "join.json").read_bytes()
        replay = self.workspace.join_prepare(grant["token"], request, run_id=grant["run_id"])
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["join_sha256"], result["join_sha256"])
        self.assertEqual((self.workspace.root / "join.json").read_bytes(), before)
        changed = deepcopy(request)
        changed["selection"]["choices"].pop()
        with self.assertRaises(ContextConflict):
            self.workspace.join_prepare(grant["token"], changed, run_id=grant["run_id"])

    def test_historical_dependency_keeps_exact_old_revision(self):
        _, request, _, bundle = self.join(conflict=True, old_dependency=True)
        self.assertIn("claim:0", {r["revision_id"] for r in bundle["graph"]["revisions"]})
        dependency = next(r for r in bundle["graph"]["relations"] if r["type"] == "depends_on")
        self.assertEqual(dependency["target"]["revision_id"], "claim:0")
        self.assertEqual(request["selection"]["historical"][0]["revision_id"], "claim:0")

    def test_noop_join_has_no_commit_request_and_source_unchanged(self):
        before = (self.source.root / "manifest.json").read_bytes()
        _, _, result, bundle = self.join()
        self.assertTrue(result["no_op"])
        self.assertIsNone(bundle["commit_request"])
        self.assertEqual((self.source.root / "manifest.json").read_bytes(), before)

    def test_exported_join_uses_original_commit_contract_and_cas(self):
        _, _, _, bundle = self.join(conflict=True)
        request = bundle["commit_request"]
        self.assertEqual(set(request), {"version", "request_id", "expected_head", "entities", "revisions", "relations", "evidence"})
        self.assertEqual(request["expected_head"], self.spec["source"]["head"])
        result = self.source.commit(self.source_token, request, run_id="source-run")
        self.assertTrue(self.source.commit(self.source_token, request, run_id="source-run")["replayed"])
        self.assertEqual(result["status"], "committed")

    def test_view_is_frozen_and_never_hydrates_broader_join_on_read(self):
        verifier, manifest, result = self.freeze_view(conflict=True)
        with mock.patch.object(self.workspace, "_load_join", side_effect=AssertionError("broad join read")):
            dto = self.workspace.view_read(verifier["token"], run_id=verifier["run_id"])
        self.assertNotIn("PRIVATE_", canonical_json(dto).decode())
        self.assertEqual(dto["roots"]["claim"]["revision_id"], "claim:left")
        self.assertTrue(self.workspace.build_view(manifest)["replayed"])
        changed = deepcopy(manifest)
        changed["request_id"] = "new-view"
        with self.assertRaises(ContextConflict):
            self.workspace.build_view(changed)
        with self.assertRaises(AuthorizationError):
            self.workspace.branch_read(verifier["token"], run_id=verifier["run_id"])

    def test_missing_public_assumption_rejects_capture_without_output(self):
        bad = deepcopy(self.spec)
        bad["public"] = [r for r in bad["public"] if r["entity_id"] != "assumption"]
        with self.assertRaises(ContextError):
            ContextWorkspace.initialize(self.base / "bad", self.source, self.source_token, run_id="source-run", spec=bad)
        self.assertFalse((self.base / "bad").exists())

    def test_wrong_source_and_production_capture_reject(self):
        for change in ("domain", "store"):
            bad = deepcopy(self.spec)
            if change == "domain":
                bad["domain"] = "production"
            else:
                bad["source"]["store_id"] = "f" * 64
            with self.subTest(change=change), self.assertRaises(ContextError):
                ContextWorkspace.initialize(self.base / "bad", self.source, self.source_token, run_id="source-run", spec=bad)

    def test_read_budget_persists_through_reopen_and_failed_read(self):
        self.read_branch()
        state = read_json(self.workspace.root / "state.json")
        key = "read." + _branch_dir("left")
        self.assertEqual(state["counts"][key], 1)
        self.workspace = ContextWorkspace(self.workspace.root)
        with self.assertRaises(MemoryConflict):
            self.read_branch(head="f" * 64)
        self.assertEqual(read_json(self.workspace.root / "state.json")["counts"][key], 2)
        state = read_json(self.workspace.root / "state.json")
        state["counts"][key] = 64
        write_private(self.workspace.root / "state.json", encoded(state))
        with self.assertRaises(ContextCapacity):
            self.read_branch()

    def test_ordinary_internal_mint_exhaustion_preserves_recovery_reserve(self):
        state = read_json(self.workspace.root / "state.json")
        state["counts"]["mint." + _branch_dir("left")] = 192
        write_private(self.workspace.root / "state.json", encoded(state))
        self.workspace = ContextWorkspace(self.workspace.root)
        with self.assertRaises(ContextCapacity):
            self.read_branch()
        self.assertEqual(self.workspace.recover_branch("left")["status"], "unsealed")
        state = read_json(self.workspace.root / "state.json")
        self.assertEqual(state["counts"]["reserve." + _branch_dir("left")], 1)

    def test_cancelled_branch_commit_does_not_publish(self):
        request = self.branch_request()
        initial = self.read_branch()["head"]
        cancelled = threading.Event()
        cancelled.set()
        grant = self.branch()
        with self.assertRaises(MemoryCancelled):
            self.workspace.branch_commit(grant["token"], request, run_id=grant["run_id"], cancel_event=cancelled)
        self.assertEqual(self.read_branch()["head"], initial)

    def test_seal_intent_crash_blocks_access_and_requires_operator_finish(self):
        def checkpoint(stage):
            if stage == "after_seal_intent":
                raise RuntimeError("synthetic stop")
        with mock.patch.object(self.workspace, "_checkpoint", side_effect=checkpoint), self.assertRaises(RuntimeError):
            self.seal_branch("left")
        with self.assertRaises(AuthorizationError):
            self.read_branch()
        self.workspace = ContextWorkspace(self.workspace.root)
        self.assertEqual(self.workspace.recover_branch("left")["status"], "sealing")
        self.assertEqual(self.workspace.recover_branch("left", apply=True)["status"], "sealed")

    def test_source_advance_rejects_exported_join_cas_without_rebase(self):
        from test_memory_store import request_fixture
        _, _, _, bundle = self.join(conflict=True)
        original = deepcopy(bundle["commit_request"])
        head = self.source.status(self.source_token, run_id="source-run")["head"]
        self.source.commit(self.source_token, request_fixture(head, label="concurrent-source"), run_id="source-run")
        with self.assertRaises(MemoryConflict):
            self.source.commit(self.source_token, original, run_id="source-run")
        self.assertEqual(bundle["commit_request"], original)
        self.assertEqual(original["expected_head"], self.spec["source"]["head"])

    def test_same_scope_other_workspace_capability_rejects_before_store(self):
        second = ContextWorkspace.initialize(self.base / "second-workspace", self.source, self.source_token,
                                             run_id="source-run", spec=self.spec)
        other = second.issue(kind="branch", branch_id="left", actions=["context.branch.read"])
        self.assertEqual(other["run_id"], self.branch()["run_id"])
        with mock.patch.object(self.workspace, "_store", side_effect=AssertionError("hydrated")), self.assertRaises(AuthorizationError):
            self.workspace.branch_read(other["token"], run_id=other["run_id"])

    def test_outer_guard_replay_cannot_skip_late_expiry(self):
        from rethlas_core.memory_store import MemoryStoreError
        request = self.branch_request()
        self.commit_branch("left", request)
        branch = self.workspace.branches["left"]
        store = self.workspace._store(branch)
        token = store.issue(run_id="context-internal", actions=["memory.commit"])
        calls = []
        def guard():
            calls.append(True)
            if len(calls) == 2:
                raise AuthorizationError("synthetic outer expiry")
        with self.assertRaises(AuthorizationError):
            store.commit(token, request, run_id="context-internal", outer_guard=guard)
        self.assertEqual(len(calls), 2)
        with self.assertRaises(MemoryStoreError):
            store.commit(token, request, run_id="context-internal", outer_guard=True)

    def test_closed_state_never_issues_old_token_and_epoch_does_not_rebind_inner_request(self):
        request = self.branch_request()
        initial = self.commit_branch("left", request)
        self.workspace.authority.bump_epoch(self.branch()["run_id"])
        self.grants["left"] = self.workspace.issue(kind="branch", branch_id="left", actions=list(ROLE_ACTIONS["context_branch"]))
        replay = self.commit_branch("left", request)
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["head"], initial["head"])

    def test_all_internal_token_slots_are_finite_with_distinct_reserve(self):
        from rethlas_core.memory_store import MemoryCapacity
        branch = self.workspace.branches["left"]
        with self.workspace._operator() as (_, state):
            for _ in range(191):
                self.workspace._tokens.clear()
                self.workspace._internal(branch, state)
            self.workspace._tokens.clear()
            with self.assertRaises(ContextCapacity):
                self.workspace._internal(branch, state)
            for _ in range(64):
                self.workspace._tokens.clear()
                store, _ = self.workspace._internal(branch, state, maintenance=True)
            self.workspace._tokens.clear()
            with self.assertRaises(ContextCapacity):
                self.workspace._internal(branch, state, maintenance=True)
            registry = read_json(store.root / "authority/registry.json")
            self.assertEqual(len(registry["tokens"]), 256)
            with self.assertRaises(MemoryCapacity):
                store.issue(run_id="context-internal", actions=["memory.read"])
        self.assertEqual(len(read_json(store.root / "authority/registry.json")["tokens"]), 256)

    def test_commit_capacity_does_not_block_exact_committed_replay(self):
        from rethlas_core.memory_store import MemoryCapacity
        from test_memory_store import request_fixture
        head = self.read_branch()["head"]
        requests = []
        for number in range(32):
            request = request_fixture(head, label="quota-%d" % number)
            requests.append(request)
            head = self.commit_branch("left", request)["head"]
        with self.assertRaises(MemoryCapacity):
            self.commit_branch("left", request_fixture(head, label="quota-overflow"))
        self.assertTrue(self.commit_branch("left", requests[0])["replayed"])
        self.assertEqual(self.read_branch()["head"], head)

    def test_published_join_replay_survives_prepare_quota_but_modified_request_does_not(self):
        grant, request, result, _ = self.join()
        state = read_json(self.workspace.root / "state.json")
        state["counts"]["prepare"] = 32
        write_private(self.workspace.root / "state.json", encoded(state))
        replay = self.workspace.join_prepare(grant["token"], request, run_id=grant["run_id"])
        self.assertEqual(replay["join_sha256"], result["join_sha256"])
        changed = deepcopy(request)
        changed["request_id"] = "replacement"
        with self.assertRaises(ContextConflict):
            self.workspace.join_prepare(grant["token"], changed, run_id=grant["run_id"])

    def test_partial_workspace_is_never_reinitialized_as_empty(self):
        (self.workspace.root / "workspace.json").unlink()
        with self.assertRaises(FileNotFoundError):
            ContextWorkspace(self.workspace.root)
        with self.assertRaises(ContextConflict):
            ContextWorkspace.initialize(self.workspace.root, self.source, self.source_token, run_id="source-run", spec=self.spec)

    def test_altered_snapshot_or_unexpected_file_rejects_before_new_write(self):
        request = self.branch_request()
        path = self.workspace.root / "workspace.json"
        original = path.read_bytes()
        config = read_json(path)
        config["capture"]["request_id"] = "altered"
        write_private(path, encoded(config))
        with self.assertRaises(ContextError):
            self.commit_branch("left", request)
        write_private(path, original)
        write_private(self.workspace.root / "unexpected.json", b"{}")
        with self.assertRaises(ContextError):
            self.commit_branch("left", request)

    def test_committed_pending_manifest_replay_remains_available_after_outer_recheck(self):
        request = self.branch_request()
        branch = self.workspace.branches["left"]
        store = self.workspace._store(branch)
        def stop(stage):
            if stage == "after_manifest":
                raise OSError("synthetic reply loss")
        from rethlas_core.memory_store import MemoryIndeterminate
        with mock.patch.object(store, "_checkpoint", side_effect=stop), mock.patch.object(self.workspace, "_store", return_value=store):
            with self.assertRaises(MemoryIndeterminate):
                self.commit_branch("left", request)
        self.workspace = ContextWorkspace(self.workspace.root)
        self.assertTrue(self.commit_branch("left", request)["replayed"])
        self.assertEqual(self.workspace.recover_branch("left", apply=True)["status"], "unsealed")

    def test_join_business_barrier_never_spends_operator_reserve(self):
        grant, request, _, _ = self.join()
        before = read_json(self.workspace.root / "state.json")["counts"]
        for _ in range(2):
            self.workspace = ContextWorkspace(self.workspace.root)
            self.workspace.join_prepare(grant["token"], request, run_id=grant["run_id"])
            self.workspace.join_read(grant["token"], run_id=grant["run_id"])
        after = read_json(self.workspace.root / "state.json")["counts"]
        for branch in ("left", "right"):
            key = _branch_dir(branch)
            self.assertEqual(after.get("reserve." + key), before.get("reserve." + key))
            self.assertGreater(after["mint." + key], before["mint." + key])

    def test_commit_shape_rejects_before_hydration_and_calls_are_bounded(self):
        request = self.branch_request()
        bad = {**request, "history": "PRIVATE"}
        with mock.patch.object(self.workspace, "_store", side_effect=AssertionError("hydrated")), self.assertRaises(ValueError):
            self.commit_branch("left", bad)
        state = read_json(self.workspace.root / "state.json")
        key = "commit." + _branch_dir("left")
        self.assertNotIn(key, state["counts"])
        self.commit_branch("left", request)
        state = read_json(self.workspace.root / "state.json")
        state["counts"][key] = 128
        write_private(self.workspace.root / "state.json", encoded(state))
        with mock.patch.object(self.workspace, "_store", side_effect=AssertionError("hydrated")), self.assertRaises(ContextCapacity):
            self.commit_branch("left", request)
        self.assertEqual(self.workspace.recover_branch("left")["status"], "unsealed")

    def test_same_new_branch_revision_with_different_edges_cannot_join(self):
        for name in ("left", "right"):
            request = self.branch_request(name)
            for revision in request["revisions"]:
                revision["revision_id"] = revision["entity_id"] + ":same"
                revision["body"] = "相同修订正文 " + revision["entity_id"]
                revision["body_sha256"] = sha(revision["body"].encode())
            for relation in request["relations"]:
                for side in ("source", "target"):
                    if relation[side]["revision_id"].endswith(":" + name):
                        relation[side]["revision_id"] = relation[side]["entity_id"] + ":same"
                relation["relation_id"] = name + "/" + relation["type"]
            self.commit_branch(name, request)
            self.seal_branch(name)
        grant = self.workspace.issue(kind="join", actions=["context.join.prepare"])
        choices = active(self.graph)
        for item in choices:
            if item["entity_id"] in ("claim", "proof"):
                item["revision_id"] = item["entity_id"] + ":same"
        request = {"version": VERSION, "request_id": "same-revision", "snapshot_sha256": self.workspace.config["snapshot_sha256"],
                   "selection": {"choices": choices, "historical": []}}
        with self.assertRaisesRegex(ContextConflict, "outgoing structure"):
            self.workspace.join_prepare(grant["token"], request, run_id=grant["run_id"])
        self.assertFalse((self.workspace.root / "join.json").exists())

    def test_published_join_conflicting_calls_have_bounded_hydration_admission(self):
        grant, request, _, _ = self.join()
        changed = deepcopy(request)
        changed["request_id"] = "conflicting"
        before = read_json(self.workspace.root / "state.json")["counts"].get("join.calls", 0)
        with self.assertRaises(ContextConflict):
            self.workspace.join_prepare(grant["token"], changed, run_id=grant["run_id"])
        state = read_json(self.workspace.root / "state.json")
        self.assertEqual(state["counts"]["join.calls"], before + 1)
        state["counts"]["join.calls"] = 128
        write_private(self.workspace.root / "state.json", encoded(state))
        with mock.patch.object(self.workspace, "_load_join", side_effect=AssertionError("hydrated")), self.assertRaises(ContextCapacity):
            self.workspace.join_prepare(grant["token"], request, run_id=grant["run_id"])

    def test_production_context_grant_is_not_accepted_in_fixture_workspace(self):
        authority = self.workspace.authority
        authority.register_run("owner", "production-context", "production")
        token = authority.issue(owner="owner", run_id="production-context", domain="production",
                                role="context_branch", actions=["context.branch.read"])
        with mock.patch.object(self.workspace, "_store", side_effect=AssertionError("hydrated")), self.assertRaises(AuthorizationError):
            self.workspace.branch_read(token, run_id="production-context")

    def test_locked_close_does_not_reenter_registry(self):
        grant = self.branch()
        with self.workspace.authority._registry() as registry:
            claims = self.workspace.authority._check_locked(registry, grant["token"], owner="owner", run_id=grant["run_id"],
                domain="fixture", action="context.branch.read")
            self.assertEqual(claims["role"], "context_branch")
            with mock.patch.object(self.workspace.authority, "_registry", side_effect=AssertionError("reentrant")):
                self.workspace.authority._close_run_locked(registry, grant["run_id"])
        with self.assertRaises(AuthorizationError):
            self.read_branch()


if __name__ == "__main__":
    unittest.main()
