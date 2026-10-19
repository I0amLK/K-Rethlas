from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from rethlas_core.authorization import AuthorizationError, ROLE_ACTIONS
from rethlas_core.context_views import exact_ref
from rethlas_core.private_fs import canonical_json, digest
from rethlas_core.project_service import ProjectService
from rethlas_core.validity import ValidityError, ValidityConflict, ValidityRecoveryRequired
from test_context_views import mathematical_fixture, edge


class ProjectFixtureMixin:
    def setUp(self):
        super().setUp()
        self.temporary = tempfile.TemporaryDirectory(prefix="research-project-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.service = ProjectService.initialize(self.base / "project", owner="owner")
        self.reader = self.service.issue(role="project_reader")
        self.writer = self.service.issue(role="project_writer")
        self.withdrawer = self.service.issue(role="project_withdrawer")
        self.runner = self.service.issue(role="research_runner")
        self.graph = mathematical_fixture()
        self.seed = {"version": 1, "request_id": "seed", "expected_head": self.service.config["genesis"]["head"],
                     **{k: self.graph[k] for k in ("entities", "revisions", "relations", "evidence")}}
        self.committed = self.call("commit", self.writer, self.seed)

    def call(self, method, grant, *args, **kwargs):
        return getattr(self.service, method)(grant["token"], *args, run_id=grant["run_id"], **kwargs)

    def ref(self, revision_id):
        return exact_ref(next(r for r in self.graph["revisions"] if r["revision_id"] == revision_id))

    def withdraw(self, revision_id, request_id="withdraw"):
        return self.call("withdraw", self.withdrawer, {"version": "project-1", "request_id": request_id,
                         "targets": [self.ref(revision_id)], "reason": "synthetic withdrawal"})


class ProjectServiceTests(ProjectFixtureMixin, unittest.TestCase):
    def test_domain_is_research_and_fixture_interfaces_remain_disabled(self):
        self.assertEqual(self.service.config["domain"], "research")
        self.assertEqual(self.service._store().config["domain"], "research")
        for name in ("register_use", "probe", "publish", "read", "lookup"):
            with self.subTest(name=name), self.assertRaises(AuthorizationError):
                getattr(self.service, name)({})

    def test_arbitrary_exact_context_without_fixed_use_registration(self):
        for _ in range(10):
            result = self.call("prepare_context", self.reader, [self.ref("proof:0")])
            self.assertTrue(result["overlay"]["input_eligible"])
        with self.service._operator() as (_, state):
            self.assertEqual(state["uses"], {})
        ids = {r["revision_id"] for r in result["dto"]["revisions"]}
        self.assertTrue({"proof:0", "claim:0", "definition:0", "assumption:0", "source:0"} <= ids)
        self.assertNotIn("PRIVATE_METADATA_ACTOR", canonical_json(result).decode())

    def test_support_withdrawal_blocks_context_and_preserves_history(self):
        before = {p.name: p.read_bytes() for p in (self.service.root / "memory/bodies").iterdir()}
        self.withdraw("assumption:0")
        with self.assertRaises(ValidityError):
            self.call("prepare_context", self.reader, [self.ref("proof:0")])
        current = self.call("status", self.reader, refs=[self.ref("proof:0")])
        self.assertTrue(current["current"]["proof:0"]["math_blocked"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in (self.service.root / "memory/bodies").iterdir()})

    def test_citation_only_withdrawal_is_not_support(self):
        self.withdraw("source:0")
        value = self.call("prepare_context", self.reader, [self.ref("claim:0")])
        self.assertTrue(value["overlay"]["input_eligible"])
        self.assertTrue(value["overlay"]["citation_only"][0]["withdrawn"])
        self.assertTrue(value["overlay"]["citation_only"][0]["not_authorized_as_premise"])

    def test_cross_project_role_revoke_and_unknown_ref_reject(self):
        other = ProjectService.initialize(self.base / "other", owner="owner")
        with mock.patch.object(other, "_store", side_effect=AssertionError("must authenticate first")):
            with self.assertRaises(AuthorizationError):
                other.status(self.reader["token"], run_id=self.reader["run_id"])
        with self.assertRaises(AuthorizationError):
            self.call("commit", self.reader, self.seed)
        invalid = self.ref("claim:0")
        invalid["revision_id"] = "uncommitted"
        with self.assertRaises(ValueError):
            self.call("get_revision", self.reader, invalid)
        claims = self.service.authority.authenticate(self.reader["token"], action="project.read")
        self.service.authority.revoke(claims["token_id"])
        with self.assertRaises(AuthorizationError):
            self.call("status", self.reader)

    def test_graph_replay_monotone_and_old_outgoing_structure_frozen(self):
        q = {"version": 1, "request_id": "backfill", "expected_head": self.committed["head"],
             "entities": [], "revisions": [], "relations": [edge("claim:0", "assumption:0", "depends_on")], "evidence": []}
        with self.assertRaises(ValueError):
            self.call("commit", self.writer, q)
        self.assertTrue(self.call("commit", self.writer, self.seed)["replayed"])

    def test_navigation_limited_but_prepare_context_complete(self):
        value = self.call("expand", self.reader, [self.ref("proof:0")], depth=0)
        self.assertEqual(len(value["revisions"]), 1)
        self.assertIn("not_complete", value["status"])
        full = self.call("prepare_context", self.reader, [self.ref("proof:0")])
        self.assertGreater(len(full["dto"]["revisions"]), 1)
        with self.assertRaises(ValueError):
            self.call("expand", self.reader, [self.ref("proof:0")], depth=True)

    def test_current_reads_do_not_append_persistent_per_search_counters(self):
        with self.service._operator() as (_, state):
            before = state["sequence"]
        for _ in range(3):
            self.call("status", self.reader)
        with self.service._operator() as (_, state):
            self.assertEqual(before, state["sequence"])

    def test_prepared_research_stage_cannot_be_completed_without_pipeline_checks(self):
        with self.service._locked(self.runner["token"], self.runner["run_id"], "research.run") as (_, state, claims, resource):
            key, _ = self.service._binding(state, "research", {"version": "project-1", "request_id": "prepared-stage"},
                self.service._identity(claims, resource), claims=claims)
            def crash(stage):
                if stage == "after_control_prepare":
                    raise RuntimeError("synthetic prepared stop")
            with mock.patch.object(self.service.journal, "checkpoint", side_effect=crash):
                with self.assertRaises(RuntimeError):
                    self.service._change(state, "RESEARCH_STARTED", lambda n: n["requests"][key].update(status="started"),
                                         key=key, claims=claims)
        with self.assertRaises(ValidityConflict):
            self.service.recover(kind="control", apply=True, token=self.runner["token"], run_id=self.runner["run_id"])
        self.service.recover(kind="control", apply=True, cancel=True)
        with self.service._operator() as (_, state):
            self.assertEqual(state["requests"][key]["status"], "aborted")

    def test_strict_reference_and_relation_containers_fail_closed(self):
        for selected in (None, "proof:0", [None], [{"revision_id": "proof:0"}]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                self.call("prepare_context", self.reader, selected)
        for relation_types in ("depends_on", [None], [], ["unknown"]):
            with self.subTest(relation_types=relation_types), self.assertRaises(ValueError):
                self.call("expand", self.reader, [self.ref("proof:0")], relation_types=relation_types)

    def test_pending_control_stays_fail_closed(self):
        def crash(stage):
            if stage == "after_anchor_pending":
                raise RuntimeError("synthetic interruption")
        with mock.patch.object(self.service.journal, "checkpoint", side_effect=crash):
            with self.assertRaises(ValueError):
                self.withdraw("source:0")
        with self.assertRaises(ValidityRecoveryRequired):
            self.call("status", self.reader)
