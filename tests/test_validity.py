"""Synthetic managed-current-use tests; not mathematical or human acceptance."""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from rethlas_core.authorization import AuthorizationError, ROLE_ACTIONS
from rethlas_core.memory_store import MemoryStore, MemoryCancelled
from rethlas_core.private_fs import canonical_json, digest, strict_json_loads
from rethlas_core.validity import (
    VERSION, ValidityError, ValidityConflict, ValidityCapacity,
    ValidityRecoveryRequired, ValidityIndeterminate,
)
from rethlas_core.validity_service import CurrentValidityService, INTERNAL_RUN
from rethlas_core.validity_protocol import protocol, STATE_BYTES
from test_context_views import mathematical_fixture, manifest_for, edge
from test_memory_store import MutableClock, sha, write_private


class ValidityFixtureMixin:
    def setUp(self):
        super().setUp()
        self.temporary = tempfile.TemporaryDirectory(prefix="validity-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "service"
        self.service = CurrentValidityService.initialize(self.root, owner="owner")
        self.graph = mathematical_fixture()
        self.writer = self.grant("validity_writer")
        self.withdrawer = self.grant("validity_withdrawer")
        self.seed = {"version": 1, "request_id": "seed", "expected_head": self.service.config["genesis"]["head"],
                     **{k: self.graph[k] for k in ("entities", "revisions", "relations", "evidence")}}
        self.committed = self.call("commit", self.writer, self.seed)
        self.registration = self.registration_request()
        self.service.register_use(self.registration)
        self.reader = self.grant("validity_reader", "main")
        self.verifier = self.grant("validity_verifier", "main")
        self.publisher = self.grant("validity_publisher", "main")

    def grant(self, role, use_id=None):
        return self.service.issue(role=role, actions=sorted(ROLE_ACTIONS[role]), use_id=use_id)

    def call(self, method, grant, request=None, **kwargs):
        args = [grant["token"]]
        if request is not None:
            args.append(request)
        return getattr(self.service, method)(*args, run_id=grant["run_id"], **kwargs)

    def registration_request(self, *, use_id="main", graph=None, head=None):
        manifest = manifest_for(graph or self.graph)
        return {"version": VERSION, "request_id": "register-" + use_id, "use_id": use_id,
                "expected_head": head or self.committed["head"],
                **{k: manifest[k] for k in ("roots", "dependencies", "references")}}

    def ref(self, revision_id):
        revision = next(r for r in self.graph["revisions"] if r["revision_id"] == revision_id)
        return {k: revision[k] for k in ("entity_id", "revision_id", "body_sha256")}

    def withdraw_request(self, target="assumption:0", request_id="withdraw"):
        return {"version": VERSION, "request_id": request_id, "targets": [self.ref(target)], "reason": "SYNTHETIC_PRIVATE_REASON"}

    def state(self):
        return strict_json_loads((self.root / "control/state.json").read_bytes())

    def current(self):
        return self.call("status", self.reader)["current"]

    def fresh(self):
        self.service = CurrentValidityService(self.root)
        return self.service

    def next_request(self, *, request_id="next", dependency="assumption:0"):
        revision = deepcopy(next(r for r in self.graph["revisions"] if r["revision_id"] == "claim:0"))
        revision.update(revision_id="claim:" + request_id, body="新的合成待审命题 " + request_id)
        revision["body_sha256"] = sha(revision["body"].encode())
        return {"version": 1, "request_id": request_id, "expected_head": self.state()["accepted_graph"]["head"],
                "entities": [], "revisions": [revision], "relations": [edge(revision["revision_id"], dependency, "depends_on")], "evidence": []}


class ValidityServiceTests(ValidityFixtureMixin, unittest.TestCase):
    def test_current_read_and_exact_lookup_do_not_promote_mathematics(self):
        value = self.call("read", self.reader)
        self.assertTrue(value["current"]["fixture_input_eligible"])
        self.assertFalse(value["current"]["mathematical_review"])
        self.assertFalse(value["current"]["production_completed"])
        self.assertNotIn("PRIVATE_", canonical_json(value).decode())
        result = self.call("lookup", self.reader, {"version": VERSION, "refs": [self.ref("source:0")], "depth": 0})
        self.assertEqual(result["retrieval_kind"], "exact_id_not_embedding")
        self.assertEqual(result["revisions"][0]["use"], "historical_citation_only")
        self.assertTrue(result["revisions"][0]["not_authorized_as_premise"])

    def test_withdrawal_preserves_graph_dto_and_blocks_current_use(self):
        graph_files = {str(p.relative_to(self.root)): p.read_bytes() for folder in ("memory/graphs", "memory/bodies", "uses")
                       for p in (self.root / folder).iterdir() if p.is_file()}
        result = self.call("withdraw", self.withdrawer, self.withdraw_request())
        self.assertEqual(result["status"], "withdrawn")
        self.assertFalse(self.current()["fixture_input_eligible"])
        with self.assertRaises(ValidityError):
            self.call("read", self.reader)
        with self.assertRaises(ValidityError):
            self.call("lookup", self.reader, {"version": VERSION, "refs": [self.ref("proof:0")], "depth": 1})
        for name, content in graph_files.items():
            self.assertEqual((self.root / name).read_bytes(), content)
        self.assertNotIn("SYNTHETIC_PRIVATE_REASON", canonical_json(self.current()).decode())
        self.fresh()
        self.assertFalse(self.current()["fixture_input_eligible"])

    def test_citation_only_withdrawal_does_not_block_math_support(self):
        before = self.call("read", self.reader)["dto"]
        self.call("withdraw", self.withdrawer, self.withdraw_request("source:0"))
        after = self.call("read", self.reader)
        self.assertEqual(before, after["dto"])
        self.assertTrue(after["current"]["fixture_input_eligible"])
        citation = after["current"]["overlay"]["citation_only"][0]
        self.assertTrue(citation["withdrawn"])
        self.assertTrue(citation["not_authorized_as_premise"])

    def test_exact_replay_new_identity_token_and_conflicting_request(self):
        request = self.withdraw_request()
        first = self.call("withdraw", self.withdrawer, request)
        second = self.call("withdraw", self.grant("validity_withdrawer"), request)
        self.assertEqual(first["withdrawal_id"], second["withdrawal_id"])
        self.assertTrue(second["replayed"])
        changed = deepcopy(request)
        changed["reason"] = "different"
        with self.assertRaises(ValidityConflict):
            self.call("withdraw", self.withdrawer, changed)
        self.assertEqual(self.state()["withdrawal_version"], 1)

    def test_request_id_binding_cross_kind_does_not_free_on_rejection(self):
        bad = self.withdraw_request()
        bad["targets"][0]["body_sha256"] = "0" * 64
        with self.assertRaises(ValidityError):
            self.call("withdraw", self.withdrawer, bad)
        with self.assertRaises(ValidityConflict):
            self.call("withdraw", self.withdrawer, self.withdraw_request())

    def test_new_candidate_depending_withdrawn_input_is_committed_but_blocked(self):
        self.call("withdraw", self.withdrawer, self.withdraw_request())
        q = self.next_request()
        self.call("commit", self.writer, q)
        entries = self.service.review_queue()["entries"]
        self.assertTrue(any(e["revision"]["revision_id"] == "claim:next" and e["kind"] == "math_dependency_blocked" for e in entries))
        audit = self.service.audit()
        self.assertTrue(audit["validity"]["claim:next"]["math_blocked"])
        self.assertEqual(audit["graph"]["graph"]["revisions"][-1]["assertion_state"], "candidate")

    def test_old_revision_cannot_gain_new_outgoing_relation(self):
        q = {"version": 1, "request_id": "backfill", "expected_head": self.committed["head"],
             "entities": [], "revisions": [], "relations": [edge("claim:0", "assumption:0", "depends_on")], "evidence": []}
        with self.assertRaises(ValueError):
            self.call("commit", self.writer, q)
        self.assertEqual(self.state()["accepted_graph"]["head"], self.committed["head"])
        self.assertIsNone(self.state()["pending_graph"])

    def test_original_committed_replay_never_regresses_accepted_floor(self):
        result = self.call("commit", self.writer, self.next_request())
        replay = self.call("commit", self.writer, self.seed)
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["head"], self.committed["head"])
        self.assertEqual(self.state()["accepted_graph"]["head"], result["head"])

    def test_current_operations_never_hydrate_active_graph(self):
        with mock.patch.object(MemoryStore, "_chain", side_effect=AssertionError("wide hydration")), \
             mock.patch.object(MemoryStore, "_graph", side_effect=AssertionError("wide hydration")):
            self.call("read", self.reader)
            self.call("status", self.reader)
            self.call("lookup", self.reader, {"version": VERSION, "refs": [self.ref("proof:0")], "depth": 2})

    def test_foreign_role_resource_scope_reject_before_dto_read(self):
        with mock.patch.object(self.service, "_use", side_effect=AssertionError("must authenticate first")):
            for grant, method in ((self.writer, "read"), (self.reader, "probe"), (self.withdrawer, "status")):
                with self.subTest(method=method), self.assertRaises(AuthorizationError):
                    self.call(method, grant, {"version": VERSION, "request_id": "x"} if method == "probe" else None)
            with self.assertRaises(AuthorizationError):
                self.service.read(self.reader["token"], run_id=self.writer["run_id"])

    def test_expired_revoked_and_old_epoch_reject(self):
        token = self.reader["token"]
        claims = self.service.authority.authenticate(token, action="validity.read")
        self.service.authority.revoke(claims["token_id"])
        with self.assertRaises(AuthorizationError):
            self.call("read", self.reader)
        reader = self.grant("validity_reader", "main")
        self.service.authority.bump_epoch(reader["run_id"])
        with self.assertRaises(AuthorizationError):
            self.call("read", reader)
        with mock.patch.object(self.service, "clock", return_value=claims["expires_at"] + 1), self.assertRaises(AuthorizationError):
            self.call("read", self.verifier)

    def test_unknown_or_modified_revision_is_not_unknown_equal_eligible(self):
        bad = deepcopy(self.registration)
        bad.update(request_id="bad-use", use_id="bad")
        bad["roots"]["proof"]["revision_id"] = "uncommitted-branch:0"
        with self.assertRaises(ValueError):
            self.service.register_use(bad)
        with self.assertRaises(ValueError):
            self.call("lookup", self.reader, {"version": VERSION, "refs": [{**self.ref("proof:0"), "body_sha256": "0" * 64}], "depth": 0})

    def test_input_shape_float_bool_and_production_reject(self):
        for kind, value in (("lookup", {"version": VERSION, "refs": [self.ref("proof:0")], "depth": 1.0}),
                            ("lookup", {"version": VERSION, "refs": [self.ref("proof:0")], "depth": True}),
                            ("publish", {"version": VERSION, "request_id": "x", "probe_id": "0" * 64, "correct": True})):
            with self.subTest(kind=kind), self.assertRaises(ValidityError):
                protocol(value, kind)
        with self.assertRaises(ValidityError):
            CurrentValidityService.initialize(self.base / "production", owner="owner", domain="production")

    def test_cancellation_before_action_leaves_no_withdrawal(self):
        event = threading.Event()
        event.set()
        with self.assertRaises(MemoryCancelled):
            self.call("withdraw", self.withdrawer, self.withdraw_request(), cancel_event=event)
        self.assertEqual(self.state()["withdrawal_version"], 0)

    def test_raw_newer_graph_and_rollback_manifest_fail_closed(self):
        old = (self.root / "memory/manifest.json").read_bytes()
        result = self.call("commit", self.writer, self.next_request())
        write_private(self.root / "memory/manifest.json", old)
        with self.assertRaises(ValidityError):
            self.call("read", self.reader)
        self.assertEqual(self.state()["accepted_graph"]["head"], result["head"])

    def test_single_control_state_or_anchor_rollback_fails_closed(self):
        old_state = (self.root / "control/state.json").read_bytes()
        old_anchor = (self.root / "control/anchor.json").read_bytes()
        self.call("withdraw", self.withdrawer, self.withdraw_request())
        new_state = (self.root / "control/state.json").read_bytes()
        write_private(self.root / "control/state.json", old_state)
        with self.assertRaises(ValidityError):
            self.call("read", self.reader)
        write_private(self.root / "control/state.json", new_state)
        write_private(self.root / "control/anchor.json", old_anchor)
        with self.assertRaises(ValidityError):
            self.call("read", self.reader)

    def test_dto_tamper_and_missing_transition_fail_closed(self):
        dto = next((self.root / "uses").glob("*.json"))
        before = dto.read_bytes()
        write_private(dto, b"{}")
        with self.assertRaises(ValidityError):
            self.call("read", self.reader)
        write_private(dto, before)
        (self.root / "control/transition.json").unlink()
        with self.assertRaises(ValidityError):
            self.call("read", self.reader)

    def test_single_use_budget_is_durable_across_reopen(self):
        key = "read." + digest("main")
        with self.service._operator() as (_, state):
            self.service._change(state, "ADMISSION", lambda n: n["counts"].update({key: 64}))
        self.fresh()
        with self.assertRaises(ValidityCapacity):
            self.call("read", self.reader)

    def test_prepared_request_requires_operator_and_cancellation_keeps_binding(self):
        original = self.service._checkpoint
        def fail(stage):
            if stage == "after_control_prepare":
                raise OSError("synthetic prepared interruption")
        self.service._checkpoint = fail
        self.service.journal.checkpoint = fail
        with self.assertRaises(OSError):
            self.call("withdraw", self.withdrawer, self.withdraw_request())
        self.service._checkpoint = original
        self.service.journal.checkpoint = original
        with self.assertRaises(ValidityRecoveryRequired):
            self.call("read", self.reader)
        self.service.recover(kind="control", apply=True, cancel=True)
        # This interruption was the admission before a request binding: no target
        # withdrawal was accepted, but the failed admission is not refunded.
        self.assertEqual(self.state()["withdrawal_version"], 0)
        self.assertGreater(self.state()["counts"].get("withdraw.calls", 0), 0)

    def test_real_sandbox_probe_publication_and_withdrawn_historical_replay(self):
        request = {"version": VERSION, "request_id": "probe"}
        observed = self.call("probe", self.verifier, request)
        self.assertFalse(observed["probe"]["mathematical_review"])
        self.assertTrue(observed["probe"]["observation"]["input_readonly"])
        published_q = {"version": VERSION, "request_id": "publish", "probe_id": observed["probe"]["probe_id"]}
        published = self.call("publish", self.publisher, published_q)
        self.assertEqual(published["status"], "fixture_published")
        self.call("withdraw", self.withdrawer, self.withdraw_request())
        replay = self.call("publish", self.publisher, published_q)
        self.assertEqual(replay["receipt"], published["receipt"])
        self.assertFalse(replay["current"]["fixture_input_eligible"])
        probe_replay = self.call("probe", self.verifier, request)
        self.assertFalse(probe_replay["current"]["fixture_input_eligible"])
        for method, grant, q in (("probe", self.verifier, {"version": VERSION, "request_id": "new-probe"}),
                                  ("publish", self.publisher, {**published_q, "request_id": "new-publish"})):
            with self.subTest(method=method), self.assertRaises(ValidityError):
                self.call(method, grant, q)

    def test_citation_refresh_requires_new_observation_not_math_retraction(self):
        first = self.call("probe", self.verifier, {"version": VERSION, "request_id": "first"})
        self.call("withdraw", self.withdrawer, self.withdraw_request("source:0"))
        with self.assertRaises(ValidityConflict):
            self.call("publish", self.publisher, {"version": VERSION, "request_id": "stale-pub", "probe_id": first["probe"]["probe_id"]})
        self.assertTrue(self.current()["fixture_input_eligible"])
        second = self.call("probe", self.verifier, {"version": VERSION, "request_id": "second"})
        self.call("publish", self.publisher, {"version": VERSION, "request_id": "fresh-pub", "probe_id": second["probe"]["probe_id"]})

    def test_external_observation_or_old_receipt_cannot_publish(self):
        with self.assertRaises(ValidityConflict):
            self.call("publish", self.publisher, {"version": VERSION, "request_id": "foreign", "probe_id": "0" * 64})
        with self.assertRaises(ValidityError):
            self.call("publish", self.publisher, {"version": VERSION, "request_id": "self-report", "correct": True, "receipt": {}})


if __name__ == "__main__":
    unittest.main()
