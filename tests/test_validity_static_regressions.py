"""Unexecuted regression definitions for statically identified T09 repairs.

These tests do not constitute a runtime result, mathematical review, or final
acceptance. Real fixture probes are required by the replay tests when authorized.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import unittest
from unittest import mock

from rethlas_core.authorization import AuthorizationError, ROLE_ACTIONS
from rethlas_core.memory_store import MemoryStore, MemoryCancelled
from rethlas_core.private_fs import digest, strict_json_loads
from rethlas_core.validity import (
    VERSION, ValidityError, ValidityConflict, ValidityCapacity, ValidityRecoveryRequired,
    ValidityIndeterminate,
)
from rethlas_core.validity_service import CurrentValidityService, ROOT_BYTES, CONTROL_BYTES, ROOT_FILES

from test_memory_store import MutableClock, write_private
from test_validity import ValidityFixtureMixin


class InterruptedControl(RuntimeError):
    pass


class InterruptedService(CurrentValidityService):
    """Trusted Python-only stop after a selected control write, not a CLI option."""
    def __init__(self, root, *, kind, stage):
        self.selected_kind = kind
        self.selected_stage = stage
        self.active_kind = None
        self.stopped = False
        super().__init__(root)

    def _change(self, state, kind, mutate, **kwargs):
        previous = self.active_kind
        self.active_kind = kind
        try:
            return super()._change(state, kind, mutate, **kwargs)
        finally:
            self.active_kind = previous

    def _checkpoint(self, stage):
        if not self.stopped and self.active_kind == self.selected_kind and stage == self.selected_stage:
            self.stopped = True
            raise InterruptedControl("synthetic control boundary stop")


class ValidityStaticRegressionTests(ValidityFixtureMixin, unittest.TestCase):
    def prepared_graph_accept(self, *, request_id="prepared-graph"):
        request = self.next_request(request_id=request_id)
        service = InterruptedService(self.root, kind="GRAPH_ACCEPT", stage="after_control_prepare")
        with self.assertRaises(InterruptedControl):
            service.commit(self.writer["token"], request, run_id=self.writer["run_id"])
        self.fresh()
        self.assertEqual(self.service.recover(kind="control")["status"], "prepared")
        self.assertIsNotNone(self.state()["pending_graph"])
        self.assertNotEqual(strict_json_loads((self.root / "memory/manifest.json").read_bytes())["head"],
                            self.state()["accepted_graph"]["head"])
        return request

    def binding(self, request_id):
        return next(record for record in self.state()["requests"].values() if record["request_id"] == request_id)

    def observe_and_publish(self):
        probe_request = {"version": VERSION, "request_id": "observed-probe"}
        observed = self.call("probe", self.verifier, probe_request)
        self.assertEqual(observed["probe"]["status"], "accepted")
        self.assertTrue(observed["probe"]["observation"]["input_readonly"])
        publication_request = {"version": VERSION, "request_id": "observed-publication",
                               "probe_id": observed["probe"]["probe_id"]}
        publication = self.call("publish", self.publisher, publication_request)
        return probe_request, observed, publication_request, publication

    def test_prepared_graph_accept_reconciles_expired_original_writer_without_new_grant(self):
        claims = self.service.authority.authenticate(self.writer["token"], action="validity.commit")
        request = self.prepared_graph_accept()
        before = (self.root / "memory/manifest.json").read_bytes()
        self.service.clock = MutableClock(claims["expires_at"] + 1)
        with self.assertRaises(ValidityConflict):
            self.service.recover(kind="control", apply=True, cancel=True)
        result = self.service.recover(kind="control", apply=True)
        self.assertEqual(result["status"], "recovered")
        self.assertEqual(self.binding(request["request_id"])["status"], "committed")
        self.assertIsNone(self.state()["pending_graph"])
        self.assertEqual((self.root / "memory/manifest.json").read_bytes(), before)
        self.assertEqual(self.state()["accepted_graph"]["head"], strict_json_loads(before)["head"])
        with self.assertRaises(AuthorizationError):
            self.call("commit", self.writer, request)

    def test_prepared_graph_accept_reconciles_closed_writer_and_cannot_terminate_commit(self):
        request = self.prepared_graph_accept()
        committed_manifest = (self.root / "memory/manifest.json").read_bytes()
        self.service.authority.close_run(self.writer["run_id"])
        with self.assertRaises(ValidityConflict):
            self.service.recover(kind="control", apply=True, cancel=True)
        self.assertEqual(self.binding(request["request_id"])["status"], "bound")
        self.service.recover(kind="control", apply=True)
        self.assertEqual(self.binding(request["request_id"])["status"], "committed")
        self.assertIsNone(self.state()["pending_graph"])
        self.assertEqual((self.root / "memory/manifest.json").read_bytes(), committed_manifest)
        self.assertTrue(self.current()["fixture_input_eligible"])

    def test_prepared_graph_accept_never_recreates_a_commit_removed_from_disk(self):
        original_manifest = (self.root / "memory/manifest.json").read_bytes()
        request = self.prepared_graph_accept()
        write_private(self.root / "memory/manifest.json", original_manifest)
        with self.assertRaises(ValidityError):
            self.service.recover(kind="control", apply=True)
        self.assertEqual((self.root / "memory/manifest.json").read_bytes(), original_manifest)
        self.assertIsNotNone(self.state()["pending_graph"])
        self.assertEqual(self.binding(request["request_id"])["status"], "bound")
        with self.assertRaises(ValidityRecoveryRequired):
            self.call("read", self.reader)

    def test_pending_recovery_materializes_accepted_operator_count64_without_extra_admission(self):
        with self.service._operator() as (_, state):
            self.service._change(state, "RECOVERY", lambda n: n["counts"].update(operator=63), reserved=True)
        service = InterruptedService(self.root, kind="ADMISSION", stage="after_control_state")
        request = self.registration_request(use_id="unpublished-extra-use")
        with self.assertRaises(ValidityIndeterminate):
            service.register_use(request)
        self.fresh()
        self.assertEqual(self.state()["counts"]["operator"], 64)
        transition = strict_json_loads((self.root / "control/transition.json").read_bytes())
        self.assertEqual(self.service.recover(kind="control")["status"], "pending")
        result = self.service.recover(kind="control", apply=True)
        self.assertEqual(result["status"], "recovered")
        self.assertEqual(self.state()["sequence"], transition["next_sequence"])
        self.assertEqual(self.state()["counts"]["operator"], 64)
        anchor = strict_json_loads((self.root / "control/anchor.json").read_bytes())
        self.assertEqual(anchor["phase"], "stable")
        self.assertEqual(anchor["state_sha256"], digest(self.state()))
        self.assertNotIn("unpublished-extra-use", self.state()["uses"])

    def test_accepted_probe_and_publication_replays_consume_read_budget_and_stop_at_limit(self):
        probe_request, observed, publication_request, publication = self.observe_and_publish()
        key = "read." + digest("main")
        before = self.state()["counts"].get(key, 0)
        probe_count = self.state()["counts"]["probe." + digest("main")]
        publish_count = self.state()["counts"]["publish." + digest("main")]
        with mock.patch("rethlas_core.worker.SandboxWorker.run_bounded", side_effect=AssertionError("no child on historical replay")):
            replayed_probe = self.call("probe", self.verifier, probe_request)
            replayed_publication = self.call("publish", self.publisher, publication_request)
            self.assertEqual(replayed_probe["probe"], observed["probe"])
            self.assertEqual(replayed_publication["receipt"], publication["receipt"])
            self.assertEqual(self.state()["counts"][key], before + 2)
            self.assertEqual(self.state()["counts"]["probe." + digest("main")], probe_count)
            self.assertEqual(self.state()["counts"]["publish." + digest("main")], publish_count)
            with self.service._operator() as (_, state):
                self.service._change(state, "RECOVERY", lambda n: n["counts"].update({key: 64}), reserved=True)
            for method, grant, request in (("probe", self.verifier, probe_request),
                                           ("publish", self.publisher, publication_request)):
                with self.subTest(method=method), self.assertRaises(ValidityCapacity):
                    self.call(method, grant, request)
        self.assertEqual(self.state()["counts"][key], 64)
        self.assertEqual(self.state()["probes"][observed["probe"]["probe_id"]], observed["probe"])
        self.assertEqual(self.state()["publications"][publication["receipt"]["publication_id"]], publication["receipt"])

    def test_replay_rechecks_outer_expiry_after_overlay_not_just_inner_memory_lifetime(self):
        probe_request, _, publication_request, _ = self.observe_and_publish()
        clock = MutableClock()
        self.service.clock = clock
        for method, role, action, request in (("probe", "validity_verifier", "validity.probe", probe_request),
                                             ("publish", "validity_publisher", "validity.publish", publication_request)):
            with self.subTest(method=method):
                grant = self.service.issue(role=role, actions=sorted(ROLE_ACTIONS[role]), use_id="main", ttl_seconds=10)
                claims = self.service.authority.authenticate(grant["token"], action=action, now=clock())
                original_use = self.service._use

                def use(state, use_id):
                    result = original_use(state, use_id)
                    clock.value = claims["expires_at"] + 1
                    return result

                with mock.patch.object(self.service, "_use", side_effect=use), self.assertRaises(AuthorizationError):
                    self.call(method, grant, request)
        self.assertEqual(self.state()["counts"]["read." + digest("main")], 2)

    def test_ordinary_capacity_preserves_bytes_and_file_reserve_but_reserved_write_can_fit(self):
        size = 4096
        usage = {"bytes": ROOT_BYTES - 2 * size - 1, "control_bytes": CONTROL_BYTES - 2 * size - 1,
                 "files": ROOT_FILES - 5, "sizes": {}}
        with mock.patch.object(self.service, "_inventory", return_value=usage):
            with self.assertRaises(ValidityCapacity):
                self.service._reserve(size)
            self.service._reserve(size, reserved=True)
        full_files = {**usage, "bytes": 0, "control_bytes": 0, "files": ROOT_FILES - 3}
        with mock.patch.object(self.service, "_inventory", return_value=full_files), self.assertRaises(ValidityCapacity):
            self.service._reserve(size, reserved=True)

    def test_withdrawal_marks_every_control_and_operation_write_reserved(self):
        calls = []

        def reserve(size, *, reserved=False):
            calls.append((size, reserved))
            if not reserved:
                raise ValidityCapacity("synthetic ordinary safety reserve exhausted")

        with mock.patch.object(self.service, "_reserve", side_effect=reserve), mock.patch.object(
            self.service.journal, "reserve", side_effect=reserve
        ):
            with self.assertRaises(ValidityCapacity):
                self.call("read", self.reader)
            denied_calls = len(calls)
            result = self.call("withdraw", self.withdrawer, self.withdraw_request())
        self.assertEqual(result["status"], "withdrawn")
        self.assertTrue(calls[:denied_calls] and not calls[0][1])
        self.assertTrue(calls[denied_calls:])
        self.assertTrue(all(reserved for _, reserved in calls[denied_calls:]))
        self.assertEqual(self.state()["withdrawal_version"], 1)
        self.assertFalse(self.current()["fixture_input_eligible"])

    def test_graph_accept_postyield_guard_failure_is_indeterminate_not_no_effect(self):
        request = self.next_request()
        original = MemoryStore.locked_checkpoint
        fired = False

        @contextmanager
        def checkpoint(store, token, *, run_id, cancel_event=None):
            nonlocal fired
            with original(store, token, run_id=run_id, cancel_event=cancel_event) as value:
                yield value
                binding = next((b for b in self.state()["requests"].values() if b["request_id"] == request["request_id"]), None)
                if not fired and binding is not None and binding["status"] == "committed":
                    fired = True
                    raise AuthorizationError("synthetic internal postyield expiry")

        with mock.patch.object(MemoryStore, "locked_checkpoint", new=checkpoint), self.assertRaises(ValidityIndeterminate):
            self.call("commit", self.writer, request)
        self.assertTrue(fired)
        self.assertEqual(self.binding(request["request_id"])["status"], "committed")
        self.assertIsNone(self.state()["pending_graph"])
        self.assertEqual(self.state()["accepted_graph"]["head"], strict_json_loads((self.root / "memory/manifest.json").read_bytes())["head"])
        replay = self.call("commit", self.writer, request)
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["head"], self.state()["accepted_graph"]["head"])

    def test_prepared_recovery_postyield_guard_failure_is_indeterminate_and_keeps_accepted_fact(self):
        request = self.prepared_graph_accept()
        original = MemoryStore.locked_checkpoint
        fired = False

        @contextmanager
        def checkpoint(store, token, *, run_id, cancel_event=None):
            nonlocal fired
            with original(store, token, run_id=run_id, cancel_event=cancel_event) as value:
                yield value
                if not fired and self.binding(request["request_id"])["status"] == "committed":
                    fired = True
                    raise MemoryCancelled("synthetic recovery postyield cancellation")

        with mock.patch.object(MemoryStore, "locked_checkpoint", new=checkpoint), self.assertRaises(ValidityIndeterminate):
            self.service.recover(kind="control", apply=True)
        self.assertTrue(fired)
        self.assertEqual(self.binding(request["request_id"])["status"], "committed")
        self.assertIsNone(self.state()["pending_graph"])
        self.assertEqual(self.service.recover(kind="control")["status"], "stable")
        self.assertTrue(self.current()["fixture_input_eligible"])

    def test_event_cursor_pages_once_then_rejects_replay_and_unrelated_control_change(self):
        with self.service._operator() as (_, state):
            semantic_count = sum(event["kind"] != "ADMISSION" for event in state["events"])
            for _ in range(max(0, 129 - semantic_count)):
                self.service._change(state, "RECOVERY", lambda n: None, reserved=True)
        expected = [deepcopy(event) for event in self.state()["events"] if event["kind"] != "ADMISSION"]
        first = self.service.events()
        self.assertEqual(first["entries"], expected[:128])
        self.assertIsNotNone(first["next_cursor"])
        second = self.service.events(cursor=first["next_cursor"])
        self.assertEqual(second["entries"], expected[128:])
        self.assertIsNone(second["next_cursor"])
        self.assertEqual(first["entries"] + second["entries"], expected)
        with self.assertRaises(ValidityConflict):
            self.service.events(cursor=first["next_cursor"])
        fresh = self.service.events()
        self.assertIsNotNone(fresh["next_cursor"])
        self.call("status", self.reader)
        with self.assertRaises(ValidityConflict):
            self.service.events(cursor=fresh["next_cursor"])

    def test_valid_but_unknown_higher_raw_manifest_is_not_automatically_accepted(self):
        accepted = deepcopy(self.state()["accepted_graph"])
        store = self.service._store()
        token = store.issue(run_id="raw-operator-negative", actions=sorted(ROLE_ACTIONS["memory"]))
        request = self.next_request(request_id="raw-unmanaged")
        raw = store.commit(token, request, run_id="raw-operator-negative")
        self.assertGreater(strict_json_loads((self.root / "memory/manifest.json").read_bytes())["generation"], accepted["generation"])
        self.assertNotEqual(raw["head"], accepted["head"])
        for method, grant, payload in (("read", self.reader, None), ("status", self.reader, None),
                                        ("probe", self.verifier, {"version": VERSION, "request_id": "must-not-dispatch"}),
                                        ("publish", self.publisher, {"version": VERSION, "request_id": "must-not-publish", "probe_id": "0" * 64})):
            with self.subTest(method=method), self.assertRaises(ValidityError):
                self.call(method, grant, payload)
        self.assertEqual(self.state()["accepted_graph"], accepted)
        self.assertIsNone(self.state()["pending_graph"])
        self.assertEqual(self.state()["publications"], {})
        self.assertEqual(strict_json_loads((self.root / "memory/manifest.json").read_bytes())["head"], raw["head"])


if __name__ == "__main__":
    unittest.main()
