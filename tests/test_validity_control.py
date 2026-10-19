"""Direct synthetic control-journal tests; not executed while runtime is blocked."""
from __future__ import annotations

from copy import deepcopy
import errno
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from rethlas_core.authorization import AuthorizationError
from rethlas_core.private_fs import PrivateDirectory, canonical_json, digest
from rethlas_core.validity import (
    VERSION, POLICY_SHA256, ValidityCapacity, ValidityConflict, ValidityError,
    ValidityIndeterminate, ValidityRecoveryRequired,
)
from rethlas_core.validity_control import ControlJournal, stable_anchor


class StopBeforeAcceptance(RuntimeError):
    pass


class ControlJournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="validity-control-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.directory = PrivateDirectory(Path(self.temp.name))
        self.floor = {"head": "1" * 64, "generation": 0, "graph_sha256": "2" * 64}
        self.config = {"version": VERSION, "service_id": "3" * 64, "owner": "owner", "domain": "fixture",
                       "policy_hash": POLICY_SHA256, "store_id": "4" * 64,
                       "store_config_sha256": "5" * 64, "genesis": self.floor}
        self.state = {"version": VERSION, "service_id": self.config["service_id"], "config_sha256": digest(self.config),
                      "sequence": 0, "events": [], "requests": {}, "accepted_graph": self.floor,
                      "pending_graph": None, "uses": {}, "probes": {}, "publications": {}, "runs": {},
                      "counts": {}, "withdrawal_version": 0}
        self.directory.write_json("state.json", self.state)
        self.directory.write_json("anchor.json", stable_anchor(self.config, self.state))
        self.reservations = []
        self.stage = None
        self.failure = StopBeforeAcceptance
        self.journal = ControlJournal(self.directory, self.config, reserve=self.reserve, checkpoint=self.checkpoint)

    def reserve(self, size, *, reserved=False):
        self.reservations.append((size, reserved))

    def checkpoint(self, stage):
        if stage == self.stage:
            raise self.failure("synthetic control interruption")

    def change(self, kind="ADMISSION", mutate=None, **kwargs):
        if mutate is None:
            mutate = lambda n: n["counts"].update({"synthetic": n["counts"].get("synthetic", 0) + 1})
        with self.directory.lock():
            return self.journal.change(self.journal.ready(), kind, mutate, **kwargs)

    def prepared(self, *, kind="ADMISSION", mutate=None, key=None):
        self.stage = "after_control_prepare"
        with self.assertRaises(StopBeforeAcceptance):
            self.change(kind, mutate, request_key=key)
        self.stage = None
        state, anchor, transition, phase = self.journal.inspect()
        self.assertEqual(phase, "prepared")
        return state, anchor, transition

    def binding(self, *, status="bound"):
        return {"request_id": "original-request", "operation_sha256": "6" * 64,
                "identity": {"owner": "owner", "run_id": "writer", "domain": "fixture",
                             "role": "validity_writer", "epoch": 0, "resource": "owned-store"},
                "kind": "commit", "bucket": "ordinary", "status": status, "result": {}}

    def test_ordinary_and_reserved_changes_propagate_space_class_to_every_write(self):
        for reserved in (False, True):
            self.reservations.clear()
            result = self.change(reserved=reserved)
            self.assertEqual(self.journal.ready(), result)
            self.assertEqual(len(self.reservations), 4)
            self.assertEqual({flag for _, flag in self.reservations}, {reserved})

    def test_prepared_transition_blocks_new_work_and_does_not_imply_effect(self):
        state, anchor, transition = self.prepared()
        self.assertEqual(state, self.state)
        self.assertEqual(anchor, stable_anchor(self.config, self.state))
        self.assertEqual(transition["next_state"]["counts"]["synthetic"], 1)
        with self.assertRaises(ValidityRecoveryRequired):
            self.journal.ready()
        with self.assertRaises(ValidityRecoveryRequired):
            self.change()
        self.assertEqual(self.directory.read_json("transition.json"), transition)

    def test_prepared_recovery_reservation_keeps_one_event_and_original_binding(self):
        key = "7" * 64
        _, _, transition = self.prepared(kind="REQUEST_BOUND",
            mutate=lambda n: n["requests"].update({key: self.binding()}), key=key)
        self.reservations.clear()
        updated = self.journal.reserve_prepared_recovery(mint=True)
        self.assertEqual(updated["next_state"]["requests"], transition["next_state"]["requests"])
        self.assertEqual(updated["next_state"]["events"], transition["next_state"]["events"])
        self.assertEqual(updated["next_state"]["counts"]["operator"], 1)
        self.assertEqual(updated["next_state"]["counts"]["internal.reserve"], 1)
        self.assertTrue(all(flag for _, flag in self.reservations))
        result = self.journal.recover(apply=True)
        self.assertEqual(result["counts"]["operator"], 1)
        self.assertEqual(result["sequence"], 1)

    def test_prepared_cancellation_atomically_preserves_original_terminal_binding(self):
        key = "7" * 64
        self.prepared(kind="REQUEST_BOUND", mutate=lambda n: n["requests"].update({key: self.binding()}), key=key)
        self.stage = "after_prepared_abort"
        with self.assertRaises(StopBeforeAcceptance):
            self.journal.recover(apply=True, cancel=True)
        self.stage = None
        state, _, transition, phase = self.journal.inspect()
        self.assertEqual(phase, "prepared")
        self.assertEqual(state["requests"], {})
        self.assertEqual(transition["kind"], "PREPARED_ABORT")
        self.assertEqual(transition["next_state"]["requests"][key]["status"], "aborted")
        result = self.journal.recover(apply=True)
        binding = result["requests"][key]
        self.assertEqual(binding["operation_sha256"], "6" * 64)
        self.assertEqual(binding["status"], "aborted")
        self.assertTrue(all(flag for _, flag in self.reservations[1:]))

    def test_historical_graph_reconciliation_cannot_be_cancelled(self):
        for kind in ("GRAPH_ACCEPT", "GRAPH_ABORT"):
            with self.subTest(kind=kind):
                _, _, transition = self.prepared(kind=kind)
                with self.assertRaises(ValidityConflict):
                    self.journal.recover(apply=True, cancel=True)
                self.assertEqual(self.directory.read_json("transition.json"), transition)
                self.journal.recover(apply=True)

    def test_pending_recovery_materializes_at_event_and_operator_quota_without_admission(self):
        self.stage = "after_anchor_pending"
        def fill(n):
            n["counts"]["operator"] = 64
            n["counts"]["ordinary.events"] = 384
        with self.assertRaises(ValidityIndeterminate):
            self.change(mutate=fill, reserved=True)
        self.stage = None
        before, _, transition, phase = self.journal.inspect()
        self.assertEqual(phase, "pending")
        self.assertEqual(before["sequence"], 0)
        self.reservations.clear()
        result = self.journal.recover(apply=True)
        self.assertEqual(result, transition["next_state"])
        self.assertEqual(result["counts"]["operator"], 64)
        self.assertEqual(result["counts"]["ordinary.events"], 384)
        self.assertTrue(all(flag for _, flag in self.reservations))
        self.assertEqual(len(self.reservations), 2)

    def test_pending_cannot_be_cancelled_or_serve_previous_state(self):
        self.stage = "after_anchor_pending"
        with self.assertRaises(ValidityIndeterminate):
            self.change()
        self.stage = None
        with self.assertRaises(ValidityRecoveryRequired):
            self.journal.ready()
        with self.assertRaises(ValidityConflict):
            self.journal.recover(apply=True, cancel=True)
        with self.assertRaises(ValidityRecoveryRequired):
            self.journal.recover(apply=False)
        self.journal.recover(apply=True)
        self.assertEqual(self.journal.ready()["counts"]["synthetic"], 1)

    def test_after_pending_or_state_or_stable_failure_is_indeterminate_and_forward_only(self):
        for stage in ("after_anchor_pending", "after_control_state", "after_anchor_stable"):
            with self.subTest(stage=stage):
                self.stage = stage
                with self.assertRaises(ValidityIndeterminate):
                    self.change()
                self.stage = None
                _, _, _, phase = self.journal.inspect()
                self.assertEqual(phase, "stable" if stage == "after_anchor_stable" else "pending")
                result = self.journal.recover(apply=True)
                self.assertEqual(self.journal.ready(), result)

    def test_final_guard_failure_retains_prepared_request_and_no_effect(self):
        key = "7" * 64
        def denied():
            raise AuthorizationError("synthetic expired grant")
        with self.assertRaises(AuthorizationError):
            self.change("REQUEST_BOUND", lambda n: n["requests"].update({key: self.binding()}),
                        request_key=key, guard=denied)
        state, _, transition, phase = self.journal.inspect()
        self.assertEqual((state["sequence"], phase), (0, "prepared"))
        self.assertEqual(transition["next_state"]["requests"][key]["request_id"], "original-request")

    def test_transition_write_replace_failure_never_reports_success(self):
        original = PrivateDirectory.write_bytes
        def fail(directory, name, data, *args, **kwargs):
            if directory.root == self.directory.root and name == "transition.json":
                raise OSError(errno.EIO, "synthetic replace failure")
            return original(directory, name, data, *args, **kwargs)
        with mock.patch.object(PrivateDirectory, "write_bytes", fail), self.assertRaises(ValidityIndeterminate):
            self.change()
        self.assertEqual(self.journal.ready(), self.state)

    def test_pending_anchor_directory_fsync_failure_requires_forward_recovery(self):
        original = PrivateDirectory.write_bytes
        failed = False
        def fail_after_write(directory, name, data, *args, **kwargs):
            nonlocal failed
            result = original(directory, name, data, *args, **kwargs)
            if directory.root == self.directory.root and name == "anchor.json" and not failed:
                failed = True
                raise OSError(errno.EIO, "synthetic directory fsync failure after replace")
            return result
        with mock.patch.object(PrivateDirectory, "write_bytes", fail_after_write), self.assertRaises(ValidityIndeterminate):
            self.change()
        self.assertEqual(self.journal.inspect()[3], "pending")
        self.journal.recover(apply=True)
        self.assertEqual(self.journal.ready()["counts"]["synthetic"], 1)

    def test_sync_barrier_failure_cannot_be_reported_as_ready(self):
        original = PrivateDirectory.sync_file
        def fail(directory, name):
            if directory.root == self.directory.root and name == "anchor.json":
                raise OSError(errno.EIO, "synthetic fsync failure")
            return original(directory, name)
        with mock.patch.object(PrivateDirectory, "sync_file", fail), self.assertRaises(ValidityIndeterminate):
            self.journal.ready()
        self.assertEqual(self.journal.ready(), self.state)

    def test_single_old_state_or_anchor_replay_is_rejected(self):
        original_state = self.directory.read_bytes("state.json")
        original_anchor = self.directory.read_bytes("anchor.json")
        committed = self.change()
        new_state = self.directory.read_bytes("state.json")
        new_anchor = self.directory.read_bytes("anchor.json")
        self.directory.write_bytes("state.json", original_state)
        with self.assertRaises(ValidityError):
            self.journal.ready()
        self.directory.write_bytes("state.json", new_state)
        self.directory.write_bytes("anchor.json", original_anchor)
        with self.assertRaises(ValidityError):
            self.journal.ready()
        self.directory.write_bytes("anchor.json", new_anchor)
        self.assertEqual(self.journal.ready(), committed)

    def test_missing_or_mismatching_transition_after_commit_fails_closed(self):
        self.change()
        path = self.directory.root / "transition.json"
        original = path.read_bytes()
        path.unlink()
        with self.assertRaises(ValidityError):
            self.journal.ready()
        path.write_bytes(original)
        path.chmod(0o600)
        transition = self.directory.read_json("transition.json")
        transition["next_sha256"] = "9" * 64
        self.directory.write_json("transition.json", transition)
        with self.assertRaises(ValidityError):
            self.journal.ready()

    def test_strict_json_noncanonical_float_and_unknown_fields_reject(self):
        original = self.directory.read_bytes("state.json")
        bad = [b'{"version":"validity-1","version":"validity-1"}', b'{"value":NaN}', b'\xff']
        floating = deepcopy(self.state)
        floating["sequence"] = 0.0
        bad.append(canonical_json(floating))
        extra = {**self.state, "ignore_revocation": True}
        bad.append(canonical_json(extra))
        for raw in bad:
            with self.subTest(raw=raw[:40]):
                self.directory.write_bytes("state.json", raw)
                with self.assertRaises((ValueError, UnicodeError)):
                    self.journal.ready()
        self.directory.write_bytes("state.json", original)
        self.assertEqual(self.journal.ready(), self.state)

    def test_terminal_bindings_and_append_only_events_cannot_change(self):
        key = "7" * 64
        initial = self.change("REQUEST_BOUND", lambda n: n["requests"].update({key: self.binding(status="aborted")}))
        for mutate in (lambda n: n["requests"][key].update(status="committed"),
                       lambda n: n["events"].clear(), lambda n: n["requests"].clear(),
                       lambda n: n["requests"][key].update(operation_sha256="8" * 64)):
            with self.subTest(mutate=mutate), self.assertRaises(ValidityError):
                self.journal.change(initial, "RECOVERY", mutate, reserved=True)
        self.assertEqual(self.journal.ready(), initial)

    def test_prepared_reservation_quota_does_not_overwrite_original_transition(self):
        _, _, transition = self.prepared(mutate=lambda n: n["counts"].update(operator=64))
        with self.assertRaises(ValidityCapacity):
            self.journal.reserve_prepared_recovery()
        self.assertEqual(self.directory.read_json("transition.json"), transition)


if __name__ == "__main__":
    unittest.main()
