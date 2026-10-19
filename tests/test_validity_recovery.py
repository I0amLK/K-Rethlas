"""Real owned spawn/SIGKILL races and synthetic T09 durability faults.

This module is not executed during implementation. Actual sandbox observations
are required when these tests run; caller-supplied child output is never accepted.
"""
from __future__ import annotations

from copy import deepcopy
import multiprocessing
import os
from pathlib import Path
import signal
import unittest
from unittest import mock

from rethlas_core.authorization import AuthorizationError
from rethlas_core.memory_store import MemoryStore
from rethlas_core.private_fs import PrivateDirectory, canonical_json, digest, strict_json_loads
from rethlas_core.validity import (
    VERSION, ValidityError, ValidityConflict, ValidityRecoveryRequired, ValidityIndeterminate,
)
from rethlas_core.validity_service import CurrentValidityService

from test_memory_recovery import ProcessFixtureMixin
from test_memory_store import write_private
from test_validity import ValidityFixtureMixin


PROCESS_TIMEOUT = 20


class CheckpointMemoryStore(MemoryStore):
    def __init__(self, root, *, observer, clock):
        self.observer = observer
        super().__init__(root, clock=clock)

    def _checkpoint(self, stage):
        self.observer("memory." + stage)


class CheckpointService(CurrentValidityService):
    """Trusted test-only instrumentation; no failpoint comes from CLI/env data."""
    def __init__(self, root, *, observer=None):
        self.active_kind = None
        self.observer = observer
        super().__init__(root)

    def _change(self, state, kind, mutate, **kwargs):
        previous = self.active_kind
        self.active_kind = kind
        try:
            return super()._change(state, kind, mutate, **kwargs)
        finally:
            self.active_kind = previous

    def _checkpoint(self, stage):
        if self.observer is not None:
            self.observer(self.active_kind, stage)

    def _store(self):
        original = super()._store()
        return CheckpointMemoryStore(original.root, clock=self.clock,
                                     observer=lambda stage: self._checkpoint(stage))


def _operation_worker(root, connection, start, release):
    """Top-level spawn target; reopen authorities and own only this child."""
    try:
        payload = connection.recv()
        matched = False

        def observer(kind, stage):
            nonlocal matched
            if (not matched and stage == payload.get("stage")
                    and (payload.get("kind") is None or kind == payload["kind"])):
                matched = True
                connection.send(("checkpoint", {"kind": kind, "stage": stage}))
                if not release.wait(PROCESS_TIMEOUT):
                    raise TimeoutError("owned checkpoint release deadline")

        service = CheckpointService(Path(root), observer=observer)
        connection.send(("ready", None))
        if not start.wait(PROCESS_TIMEOUT):
            raise TimeoutError("owned operation start deadline")
        connection.send(("attempting", None))
        if payload["method"] == "recover":
            result = service.recover(**payload["options"])
        else:
            grant = payload["grant"]
            args = [grant["token"]]
            if payload.get("request") is not None:
                args.append(payload["request"])
            result = getattr(service, payload["method"])(*args, run_id=grant["run_id"])
        connection.send(("result", result))
    except Exception as exc:
        connection.send(("error", type(exc).__name__))
    finally:
        connection.close()


class ValidityRecoveryTests(ProcessFixtureMixin, ValidityFixtureMixin, unittest.TestCase):
    def worker(self, method, grant=None, request=None, *, kind=None, stage=None, options=None):
        context = multiprocessing.get_context("spawn")
        start, release = context.Event(), context.Event()
        process, connection = self.spawn_worker(_operation_worker, self.root, start, release)
        connection.send({"method": method, "grant": grant, "request": request,
                         "kind": kind, "stage": stage, "options": options})
        self.assertEqual(self.receive(connection), ("ready", None))
        # A process killed inside Event.wait can leave its condition lock held;
        # setting that Event during cleanup would then hang the whole test suite.
        return {"process": process, "connection": connection, "start": start, "release": release}

    def begin(self, child, *, checkpoint=False):
        child["start"].set()
        self.assertEqual(self.receive(child["connection"]), ("attempting", None))
        if checkpoint:
            message = self.receive(child["connection"])
            self.assertEqual(message[0], "checkpoint", message)
            self.assertTrue(child["process"].is_alive())
            return message[1]
        return None

    def result(self, child, *, error=None):
        message = self.receive(child["connection"])
        self.join_cleanly(child["process"])
        self.assertEqual(message[0], "error" if error is not None else "result", message)
        if error is not None:
            self.assertEqual(message[1], error)
        return message[1]

    def kill(self, child):
        process = child["process"]
        self.assertTrue(process.is_alive())
        self.assertIsNotNone(process.pid)
        os.kill(process.pid, signal.SIGKILL)
        process.join(PROCESS_TIMEOUT)
        self.assertFalse(process.is_alive(), "owned checkpoint child did not die")
        self.assertEqual(process.exitcode, -signal.SIGKILL)

    def request_binding(self, request_id):
        return next(record for record in self.state()["requests"].values() if record["request_id"] == request_id)

    def recreate_fixture(self, name):
        self.root = self.base / name
        self.service = CurrentValidityService.initialize(self.root, owner="owner")
        self.writer = self.grant("validity_writer")
        self.withdrawer = self.grant("validity_withdrawer")
        self.seed = {**self.seed, "expected_head": self.service.config["genesis"]["head"]}
        self.committed = self.call("commit", self.writer, self.seed)
        self.registration = self.registration_request()
        self.service.register_use(self.registration)
        self.reader = self.grant("validity_reader", "main")
        self.verifier = self.grant("validity_verifier", "main")
        self.publisher = self.grant("validity_publisher", "main")

    def probe_request(self, request_id="probe"):
        return {"version": VERSION, "request_id": request_id}

    def accepted_probe(self, request_id="probe"):
        return self.call("probe", self.verifier, self.probe_request(request_id))["probe"]

    def publication_request(self, probe, request_id="publish"):
        return {"version": VERSION, "request_id": request_id, "probe_id": probe["probe_id"]}

    def test_real_sigkill_four_withdrawal_control_boundaries_only_recovers_forward(self):
        original_graph = (self.root / "memory/manifest.json").read_bytes()
        for number, stage in enumerate(("after_control_prepare", "after_anchor_pending", "after_control_state", "after_anchor_stable"), 1):
            with self.subTest(stage=stage):
                request = self.withdraw_request(request_id="withdraw-" + stage)
                child = self.worker("withdraw", self.withdrawer, request, kind="WITHDRAW", stage=stage)
                self.begin(child, checkpoint=True)
                self.kill(child)
                self.fresh()
                inspected = self.service.recover(kind="control")
                phase = "prepared" if stage == "after_control_prepare" else "stable" if stage == "after_anchor_stable" else "pending"
                self.assertEqual(inspected["status"], phase)
                if phase != "stable":
                    with self.assertRaises(ValidityRecoveryRequired):
                        self.call("status", self.reader)
                self.service.recover(kind="control", apply=True, token=self.withdrawer["token"], run_id=self.withdrawer["run_id"])
                self.assertEqual(self.request_binding(request["request_id"])["status"], "withdrawn")
                self.assertEqual(self.state()["withdrawal_version"], number)
                self.assertFalse(self.current()["fixture_input_eligible"])
                replay = self.call("withdraw", self.withdrawer, request)
                self.assertTrue(replay["replayed"])
                self.assertEqual(self.state()["withdrawal_version"], number)
                self.assertEqual((self.root / "memory/manifest.json").read_bytes(), original_graph)

    def test_real_prepared_cancellation_preserves_original_permanent_request_binding(self):
        request = self.withdraw_request()
        child = self.worker("withdraw", self.withdrawer, request, kind="WITHDRAW", stage="after_control_prepare")
        self.begin(child, checkpoint=True)
        self.kill(child)
        self.fresh()
        original = deepcopy(self.request_binding(request["request_id"]))
        self.assertEqual(original["status"], "bound")
        self.service.recover(kind="control", apply=True, cancel=True)
        aborted = self.request_binding(request["request_id"])
        self.assertEqual(aborted["status"], "aborted")
        for field in ("operation_sha256", "identity", "request_id", "kind", "bucket"):
            self.assertEqual(aborted[field], original[field])
        self.assertEqual(self.state()["withdrawal_version"], 0)
        self.assertTrue(self.current()["fixture_input_eligible"])
        for modified in (request, {**request, "reason": "changed request must not reuse ID"}):
            with self.subTest(modified=modified != request), self.assertRaises(ValidityConflict):
                self.call("withdraw", self.withdrawer, modified)
        self.fresh()
        self.assertEqual(self.request_binding(request["request_id"])["status"], "aborted")

    def test_real_cancel_replacement_kill_keeps_original_binding_before_state_install(self):
        request = self.withdraw_request(request_id="prepared-binding")
        child = self.worker("withdraw", self.withdrawer, request, kind="REQUEST_BOUND", stage="after_control_prepare")
        self.begin(child, checkpoint=True)
        self.kill(child)
        self.assertFalse(any(b["request_id"] == request["request_id"] for b in self.state()["requests"].values()))
        cancel = self.worker("recover", options={"kind": "control", "apply": True, "cancel": True}, stage="after_prepared_abort")
        self.begin(cancel, checkpoint=True)
        self.kill(cancel)
        self.fresh()
        self.assertEqual(self.service.recover(kind="control")["status"], "prepared")
        with self.assertRaises(ValidityRecoveryRequired):
            self.call("read", self.reader)
        self.service.recover(kind="control", apply=True)
        self.assertEqual(self.request_binding(request["request_id"])["status"], "aborted")
        with self.assertRaises(ValidityConflict):
            self.call("withdraw", self.withdrawer, request)
        self.assertTrue(self.current()["fixture_input_eligible"])

    def test_real_pending_anchor_cannot_be_cancelled_or_restore_old_permission(self):
        request = self.withdraw_request()
        child = self.worker("withdraw", self.withdrawer, request, kind="WITHDRAW", stage="after_anchor_pending")
        self.begin(child, checkpoint=True)
        self.kill(child)
        self.fresh()
        with self.assertRaises(ValidityConflict):
            self.service.recover(kind="control", apply=True, cancel=True)
        with self.assertRaises(ValidityRecoveryRequired):
            self.call("read", self.reader)
        self.service.authority.close_run(self.withdrawer["run_id"])
        self.service.recover(kind="control", apply=True)
        self.assertEqual(self.request_binding(request["request_id"])["status"], "withdrawn")
        self.assertFalse(self.current()["fixture_input_eligible"])

    def test_real_prepared_completion_needs_live_original_identity_not_new_authority(self):
        request = self.withdraw_request()
        child = self.worker("withdraw", self.withdrawer, request, kind="WITHDRAW", stage="after_control_prepare")
        self.begin(child, checkpoint=True)
        self.kill(child)
        self.fresh()
        with self.assertRaises(AuthorizationError):
            self.service.recover(kind="control", apply=True)
        with self.assertRaises(AuthorizationError):
            self.service.recover(kind="control", apply=True, token=self.writer["token"], run_id=self.writer["run_id"])
        self.service.authority.close_run(self.withdrawer["run_id"])
        with self.assertRaises(AuthorizationError):
            self.service.recover(kind="control", apply=True, token=self.withdrawer["token"], run_id=self.withdrawer["run_id"])
        self.assertEqual(self.state()["withdrawal_version"], 0)
        self.service.recover(kind="control", apply=True, cancel=True)
        self.assertEqual(self.request_binding(request["request_id"])["status"], "aborted")

    def test_real_graph_prepare_kill_requires_original_q_then_can_explicitly_continue(self):
        request = self.next_request()
        child = self.worker("commit", self.writer, request, kind="GRAPH_PREPARE", stage="after_graph_prepare")
        self.begin(child, checkpoint=True)
        self.kill(child)
        self.fresh()
        self.assertIsNotNone(self.state()["pending_graph"])
        self.assertEqual(self.state()["accepted_graph"]["head"], self.committed["head"])
        with self.assertRaises(ValidityRecoveryRequired):
            self.call("read", self.reader)
        with self.assertRaises(ValidityConflict):
            self.service.recover(kind="graph", request={**request, "request_id": "different"})
        inspected = self.service.recover(kind="graph", request=request)
        self.assertEqual(inspected["status"], "graph_prepared")
        with self.assertRaises(AuthorizationError):
            self.service.recover(kind="graph", apply=True, request=request)
        result = self.service.recover(kind="graph", apply=True, request=request,
                                      token=self.writer["token"], run_id=self.writer["run_id"])
        self.assertEqual(result["status"], "committed")
        self.assertIsNone(self.state()["pending_graph"])
        self.assertEqual(self.state()["accepted_graph"]["head"], result["head"])
        replay = self.call("commit", self.writer, request)
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["head"], result["head"])
        self.assertTrue(self.current()["fixture_input_eligible"])

    def test_real_graph_prepare_can_abort_but_terminated_id_never_rebinds(self):
        request = self.next_request()
        child = self.worker("commit", self.writer, request, kind="GRAPH_PREPARE", stage="after_graph_prepare")
        self.begin(child, checkpoint=True)
        self.kill(child)
        self.fresh()
        self.service.recover(kind="graph", apply=True, cancel=True, request=request)
        self.assertEqual(self.request_binding(request["request_id"])["status"], "aborted")
        self.assertIsNone(self.state()["pending_graph"])
        self.assertEqual(self.state()["accepted_graph"]["head"], self.committed["head"])
        with self.assertRaises(ValidityConflict):
            self.call("commit", self.writer, request)
        with self.assertRaises(ValidityConflict):
            self.call("commit", self.writer, {**request, "relations": []})
        self.assertTrue(self.current()["fixture_input_eligible"])

    def test_real_sigkill_before_and_after_memory_manifest_distinguishes_fact_from_fence(self):
        for number, stage in enumerate(("memory.before_manifest", "memory.after_manifest")):
            if number:
                self.recreate_fixture("after-manifest")
            with self.subTest(stage=stage):
                request = self.next_request()
                child = self.worker("commit", self.writer, request, stage=stage)
                self.begin(child, checkpoint=True)
                self.kill(child)
                self.fresh()
                disk = strict_json_loads((self.root / "memory/manifest.json").read_bytes())
                self.assertEqual(self.state()["accepted_graph"]["head"], self.committed["head"])
                self.assertIsNotNone(self.state()["pending_graph"])
                with self.assertRaises(ValidityRecoveryRequired):
                    self.call("read", self.reader)
                if stage == "memory.before_manifest":
                    self.assertEqual(disk["head"], self.committed["head"])
                    inspected = self.service.recover(kind="graph", request=request)
                    self.assertEqual(inspected["status"], "graph_prepared")
                    result = self.service.recover(kind="graph", apply=True, request=request,
                                                  token=self.writer["token"], run_id=self.writer["run_id"])
                else:
                    self.assertNotEqual(disk["head"], self.committed["head"])
                    self.service.authority.close_run(self.writer["run_id"])
                    inspected = self.service.recover(kind="graph", request=request)
                    self.assertEqual(inspected["status"], "graph_committed_control_pending")
                    with self.assertRaises(ValidityConflict):
                        self.service.recover(kind="graph", apply=True, cancel=True, request=request)
                    result = self.service.recover(kind="graph", apply=True, request=request)
                self.assertEqual(result["status"], "committed")
                self.assertEqual(self.state()["accepted_graph"]["head"], result["head"])
                self.assertIsNone(self.state()["pending_graph"])
                self.assertTrue(self.current()["fixture_input_eligible"])

    def test_real_pending_graph_accept_recovers_control_before_graph_and_after_writer_close(self):
        request = self.next_request()
        child = self.worker("commit", self.writer, request, kind="GRAPH_ACCEPT", stage="after_anchor_pending")
        self.begin(child, checkpoint=True)
        self.kill(child)
        self.fresh()
        with self.assertRaises(ValidityRecoveryRequired):
            self.service.recover(kind="graph", apply=True, request=request)
        self.service.authority.close_run(self.writer["run_id"])
        self.service.recover(kind="control", apply=True)
        self.assertIsNone(self.state()["pending_graph"])
        self.assertEqual(self.request_binding(request["request_id"])["status"], "committed")
        self.assertEqual(self.service.recover(kind="graph", request=request)["status"], "no_pending_graph")
        self.assertTrue(self.current()["fixture_input_eligible"])

    def test_real_probe_result_lost_before_accept_is_never_reconstructed_from_stdout(self):
        request = self.probe_request()
        child = self.worker("probe", self.verifier, request, stage="before_probe_accept")
        self.begin(child, checkpoint=True)
        self.kill(child)
        self.fresh()
        binding = self.request_binding(request["request_id"])
        self.assertEqual(binding["status"], "started")
        probe = self.state()["probes"][binding["result"]["probe_id"]]
        self.assertEqual(probe["status"], "started")
        self.assertNotIn("observation", probe)
        with self.assertRaises(ValidityRecoveryRequired):
            self.call("probe", self.verifier, request)
        accepted = self.accepted_probe("new-probe")
        self.assertEqual(accepted["status"], "accepted")
        self.assertEqual(self.request_binding(request["request_id"])["status"], "started")

    def test_real_pending_probe_accept_and_publication_recover_exact_historical_records(self):
        request = self.probe_request()
        child = self.worker("probe", self.verifier, request, kind="PROBE_ACCEPTED", stage="after_anchor_pending")
        self.begin(child, checkpoint=True)
        self.kill(child)
        self.fresh()
        self.service.recover(kind="control", apply=True)
        observed = self.call("probe", self.verifier, request)
        self.assertTrue(observed["replayed"])
        self.assertEqual(observed["probe"]["status"], "accepted")
        self.assertFalse(observed["probe"]["observation"]["mathematical_review"])
        publication = self.publication_request(observed["probe"])
        child = self.worker("publish", self.publisher, publication, kind="FIXTURE_PUBLISHED", stage="after_anchor_pending")
        self.begin(child, checkpoint=True)
        self.kill(child)
        self.fresh()
        self.service.recover(kind="control", apply=True)
        published = self.call("publish", self.publisher, publication)
        self.assertTrue(published["replayed"])
        self.assertEqual(len(self.state()["publications"]), 1)
        receipt = deepcopy(published["receipt"])
        self.call("withdraw", self.withdrawer, self.withdraw_request())
        replay = self.call("publish", self.publisher, publication)
        self.assertEqual(replay["receipt"], receipt)
        self.assertFalse(replay["current"]["fixture_input_eligible"])

    def test_real_withdraw_read_race_both_linearization_orders(self):
        for number, first in enumerate(("read", "withdraw")):
            if number:
                self.recreate_fixture("withdraw-first-read")
            with self.subTest(first=first):
                reader = self.worker("read", self.reader, stage="before_read_return" if first == "read" else None)
                withdrawer = self.worker("withdraw", self.withdrawer, self.withdraw_request(),
                                         kind="WITHDRAW", stage="after_anchor_pending" if first == "withdraw" else None)
                first_child, second_child = (reader, withdrawer) if first == "read" else (withdrawer, reader)
                self.begin(first_child, checkpoint=True)
                self.begin(second_child)
                first_child["release"].set()
                if first == "read":
                    old = self.result(reader)
                    self.assertTrue(old["current"]["fixture_input_eligible"])
                    self.assertEqual(self.result(withdrawer)["status"], "withdrawn")
                else:
                    self.assertEqual(self.result(withdrawer)["status"], "withdrawn")
                    self.result(reader, error="ValidityError")
                self.fresh()
                self.assertFalse(self.current()["fixture_input_eligible"])

    def test_real_withdraw_publish_race_both_orders_never_leave_current_valid_receipt(self):
        for number, first in enumerate(("publish", "withdraw")):
            if number:
                self.recreate_fixture("withdraw-first-publish")
            with self.subTest(first=first):
                probe = self.accepted_probe()
                request = self.publication_request(probe)
                publisher = self.worker("publish", self.publisher, request,
                                        kind="FIXTURE_PUBLISHED", stage="after_anchor_pending" if first == "publish" else None)
                withdrawer = self.worker("withdraw", self.withdrawer, self.withdraw_request(),
                                         kind="WITHDRAW", stage="after_anchor_pending" if first == "withdraw" else None)
                first_child, second_child = (publisher, withdrawer) if first == "publish" else (withdrawer, publisher)
                self.begin(first_child, checkpoint=True)
                self.begin(second_child)
                first_child["release"].set()
                if first == "publish":
                    result = self.result(publisher)
                    self.assertEqual(result["status"], "fixture_published")
                    self.assertTrue(result["current"]["fixture_input_eligible"])
                    self.assertEqual(self.result(withdrawer)["status"], "withdrawn")
                    self.fresh()
                    replay = self.call("publish", self.publisher, request)
                    self.assertEqual(replay["receipt"], result["receipt"])
                    self.assertFalse(replay["current"]["fixture_input_eligible"])
                else:
                    self.assertEqual(self.result(withdrawer)["status"], "withdrawn")
                    self.result(publisher, error="ValidityError")
                    self.fresh()
                    self.assertEqual(self.state()["publications"], {})
                self.assertFalse(self.current()["fixture_input_eligible"])

    def test_real_graph_accept_withdraw_race_both_orders_enqueues_new_blocked_candidate(self):
        for number, first in enumerate(("graph", "withdraw")):
            if number:
                self.recreate_fixture("withdraw-first-graph")
            with self.subTest(first=first):
                request = self.next_request()
                writer = self.worker("commit", self.writer, request,
                                     kind="GRAPH_ACCEPT", stage="after_anchor_pending" if first == "graph" else None)
                withdrawer = self.worker("withdraw", self.withdrawer, self.withdraw_request(),
                                         kind="WITHDRAW", stage="after_anchor_pending" if first == "withdraw" else None)
                first_child, second_child = (writer, withdrawer) if first == "graph" else (withdrawer, writer)
                self.begin(first_child, checkpoint=True)
                self.begin(second_child)
                first_child["release"].set()
                self.assertEqual(self.result(writer)["status"], "committed")
                self.assertEqual(self.result(withdrawer)["status"], "withdrawn")
                self.fresh()
                self.assertFalse(self.current()["fixture_input_eligible"])
                entries = self.service.review_queue()["entries"]
                self.assertTrue(any(row["revision"]["revision_id"] == "claim:next" and row["kind"] == "math_dependency_blocked" for row in entries))
                self.assertIsNone(self.state()["pending_graph"])

    def test_real_actual_probe_acceptance_window_allows_withdrawal_and_checks_overlay(self):
        for number, target in enumerate(("assumption:0", "source:0", "unrelated")):
            if number:
                self.recreate_fixture("probe-race-" + str(number))
            with self.subTest(target=target):
                if target == "unrelated":
                    addition = self.next_request(request_id="unrelated")
                    self.call("commit", self.writer, addition)
                    exact = {key: addition["revisions"][0][key] for key in ("entity_id", "revision_id", "body_sha256")}
                    withdrawal = {"version": VERSION, "request_id": "withdraw-unrelated", "targets": [exact], "reason": "synthetic unrelated withdrawal"}
                else:
                    withdrawal = self.withdraw_request(target)
                request = self.probe_request()
                child = self.worker("probe", self.verifier, request, stage="before_probe_accept")
                self.begin(child, checkpoint=True)
                # This checkpoint follows a real sandbox result and holds no
                # service/store locks. Withdrawal must finish before release.
                withdrawn = self.call("withdraw", self.withdrawer, withdrawal)
                self.assertEqual(withdrawn["status"], "withdrawn")
                child["release"].set()
                if target == "unrelated":
                    observed = self.result(child)
                    self.assertEqual(observed["probe"]["status"], "accepted")
                    self.assertTrue(observed["current"]["fixture_input_eligible"])
                else:
                    self.result(child, error="ValidityError" if target == "assumption:0" else "ValidityConflict")
                    self.assertEqual(self.request_binding(request["request_id"])["status"], "started")
                    self.assertEqual(self.current()["fixture_input_eligible"], target == "source:0")

    def test_real_probe_accept_before_withdraw_keeps_observation_only_as_history(self):
        request = self.probe_request()
        probe = self.worker("probe", self.verifier, request, kind="PROBE_ACCEPTED", stage="after_anchor_stable")
        withdrawer = self.worker("withdraw", self.withdrawer, self.withdraw_request())
        self.begin(probe, checkpoint=True)
        self.begin(withdrawer)
        probe["release"].set()
        accepted = self.result(probe)
        self.assertEqual(accepted["probe"]["status"], "accepted")
        self.assertEqual(self.result(withdrawer)["status"], "withdrawn")
        self.fresh()
        replay = self.call("probe", self.verifier, request)
        self.assertEqual(replay["probe"], accepted["probe"])
        self.assertFalse(replay["current"]["fixture_input_eligible"])

    def test_control_replace_and_directory_fsync_failures_do_not_report_withdrawal_success(self):
        for number, failure in enumerate(("replace", "directory-fsync")):
            if number:
                self.recreate_fixture("fsync-failure")
            with self.subTest(failure=failure):
                service = CheckpointService(self.root)
                request = self.withdraw_request()
                original_replace, original_fsync = os.replace, os.fsync
                control_inode = (self.root / "control").stat().st_ino
                armed = False
                failed = False

                def replace(source, destination, **kwargs):
                    nonlocal armed, failed
                    selected = (service.active_kind == "WITHDRAW" and destination == "anchor.json"
                                and os.fstat(kwargs["dst_dir_fd"]).st_ino == control_inode)
                    if selected and failure == "replace" and not failed:
                        failed = True
                        raise OSError("synthetic pending anchor replace failure")
                    result = original_replace(source, destination, **kwargs)
                    if selected and not failed:
                        armed = True
                    return result

                def fsync(fd):
                    nonlocal armed, failed
                    if armed and not failed and failure == "directory-fsync" and os.fstat(fd).st_ino == control_inode:
                        failed = True
                        armed = False
                        raise OSError("synthetic pending anchor directory fsync failure")
                    return original_fsync(fd)

                with mock.patch("rethlas_core.private_fs.os.replace", side_effect=replace), mock.patch(
                    "rethlas_core.private_fs.os.fsync", side_effect=fsync
                ), self.assertRaises(ValidityIndeterminate):
                    service.withdraw(self.withdrawer["token"], request, run_id=self.withdrawer["run_id"])
                self.assertTrue(failed)
                self.fresh()
                phase = self.service.recover(kind="control")["status"]
                self.assertEqual(phase, "prepared" if failure == "replace" else "pending")
                with self.assertRaises(ValidityRecoveryRequired):
                    self.call("read", self.reader)
                self.service.recover(kind="control", apply=True, token=self.withdrawer["token"], run_id=self.withdrawer["run_id"])
                self.assertFalse(self.current()["fixture_input_eligible"])
                self.assertEqual(self.state()["withdrawal_version"], 1)

    def test_missing_transition_state_or_anchor_never_falls_back_to_old_validity(self):
        self.call("withdraw", self.withdrawer, self.withdraw_request())
        for name in ("transition.json", "state.json", "anchor.json"):
            with self.subTest(name=name):
                path = self.root / "control" / name
                raw = path.read_bytes()
                path.unlink()
                with self.assertRaises((OSError, ValueError)):
                    self.call("read", self.reader)
                with self.assertRaises((OSError, ValueError)):
                    self.service.recover(kind="control", apply=True)
                self.assertFalse(path.exists(), "missing authority was silently repaired")
                write_private(path, raw)
        self.fresh()
        self.assertFalse(self.current()["fixture_input_eligible"])


if __name__ == "__main__":
    unittest.main()
