"""Local transaction, process-death, recovery and fault-injection fixtures.

SIGKILL tests kill only children created by this TestCase and do not certify
power-loss durability. Tokens travel over anonymous multiprocessing pipes, never
argv, environment, logs or assertion messages. No tests are executed on import.
"""
from __future__ import annotations

from copy import deepcopy
import errno
import json
import multiprocessing
import os
from pathlib import Path
import signal
import stat
import threading
import time
import unittest
from unittest import mock

# Import helpers explicitly: importing a TestCase class here duplicates discovery.
from test_memory_store import (
    ACTIONS, COLLECTION_IDS, RUN_ID, MemoryFixtureMixin, MutableClock,
    encoded, graph_from_requests, read_json, receipt_head, request_fixture, sha,
    write_private,
)
from rethlas_core.authorization import Authority, AuthorizationError
from rethlas_core.memory_store import (
    MemoryCancelled, MemoryCapacity, MemoryConflict, MemoryCorruption,
    MemoryIndeterminate, MemoryRecoveryRequired, MemoryStore, MemoryStoreError,
)
from rethlas_core.private_fs import PrivateDirectory, PrivateFilesystemError


PROCESS_TIMEOUT = 20


class CheckpointStore(MemoryStore):
    """Trusted Python-only test hook; no request/CLI/environment failpoints."""

    def __init__(self, root, *, checkpoint, clock=time.time):
        super().__init__(root, clock=clock)
        self.checkpoint = checkpoint

    def _checkpoint(self, stage):
        self.checkpoint(stage)


class InjectedCheckpointFailure(RuntimeError):
    pass


def _race_commit_worker(root, connection, start_event):
    """Spawn target must stay top-level; reopen the store and its authority."""
    try:
        payload = connection.recv()
        store = MemoryStore(Path(root))
        connection.send(("ready", None))
        if not start_event.wait(PROCESS_TIMEOUT):
            connection.send(("error", "start_timeout"))
            return
        try:
            result = store.commit(payload["token"], payload["request"], run_id=payload["run_id"])
        except MemoryConflict:
            connection.send(("conflict", None))
        except Exception as exc:
            connection.send(("error", type(exc).__name__))
        else:
            connection.send(("committed", result))
    except Exception as exc:
        connection.send(("error", type(exc).__name__))
    finally:
        connection.close()


def _checkpoint_commit_worker(root, connection, blocked):
    """Reach a real commit stage and block for parent-issued SIGKILL."""
    try:
        payload = connection.recv()

        def checkpoint(stage):
            if stage == payload["stage"]:
                connection.send(("checkpoint", stage))
                if not blocked.wait(PROCESS_TIMEOUT * 2):
                    raise TimeoutError("parent did not kill the checkpoint child")

        store = CheckpointStore(Path(root), checkpoint=checkpoint)
        result = store.commit(payload["token"], payload["request"], run_id=payload["run_id"])
        connection.send(("unexpected_commit_return", result["head"]))
    except Exception as exc:
        connection.send(("error", type(exc).__name__))
    finally:
        connection.close()


def _controlled_commit_worker(root, connection, release):
    try:
        payload = connection.recv()

        def checkpoint(stage):
            if stage == "before_manifest":
                connection.send(("at_final_guard", None))
                if not release.wait(PROCESS_TIMEOUT):
                    raise TimeoutError("parent did not release owned commit child")

        store = CheckpointStore(Path(root), checkpoint=checkpoint)
        result = store.commit(payload["token"], payload["request"], run_id=payload["run_id"])
        connection.send(("committed", result["head"]))
    except Exception as exc:
        connection.send(("error", type(exc).__name__))
    finally:
        connection.close()


def _revoke_worker(root, connection, attempted, finished):
    try:
        authority = Authority(Path(root) / "authority")
        # The controller retains an already-open authority, as the real lifecycle
        # caller does. Its later revoke() is the operation that must wait.
        connection.send(("ready", None))
        token_id = connection.recv()
        attempted.set()
        authority.revoke(token_id)
        finished.set()
        connection.send(("revoked", None))
    except Exception as exc:
        connection.send(("error", type(exc).__name__))
    finally:
        connection.close()


def _recover_reopen_worker(root, connection):
    """Independent interpreter reauthenticates and performs explicit recovery."""
    try:
        payload = connection.recv()
        store = MemoryStore(Path(root))
        token, run_id = payload["token"], payload["run_id"]
        before = store.status(token, run_id=run_id)
        exported_before = store.export(token, run_id=run_id)
        inspected = store.recover(token, run_id=run_id, apply=False)
        recovered = store.recover(token, run_id=run_id, apply=True)
        exported_after = store.export(token, run_id=run_id)
        replay = store.commit(token, payload["request"], run_id=run_id)
        after = store.status(token, run_id=run_id)
        final = store.export(token, run_id=run_id)
        connection.send(("recovered", {
            "before": before, "exported_before": exported_before,
            "inspected": inspected, "recovered": recovered,
            "exported_after": exported_after, "replay": replay,
            "after": after, "final": final,
        }))
    except Exception as exc:
        connection.send(("error", type(exc).__name__))
    finally:
        connection.close()


class ProcessFixtureMixin:
    """Own every process and every timeout; never search for unrelated PIDs."""

    def spawn_worker(self, target, root, *args):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        process = context.Process(target=target, args=(str(root), child, *args))
        process.start()
        child.close()
        self.addCleanup(parent.close)
        self.addCleanup(self.cleanup_process, process)
        return process, parent

    @staticmethod
    def cleanup_process(process):
        if process.is_alive():
            process.kill()
        process.join(PROCESS_TIMEOUT)
        if process.is_alive():
            raise AssertionError("owned test child did not exit after SIGKILL")
        process.close()

    def receive(self, connection):
        if not connection.poll(PROCESS_TIMEOUT):
            self.fail("owned test child did not send its stage/result before timeout")
        try:
            return connection.recv()
        except EOFError:
            self.fail("owned test child closed its pipe without a stage/result")

    def join_cleanly(self, process):
        process.join(PROCESS_TIMEOUT)
        self.assertFalse(process.is_alive(), "owned test child did not exit before timeout")
        self.assertEqual(process.exitcode, 0)

    def kill_at_stage(self, store, token, request, stage):
        blocked = multiprocessing.get_context("spawn").Event()
        process, connection = self.spawn_worker(_checkpoint_commit_worker, store.root, blocked)
        connection.send({"token": token, "request": request, "run_id": RUN_ID, "stage": stage})
        message = self.receive(connection)
        self.assertEqual(message, ("checkpoint", stage))
        self.assertTrue(process.is_alive())
        # This is the PID just created by spawn_worker, not a discovered/system PID.
        self.assertIsNotNone(process.pid)
        os.kill(process.pid, signal.SIGKILL)
        process.join(PROCESS_TIMEOUT)
        self.assertFalse(process.is_alive(), "SIGKILL child failed to exit")
        self.assertEqual(process.exitcode, -signal.SIGKILL)
        return process


class MemoryMultiprocessTests(ProcessFixtureMixin, MemoryFixtureMixin, unittest.TestCase):
    def race(self, requests):
        start = multiprocessing.get_context("spawn").Event()
        children = [self.spawn_worker(_race_commit_worker, self.root, start) for _ in requests]
        for (_, connection), request in zip(children, requests):
            connection.send({"token": self.token, "run_id": RUN_ID, "request": request})
        for _, connection in children:
            self.assertEqual(self.receive(connection), ("ready", None))
        start.set()
        results = [self.receive(connection) for _, connection in children]
        for process, _ in children:
            self.join_cleanly(process)
        return results

    def test_spawn_same_base_two_writers_one_conflict_then_retry_has_no_loss_or_duplicates(self):
        first = request_fixture(self.initial_head, label="writer-a", body="writer A\n")
        second = request_fixture(self.initial_head, label="writer-b", body="writer B\n")
        results = self.race([first, second])
        self.assertEqual(sorted(item[0] for item in results), ["committed", "conflict"])
        winner = next(index for index, item in enumerate(results) if item[0] == "committed")
        loser = 1 - winner
        requests = [first, second]
        visible = self.export()["graph"]
        self.assertEqual(visible, graph_from_requests(requests[winner]))
        self.assertEqual(self.status()["generation"], 1)
        retry = deepcopy(requests[loser])
        retry["request_id"] += "/retry-new-id"
        retry["expected_head"] = self.status()["head"]
        # Retry in a fresh interpreter with a new explicit synchronization gate.
        start = multiprocessing.get_context("spawn").Event()
        retry_process, retry_connection = self.spawn_worker(_race_commit_worker, self.root, start)
        retry_connection.send({"token": self.token, "run_id": RUN_ID, "request": retry})
        self.assertEqual(self.receive(retry_connection), ("ready", None))
        start.set()
        kind, result = self.receive(retry_connection)
        self.assertEqual(kind, "committed")
        self.assertFalse(result["replayed"])
        self.join_cleanly(retry_process)
        final = self.export()["graph"]
        expected = graph_from_requests(requests[winner], retry)
        for collection, identity in COLLECTION_IDS.items():
            self.assertEqual(final[collection], expected[collection])
            self.assertEqual(len({record[identity] for record in final[collection]}), 2)
        self.assertEqual(self.status()["generation"], 2)
        self.assertEqual(len(list((self.root / "receipts").glob("*.json"))), 2)

    def test_spawn_same_request_has_exactly_one_new_commit_and_one_exact_replay(self):
        request = request_fixture(self.initial_head, label="same-request")
        results = self.race([request, deepcopy(request)])
        self.assertEqual([kind for kind, _ in results], ["committed", "committed"])
        receipts = [result for _, result in results]
        self.assertEqual(sorted(result["replayed"] for result in receipts), [False, True])
        self.assertEqual(receipts[0]["receipt"], receipts[1]["receipt"])
        self.assertEqual(receipts[0]["head"], receipts[1]["head"])
        self.assertEqual(self.status()["generation"], 1)
        self.assertEqual(self.export()["graph"], graph_from_requests(request))
        self.assertEqual(len(list((self.root / "receipts").glob("*.json"))), 1)

    def test_revoke_waits_for_inflight_commit_then_blocks_later_operations(self):
        claims = self.store.authority.authenticate(self.token, action="memory.commit", now=self.clock())
        context = multiprocessing.get_context("spawn")
        attempted, finished, release = context.Event(), context.Event(), context.Event()
        revoker, revoker_pipe = self.spawn_worker(_revoke_worker, self.root, attempted, finished)
        self.assertEqual(self.receive(revoker_pipe), ("ready", None))
        writer, writer_pipe = self.spawn_worker(_controlled_commit_worker, self.root, release)
        payload = request_fixture(self.initial_head, label="serialized-revoke")
        writer_pipe.send({"token": self.token, "request": payload, "run_id": RUN_ID})
        self.assertEqual(self.receive(writer_pipe), ("at_final_guard", None))
        revoker_pipe.send(claims["token_id"])
        self.assertTrue(attempted.wait(PROCESS_TIMEOUT))
        try:
            self.assertFalse(finished.wait(0.1), "revocation overtook a held authority commit lock")
        finally:
            release.set()
        kind, head = self.receive(writer_pipe)
        self.assertEqual(kind, "committed")
        self.assertEqual(self.receive(revoker_pipe), ("revoked", None))
        self.join_cleanly(writer)
        self.join_cleanly(revoker)
        self.assertEqual(self.head(), head)
        with self.assertRaises(AuthorizationError):
            self.store.status(self.token, run_id=RUN_ID)
        fresh = self.store.issue(run_id=RUN_ID, actions=ACTIONS)
        visible = self.store.export(fresh, run_id=RUN_ID)
        self.assertEqual(visible["graph"], graph_from_requests(payload))
        with self.assertRaises(AuthorizationError):
            self.store.commit(self.token, payload, run_id=RUN_ID)

    @unittest.skipUnless(os.name == "posix" and hasattr(signal, "SIGKILL"), "requires actual SIGKILL")
    def test_sigkill_each_checkpoint_reopen_recover_and_replay_is_complete_old_or_new(self):
        for stage in ("after_bodies", "after_graph", "before_manifest", "after_manifest"):
            with self.subTest(stage=stage):
                store, token = self.make_store("kill-" + stage)
                seed = request_fixture(self.head(store), label="seed/" + stage, body="old committed body\n")
                seed_result = self.commit(seed, store=store, token=token)
                request = request_fixture(seed_result["head"], label="new/" + stage,
                                          body="new transaction body\n")
                self.kill_at_stage(store, token, request, stage)
                process, connection = self.spawn_worker(_recover_reopen_worker, store.root)
                connection.send({"token": token, "run_id": RUN_ID, "request": request})
                kind, report = self.receive(connection)
                self.assertEqual(kind, "recovered")
                self.join_cleanly(process)
                committed = stage == "after_manifest"
                before_graph = graph_from_requests(seed, request) if committed else graph_from_requests(seed)
                expected_before = receipt_head(read_json(next(
                    path for path in (store.root / "receipts").glob("*.json")
                    if read_json(path)["binding"]["request_id"] == request["request_id"]))) if committed else seed_result["head"]
                self.assertEqual(report["before"]["head"], expected_before)
                self.assertEqual(report["before"]["generation"], 2 if committed else 1)
                self.assertEqual(report["exported_before"]["graph"], before_graph)
                self.assertEqual(report["exported_after"]["graph"], before_graph)
                pending = "committed_pending" if committed else "uncommitted_pending"
                self.assertEqual(report["inspected"]["pending_before"]["status"], pending)
                self.assertFalse(report["inspected"]["applied"])
                self.assertFalse(report["inspected"]["head_changed"])
                self.assertEqual(report["recovered"]["head"], expected_before)
                self.assertFalse(report["recovered"]["head_changed"])
                self.assertFalse(report["recovered"]["external_side_effects_replayed"])
                quarantine_name = report["recovered"]["quarantined"]
                self.assertIsNotNone(quarantine_name)
                self.assertEqual(report["inspected"]["pending_before"],
                                 report["recovered"]["pending_before"])
                raw_pending = (store.root / "quarantine" / quarantine_name).read_bytes()
                self.assertEqual(sha(raw_pending), report["recovered"]["pending_before"]["sha256"])
                self.assertEqual(report["inspected"]["prepared_only_receipts"], 0 if committed else 1)
                self.assertEqual(report["replay"]["replayed"], committed)
                self.assertEqual(report["after"]["generation"], 2)
                self.assertEqual(report["final"]["graph"], graph_from_requests(seed, request))
                self.assertEqual(report["after"]["head"], report["replay"]["head"])
                self.assertTrue((store.root / "bodies" / (seed["revisions"][0]["body_sha256"] + ".md")).exists())
                self.assertEqual(len(list((store.root / "receipts").glob("*.json"))), 2)
                # The application intentionally does not roll back/replay external effects.
                self.assertEqual(report["after"]["pending"]["status"], "empty")


class MemoryRecoveryTests(MemoryFixtureMixin, unittest.TestCase):
    def interrupted(self, stage, request=None, *, store=None, token=None):
        store = store or self.store
        token = token or self.token
        request = request or request_fixture(self.head(store))

        def checkpoint(observed):
            if observed == stage:
                raise InjectedCheckpointFailure(stage)

        failing = CheckpointStore(store.root, checkpoint=checkpoint, clock=self.clock)
        with self.assertRaises(InjectedCheckpointFailure):
            failing.commit(token, request, run_id=RUN_ID)
        return request

    def test_inspect_is_read_only_and_apply_quarantines_original_pending_bytes(self):
        request = self.interrupted("after_graph")
        before = self.authoritative_files()
        raw = (self.root / "journal/pending.log").read_bytes()
        inspected = self.store.recover(self.token, run_id=RUN_ID, apply=False)
        self.assertEqual(inspected["pending_before"]["status"], "uncommitted_pending")
        self.assertEqual(inspected["prepared_only_receipts"], 1)
        self.assertEqual(self.authoritative_files(), before)
        self.assertEqual(self.export()["graph"], graph_from_requests())
        with self.assertRaises(MemoryRecoveryRequired):
            self.commit(request)
        recovered = self.store.recover(self.token, run_id=RUN_ID, apply=True)
        self.assertTrue(recovered["applied"])
        self.assertFalse(recovered["head_changed"])
        self.assertEqual((self.root / "quarantine" / recovered["quarantined"]).read_bytes(), raw)
        self.assertEqual((self.root / "journal/pending.log").read_bytes(), b"")
        self.assert_unchanged_head()
        self.assertFalse(self.commit(request)["replayed"])
        self.assertEqual(self.export()["graph"], graph_from_requests(request))

    def test_prepared_request_id_remains_bound_after_final_cancellation_and_recovery(self):
        request = request_fixture(self.initial_head)
        cancel = threading.Event()

        def checkpoint(stage):
            if stage == "before_manifest":
                cancel.set()

        store = CheckpointStore(self.root, checkpoint=checkpoint, clock=self.clock)
        with self.assertRaises(MemoryCancelled):
            store.commit(self.token, request, run_id=RUN_ID, cancel_event=cancel)
        self.assert_unchanged_head()
        receipts_before = {path.name: path.read_bytes() for path in (self.root / "receipts").iterdir()}
        self.assertEqual(len(receipts_before), 1)
        self.store.recover(self.token, run_id=RUN_ID, apply=True)
        changed = deepcopy(request)
        changed["entities"][0]["name"] = "cancelled ID cannot be rebound"
        self.assert_rejected_without_head_change(changed, MemoryConflict)
        self.assertEqual({path.name: path.read_bytes() for path in (self.root / "receipts").iterdir()}, receipts_before)
        cancel.clear()
        result = self.commit(request, cancel_event=cancel)
        self.assertFalse(result["replayed"])
        self.assertEqual(self.export()["graph"], graph_from_requests(request))

    def test_final_expiry_guard_observes_clock_changed_inside_before_manifest_hook(self):
        request = request_fixture(self.initial_head)
        original_time = self.clock.value

        def checkpoint(stage):
            if stage == "before_manifest":
                self.clock.value = original_time + 300

        store = CheckpointStore(self.root, checkpoint=checkpoint, clock=self.clock)
        with self.assertRaises(AuthorizationError):
            store.commit(self.token, request, run_id=RUN_ID)
        self.assert_unchanged_head()
        self.clock.value = original_time
        self.assertEqual(self.status()["pending"]["status"], "uncommitted_pending")
        self.store.recover(self.token, run_id=RUN_ID, apply=True)
        self.assertFalse(self.commit(request)["replayed"])

    def test_exact_committed_replay_precedes_pending_slot_recovery(self):
        request = request_fixture(self.initial_head)

        def checkpoint(stage):
            if stage == "after_manifest":
                raise OSError(errno.EIO, "synthetic post-manifest failure")

        store = CheckpointStore(self.root, checkpoint=checkpoint, clock=self.clock)
        with self.assertRaises(MemoryIndeterminate):
            store.commit(self.token, request, run_id=RUN_ID)
        reopened = MemoryStore(self.root, clock=self.clock)
        before = self.authoritative_files()
        replay = self.commit(request, store=reopened)
        self.assertTrue(replay["replayed"])
        self.assertEqual(self.status(reopened)["pending"]["status"], "committed_pending")
        self.assertEqual(self.authoritative_files(), before)
        self.assertEqual(self.export(reopened)["graph"], graph_from_requests(request))
        with self.assertRaises(MemoryRecoveryRequired):
            self.commit(request_fixture(replay["head"], label="next"), store=reopened)
        self.assertFalse(reopened.recover(self.token, run_id=RUN_ID, apply=True)["head_changed"])
        self.assertEqual(self.status(reopened)["status"], "ready")

    def test_uncommitted_graph_and_prepared_receipt_never_confer_visibility(self):
        request = self.interrupted("after_graph")
        receipt_path = next((self.root / "receipts").glob("*.json"))
        receipt = read_json(receipt_path)
        prospective_head = receipt_head(receipt)
        self.assertTrue((self.root / "graphs" / (receipt["graph_sha256"] + ".json")).exists())
        self.assertEqual(self.export()["graph"], graph_from_requests())
        for head in (prospective_head, receipt["request_key"], receipt["graph_sha256"]):
            with self.subTest(head=head), self.assertRaises(MemoryConflict):
                self.export(head=head)
        self.store.rebuild_projection(self.token, run_id=RUN_ID)
        self.assertEqual(read_json(self.root / "projection/ids.json")["revisions"], [])
        self.assert_unchanged_head()
        self.assertEqual(request["expected_head"], self.initial_head)

    def test_corrupt_tail_is_quarantined_byte_exactly_without_advancing_head(self):
        self.interrupted("after_graph")
        original = (self.root / "journal/pending.log").read_bytes()
        corrupt = original + b'{"torn":true,\xff\x00'
        write_private(self.root / "journal/pending.log", corrupt)
        before = self.head()
        inspected = self.store.recover(self.token, run_id=RUN_ID)
        self.assertEqual(inspected["pending_before"]["status"], "corrupt")
        self.assertEqual((self.root / "journal/pending.log").read_bytes(), corrupt)
        recovered = self.store.recover(self.token, run_id=RUN_ID, apply=True)
        self.assertEqual((self.root / "quarantine" / recovered["quarantined"]).read_bytes(), corrupt)
        self.assertEqual(self.head(), before)
        self.assertEqual(self.status()["pending"]["status"], "empty")
        self.assertEqual(self.export()["graph"], graph_from_requests())

    def test_oversized_pending_is_not_read_and_can_be_quarantined_without_copying(self):
        oversized = b"not json\xff" * 4096
        self.assertGreater(len(oversized), 16384)
        journal = self.root / "journal/pending.log"
        write_private(journal, oversized)
        original_read = PrivateDirectory.read_bytes

        def bounded_read(directory, name, *, max_bytes=None):
            if directory.root == self.root / "journal" and name == "pending.log":
                self.fail("oversized pending journal contents must not be read")
            return original_read(directory, name, max_bytes=max_bytes)

        with mock.patch.object(PrivateDirectory, "read_bytes", new=bounded_read):
            inspected = self.store.recover(self.token, run_id=RUN_ID, apply=False)
            self.assertEqual(inspected["pending_before"]["reason"], "journal_size")
            self.assertEqual(inspected["pending_before"]["bytes"], len(oversized))
            recovered = self.store.recover(self.token, run_id=RUN_ID, apply=True)
        self.assertEqual((self.root / "quarantine" / recovered["quarantined"]).read_bytes(), oversized)
        self.assertEqual(journal.read_bytes(), b"")
        self.assert_unchanged_head()

    def test_missing_pending_requires_explicit_recovery_and_never_changes_head(self):
        (self.root / "journal/pending.log").unlink()
        self.assertEqual(self.status()["pending"]["status"], "missing")
        self.assert_rejected_without_head_change(request_fixture(self.initial_head), MemoryRecoveryRequired)
        inspected = self.store.recover(self.token, run_id=RUN_ID)
        self.assertEqual(inspected["pending_before"]["status"], "missing")
        self.assertFalse((self.root / "journal/pending.log").exists())
        recovered = self.store.recover(self.token, run_id=RUN_ID, apply=True)
        self.assertIsNone(recovered["quarantined"])
        self.assertEqual((self.root / "journal/pending.log").read_bytes(), b"")
        self.assert_unchanged_head()

    def test_quarantined_bytes_remain_in_store_capacity_accounting(self):
        before = self.status()["bytes"]
        raw = b"corrupt pending retained as evidence\xff" * 23
        write_private(self.root / "journal/pending.log", raw)
        pending_bytes = self.status()["bytes"]
        self.assertEqual(pending_bytes - before, len(raw))
        recovered = self.store.recover(self.token, run_id=RUN_ID, apply=True)
        self.assertEqual(self.status()["bytes"], pending_bytes)
        self.assertEqual((self.root / "quarantine" / recovered["quarantined"]).read_bytes(), raw)
        self.store.recover(self.token, run_id=RUN_ID, apply=True)
        self.assertEqual(self.status()["bytes"], pending_bytes)
        self.assert_unchanged_head()

    def test_quarantine_capacity_is_explicit_and_never_discards_old_evidence(self):
        store, token = self.make_store("quarantine-cap", limits={"quarantine_files": 1})
        journal = store.root / "journal/pending.log"
        write_private(journal, b"first corrupt pending\x00")
        first = store.recover(token, run_id=RUN_ID, apply=True)
        first_path = store.root / "quarantine" / first["quarantined"]
        first_bytes = first_path.read_bytes()
        write_private(journal, b"second corrupt pending\xff")
        before = self.authoritative_files(store.root)
        with self.assertRaises(MemoryCapacity):
            store.recover(token, run_id=RUN_ID, apply=True)
        self.assertEqual(self.authoritative_files(store.root), before)
        self.assertEqual(first_path.read_bytes(), first_bytes)
        self.assertEqual(len(list((store.root / "quarantine").iterdir())), 1)
        self.assertEqual(self.status(store, token)["generation"], 0)

    def test_prepared_receipts_count_against_capacity_after_recovery(self):
        store, token = self.make_store("prepared-cap", limits={"receipts": 1})
        request = self.interrupted("after_bodies", store=store, token=token)
        recovered = store.recover(token, run_id=RUN_ID, apply=True)
        self.assertEqual(recovered["prepared_only_receipts"], 1)
        self.assert_rejected_without_head_change(request_fixture(self.head(store), label="different"),
                                                MemoryCapacity, store=store, token=token)
        self.assertEqual(len(list((store.root / "receipts").iterdir())), 1)
        self.assertFalse(self.commit(request, store=store, token=token)["replayed"])
        self.assertEqual(self.status(store, token)["generation"], 1)

    def test_committed_body_graph_receipt_or_manifest_damage_fails_closed_no_fallback(self):
        for kind in ("body", "graph", "receipt", "manifest"):
            with self.subTest(kind=kind):
                store, token = self.make_store("damage-" + kind)
                seed = request_fixture(self.head(store), label="old")
                first = self.commit(seed, store=store, token=token)
                second = self.commit(request_fixture(first["head"], label="new"), store=store, token=token)
                if kind == "body":
                    # Damage an ancestor's body: latest-state fallback is forbidden.
                    path = store.root / "bodies" / (seed["revisions"][0]["body_sha256"] + ".md")
                elif kind == "graph":
                    path = store.root / "graphs" / (first["receipt"]["graph_sha256"] + ".json")
                elif kind == "receipt":
                    path = store.root / "receipts" / (first["receipt"]["request_key"] + ".json")
                else:
                    path = store.root / "manifest.json"
                write_private(path, b"{\"corrupt\":true}\n")
                before = self.authoritative_files(store.root)
                reopened = MemoryStore(store.root, clock=self.clock)
                operations = [
                    lambda: reopened.status(token, run_id=RUN_ID),
                    lambda: reopened.export(token, run_id=RUN_ID),
                    lambda: reopened.export(token, run_id=RUN_ID, head=first["head"]),
                    lambda: reopened.recover(token, run_id=RUN_ID, apply=False),
                    lambda: reopened.recover(token, run_id=RUN_ID, apply=True),
                    lambda: reopened.rebuild_projection(token, run_id=RUN_ID),
                    lambda: reopened.commit(token, request_fixture(second["head"], label="refuse"), run_id=RUN_ID),
                ]
                for index, operation in enumerate(operations):
                    with self.subTest(operation=index), self.assertRaises(MemoryCorruption):
                        operation()
                self.assertEqual(self.authoritative_files(store.root), before)

    def test_missing_committed_object_cannot_be_recovered_as_genesis(self):
        request = request_fixture(self.initial_head)
        result = self.commit(request)
        (self.root / "receipts" / (result["receipt"]["request_key"] + ".json")).unlink()
        before = self.authoritative_files()
        with self.assertRaises(MemoryCorruption):
            self.store.recover(self.token, run_id=RUN_ID, apply=True)
        self.assertEqual(self.authoritative_files(), before)
        self.assertEqual(self.head(), result["head"])

    def test_duplicate_json_keys_nonfinite_or_noncanonical_manifest_fail_closed(self):
        original = (self.root / "manifest.json").read_bytes()
        value = read_json(self.root / "manifest.json")
        corruptions = [
            b'{"generation":0,' + original[1:],
            original.replace(b'"generation":0', b'"generation":NaN'),
            original + b"\n",
            b"\xff" + original,
            encoded({**value, "generation": True}),
        ]
        for index, raw in enumerate(corruptions):
            with self.subTest(case=index):
                write_private(self.root / "manifest.json", raw)
                with self.assertRaises(MemoryCorruption):
                    self.status()
                with self.assertRaises(MemoryCorruption):
                    self.store.recover(self.token, run_id=RUN_ID, apply=True)
                self.assertEqual((self.root / "manifest.json").read_bytes(), raw)
        write_private(self.root / "manifest.json", original)

    def test_corrupt_prepared_request_binding_is_not_silently_deleted(self):
        self.interrupted("after_bodies")
        receipt_path = next((self.root / "receipts").glob("*.json"))
        write_private(receipt_path, b'{"request_key":"broken"}')
        before = self.authoritative_files()
        # A damaged persistent binding must stop apply; it must not be forgotten.
        with self.assertRaises((MemoryStoreError, PrivateFilesystemError)):
            self.store.recover(self.token, run_id=RUN_ID, apply=True)
        self.assertEqual(self.authoritative_files(), before)
        self.assert_unchanged_head()

    def test_missing_permanent_binding_referenced_by_valid_intent_cannot_be_consumed(self):
        self.interrupted("after_bodies")
        receipt_path = next((self.root / "receipts").glob("*.json"))
        receipt_path.unlink()
        # A valid first frame is still evidence of this request key, even with a
        # damaged tail; do not erase it and allow a new payload to take its ID.
        original = (self.root / "journal/pending.log").read_bytes()
        for raw in (original, original + b'{"torn":', original + b"x" * 16384):
            with self.subTest(torn=raw != original):
                write_private(self.root / "journal/pending.log", raw)
                before = self.authoritative_files()
                with self.assertRaises(MemoryCorruption):
                    self.store.recover(self.token, run_id=RUN_ID, apply=True)
                self.assertEqual(self.authoritative_files(), before)
                self.assert_unchanged_head()

    def test_pending_frame_checksum_order_duplicate_keys_and_torn_lines_are_corrupt(self):
        self.interrupted("after_graph")
        original = (self.root / "journal/pending.log").read_bytes()
        journal_frame = json.loads(original.splitlines()[0])
        checksum = {**journal_frame, "sha256": "0" * 64}
        index = {**journal_frame, "index": 1}
        cases = [original[:-1], encoded(checksum) + b"\n", encoded(index) + b"\n",
                 b'{"index":0,' + original[1:], original * 4]
        for number, raw in enumerate(cases):
            with self.subTest(case=number):
                write_private(self.root / "journal/pending.log", raw)
                inspected = self.store.recover(self.token, run_id=RUN_ID)
                self.assertEqual(inspected["pending_before"]["status"], "corrupt")
                self.assertEqual((self.root / "journal/pending.log").read_bytes(), raw)
                self.assert_unchanged_head()


class MemoryWriteFaultTests(MemoryFixtureMixin, unittest.TestCase):
    def test_body_write_failure_preserves_old_head_and_allows_explicit_recovery_retry(self):
        request = request_fixture(self.initial_head)
        original_write = PrivateDirectory.write_bytes

        def fail_body(directory, name, data, exclusive=False, *, before_replace=None):
            if directory.root == self.root / "bodies":
                raise OSError(errno.ENOSPC, "synthetic body write failure")
            return original_write(directory, name, data, exclusive=exclusive,
                                  before_replace=before_replace)

        with mock.patch.object(PrivateDirectory, "write_bytes", new=fail_body):
            with self.assertRaises(OSError):
                self.commit(request)
        self.assert_unchanged_head()
        self.assertEqual(self.export()["graph"], graph_from_requests())
        self.store.recover(self.token, run_id=RUN_ID, apply=True)
        self.assertFalse(self.commit(request)["replayed"])

    def test_graph_write_failure_preserves_old_head_and_keeps_unreferenced_body(self):
        request = request_fixture(self.initial_head)
        original_write = PrivateDirectory.write_bytes

        def fail_graph(directory, name, data, exclusive=False, *, before_replace=None):
            if directory.root == self.root / "graphs":
                raise OSError(errno.EIO, "synthetic graph write failure")
            return original_write(directory, name, data, exclusive=exclusive,
                                  before_replace=before_replace)

        with mock.patch.object(PrivateDirectory, "write_bytes", new=fail_graph):
            with self.assertRaises(OSError):
                self.commit(request)
        self.assert_unchanged_head()
        body_path = self.root / "bodies" / (request["revisions"][0]["body_sha256"] + ".md")
        self.assertTrue(body_path.exists())
        self.assertEqual(self.export()["graph"], graph_from_requests())
        self.store.recover(self.token, run_id=RUN_ID, apply=True)
        self.assertTrue(body_path.exists())
        self.assertFalse(self.commit(request)["replayed"])

    def test_manifest_replace_failure_is_indeterminate_but_reopen_finds_old_head(self):
        request = request_fixture(self.initial_head)
        original_replace = os.replace

        def fail_manifest(source, destination, *args, **kwargs):
            if destination == "manifest.json":
                raise OSError(errno.EIO, "synthetic manifest replace failure")
            return original_replace(source, destination, *args, **kwargs)

        with mock.patch("rethlas_core.private_fs.os.replace", side_effect=fail_manifest):
            with self.assertRaises(MemoryIndeterminate):
                self.commit(request)
        reopened = MemoryStore(self.root, clock=self.clock)
        self.assert_unchanged_head()
        self.assertEqual(self.export(reopened)["graph"], graph_from_requests())
        reopened.recover(self.token, run_id=RUN_ID, apply=True)
        self.assertFalse(self.commit(request, store=reopened)["replayed"])

    def test_manifest_file_fsync_failure_precedes_replace_and_reopen_retries_after_recovery(self):
        request = request_fixture(self.initial_head)
        original_write = PrivateDirectory.write_bytes
        original_fsync = os.fsync
        injected = []

        def fail_manifest_fsync(directory, name, data, exclusive=False, *, before_replace=None):
            if directory.root != self.root or name != "manifest.json":
                return original_write(directory, name, data, exclusive=exclusive,
                                      before_replace=before_replace)

            def file_fsync(fd):
                if stat.S_ISREG(os.fstat(fd).st_mode):
                    injected.append("file-fsync")
                    raise OSError(errno.EIO, "synthetic manifest file fsync failure")
                return original_fsync(fd)

            with mock.patch("rethlas_core.private_fs.os.fsync", side_effect=file_fsync):
                return original_write(directory, name, data, exclusive=exclusive,
                                      before_replace=before_replace)

        with mock.patch.object(PrivateDirectory, "write_bytes", new=fail_manifest_fsync):
            with self.assertRaises(MemoryIndeterminate):
                self.commit(request)
        self.assertEqual(injected, ["file-fsync"])
        reopened = MemoryStore(self.root, clock=self.clock)
        self.assert_unchanged_head()
        reopened.recover(self.token, run_id=RUN_ID, apply=True)
        self.assertFalse(self.commit(request, store=reopened)["replayed"])

    def test_manifest_directory_fsync_failure_may_be_committed_reopen_exact_replay_decides(self):
        request = request_fixture(self.initial_head)
        original_write = PrivateDirectory.write_bytes
        original_fsync = os.fsync
        injected = []

        def fail_manifest_dirsync(directory, name, data, exclusive=False, *, before_replace=None):
            if directory.root != self.root or name != "manifest.json":
                return original_write(directory, name, data, exclusive=exclusive,
                                      before_replace=before_replace)

            def directory_fsync(fd):
                if stat.S_ISDIR(os.fstat(fd).st_mode):
                    injected.append("directory-fsync")
                    raise OSError(errno.EIO, "synthetic manifest directory fsync failure")
                return original_fsync(fd)

            with mock.patch("rethlas_core.private_fs.os.fsync", side_effect=directory_fsync):
                return original_write(directory, name, data, exclusive=exclusive,
                                      before_replace=before_replace)

        with mock.patch.object(PrivateDirectory, "write_bytes", new=fail_manifest_dirsync):
            with self.assertRaises(MemoryIndeterminate):
                self.commit(request)
        self.assertEqual(injected, ["directory-fsync"])
        # Do not assert that a failed dir-fsync means no commit: replace happened.
        reopened = MemoryStore(self.root, clock=self.clock)
        before = self.authoritative_files()
        replay = self.commit(request, store=reopened)
        self.assertTrue(replay["replayed"])
        self.assertEqual(self.head(), replay["head"])
        self.assertEqual(self.export(reopened)["graph"], graph_from_requests(request))
        self.assertEqual(self.authoritative_files(), before)
        reopened.recover(self.token, run_id=RUN_ID, apply=True)
        self.assertEqual(self.status(reopened)["generation"], 1)

    def test_journal_final_clear_failure_keeps_committed_head_replayable(self):
        request = request_fixture(self.initial_head)
        original_write = PrivateDirectory.write_bytes

        def fail_final_clear(directory, name, data, exclusive=False, *, before_replace=None):
            if directory.root == self.root / "journal" and name == "pending.log" and data == b"":
                raise OSError(errno.EIO, "synthetic final pending clear failure")
            return original_write(directory, name, data, exclusive=exclusive,
                                  before_replace=before_replace)

        with mock.patch.object(PrivateDirectory, "write_bytes", new=fail_final_clear):
            with self.assertRaises(MemoryIndeterminate):
                self.commit(request)
        reopened = MemoryStore(self.root, clock=self.clock)
        self.assertEqual(self.status(reopened)["pending"]["status"], "committed_pending")
        replay = self.commit(request, store=reopened)
        self.assertTrue(replay["replayed"])
        self.assertEqual(self.export(reopened)["graph"], graph_from_requests(request))
        reopened.recover(self.token, run_id=RUN_ID, apply=True)
        self.assertEqual(self.status(reopened)["generation"], 1)
        self.assertEqual(self.status(reopened)["pending"]["status"], "empty")

    def test_existing_immutable_body_requires_fresh_sync_before_publishing_new_head(self):
        request = self.interrupted_after_body_for_sync()
        original_sync = PrivateDirectory.sync_file
        observed = []

        def fail_existing_body(directory, name):
            if directory.root == self.root / "bodies":
                observed.append(name)
                raise OSError(errno.EIO, "synthetic existing immutable sync failure")
            return original_sync(directory, name)

        with mock.patch.object(PrivateDirectory, "sync_file", new=fail_existing_body):
            with self.assertRaises(OSError):
                self.commit(request)
        self.assertEqual(observed, [request["revisions"][0]["body_sha256"] + ".md"])
        self.assert_unchanged_head()
        self.assertEqual(self.export()["graph"], graph_from_requests())
        self.store.recover(self.token, run_id=RUN_ID, apply=True)
        self.assertFalse(self.commit(request)["replayed"])

    def interrupted_after_body_for_sync(self):
        request = request_fixture(self.initial_head)

        def checkpoint(stage):
            if stage == "after_bodies":
                raise InjectedCheckpointFailure(stage)

        interrupted = CheckpointStore(self.root, checkpoint=checkpoint, clock=self.clock)
        with self.assertRaises(InjectedCheckpointFailure):
            interrupted.commit(self.token, request, run_id=RUN_ID)
        self.store.recover(self.token, run_id=RUN_ID, apply=True)
        return request

    def test_committed_replay_sync_failure_is_indeterminate_without_changing_any_bytes(self):
        request = request_fixture(self.initial_head)
        committed = self.commit(request)
        advanced = self.commit(request_fixture(committed["head"], label="advanced-sync"))
        before = self.authoritative_files()
        original_sync = PrivateDirectory.sync_file
        observed = []

        def fail_manifest_sync(directory, name):
            if directory.root == self.root and name == "manifest.json":
                observed.append(name)
                raise OSError(errno.EIO, "synthetic replay durability barrier failure")
            return original_sync(directory, name)

        with mock.patch.object(PrivateDirectory, "sync_file", new=fail_manifest_sync):
            with self.assertRaises(MemoryIndeterminate):
                self.commit(request)
        self.assertEqual(observed, ["manifest.json"])
        self.assertEqual(self.authoritative_files(), before)
        self.assertEqual(self.head(), advanced["head"])
        replay = self.commit(request)
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["head"], committed["head"])
        self.assertEqual(self.head(), advanced["head"])

    def test_recovery_sync_failure_preserves_pending_evidence_and_head(self):
        request = self.interrupted_after_body_for_sync()
        # Create another pending attempt with a real body already present.
        def checkpoint(stage):
            if stage == "after_graph":
                raise InjectedCheckpointFailure(stage)

        interrupted = CheckpointStore(self.root, checkpoint=checkpoint, clock=self.clock)
        with self.assertRaises(InjectedCheckpointFailure):
            interrupted.commit(self.token, request, run_id=RUN_ID)
        before = self.authoritative_files()
        original_sync = PrivateDirectory.sync_file

        def fail_manifest_sync(directory, name):
            if directory.root == self.root and name == "manifest.json":
                raise OSError(errno.EIO, "synthetic recovery durability barrier failure")
            return original_sync(directory, name)

        with mock.patch.object(PrivateDirectory, "sync_file", new=fail_manifest_sync):
            with self.assertRaises(MemoryIndeterminate):
                self.store.recover(self.token, run_id=RUN_ID, apply=True)
        self.assertEqual(self.authoritative_files(), before)
        self.assert_unchanged_head()
        self.store.recover(self.token, run_id=RUN_ID, apply=True)
        self.assertFalse(self.commit(request)["replayed"])

    def test_manifest_temp_write_failure_never_publishes_partial_state(self):
        request = request_fixture(self.initial_head)
        original_write = PrivateDirectory.write_bytes

        def fail_manifest_write(directory, name, data, exclusive=False, *, before_replace=None):
            if directory.root != self.root or name != "manifest.json":
                return original_write(directory, name, data, exclusive=exclusive,
                                      before_replace=before_replace)
            with mock.patch("rethlas_core.private_fs._write_all", side_effect=OSError(errno.ENOSPC, "synthetic temp write")):
                return original_write(directory, name, data, exclusive=exclusive,
                                      before_replace=before_replace)

        with mock.patch.object(PrivateDirectory, "write_bytes", new=fail_manifest_write):
            with self.assertRaises(MemoryIndeterminate):
                self.commit(request)
        self.assert_unchanged_head()
        self.assertEqual(self.export()["graph"], graph_from_requests())
        self.store.recover(self.token, run_id=RUN_ID, apply=True)
        self.assertFalse(self.commit(request)["replayed"])


if __name__ == "__main__":
    unittest.main()
