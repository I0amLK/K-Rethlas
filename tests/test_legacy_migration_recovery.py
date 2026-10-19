"""Real owned-process migration replay, interruption and I/O fault fixtures.

Only synthetic fixture stores and packages are used. Checkpoints are trusted
Python subclasses, not request/environment/CLI switches. SIGKILL is not a claim
about arbitrary power loss, NFS, multi-master operation or mathematical review.
"""
from __future__ import annotations

import errno
import json
import multiprocessing
import os
from pathlib import Path
import signal
import threading
import time
import unittest
from unittest import mock

from test_legacy_migration import MigrationFixtureMixin, legacy_line
from test_memory_recovery import CheckpointStore, ProcessFixtureMixin, PROCESS_TIMEOUT
from test_memory_store import RUN_ID, encoded, request_fixture, sha
from rethlas_core.legacy_migration import MigrationError, apply, prepare
from rethlas_core.memory_store import (
    MemoryCancelled, MemoryConflict, MemoryIndeterminate, MemoryRecoveryRequired, MemoryStore,
)
from rethlas_core.private_fs import PrivateDirectory


def _apply_worker(root, connection, start):
    try:
        payload = connection.recv()
        store = MemoryStore(Path(root))
        connection.send(("ready", None))
        if not start.wait(PROCESS_TIMEOUT):
            raise TimeoutError("parent did not release owned apply child")
        result = apply(Path(payload["bundle"]), payload["pin"], store, payload["token"], run_id=RUN_ID)
        connection.send(("committed", {key: result[key] for key in ("head", "replayed", "receipt")}))
    except Exception as exc:
        connection.send(("error", type(exc).__name__))
    finally:
        connection.close()


def _checkpoint_apply_worker(root, connection, release):
    try:
        payload = connection.recv()

        def checkpoint(stage):
            if stage == payload["stage"]:
                connection.send(("checkpoint", stage))
                if not release.wait(PROCESS_TIMEOUT * 2):
                    raise TimeoutError("parent did not terminate owned checkpoint child")

        store = CheckpointStore(Path(root), checkpoint=checkpoint)
        result = apply(Path(payload["bundle"]), payload["pin"], store, payload["token"], run_id=RUN_ID)
        connection.send(("unexpected_return", result["head"]))
    except Exception as exc:
        connection.send(("error", type(exc).__name__))
    finally:
        connection.close()


def _reopen_migration_worker(root, connection):
    try:
        payload = connection.recv()
        store = MemoryStore(Path(root))
        before = store.status(payload["token"], run_id=RUN_ID)
        inspected = store.recover(payload["token"], run_id=RUN_ID)
        recovered = store.recover(payload["token"], run_id=RUN_ID, apply=True)
        first = apply(Path(payload["bundle"]), payload["pin"], store, payload["token"], run_id=RUN_ID)
        second = apply(Path(payload["bundle"]), payload["pin"], store, payload["token"], run_id=RUN_ID)
        after = store.status(payload["token"], run_id=RUN_ID)
        connection.send(("recovered", {"before": before, "inspected": inspected, "recovered": recovered,
                                       "first": first, "second": second, "after": after}))
    except Exception as exc:
        connection.send(("error", type(exc).__name__))
    finally:
        connection.close()


class LegacyMigrationProcessTests(ProcessFixtureMixin, MigrationFixtureMixin, unittest.TestCase):
    def test_spawn_same_package_race_commits_once_and_replays_once(self):
        prepared = self.prepared()
        start = multiprocessing.get_context("spawn").Event()
        children = [self.spawn_worker(_apply_worker, self.root, start) for _ in range(2)]
        for _, connection in children:
            connection.send({"token": self.token, "bundle": str(self.bundle), "pin": prepared["bundle_sha256"]})
            self.assertEqual(self.receive(connection), ("ready", None))
        start.set()
        results = [self.receive(connection) for _, connection in children]
        for process, _ in children:
            self.join_cleanly(process)
        self.assertTrue(all(tag == "committed" for tag, _ in results), [tag for tag, _ in results])
        self.assertEqual(sorted(value["replayed"] for _, value in results), [False, True])
        self.assertEqual(results[0][1]["receipt"], results[1][1]["receipt"])
        self.assertEqual(self.status()["generation"], 1)
        self.assertEqual(len(list((self.root / "receipts").iterdir())), 1)

    def test_manifest_before_and_after_sigkill_fresh_process_recovery_and_exact_replay(self):
        for stage in ("after_intent", "after_graph", "before_manifest", "after_manifest"):
            with self.subTest(stage=stage):
                store, token = self.make_store("kill-" + stage)
                self.spec()
                bundle = self.base / ("bundle-" + stage)
                prepared = prepare(self.spec_path, bundle, store, token, run_id=RUN_ID)
                release = multiprocessing.get_context("spawn").Event()
                process, connection = self.spawn_worker(_checkpoint_apply_worker, store.root, release)
                connection.send({"token": token, "bundle": str(bundle), "pin": prepared["bundle_sha256"], "stage": stage})
                self.assertEqual(self.receive(connection), ("checkpoint", stage))
                self.assertTrue(process.is_alive())
                os.kill(process.pid, signal.SIGKILL)
                process.join(PROCESS_TIMEOUT)
                self.assertEqual(process.exitcode, -signal.SIGKILL)
                reopened, channel = self.spawn_worker(_reopen_migration_worker, store.root)
                channel.send({"token": token, "bundle": str(bundle), "pin": prepared["bundle_sha256"]})
                tag, result = self.receive(channel)
                self.assertEqual(tag, "recovered", result if type(result) is str else "unexpected outcome")
                self.join_cleanly(reopened)
                committed_before_kill = stage == "after_manifest"
                self.assertEqual(result["before"]["generation"], int(committed_before_kill))
                self.assertEqual(result["first"]["replayed"], committed_before_kill)
                self.assertTrue(result["second"]["replayed"])
                self.assertEqual(result["first"]["receipt"], result["second"]["receipt"])
                self.assertEqual(result["after"]["generation"], 1)
                self.assertEqual(result["after"]["pending"]["status"], "empty")
                self.assertEqual(result["after"]["counts"]["revisions"], 1)


class LegacyMigrationInterruptionTests(MigrationFixtureMixin, unittest.TestCase):
    def test_final_guard_cancel_no_publication_requires_explicit_recovery(self):
        prepared = self.prepared()
        cancel = threading.Event()
        store = CheckpointStore(self.root, checkpoint=lambda stage: cancel.set() if stage == "before_manifest" else None,
                                clock=self.clock)
        with self.assertRaises(MemoryCancelled):
            apply(self.bundle, prepared["bundle_sha256"], store, self.token, run_id=RUN_ID, cancel_event=cancel)
        self.assert_unchanged_head()
        self.assertEqual(self.status()["pending"]["status"], "uncommitted_pending")
        with self.assertRaises(MemoryRecoveryRequired):
            self.applied(prepared)
        self.store.recover(self.token, run_id=RUN_ID, apply=True)
        first = self.applied(prepared)
        self.assertFalse(first["replayed"])
        self.assertTrue(self.applied(prepared)["replayed"])
        self.assertEqual(len(list((self.root / "receipts").iterdir())), 1)

    def test_signal_event_after_manifest_is_not_reported_as_rollback(self):
        prepared = self.prepared()
        cancel = threading.Event()
        store = CheckpointStore(self.root, checkpoint=lambda stage: cancel.set() if stage == "after_manifest" else None,
                                clock=self.clock)
        result = apply(self.bundle, prepared["bundle_sha256"], store, self.token, run_id=RUN_ID, cancel_event=cancel)
        self.assertEqual(result["status"], "committed")
        self.assertEqual(self.status()["generation"], 1)
        self.assertTrue(self.applied(prepared)["replayed"])

    def test_after_manifest_response_failure_is_indeterminate_then_exact_replay(self):
        prepared = self.prepared()

        def checkpoint(stage):
            if stage == "after_manifest":
                raise OSError(errno.EIO, "synthetic response-side I/O failure")

        store = CheckpointStore(self.root, checkpoint=checkpoint, clock=self.clock)
        with self.assertRaises(MemoryIndeterminate):
            apply(self.bundle, prepared["bundle_sha256"], store, self.token, run_id=RUN_ID)
        self.assertEqual(self.status()["generation"], 1)
        replay = self.applied(prepared)
        self.assertTrue(replay["replayed"])
        self.assertEqual(self.status()["pending"]["status"], "committed_pending")
        self.store.recover(self.token, run_id=RUN_ID, apply=True)
        self.assertEqual(self.status()["pending"]["status"], "empty")

    def test_manifest_dir_fsync_failure_does_not_false_report_uncommitted(self):
        prepared = self.prepared()
        real_replace, real_fsync = os.replace, os.fsync
        state = {"published": False, "failed": False}

        def replace(source, destination, *args, **kwargs):
            result = real_replace(source, destination, *args, **kwargs)
            if destination == "manifest.json":
                state["published"] = True
            return result

        def fsync(fd):
            if state["published"] and not state["failed"]:
                state["failed"] = True
                raise OSError(errno.EIO, "synthetic manifest directory barrier failure")
            return real_fsync(fd)

        with mock.patch("rethlas_core.private_fs.os.replace", side_effect=replace), \
                mock.patch("rethlas_core.private_fs.os.fsync", side_effect=fsync):
            with self.assertRaises(MemoryIndeterminate):
                self.applied(prepared)
        self.assertTrue(state["published"] and state["failed"])
        self.assertEqual(self.status()["generation"], 1)
        self.assertTrue(self.applied(prepared)["replayed"])

    def test_before_manifest_replace_failure_explicit_recovery_preserves_binding(self):
        prepared = self.prepared()
        real_replace = os.replace

        def replace(source, destination, *args, **kwargs):
            if destination == "manifest.json":
                raise OSError(errno.EIO, "synthetic pre-publication replace failure")
            return real_replace(source, destination, *args, **kwargs)

        with mock.patch("rethlas_core.private_fs.os.replace", side_effect=replace):
            with self.assertRaises(MemoryIndeterminate):
                self.applied(prepared)
        self.assert_unchanged_head()
        self.store.recover(self.token, run_id=RUN_ID, apply=True)
        original = json.loads((self.bundle / "request.json").read_bytes())
        altered = json.loads(encoded(original))
        altered["expected_head"] = "0" * 64
        with self.assertRaises(MemoryConflict):
            self.commit(altered)
        self.assertFalse(self.applied(prepared)["replayed"])
        self.assertEqual(len(list((self.root / "receipts").iterdir())), 1)

    def test_package_write_file_fsync_failure_leaves_no_usable_manifest(self):
        self.spec()
        real_fsync, real_write = os.fsync, PrivateDirectory.write_bytes
        state = {"writing_snapshot": False, "failed": False}

        def write(directory, name, data, *args, **kwargs):
            state["writing_snapshot"] = directory.root == self.bundle and name == "snapshot.json"
            try:
                return real_write(directory, name, data, *args, **kwargs)
            finally:
                state["writing_snapshot"] = False

        def fsync(fd):
            if state["writing_snapshot"] and not state["failed"]:
                state["failed"] = True
                raise OSError(errno.EIO, "synthetic package file barrier failure")
            return real_fsync(fd)

        with mock.patch.object(PrivateDirectory, "write_bytes", new=write), \
                mock.patch("rethlas_core.private_fs.os.fsync", side_effect=fsync):
            with self.assertRaises(OSError):
                self.prepared()
        self.assertTrue(state["failed"])
        self.assertFalse((self.bundle / "manifest.json").exists())
        self.assert_unchanged_head()

    def test_cancelled_pending_cannot_be_rebased_under_same_request_id(self):
        prepared = self.prepared()
        cancel = threading.Event()
        store = CheckpointStore(self.root, checkpoint=lambda stage: cancel.set() if stage == "before_manifest" else None,
                                clock=self.clock)
        with self.assertRaises(MemoryCancelled):
            apply(self.bundle, prepared["bundle_sha256"], store, self.token, run_id=RUN_ID, cancel_event=cancel)
        self.store.recover(self.token, run_id=RUN_ID, apply=True)
        self.commit(request_fixture(self.initial_head, label="someone-else"))
        with self.assertRaises(MemoryConflict):
            self.applied(prepared)
        self.assertEqual(self.status()["generation"], 1)
        self.assertEqual(len(self.export()["graph"]["revisions"]), 1)
