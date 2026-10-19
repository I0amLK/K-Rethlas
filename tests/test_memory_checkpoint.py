"""Metadata-only locking tests for a trusted current-use controller.

These synthetic tests require a separately authorized run. A checkpoint does not
validate the graph chain or confer mathematical/current-use validity by itself.
"""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import stat
import threading
import unittest
from unittest import mock

from rethlas_core.authorization import Authority, AuthorizationError
from rethlas_core.memory_store import (
    MemoryCancelled, MemoryCorruption, MemoryIndeterminate, MemoryStore,
)
from rethlas_core.private_fs import PrivateDirectory, canonical_json, digest

from test_memory_store import MemoryFixtureMixin, RUN_ID, read_json, request_fixture, write_private
from test_memory_recovery import ProcessFixtureMixin


CHECKPOINT_FIELDS = {"store_id", "config_sha256", "head", "generation", "graph_sha256",
                     "receipt_key", "receipt_sha256", "parent_head"}


def _checkpoint_commit_worker(root, connection):
    try:
        payload = connection.recv()
        store = MemoryStore(Path(root))
        connection.send(("ready", None))
        if connection.recv() != "commit":
            raise ValueError("unexpected synthetic command")
        connection.send(("attempting", None))
        result = store.commit(payload["token"], payload["request"], run_id=RUN_ID)
        connection.send(("committed", result["head"]))
    except Exception as exc:
        connection.send(("error", type(exc).__name__))
    finally:
        connection.close()


def _checkpoint_store_lock_worker(root, connection):
    try:
        directory = PrivateDirectory(Path(root))
        connection.send(("ready", None))
        if connection.recv() != "lock":
            raise ValueError("unexpected synthetic command")
        connection.send(("attempting", None))
        with directory.lock():
            connection.send(("acquired", None))
    except Exception as exc:
        connection.send(("error", type(exc).__name__))
    finally:
        connection.close()


def _checkpoint_revoke_worker(root, connection):
    try:
        authority = Authority(Path(root) / "authority")
        token_id = connection.recv()
        connection.send(("ready", None))
        if connection.recv() != "revoke":
            raise ValueError("unexpected synthetic command")
        connection.send(("attempting", None))
        authority.revoke(token_id)
        connection.send(("revoked", None))
    except Exception as exc:
        connection.send(("error", type(exc).__name__))
    finally:
        connection.close()


class MemoryCheckpointTests(ProcessFixtureMixin, MemoryFixtureMixin, unittest.TestCase):
    def checkpoint(self, token=None, **kwargs):
        return self.store.locked_checkpoint(token or self.token, run_id=RUN_ID, **kwargs)

    def test_genesis_metadata_has_exact_fields_without_hydrating_any_graph_or_receipt(self):
        manifest = read_json(self.root / "manifest.json")
        before = self.authoritative_files()
        reads = []
        original_read = self.store.directory.read_bytes

        def read(name, **kwargs):
            reads.append(name)
            return original_read(name, **kwargs)

        with mock.patch.object(self.store, "_chain", side_effect=AssertionError("no chain hydration")), mock.patch.object(
            self.store, "_graph", side_effect=AssertionError("no graph hydration")
        ), mock.patch.object(self.store, "_receipt", side_effect=AssertionError("no receipt read")), mock.patch.object(
            self.store, "_inventory", side_effect=AssertionError("no graph inventory")
        ), mock.patch.object(self.store.parts["bodies"], "read_bytes", side_effect=AssertionError("no body read")), mock.patch.object(
            self.store.parts["graphs"], "read_bytes", side_effect=AssertionError("no graph read")
        ), mock.patch.object(self.store.parts["projection"], "read_bytes", side_effect=AssertionError("no projection read")), mock.patch.object(
            self.store.directory, "read_bytes", side_effect=read
        ), mock.patch.object(self.store.directory, "write_bytes", side_effect=AssertionError("read only")):
            with self.checkpoint() as value:
                self.assertEqual(set(value), CHECKPOINT_FIELDS)
                self.assertEqual(value, {"config_sha256": self.store.config_sha256,
                                        **{key: manifest[key] for key in CHECKPOINT_FIELDS - {"config_sha256"}}})
        self.assertEqual(reads, ["store.json", "store.json", "manifest.json"])
        self.assertEqual(self.authoritative_files(), before)

    def test_checkpoint_rereads_new_committed_metadata_and_does_not_persist_caller_mutations(self):
        with self.checkpoint() as initial:
            self.assertEqual(initial["head"], self.initial_head)
        result = self.commit(request_fixture(self.initial_head))
        manifest = read_json(self.root / "manifest.json")
        with self.checkpoint() as value:
            self.assertEqual(value["head"], result["head"])
            self.assertEqual(value["generation"], 1)
            self.assertEqual(value["receipt_key"], manifest["receipt_key"])
            self.assertEqual(value["receipt_sha256"], manifest["receipt_sha256"])
            self.assertEqual(value["parent_head"], self.initial_head)
            value["head"] = "0" * 64
            value["unexpected"] = True
        with self.checkpoint() as value:
            self.assertEqual(value["head"], result["head"])
            self.assertEqual(set(value), CHECKPOINT_FIELDS)

    def test_authentication_scope_action_revocation_and_epoch_reject_before_store_metadata(self):
        write_only = self.store.issue(run_id=RUN_ID, actions=["memory.commit"])
        _, foreign = self.make_store("foreign")
        for token, run_id in ((write_only, RUN_ID), (foreign, RUN_ID), (self.token, "wrong-run")):
            with self.subTest(run_id=run_id), mock.patch.object(
                self.store.directory, "read_bytes", side_effect=AssertionError("unauthorized metadata read")
            ):
                with self.assertRaises(AuthorizationError):
                    with self.store.locked_checkpoint(token, run_id=run_id):
                        self.fail("unauthorized checkpoint body")
        token_id = self.store.authority.authenticate(self.token, action="memory.read", now=self.clock())["token_id"]
        self.store.authority.revoke(token_id)
        with mock.patch.object(self.store.directory, "read_bytes", side_effect=AssertionError("revoked metadata read")):
            with self.assertRaises(AuthorizationError):
                with self.checkpoint():
                    self.fail("revoked checkpoint body")
        replacement = self.store.issue(run_id=RUN_ID, actions=["memory.read"])
        self.store.authority.bump_epoch(RUN_ID)
        with mock.patch.object(self.store.directory, "read_bytes", side_effect=AssertionError("old epoch metadata read")):
            with self.assertRaises(AuthorizationError):
                with self.checkpoint(replacement):
                    self.fail("old epoch checkpoint body")
        current = self.store.issue(run_id=RUN_ID, actions=["memory.read"])
        self.store.authority.close_run(RUN_ID)
        with mock.patch.object(self.store.directory, "read_bytes", side_effect=AssertionError("closed metadata read")):
            with self.assertRaises(AuthorizationError):
                with self.checkpoint(current):
                    self.fail("closed run checkpoint body")

    def test_config_changes_noncanonical_encoding_and_bad_versions_fail_closed_without_repair(self):
        path = self.root / "store.json"
        original = path.read_bytes()
        config = read_json(path)
        changed = deepcopy(config)
        changed["limits"]["entries"] -= 1
        extra = {**config, "unexpected": True}
        for data in (canonical_json(changed), canonical_json(extra), b" " + original,
                     canonical_json({**config, "operation_version": 1.0}), b'{"duplicate":1,"duplicate":2}', b'"\xff"'):
            with self.subTest(data=data[:40]):
                write_private(path, data)
                with self.assertRaises((OSError, ValueError)):
                    with self.checkpoint():
                        self.fail("damaged configuration reached body")
                self.assertEqual(path.read_bytes(), data)
                write_private(path, original)
        with self.checkpoint() as value:
            self.assertEqual(value["head"], self.initial_head)

    def test_manifest_closed_fields_hash_identity_canonical_encoding_and_strict_generation(self):
        path = self.root / "manifest.json"
        original = path.read_bytes()
        manifest = read_json(path)
        variants = []
        for changes in ({"unexpected": True}, {"generation": True}, {"generation": 0.0},
                        {"generation": 1.0}, {"generation": -1}, {"generation": 257},
                        {"store_id": "1" * 64}, {"graph_sha256": "2" * 64},
                        {"generation": 1}, {"receipt_key": "3" * 64}):
            fields = {key: value for key, value in {**manifest, **changes}.items() if key != "head"}
            variants.append(canonical_json({**fields, "head": digest(fields)}))
        variants += [canonical_json({**manifest, "head": "0" * 64}), original + b"\n",
                     b'{"generation":0,"generation":0}', b"NaN", b'"\xff"', b"{}"]
        for data in variants:
            with self.subTest(data=data[:80]):
                write_private(path, data)
                with self.assertRaises(MemoryCorruption):
                    with self.checkpoint():
                        self.fail("invalid checkpoint manifest reached body")
                self.assertEqual(path.read_bytes(), data)
                write_private(path, original)
        self.assertEqual(self.head(), self.initial_head)

    def test_unsafe_missing_or_oversized_metadata_is_not_repaired(self):
        for filename in ("store.json", "manifest.json"):
            path = self.root / filename
            original = path.read_bytes()
            outside = self.base / (filename + ".outside")
            write_private(outside, original)
            with self.subTest(filename=filename, kind="symlink"):
                path.unlink()
                path.symlink_to(outside)
                with self.assertRaises((OSError, ValueError)):
                    with self.checkpoint():
                        self.fail("symlink reached checkpoint body")
                self.assertTrue(path.is_symlink())
                path.unlink()
                write_private(path, original)
            with self.subTest(filename=filename, kind="hardlink"):
                alias = self.base / (filename + ".hardlink")
                os.link(path, alias)
                with self.assertRaises((OSError, ValueError)):
                    with self.checkpoint():
                        self.fail("hardlink reached checkpoint body")
                self.assertEqual(path.stat().st_nlink, 2)
                alias.unlink()
            with self.subTest(filename=filename, kind="mode"):
                path.chmod(0o644)
                with self.assertRaises((OSError, ValueError)):
                    with self.checkpoint():
                        self.fail("unsafe mode reached checkpoint body")
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
                path.chmod(0o600)
            with self.subTest(filename=filename, kind="missing"):
                path.unlink()
                with self.assertRaises((OSError, ValueError)):
                    with self.checkpoint():
                        self.fail("missing metadata reached checkpoint body")
                self.assertFalse(path.exists())
                write_private(path, original)
            with self.subTest(filename=filename, kind="oversized"):
                write_private(path, b" " * 16385)
                with self.assertRaises((OSError, ValueError)):
                    with self.checkpoint():
                        self.fail("oversized metadata reached checkpoint body")
                self.assertEqual(path.stat().st_size, 16385)
                write_private(path, original)
            self.assertEqual(outside.read_bytes(), original)
        with self.checkpoint() as value:
            self.assertEqual(value["head"], self.initial_head)

    def test_checkpoint_fsyncs_config_and_manifest_with_parent_directory_barriers(self):
        events = []
        original_fsync = os.fsync
        names = {(self.root / name).stat().st_ino: name for name in ("store.json", "manifest.json")}
        root_inode = self.root.stat().st_ino

        def fsync(fd):
            info = os.fstat(fd)
            events.append("directory" if info.st_ino == root_inode else names.get(info.st_ino, "unexpected"))
            original_fsync(fd)

        with mock.patch("rethlas_core.private_fs.os.fsync", side_effect=fsync):
            with self.checkpoint():
                self.assertEqual(events, ["store.json", "directory", "manifest.json", "directory"])
        self.assertEqual(events, ["store.json", "directory", "manifest.json", "directory"])

    def test_either_metadata_sync_failure_is_indeterminate_and_never_reaches_body(self):
        original_sync = self.store.directory.sync_file
        before = self.authoritative_files()
        for failed in ("store.json", "manifest.json"):
            calls = []

            def sync(name):
                calls.append(name)
                if name == failed:
                    raise OSError("synthetic checkpoint barrier failure")
                original_sync(name)

            with self.subTest(failed=failed), mock.patch.object(self.store.directory, "sync_file", side_effect=sync):
                with self.assertRaises(MemoryIndeterminate):
                    with self.checkpoint():
                        self.fail("unconfirmed durability reached checkpoint body")
            self.assertEqual(calls, ["store.json"] if failed == "store.json" else ["store.json", "manifest.json"])
            self.assertEqual(self.authoritative_files(), before)

    def test_each_file_or_directory_fsync_failure_closes_descriptors_and_rejects_body(self):
        original_fsync = os.fsync
        original_open = os.open
        before = self.authoritative_files()
        for failed_call in range(1, 5):
            calls = []
            opened = []

            def open_file(name, flags, *args, **kwargs):
                fd = original_open(name, flags, *args, **kwargs)
                if name in ("store.json", "manifest.json"):
                    opened.append(fd)
                return fd

            def fsync(fd):
                calls.append("file" if stat.S_ISREG(os.fstat(fd).st_mode) else "directory")
                if len(calls) == failed_call:
                    raise OSError("synthetic partial checkpoint fsync failure")
                original_fsync(fd)

            with self.subTest(failed_call=failed_call), mock.patch(
                "rethlas_core.private_fs.os.open", side_effect=open_file
            ), mock.patch("rethlas_core.private_fs.os.fsync", side_effect=fsync):
                with self.assertRaises(MemoryIndeterminate):
                    with self.checkpoint():
                        self.fail("partial fsync reached checkpoint body")
            self.assertEqual(calls, ["file", "directory", "file", "directory"][:failed_call])
            for fd in set(opened):
                with self.assertRaises(OSError):
                    os.fstat(fd)
            self.assertEqual(self.authoritative_files(), before)

    def test_guard_is_rechecked_after_metadata_barriers_before_yield(self):
        cancel = threading.Event()
        original_sync = self.store.directory.sync_file
        original_time = self.clock.value
        for changed in ("cancel", "expiry"):
            def sync(name):
                original_sync(name)
                if name == "manifest.json":
                    if changed == "cancel":
                        cancel.set()
                    else:
                        self.clock.value += 301

            expected = MemoryCancelled if changed == "cancel" else AuthorizationError
            with self.subTest(changed=changed), mock.patch.object(self.store.directory, "sync_file", side_effect=sync):
                with self.assertRaises(expected):
                    with self.checkpoint(cancel_event=cancel):
                        self.fail("changed guard reached checkpoint body")
            cancel.clear()
            self.clock.value = original_time

    def test_expiry_and_cancellation_before_and_after_body_are_not_hidden(self):
        original_time = self.clock.value
        self.clock.value += 301
        with self.assertRaises(AuthorizationError):
            with self.checkpoint():
                self.fail("expired checkpoint body")
        self.clock.value = original_time
        with self.assertRaises(AuthorizationError):
            with self.checkpoint():
                self.clock.value += 301
        self.clock.value = original_time
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(MemoryCancelled):
            with self.checkpoint(cancel_event=cancel):
                self.fail("cancelled checkpoint body")
        cancel.clear()
        with self.assertRaises(MemoryCancelled):
            with self.checkpoint(cancel_event=cancel):
                cancel.set()
        cancel.clear()
        with self.checkpoint(cancel_event=cancel) as value:
            self.assertEqual(value["head"], self.initial_head)

    def test_caller_body_exception_propagates_intact_and_locks_release(self):
        for failure in (ValueError("trusted caller failed"), OSError("trusted caller durability failed")):
            with self.subTest(failure=type(failure).__name__):
                with self.assertRaises(type(failure)) as raised:
                    with self.checkpoint():
                        raise failure
                self.assertIs(raised.exception, failure)
        with self.checkpoint() as value:
            self.assertEqual(value["head"], self.initial_head)

    def test_real_commit_waits_until_checkpoint_control_operation_has_finished(self):
        request = request_fixture(self.initial_head)
        process, connection = self.spawn_worker(_checkpoint_commit_worker, self.root)
        connection.send({"token": self.token, "request": request})
        self.assertEqual(self.receive(connection), ("ready", None))
        control = PrivateDirectory(self.base / "control", create=True)
        with self.checkpoint() as value:
            connection.send("commit")
            self.assertEqual(self.receive(connection), ("attempting", None))
            self.assertFalse(connection.poll(0.2), "commit passed checkpoint locks")
            control.write_json("observed.json", value)
            self.assertEqual(self.head(), self.initial_head)
        kind, head = self.receive(connection)
        self.assertEqual(kind, "committed")
        self.join_cleanly(process)
        self.assertNotEqual(head, self.initial_head)
        self.assertEqual(self.head(), head)
        self.assertEqual(control.read_json("observed.json")["head"], self.initial_head)

    def test_checkpoint_also_holds_store_flock_against_independent_process(self):
        process, connection = self.spawn_worker(_checkpoint_store_lock_worker, self.root)
        self.assertEqual(self.receive(connection), ("ready", None))
        with self.checkpoint():
            connection.send("lock")
            self.assertEqual(self.receive(connection), ("attempting", None))
            self.assertFalse(connection.poll(0.2), "independent store flock passed checkpoint")
        self.assertEqual(self.receive(connection), ("acquired", None))
        self.join_cleanly(process)

    def test_checkpoint_holds_authority_against_real_revocation(self):
        token_id = self.store.authority.authenticate(self.token, action="memory.read", now=self.clock())["token_id"]
        process, connection = self.spawn_worker(_checkpoint_revoke_worker, self.root)
        connection.send(token_id)
        self.assertEqual(self.receive(connection), ("ready", None))
        with self.checkpoint():
            connection.send("revoke")
            self.assertEqual(self.receive(connection), ("attempting", None))
            self.assertFalse(connection.poll(0.2), "revocation passed checkpoint authority lock")
        self.assertEqual(self.receive(connection), ("revoked", None))
        self.join_cleanly(process)
        with self.assertRaises(AuthorizationError):
            with self.checkpoint():
                self.fail("revoked checkpoint body")
        self.assertEqual(self.head(), self.initial_head)


if __name__ == "__main__":
    unittest.main()
