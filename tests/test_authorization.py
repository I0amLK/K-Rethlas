from __future__ import annotations

import base64
import errno
import fcntl
import hashlib
import hmac
import math
import multiprocessing
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rethlas_core.authorization import Authority, AuthorizationError, ROLE_ACTIONS
from rethlas_core.private_fs import (
    PrivateDirectory,
    PrivateFilesystemError,
    canonical_json,
    digest,
    strict_json_loads,
)


def _encoded(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _revoke_in_process(root: str, token_id: str, started, finished) -> None:
    started.set()
    Authority(Path(root)).revoke(token_id)
    finished.set()


def _lock_in_process(root: str, started, finished) -> None:
    started.set()
    with PrivateDirectory(Path(root)).lock():
        finished.set()


class PrivateFilesystemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "private"
        self.directory = PrivateDirectory(self.root, create=True)

    def test_strict_json_and_canonical_digest(self) -> None:
        self.assertEqual(canonical_json({"b": 2, "a": [True, None]}), b'{"a":[true,null],"b":2}')
        self.assertEqual(strict_json_loads('{"a":1}'), {"a": 1})
        self.assertEqual(digest({"b": 2, "a": 1}), digest({"a": 1, "b": 2}))
        for value in ('{"a":1,"a":2}', '{"a":{"b":1,"b":2}}', "NaN", "Infinity", "-Infinity", "1e999", "{} {}", "﻿{}", '"\\ud800"'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                strict_json_loads(value)
        with self.assertRaises(UnicodeError):
            strict_json_loads(b'"\xff"')
        for value in (math.nan, math.inf, {1: "coerced"}, (1, 2)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                canonical_json(value)

    def test_private_modes_atomic_json_and_exclusive_creation(self) -> None:
        self.directory.write_json("state.json", {"revision": 1})
        self.assertEqual(self.directory.read_json("state.json"), {"revision": 1})
        self.directory.write_json("state.json", {"revision": 2})
        self.assertEqual(self.directory.read_json("state.json"), {"revision": 2})
        self.directory.write_bytes("new", b"first", exclusive=True)
        with self.assertRaises(FileExistsError):
            self.directory.write_bytes("new", b"second", exclusive=True)
        self.assertEqual(self.directory.read_bytes("new"), b"first")
        self.assertEqual(stat.S_IMODE(self.root.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE((self.root / "state.json").stat().st_mode), 0o600)
        self.assertFalse(any(path.name.startswith(".tmp-") for path in self.root.iterdir()))
        with mock.patch("rethlas_core.private_fs.os.replace", side_effect=OSError("injected commit failure")):
            with self.assertRaises(OSError):
                self.directory.write_json("state.json", {"revision": 3})
        self.assertEqual(self.directory.read_json("state.json"), {"revision": 2})
        self.assertFalse(any(path.name.startswith(".tmp-") for path in self.root.iterdir()))

    def test_bounded_prefix_reads_never_treat_a_large_file_as_a_complete_object(self) -> None:
        self.directory.write_bytes("tail.log", b"INTENT\n" + b"x" * 65536)
        self.assertEqual(self.directory.read_prefix("tail.log", max_bytes=7), b"INTENT\n")
        self.assertEqual(self.directory.read_prefix("tail.log", max_bytes=0), b"")
        with self.assertRaises(PrivateFilesystemError):
            self.directory.read_bytes("tail.log", max_bytes=7)
        for value in (True, -1, 1.0):
            with self.subTest(value=value), self.assertRaises(PrivateFilesystemError):
                self.directory.read_prefix("tail.log", max_bytes=value)
        (self.root / "linked.log").symlink_to(self.root / "tail.log")
        with self.assertRaises(PrivateFilesystemError):
            self.directory.read_prefix("linked.log", max_bytes=7)
        os.link(self.root / "tail.log", self.root / "hard.log")
        with self.assertRaises(PrivateFilesystemError):
            self.directory.read_prefix("tail.log", max_bytes=7)

    def test_bounded_reads_preserve_defaults_and_reject_invalid_limits(self) -> None:
        self.directory.write_bytes("data", b"12345")
        self.directory.write_bytes("empty", b"")
        self.directory.write_json("data.json", {"n": 1})
        self.assertEqual(self.directory.read_bytes("data"), b"12345")
        self.assertEqual(self.directory.read_bytes("data", max_bytes=5), b"12345")
        self.assertEqual(self.directory.read_bytes("empty", max_bytes=0), b"")
        self.assertEqual(self.directory.read_json("data.json", max_bytes=7), {"n": 1})
        with mock.patch("rethlas_core.private_fs.os.read") as read:
            for name, limit in (("data", 4), ("data", 0), ("data.json", 6)):
                with self.subTest(name=name, limit=limit), self.assertRaises(PrivateFilesystemError):
                    self.directory.read_bytes(name, max_bytes=limit)
            read.assert_not_called()  # The initial fstat rejects large files.
        with self.assertRaises(PrivateFilesystemError):
            self.directory.read_json("data.json", max_bytes=6)
        for limit in (-1, True, 1.0, "5", [], {}):
            with self.subTest(limit=limit), self.assertRaises(PrivateFilesystemError):
                self.directory.read_bytes("data", max_bytes=limit)

    def test_bounded_reads_detect_growth_after_initial_fstat(self) -> None:
        self.directory.write_bytes("data", b"abc")
        original_read = os.read
        reads = []

        def growing_read(fd, count):
            reads.append(count)
            chunk = original_read(fd, count)
            if len(reads) == 1:
                with (self.root / "data").open("ab") as stream:
                    stream.write(b"de")
            return chunk

        with mock.patch("rethlas_core.private_fs.os.read", side_effect=growing_read):
            with self.assertRaisesRegex(PrivateFilesystemError, "max_bytes"):
                self.directory.read_bytes("data", max_bytes=4)
        self.assertEqual(reads, [5, 2])

    def test_sync_file_fsyncs_file_then_directory_without_content_access_or_replacement(self) -> None:
        self.directory.write_bytes("data", b"existing bytes")
        before = (self.root / "data").stat()
        events = []
        original_open, original_fsync = os.open, os.fsync
        opened_file_flags = []

        def opened(path, flags, *args, **kwargs):
            if path == "data":
                opened_file_flags.append(flags)
            return original_open(path, flags, *args, **kwargs)

        def fsync(fd):
            events.append("file" if stat.S_ISREG(os.fstat(fd).st_mode) else "directory")
            original_fsync(fd)

        with self.directory.lock("store.lock"), mock.patch.object(
            PrivateDirectory, "lock", side_effect=AssertionError("must not reacquire lock")
        ), mock.patch("rethlas_core.private_fs.os.open", side_effect=opened), mock.patch(
            "rethlas_core.private_fs.os.fsync", side_effect=fsync
        ), mock.patch("rethlas_core.private_fs.os.read", side_effect=AssertionError("must not read")), mock.patch(
            "rethlas_core.private_fs.os.write", side_effect=AssertionError("must not write")
        ), mock.patch("rethlas_core.private_fs.os.replace", side_effect=AssertionError("must not replace")):
            self.directory.sync_file("data")
        self.assertEqual(events, ["file", "directory"])
        self.assertEqual(len(opened_file_flags), 1)
        flags = opened_file_flags[0]
        self.assertEqual(flags & os.O_ACCMODE, os.O_RDONLY)
        self.assertEqual(flags & (os.O_NOFOLLOW | os.O_NONBLOCK), os.O_NOFOLLOW | os.O_NONBLOCK)
        self.assertEqual(flags & (os.O_CREAT | os.O_TRUNC), 0)
        after = (self.root / "data").stat()
        self.assertEqual((after.st_dev, after.st_ino, after.st_mode, after.st_nlink), (before.st_dev, before.st_ino, before.st_mode, before.st_nlink))
        self.assertEqual(self.directory.read_bytes("data"), b"existing bytes")

    def test_sync_file_rejects_unsafe_files_names_and_replaced_root(self) -> None:
        self.directory.write_bytes("data", b"unchanged")
        outside = self.base / "outside"
        outside.write_bytes(b"outside untouched")
        outside.chmod(0o600)
        (self.root / "linked").symlink_to(outside)
        os.mkfifo(self.root / "fifo", mode=0o600)
        (self.root / "child").mkdir(mode=0o700)
        self.directory.write_bytes("mode", b"unsafe permissions")
        (self.root / "mode").chmod(0o644)
        for name in ("linked", "fifo", "child", "mode", "", ".", "..", "../escape", "a/b", "a\\b", "a\0b"):
            with self.subTest(name=name), mock.patch("rethlas_core.private_fs.os.fsync") as fsync:
                with self.assertRaises(PrivateFilesystemError):
                    self.directory.sync_file(name)
                fsync.assert_not_called()
        os.link(self.root / "data", self.root / "hardlink")
        with mock.patch("rethlas_core.private_fs.os.fsync") as fsync:
            with self.assertRaises(PrivateFilesystemError):
                self.directory.sync_file("hardlink")
            fsync.assert_not_called()
        (self.root / "hardlink").unlink()
        original_fstat = os.fstat
        file_inode = (self.root / "data").stat().st_ino

        def wrong_owner(fd):
            info = original_fstat(fd)
            if info.st_ino == file_inode:
                values = list(info)
                values[4] = os.getuid() + 1
                return os.stat_result(values)
            return info

        with mock.patch("rethlas_core.private_fs.os.fstat", side_effect=wrong_owner), mock.patch(
            "rethlas_core.private_fs.os.fsync"
        ) as fsync:
            with self.assertRaises(PrivateFilesystemError):
                self.directory.sync_file("data")
            fsync.assert_not_called()
        with mock.patch("rethlas_core.private_fs.os.fsync") as fsync:
            with self.assertRaises(FileNotFoundError):
                self.directory.sync_file("missing")
            fsync.assert_not_called()
        self.assertFalse((self.root / "missing").exists())
        self.assertEqual(outside.read_bytes(), b"outside untouched")
        self.assertEqual(self.directory.read_bytes("data"), b"unchanged")
        self.root.rename(self.base / "old")
        self.root.mkdir(mode=0o700)
        with mock.patch("rethlas_core.private_fs.os.fsync") as fsync:
            with self.assertRaises(PrivateFilesystemError):
                self.directory.sync_file("data")
            fsync.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_sync_file_barrier_failures_propagate_and_close_file_descriptor(self) -> None:
        self.directory.write_bytes("data", b"unchanged")
        for failed_barrier in ("file", "directory"):
            with self.subTest(failed_barrier=failed_barrier):
                events = []
                opened_file = []
                failure = OSError("injected sync barrier failure")
                original_open = os.open

                def opened(path, flags, *args, **kwargs):
                    fd = original_open(path, flags, *args, **kwargs)
                    if path == "data":
                        opened_file.append(fd)
                    return fd

                def fsync(fd):
                    kind = "file" if stat.S_ISREG(os.fstat(fd).st_mode) else "directory"
                    events.append(kind)
                    if kind == failed_barrier:
                        raise failure

                with mock.patch("rethlas_core.private_fs.os.open", side_effect=opened), mock.patch(
                    "rethlas_core.private_fs.os.fsync", side_effect=fsync
                ):
                    with self.assertRaises(OSError) as raised:
                        self.directory.sync_file("data")
                self.assertIs(raised.exception, failure)
                self.assertEqual(events, ["file"] if failed_barrier == "file" else ["file", "directory"])
                self.assertEqual(len(opened_file), 1)
                with self.assertRaises(OSError):
                    os.fstat(opened_file[0])
                self.assertEqual(self.directory.read_bytes("data"), b"unchanged")

    def test_before_replace_runs_after_fsync_and_recheck_immediately_before_replace(self) -> None:
        self.directory.write_json("state.json", {"revision": 1})
        events = []
        original_fsync, original_stat, original_replace = os.fsync, os.stat, os.replace

        def fsync(fd):
            events.append("file-fsync" if stat.S_ISREG(os.fstat(fd).st_mode) else "dir-fsync")
            return original_fsync(fd)

        def inspect(name, **kwargs):
            result = original_stat(name, **kwargs)
            if name == "state.json":
                events.append("target-check")
            return result

        def guard():
            self.assertEqual(events[-2:], ["file-fsync", "target-check"])
            self.assertEqual(self.directory.read_json("state.json"), {"revision": 1})
            temporaries = list(self.root.glob(".tmp-*"))
            self.assertEqual(len(temporaries), 1)
            self.assertEqual(temporaries[0].read_bytes(), b'{"revision":2}')
            events.append("guard")

        def replace(*args, **kwargs):
            self.assertEqual(events[-1], "guard")
            events.append("replace")
            return original_replace(*args, **kwargs)

        with mock.patch("rethlas_core.private_fs.os.fsync", side_effect=fsync), mock.patch(
            "rethlas_core.private_fs.os.stat", side_effect=inspect
        ), mock.patch("rethlas_core.private_fs.os.replace", side_effect=replace):
            self.directory.write_json("state.json", {"revision": 2}, before_replace=guard)
        self.assertEqual(events[-3:], ["guard", "replace", "dir-fsync"])
        self.assertEqual(self.directory.read_json("state.json"), {"revision": 2})

    def test_before_replace_failure_cleans_temporary_and_never_commits(self) -> None:
        self.directory.write_bytes("data", b"old")
        guard = mock.Mock(side_effect=RuntimeError("guard denied"))
        with mock.patch("rethlas_core.private_fs.os.replace") as replace:
            with self.assertRaisesRegex(RuntimeError, "guard denied"):
                self.directory.write_bytes("data", b"new", before_replace=guard)
            replace.assert_not_called()
        guard.assert_called_once_with()
        self.assertEqual(self.directory.read_bytes("data"), b"old")
        self.assertFalse(list(self.root.glob(".tmp-*")))
        for name, exclusive, callback in (("absent", True, guard), ("data", True, guard), ("data", False, True)):
            with self.subTest(name=name, exclusive=exclusive), self.assertRaises(PrivateFilesystemError):
                self.directory.write_bytes(name, b"new", exclusive=exclusive, before_replace=callback)
        self.assertFalse((self.root / "absent").exists())
        self.assertEqual(guard.call_count, 1)
        with mock.patch("rethlas_core.private_fs.os.fsync", side_effect=OSError("fsync failed")):
            with self.assertRaises(OSError):
                self.directory.write_bytes("data", b"new", before_replace=guard)
        self.assertEqual(guard.call_count, 1)
        original_fsync = os.fsync

        def unsafe_after_fsync(fd):
            original_fsync(fd)
            (self.root / "data").chmod(0o644)

        with mock.patch("rethlas_core.private_fs.os.fsync", side_effect=unsafe_after_fsync):
            with self.assertRaises(PrivateFilesystemError):
                self.directory.write_bytes("data", b"new", before_replace=guard)
        self.assertEqual(guard.call_count, 1)
        self.assertEqual((self.root / "data").read_bytes(), b"old")
        self.assertFalse(list(self.root.glob(".tmp-*")))

    def test_entries_are_bounded_private_nonrecursive_and_count_temporaries(self) -> None:
        self.assertEqual(self.directory.entries(max_entries=1), [])
        self.directory.write_bytes("data", b"abc")
        self.directory.write_bytes(".tmp-interrupted", b"12345")
        child = self.directory.child("child", create=True)
        (child.root / "unsafe-below-boundary").symlink_to(self.base / "outside")
        self.assertEqual(set(self.directory.entries(max_entries=4)), {
            ("data", False, 3), (".tmp-interrupted", False, 5), ("child", True, 0),
        })
        for limit in (1, 2, 3):
            with self.subTest(limit=limit), self.assertRaises(PrivateFilesystemError):
                self.directory.entries(max_entries=limit)
        for limit in (0, -1, True, None, 4.0, "4"):
            with self.subTest(limit=limit), self.assertRaises(PrivateFilesystemError):
                self.directory.entries(max_entries=limit)

    def test_entries_reject_unsafe_children_without_following_them(self) -> None:
        outside = self.base / "outside"
        outside.write_bytes(b"unchanged")
        outside.chmod(0o600)
        child = self.root / "unsafe"
        for kind in ("symlink", "directory-link", "hardlink", "fifo", "mode", "directory-mode", "temporary-fifo", "owner", "directory-owner"):
            with self.subTest(kind=kind):
                if kind in ("symlink", "directory-link"):
                    child.symlink_to(self.root if kind == "directory-link" else outside)
                elif kind == "hardlink":
                    os.link(outside, child)
                elif kind in ("fifo", "temporary-fifo"):
                    child = self.root / (".tmp-unsafe" if kind == "temporary-fifo" else "unsafe")
                    os.mkfifo(child, mode=0o600)
                elif kind.startswith("directory-"):
                    child.mkdir(mode=0o700)
                    if kind == "directory-mode":
                        child.chmod(0o755)
                else:
                    self.directory.write_bytes(child.name, b"x")
                    if kind == "mode":
                        child.chmod(0o644)
                try:
                    if kind in ("owner", "directory-owner"):
                        original = os.stat

                        def wrong_owner(name, **kwargs):
                            info = original(name, **kwargs)
                            values = list(info)
                            values[4] = os.getuid() + 1
                            return os.stat_result(values)

                        with mock.patch("rethlas_core.private_fs.os.stat", side_effect=wrong_owner):
                            with self.assertRaises(PrivateFilesystemError):
                                self.directory.entries(max_entries=10)
                    else:
                        with self.assertRaises(PrivateFilesystemError):
                            self.directory.entries(max_entries=10)
                finally:
                    if child.is_dir() and not child.is_symlink():
                        child.rmdir()
                    else:
                        child.unlink()
                    child = self.root / "unsafe"
        self.assertEqual(outside.read_bytes(), b"unchanged")

    def test_quarantine_moves_same_inode_and_fsyncs_both_directories_without_read_or_lock(self) -> None:
        destination = self.directory.child("quarantine", create=True)
        self.directory.write_bytes("data", b"opaque payload")
        before = (self.root / "data").stat()
        with self.directory.lock("store.lock"), mock.patch(
            "rethlas_core.private_fs.os.read", side_effect=AssertionError("must not read")
        ), mock.patch.object(PrivateDirectory, "lock", side_effect=AssertionError("must not acquire lock")), mock.patch(
            "rethlas_core.private_fs.os.fsync", wraps=os.fsync
        ) as fsync:
            self.directory.quarantine_to("data", destination, "held")
        self.assertEqual(fsync.call_count, 2)
        self.assertFalse((self.root / "data").exists())
        after = (destination.root / "held").stat()
        self.assertEqual((after.st_dev, after.st_ino), (before.st_dev, before.st_ino))
        self.assertEqual(after.st_nlink, 1)
        self.assertEqual(stat.S_IMODE(after.st_mode), 0o600)
        self.assertEqual(destination.read_bytes("held"), b"opaque payload")

    def test_quarantine_never_overwrites_including_destination_creation_race(self) -> None:
        destination = self.directory.child("quarantine", create=True)
        self.directory.write_bytes("data", b"source")
        destination.write_bytes("existing", b"destination")
        with self.assertRaises(FileExistsError):
            self.directory.quarantine_to("data", destination, "existing")
        self.assertEqual(destination.read_bytes("existing"), b"destination")
        original_stat = os.stat
        raced = False

        def racing_stat(name, **kwargs):
            nonlocal raced
            try:
                return original_stat(name, **kwargs)
            except FileNotFoundError:
                if name == "raced" and not raced:
                    raced = True
                    destination.write_bytes("raced", b"race winner", exclusive=True)
                raise

        # Create a regular target after the explicit lstat. renameat2, not
        # a precheck, must enforce no-overwrite against this creation race.
        with mock.patch("rethlas_core.private_fs.os.stat", side_effect=racing_stat):
            with self.assertRaises(FileExistsError):
                self.directory.quarantine_to("data", destination, "raced")
        self.assertEqual(destination.read_bytes("raced"), b"race winner")
        self.assertEqual(self.directory.read_bytes("data"), b"source")

    def test_quarantine_rejects_unsafe_files_roots_names_and_remembered_locks(self) -> None:
        destination = self.directory.child("quarantine", create=True)
        self.directory.write_bytes("data", b"source")
        (destination.root / "link").symlink_to(self.root / "data")
        with self.assertRaises(PrivateFilesystemError):
            self.directory.quarantine_to("data", destination, "link")
        source = self.root / "unsafe"
        for kind in ("symlink", "fifo", "directory", "mode", "hardlink", "owner"):
            with self.subTest(kind=kind):
                if kind == "symlink":
                    source.symlink_to(self.root / "data")
                elif kind == "fifo":
                    os.mkfifo(source, mode=0o600)
                elif kind == "directory":
                    source.mkdir(mode=0o700)
                elif kind == "hardlink":
                    os.link(self.root / "data", source)
                else:
                    self.directory.write_bytes("unsafe", b"x")
                    source.chmod(0o644 if kind == "mode" else 0o600)
                try:
                    if kind == "owner":
                        original = os.stat

                        def wrong_owner(name, **kwargs):
                            info = original(name, **kwargs)
                            values = list(info)
                            values[4] = os.getuid() + 1
                            return os.stat_result(values)

                        with mock.patch("rethlas_core.private_fs.os.stat", side_effect=wrong_owner):
                            with self.assertRaises(PrivateFilesystemError):
                                self.directory.quarantine_to("unsafe", destination, "held")
                    else:
                        with self.assertRaises(PrivateFilesystemError):
                            self.directory.quarantine_to("unsafe", destination, "held")
                finally:
                    if source.is_dir() and not source.is_symlink():
                        source.rmdir()
                    else:
                        source.unlink()
        for name in ("", ".", "..", "../escape", "a/b", "a\\b", "a\0b"):
            for args in ((name, destination, "held"), ("data", destination, name)):
                with self.subTest(args=args), self.assertRaises(PrivateFilesystemError):
                    self.directory.quarantine_to(*args)
        with self.assertRaises(PrivateFilesystemError):
            self.directory.quarantine_to("data", self.root, "held")
        with self.directory.lock("source.lock"):
            with self.assertRaises(PrivateFilesystemError):
                self.directory.quarantine_to("source.lock", destination, "held")
        with destination.lock("destination.lock"):
            with self.assertRaises(PrivateFilesystemError):
                self.directory.quarantine_to("data", destination, "destination.lock")
        destination.root.rename(self.root / "old-quarantine")
        destination.root.mkdir(mode=0o700)
        with self.assertRaises(PrivateFilesystemError):
            self.directory.quarantine_to("data", destination, "held")
        self.assertEqual(self.directory.read_bytes("data"), b"source")
        self.assertFalse((destination.root / "held").exists())

    def test_quarantine_cross_filesystem_or_unsupported_rename_fails_without_fallback(self) -> None:
        destination = self.directory.child("quarantine", create=True)
        self.directory.write_bytes("data", b"source")
        original_fstat = os.fstat
        destination_inode = destination.root.stat().st_ino
        destination_checks = 0

        def other_device(fd):
            nonlocal destination_checks
            info = original_fstat(fd)
            if info.st_ino == destination_inode:
                destination_checks += 1
                # Preserve _walk/_fd identity validation, change only the
                # subsequent source/destination filesystem comparison.
                if destination_checks > 2:
                    values = list(info)
                    values[2] = info.st_dev + 1
                    return os.stat_result(values)
            return info

        with mock.patch("rethlas_core.private_fs.os.fstat", side_effect=other_device):
            with self.assertRaisesRegex(PrivateFilesystemError, "same filesystem"):
                self.directory.quarantine_to("data", destination, "held")
        with mock.patch("rethlas_core.private_fs.ctypes.CDLL", return_value=object()), mock.patch(
            "rethlas_core.private_fs.os.rename"
        ) as rename, mock.patch("rethlas_core.private_fs.os.link") as link:
            with self.assertRaisesRegex(PrivateFilesystemError, "no-replace rename is unavailable"):
                self.directory.quarantine_to("data", destination, "held")
            rename.assert_not_called()
            link.assert_not_called()
        self.assertEqual(self.directory.read_bytes("data"), b"source")
        self.assertFalse((destination.root / "held").exists())

    def test_child_and_all_names_are_single_components(self) -> None:
        child = self.directory.child("run-1", create=True)
        child.write_bytes("evidence", b"data")
        self.assertEqual(child.read_bytes("evidence"), b"data")
        for name in ("", ".", "..", "../escape", "/absolute", "a/b", "a\\b", "a\0b"):
            with self.subTest(name=name):
                for operation in (
                    lambda: self.directory.read_bytes(name),
                    lambda: self.directory.write_bytes(name, b"x"),
                    lambda: self.directory.child(name, create=True),
                    lambda: self.directory.lock(name).__enter__(),
                ):
                    with self.assertRaises(PrivateFilesystemError):
                        operation()
        self.assertFalse((self.base / "escape").exists())

    def test_symlink_root_ancestor_child_and_file_rejected(self) -> None:
        alias = self.base / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(PrivateFilesystemError):
            PrivateDirectory(alias)
        (self.root / "nested").mkdir(mode=0o700)
        with self.assertRaises(PrivateFilesystemError):
            PrivateDirectory(alias / "nested")
        with self.assertRaises(PrivateFilesystemError):
            PrivateDirectory(alias / "new", create=True)
        self.assertFalse((self.root / "new").exists())
        (self.root / "child-link").symlink_to(self.root / "nested", target_is_directory=True)
        with self.assertRaises(PrivateFilesystemError):
            self.directory.child("child-link", create=True)
        outside = self.base / "outside"
        outside.write_bytes(b"untouched")
        outside.chmod(0o600)
        (self.root / "linked").symlink_to(outside)
        for operation in (
            lambda: self.directory.read_bytes("linked"),
            lambda: self.directory.write_bytes("linked", b"bad"),
            lambda: self.directory.lock("linked").__enter__(),
        ):
            with self.assertRaises(PrivateFilesystemError):
                operation()
        self.assertEqual(outside.read_bytes(), b"untouched")

    def test_unsafe_permissions_owner_hardlinks_and_special_files_rejected(self) -> None:
        unsafe = self.base / "unsafe"
        unsafe.mkdir(mode=0o755)
        unsafe.chmod(0o755)
        for create in (False, True):
            with self.assertRaises(PrivateFilesystemError):
                PrivateDirectory(unsafe, create=create)
        self.assertEqual(stat.S_IMODE(unsafe.stat().st_mode), 0o755)
        self.directory.write_bytes("data", b"safe")
        (self.root / "data").chmod(0o644)
        with self.assertRaises(PrivateFilesystemError):
            self.directory.read_bytes("data")
        with self.assertRaises(PrivateFilesystemError):
            self.directory.write_bytes("data", b"unsafe")
        (self.root / "data").chmod(0o600)
        os.link(self.root / "data", self.root / "hardlink")
        with self.assertRaises(PrivateFilesystemError):
            self.directory.read_bytes("hardlink")
        os.mkfifo(self.root / "fifo", mode=0o600)
        with self.assertRaises(PrivateFilesystemError):
            self.directory.read_bytes("fifo")
        with mock.patch("rethlas_core.private_fs.os.getuid", return_value=os.getuid() + 1):
            with self.assertRaises(PrivateFilesystemError):
                PrivateDirectory(self.root)

    def test_parent_traversal_and_replaced_root_rejected(self) -> None:
        with self.assertRaises(PrivateFilesystemError):
            PrivateDirectory(self.root / ".." / "private")
        self.root.rename(self.base / "old")
        self.root.mkdir(mode=0o700)
        with self.assertRaises(PrivateFilesystemError):
            self.directory.write_bytes("data", b"wrong root")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_lock_keeps_inode_and_blocks_another_process(self) -> None:
        context = multiprocessing.get_context("spawn")
        started, finished = context.Event(), context.Event()
        with self.directory.lock():
            inode = (self.root / "lock").stat().st_ino
            process = context.Process(target=_lock_in_process, args=(str(self.root), started, finished))
            process.start()
            try:
                self.assertTrue(started.wait(10), "child did not start")
                self.assertFalse(finished.wait(0.2), "second process passed a held flock")
            except BaseException:
                process.terminate()
                process.join(10)
                raise
        try:
            self.assertTrue(finished.wait(10), "child did not obtain released flock")
            process.join(10)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.terminate()
                process.join(10)
        with self.directory.lock():
            self.assertEqual((self.root / "lock").stat().st_ino, inode)
        with self.assertRaises(PrivateFilesystemError):
            self.directory.write_bytes("lock", b"replacement")

    def test_lock_acquisition_has_bounded_deadline(self) -> None:
        with mock.patch("rethlas_core.private_fs.time.monotonic", side_effect=[10.0, 10.0, 15.0]), mock.patch(
            "rethlas_core.private_fs.time.sleep"
        ) as sleep, mock.patch(
            "rethlas_core.private_fs.fcntl.flock", side_effect=BlockingIOError(errno.EAGAIN, "held")
        ) as flock:
            with self.assertRaisesRegex(PrivateFilesystemError, "private state lock deadline exceeded"):
                with self.directory.lock():
                    self.fail("timed-out lock must not enter its protected body")
            self.assertEqual(flock.call_count, 2)
            self.assertTrue(all(call.args[1] == fcntl.LOCK_EX | fcntl.LOCK_NB for call in flock.call_args_list))
            sleep.assert_called_once_with(0.01)
        # Timeout closes the descriptor; the same inode remains usable later.
        inode = (self.root / "lock").stat().st_ino
        with self.directory.lock():
            self.assertEqual((self.root / "lock").stat().st_ino, inode)

    def test_lock_replacement_is_rejected_by_existing_instance(self) -> None:
        with self.directory.lock():
            pass
        (self.root / "lock").rename(self.root / "old-lock")
        self.directory.write_bytes("replacement", b"")
        (self.root / "replacement").rename(self.root / "lock")
        with self.assertRaises(PrivateFilesystemError):
            with self.directory.lock():
                self.fail("replaced lock must not be used")


class AuthorizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "authority"
        self.authority = Authority(self.root, create=True)
        self.authority.register_run("owner-1", "run-1", "fixture")
        self.directory = PrivateDirectory(self.root)

    def token(self, **changes) -> str:
        values = {
            "owner": "owner-1", "run_id": "run-1", "domain": "fixture",
            "role": "verification", "actions": ["finding.append", "status"], "now": 100,
        }
        values.update(changes)
        return self.authority.issue(**values)

    def check(self, token: str, **changes) -> dict:
        values = {"owner": "owner-1", "run_id": "run-1", "domain": "fixture", "action": "finding.append", "now": 101}
        values.update(changes)
        return self.authority.check(token, **values)

    def signed(self, payload: bytes) -> str:
        key = self.directory.read_bytes("key")
        return f"{_encoded(payload)}.{_encoded(hmac.new(key, payload, hashlib.sha256).digest())}"

    def test_persistent_authority_exact_claims_and_initialization(self) -> None:
        token = self.token()
        claims = self.check(token)
        self.assertEqual(set(claims), {"version", "token_id", "owner", "run_id", "domain", "role", "epoch", "actions", "issued_at", "expires_at"})
        self.assertEqual(claims["role"], "verification")
        self.assertEqual(claims["issued_at"], 100)
        self.assertEqual(claims["expires_at"], 400)
        self.assertEqual(claims["epoch"], 0)
        key, registry = self.directory.read_bytes("key"), self.directory.read_bytes("registry.json")
        again = Authority(self.root, create=True)
        self.assertEqual(again.authenticate(token, action="status", now=102), claims)
        self.assertEqual(self.directory.read_bytes("key"), key)
        self.assertEqual(self.directory.read_bytes("registry.json"), registry)
        claims["owner"] = "not-persistent"
        self.assertEqual(self.check(token)["owner"], "owner-1")
        with self.assertRaises(AuthorizationError):
            Authority(self.root.parent / "absent")
        incomplete = PrivateDirectory(self.root.parent / "incomplete", create=True)
        incomplete.write_bytes("key", b"x" * 32)
        with self.assertRaises(AuthorizationError):
            Authority(incomplete.root, create=True)
        self.assertEqual(incomplete.read_bytes("key"), b"x" * 32)
        self.assertFalse((incomplete.root / "registry.json").exists())

    def test_authority_read_limits_fail_closed_without_repair_or_rewrite(self) -> None:
        token = self.token()
        original_key = self.directory.read_bytes("key")
        original_registry = self.directory.read_bytes("registry.json")
        oversized_registry = original_registry + b" " * (1048577 - len(original_registry))
        for name, data in (("key", original_key + b"x"), ("registry.json", oversized_registry)):
            with self.subTest(name=name):
                self.directory.write_bytes(name, data)
                with self.assertRaises(AuthorizationError):
                    Authority(self.root, create=True)
                with self.assertRaises(AuthorizationError):
                    self.check(token)
                self.assertEqual(self.directory.read_bytes(name), data)
                self.directory.write_bytes("key", original_key)
                self.directory.write_bytes("registry.json", original_registry)
        # Exactly one MiB remains legal; JSON whitespace is not normalized on read.
        exact_registry = original_registry + b" " * (1048576 - len(original_registry))
        self.directory.write_bytes("registry.json", exact_registry)
        Authority(self.root).authenticate(token, action="status", now=101)
        self.assertEqual(self.directory.read_bytes("registry.json"), exact_registry)

    def test_issue_locked_guard_sees_proposed_registry_and_saves_once_without_relocking(self) -> None:
        events = []
        persisted = self.directory.read_bytes("registry.json")
        with self.authority._registry() as registry:
            registry["runs"]["run-2"] = {"owner": "owner-1", "domain": "fixture", "state": "open", "epoch": 0}
            original_tokens = set(registry["tokens"])
            original_save = self.authority._save

            def before_save(proposed):
                self.assertIs(proposed, registry)
                self.assertEqual(len(set(proposed["tokens"]) - original_tokens), 1)
                record = next(value for key, value in proposed["tokens"].items() if key not in original_tokens)
                self.assertEqual(record["claims"]["role"], "memory")
                self.assertEqual(record["claims"]["actions"], ["memory.read"])
                self.assertEqual(self.directory.read_bytes("registry.json"), persisted)
                events.append("guard")

            def save(proposed):
                self.assertEqual(events, ["guard"])
                events.append("save")
                original_save(proposed)

            with mock.patch.object(self.authority, "_registry", side_effect=AssertionError("must not reacquire lock")), mock.patch.object(
                self.authority, "_save", side_effect=save
            ) as save_mock:
                token = self.authority._issue_locked(
                    registry, owner="owner-1", run_id="run-2", domain="fixture", role="memory",
                    actions=["memory.read"], now=100, before_save=before_save,
                )
                save_mock.assert_called_once_with(registry)
        self.assertEqual(events, ["guard", "save"])
        self.authority.check(token, owner="owner-1", run_id="run-2", domain="fixture", action="memory.read", now=101)
        with self.assertRaises(TypeError):
            self.token(before_save=lambda registry: None)  # Not a public issue option.

    def test_issue_locked_guard_failure_or_invalid_arguments_do_not_persist(self) -> None:
        persisted = self.directory.read_bytes("registry.json")
        with self.authority._registry() as registry:
            registry["runs"]["run-2"] = {"owner": "owner-1", "domain": "fixture", "state": "open", "epoch": 0}
            guard = mock.Mock(side_effect=ValueError("capacity denied"))
            with mock.patch.object(self.authority, "_save") as save:
                with self.assertRaisesRegex(ValueError, "capacity denied"):
                    self.authority._issue_locked(
                        registry, owner="owner-1", run_id="run-2", domain="fixture", role="memory",
                        actions=["memory.read"], now=100, before_save=guard,
                    )
                guard.assert_called_once_with(registry)
                save.assert_not_called()
        self.assertEqual(self.directory.read_bytes("registry.json"), persisted)
        for changes in ({"role": "unknown"}, {"actions": ["publish"]}, {"ttl_seconds": True}, {"before_save": True}):
            with self.subTest(changes=changes), self.authority._registry() as registry:
                values = {"owner": "owner-1", "run_id": "run-1", "domain": "fixture", "role": "memory", "actions": ["memory.read"], "now": 100}
                values.update(changes)
                with mock.patch.object(self.authority, "_save") as save:
                    with self.assertRaises(AuthorizationError):
                        self.authority._issue_locked(registry, **values)
                    self.assertEqual(registry["tokens"], {})
                    save.assert_not_called()
        self.assertEqual(self.directory.read_bytes("registry.json"), persisted)

    def test_role_matrix_is_explicit_and_fails_closed(self) -> None:
        expected = {
            "generation": {"verify", "status"},
            "verification": {"finding.append", "report.submit", "status"},
            "publisher": {"publish", "status"},
            "memory": {"memory.read", "memory.commit", "memory.maintain"},
            "context_branch": {"context.branch.read", "context.branch.commit", "context.branch.seal"},
            "context_join": {"context.join.read", "context.join.prepare"},
            "context_verifier": {"context.view.read", "context.view.probe"},
            "validity_writer": {"validity.commit"},
            "validity_withdrawer": {"validity.withdraw"},
            "validity_reader": {"validity.read", "validity.status"},
            "validity_verifier": {"validity.probe", "validity.status"},
            "validity_publisher": {"validity.publish", "validity.status"},
            "project_reader": {"project.read", "project.search"},
            "project_writer": {"project.read", "project.search", "project.commit", "project.index"},
            "project_withdrawer": {"project.withdraw"},
            "research_runner": {"project.read", "project.search", "project.commit", "project.index",
                                "project.withdraw", "research.run", "research.status"},
        }
        self.assertEqual(dict(ROLE_ACTIONS), expected)
        all_actions = set().union(*expected.values())
        for role, allowed in expected.items():
            token = self.token(role=role, actions=sorted(allowed))
            for action in all_actions | {"unknown.action"}:
                with self.subTest(role=role, action=action):
                    if action in allowed:
                        self.authority.authenticate(token, action=action, now=101)
                    else:
                        with self.assertRaises(AuthorizationError):
                            self.authority.authenticate(token, action=action, now=101)
            for action in all_actions - allowed:
                with self.subTest(role=role, issue_action=action), self.assertRaises(AuthorizationError):
                    self.token(role=role, actions=[action])
        for role in ("worker", "admin", "", None, ["publisher"]):
            with self.subTest(role=role), self.assertRaises(AuthorizationError):
                self.token(role=role, actions=["status"])
        for actions in (["unknown"], [], ["status", "status"], "status", [True], None):
            with self.subTest(actions=actions), self.assertRaises(AuthorizationError):
                self.token(actions=actions)
        with self.assertRaises(TypeError):
            self.token(extra_field="not allowed")

    def test_validity_extension_preserves_v1_claims_and_old_roles_cannot_borrow_actions(self) -> None:
        expected = {
            "validity_writer": {"validity.commit"},
            "validity_withdrawer": {"validity.withdraw"},
            "validity_reader": {"validity.read", "validity.status"},
            "validity_verifier": {"validity.probe", "validity.status"},
            "validity_publisher": {"validity.publish", "validity.status"},
        }
        new_actions = set().union(*expected.values())
        old_roles = ("generation", "verification", "publisher", "memory",
                     "context_branch", "context_join", "context_verifier")
        for role in old_roles:
            token = self.token(role=role, actions=sorted(ROLE_ACTIONS[role]))
            for action in new_actions:
                with self.subTest(role=role, action=action), self.assertRaises(AuthorizationError):
                    self.authority.authenticate(token, action=action, now=101)
        for role, actions in expected.items():
            token = self.token(role=role, actions=sorted(actions))
            claims = self.authority.authenticate(token, action=sorted(actions)[0], now=101)
            self.assertEqual(set(claims), {"version", "token_id", "owner", "run_id", "domain", "role", "epoch",
                                          "actions", "issued_at", "expires_at"})
            self.assertEqual(claims["version"], 1)
            self.assertEqual(set(claims["actions"]), actions)
            for extra in ({"resource_scope": "other-use"}, {"use_id": "other-use"}, {"ignore_revocation": True}):
                with self.subTest(role=role, extra=extra), self.assertRaises(AuthorizationError):
                    self.authority.authenticate(self.signed(canonical_json({**claims, **extra})),
                                                action=sorted(actions)[0], now=101)
            for old_action in {"memory.read", "memory.commit", "memory.maintain", "publish", "context.view.probe"}:
                with self.subTest(role=role, old_action=old_action), self.assertRaises(AuthorizationError):
                    self.authority.authenticate(token, action=old_action, now=101)

    def test_validity_policy_schema_and_explicit_role_matrix_agree(self) -> None:
        from importlib.resources import files

        from rethlas_core.contracts import validate_policy

        schema = strict_json_loads(files("rethlas_core").joinpath("schemas", "policy.schema.json").read_bytes())
        properties = schema["properties"]["rules"]["items"]["properties"]
        self.assertEqual(set(properties["role"]["enum"]), set(ROLE_ACTIONS))
        all_actions = set().union(*ROLE_ACTIONS.values())
        self.assertEqual(set(properties["action"]["enum"]), all_actions)
        for role in ("validity_writer", "validity_withdrawer", "validity_reader", "validity_verifier", "validity_publisher"):
            for action in all_actions:
                policy = {"schema_version": "1.0", "default": "deny", "max_model_concurrency": 3,
                          "nested_agents": False, "network": "disabled",
                          "rules": [{"role": role, "action": action, "domains": ["fixture"]}]}
                with self.subTest(role=role, action=action):
                    self.assertEqual(validate_policy(policy)["valid"], action in ROLE_ACTIONS[role])

    def test_research_extension_preserves_claims_and_cannot_borrow_other_actions(self) -> None:
        expected = {
            "project_reader": {"project.read", "project.search"},
            "project_writer": {"project.read", "project.search", "project.commit", "project.index"},
            "project_withdrawer": {"project.withdraw"},
            "research_runner": {"project.read", "project.search", "project.commit", "project.index",
                                "project.withdraw", "research.run", "research.status"},
        }
        new_actions = set().union(*expected.values())
        self.authority.register_run("owner-1", "research-1", "research")
        for role, actions in expected.items():
            token = self.token(run_id="research-1", domain="research", role=role, actions=sorted(actions))
            claims = self.authority.check(token, owner="owner-1", run_id="research-1", domain="research",
                                          action=sorted(actions)[0], now=101)
            self.assertEqual(set(claims), {"version", "token_id", "owner", "run_id", "domain", "role", "epoch",
                                          "actions", "issued_at", "expires_at"})
            for domain in ("fixture", "production"):
                with self.subTest(role=role, domain=domain), self.assertRaises(AuthorizationError):
                    self.authority.check(token, owner="owner-1", run_id="research-1", domain=domain,
                                         action=sorted(actions)[0], now=101)
            for action in {"memory.read", "memory.commit", "memory.maintain", "validity.publish", "publish"}:
                with self.subTest(role=role, action=action), self.assertRaises(AuthorizationError):
                    self.authority.authenticate(token, action=action, now=101)
        for role in set(ROLE_ACTIONS) - set(expected):
            token = self.token(role=role, actions=sorted(ROLE_ACTIONS[role]))
            for action in new_actions:
                with self.subTest(role=role, action=action), self.assertRaises(AuthorizationError):
                    self.authority.authenticate(token, action=action, now=101)

    def test_research_policy_keeps_network_disabled_and_enforces_fixed_roles(self) -> None:
        from rethlas_core.contracts import validate_policy

        for role in ("project_reader", "project_writer", "project_withdrawer", "research_runner"):
            for action in set().union(*ROLE_ACTIONS.values()):
                policy = {"schema_version": "1.0", "default": "deny", "max_model_concurrency": 3,
                          "nested_agents": False, "network": "disabled",
                          "rules": [{"role": role, "action": action, "domains": ["research"]}]}
                with self.subTest(role=role, action=action):
                    self.assertEqual(validate_policy(policy)["valid"], action in ROLE_ACTIONS[role])
            policy["rules"][0]["action"] = sorted(ROLE_ACTIONS[role])[0]
            self.assertTrue(validate_policy(policy)["valid"])
            policy["network"] = "enabled"
            self.assertFalse(validate_policy(policy)["valid"])

    def test_legacy_v1_tokens_cannot_borrow_memory_permissions(self) -> None:
        tokens = {
            role: self.token(role=role, actions=sorted(ROLE_ACTIONS[role]))
            for role in ("generation", "verification", "publisher")
        }
        memory = self.token(role="memory", actions=sorted(ROLE_ACTIONS["memory"]))
        before = self.directory.read_bytes("registry.json")
        reopened = Authority(self.root)
        for role, token in tokens.items():
            claims = reopened.authenticate(token, action="status", now=101)
            self.assertEqual(claims["version"], 1)
            for action in ROLE_ACTIONS["memory"]:
                with self.subTest(role=role, action=action), self.assertRaises(AuthorizationError):
                    reopened.authenticate(token, action=action, now=101)
            # Even a correctly signed role/action rewrite must match the old
            # registered claims; v1 does not gain permissions by reinterpretation.
            forged = {**claims, "role": "memory", "actions": ["memory.read"]}
            with self.assertRaises(AuthorizationError):
                reopened.authenticate(self.signed(canonical_json(forged)), action="memory.read", now=101)
        claims = reopened.authenticate(memory, action="memory.read", now=101)
        self.assertEqual(set(claims), {"version", "token_id", "owner", "run_id", "domain", "role", "epoch", "actions", "issued_at", "expires_at"})
        self.assertEqual(claims["version"], 1)
        for action in ("publish", "status", "memory.unknown"):
            with self.subTest(action=action), self.assertRaises(AuthorizationError):
                reopened.authenticate(memory, action=action, now=101)
        for extra in ({"resource_scope": "anything"}, {"scope": "anything"}):
            with self.subTest(extra=extra), self.assertRaises(AuthorizationError):
                reopened.authenticate(self.signed(canonical_json({**claims, **extra})), action="memory.read", now=101)
        self.assertEqual(self.directory.read_bytes("registry.json"), before)

    def test_scope_mismatch_never_reaches_business_write(self) -> None:
        token = self.token()
        before = self.directory.read_bytes("registry.json")
        output = self.root.parent / "business-output"
        for changes in ({"owner": "owner-2"}, {"run_id": "run-2"}, {"domain": "production"}, {"action": "publish"}, {"action": "unknown"}):
            with self.subTest(changes=changes), self.assertRaises(AuthorizationError):
                args = {"owner": "owner-1", "run_id": "run-1", "domain": "fixture", "action": "finding.append", "now": 101}
                args.update(changes)
                with self.authority.locked_check(token, **args):
                    output.write_bytes(b"must not happen")
        self.assertFalse(output.exists())
        self.assertEqual(self.directory.read_bytes("registry.json"), before)
        for changes in ({"owner": "owner-2"}, {"run_id": "run-2"}, {"domain": "production"}):
            with self.subTest(issue=changes), self.assertRaises(AuthorizationError):
                self.token(**changes)

    def test_identifier_domain_and_ttl_validation(self) -> None:
        invalid_identifiers = ("", ".", "..", "a/b", "a\\b", "a b", "测试", "a" * 129, True, None)
        for value in invalid_identifiers:
            with self.subTest(identifier=value), self.assertRaises(AuthorizationError):
                self.authority.register_run(value, "new", "fixture")
            with self.subTest(run=value), self.assertRaises(AuthorizationError):
                self.authority.register_run("owner-1", value, "fixture")
        for domain in ("test", "Fixture", "", None, True):
            with self.subTest(domain=domain), self.assertRaises(AuthorizationError):
                self.token(domain=domain)
        for ttl in (0, -1, 3600.1, math.nan, math.inf, True, "300", None, 10 ** 1000):
            with self.subTest(ttl=ttl), self.assertRaises(AuthorizationError):
                self.token(ttl_seconds=ttl)
        for now in (True, math.nan, math.inf, "100", 10 ** 1000):
            with self.subTest(now=now), self.assertRaises(AuthorizationError):
                self.token(now=now)
        token = self.token(ttl_seconds=3600)
        self.check(token, now=3699)
        with self.assertRaises(AuthorizationError):
            self.check(token, now=3700)
        self.authority.register_run("owner_1.test", "safe-_.id", "production")
        with self.assertRaises(AuthorizationError):
            self.authority.register_run("owner_1.test", "safe-_.id", "production")

    def test_expiry_future_issue_revocation_epoch_and_closed_run(self) -> None:
        token = self.token(ttl_seconds=2)
        self.check(token, now=100)
        for now in (99, 102, 1000, True, math.inf):
            with self.subTest(now=now), self.assertRaises(AuthorizationError):
                self.check(token, now=now)
        token_id = self.check(token)["token_id"]
        self.authority.revoke(token_id)
        self.authority.revoke(token_id)
        with self.assertRaises(AuthorizationError):
            self.check(token)
        token = self.token()
        self.assertEqual(self.authority.bump_epoch("run-1"), 1)
        with self.assertRaises(AuthorizationError):
            self.check(token)
        replacement = self.token()
        self.assertEqual(self.check(replacement)["epoch"], 1)
        self.authority.close_run("run-1")
        self.authority.close_run("run-1")
        for operation in (lambda: self.check(replacement), lambda: self.token(), lambda: self.authority.bump_epoch("run-1")):
            with self.assertRaises(AuthorizationError):
                operation()
        for operation in (lambda: self.authority.revoke("unknown"), lambda: self.authority.close_run("unknown"), lambda: self.authority.bump_epoch("unknown")):
            with self.assertRaises(AuthorizationError):
                operation()

    def test_forged_tokens_and_unregistered_valid_signature_rejected(self) -> None:
        token = self.token()
        claims = self.check(token)
        payload, signature = token.split(".")
        forged_signature = ("A" if signature[0] != "A" else "B") + signature[1:]
        for malformed in (None, "", "one", "a.b.c", "!.!", payload + "." + forged_signature, token + "=", "x" * 16385):
            with self.subTest(token=malformed), self.assertRaises(AuthorizationError):
                self.check(malformed)
        claims["token_id"] = "unregistered-token"
        with self.assertRaises(AuthorizationError):
            self.check(self.signed(canonical_json(claims)))
        other_root = self.root.parent / "other"
        other = Authority(other_root, create=True)
        other.register_run("owner-1", "run-1", "fixture")
        with self.assertRaises(AuthorizationError):
            other.authenticate(token, action="status", now=101)

    def test_signed_malformed_claims_unknown_fields_and_registry_mismatch(self) -> None:
        token = self.token()
        claims = self.check(token)
        malformed_payloads = [b'{"version":1,"version":1}', b'NaN', b'[]', b'{"x":1e999}', b'"\xff"']
        for payload in malformed_payloads:
            with self.subTest(payload=payload), self.assertRaises(AuthorizationError):
                self.check(self.signed(payload))
        invalid_changes = (
            {"extra": "unknown"}, {"version": True}, {"role": "admin"}, {"actions": ["publish"]},
            {"actions": ["unknown"]}, {"epoch": True}, {"issued_at": True}, {"expires_at": 10000},
        )
        for changes in invalid_changes:
            with self.subTest(changes=changes), self.assertRaises(AuthorizationError):
                self.check(self.signed(canonical_json({**claims, **changes})))
        changed = {**claims, "actions": ["status"]}
        with self.assertRaises(AuthorizationError):
            self.authority.authenticate(self.signed(canonical_json(changed)), action="status", now=101)
        with self.assertRaises(AuthorizationError):
            self.check(self.signed(b" " + canonical_json(claims)))
        registry = self.directory.read_json("registry.json")
        registry["tokens"][claims["token_id"]]["claims"]["expires_at"] += 1
        self.directory.write_json("registry.json", registry)
        with self.assertRaises(AuthorizationError):
            self.check(token)

    def test_missing_token_registry_entry_rejects_valid_signature(self) -> None:
        token = self.token()
        claims = self.check(token)
        registry = self.directory.read_json("registry.json")
        del registry["tokens"][claims["token_id"]]
        self.directory.write_json("registry.json", registry)
        with self.assertRaises(AuthorizationError):
            self.check(token)

    def test_registry_symlink_and_replaced_lock_deny_without_writes(self) -> None:
        token = self.token()
        registry = self.directory.read_bytes("registry.json")
        outside = self.root.parent / "outside-registry"
        outside.write_bytes(registry)
        outside.chmod(0o600)
        (self.root / "registry.json").unlink()
        (self.root / "registry.json").symlink_to(outside)
        with self.assertRaises(AuthorizationError):
            self.check(token)
        self.assertEqual(outside.read_bytes(), registry)
        (self.root / "registry.json").unlink()
        self.directory.write_bytes("registry.json", registry)
        (self.root / "lock").rename(self.root / "old-lock")
        self.directory.write_bytes("replacement", b"")
        (self.root / "replacement").rename(self.root / "lock")
        with self.assertRaises(AuthorizationError):
            self.check(token)
        self.assertEqual(self.directory.read_bytes("registry.json"), registry)

    def test_invalid_or_changed_authority_state_denies(self) -> None:
        token = self.token()
        original = self.directory.read_bytes("registry.json")
        registry = self.directory.read_json("registry.json")
        registries = [
            {**registry, "extra": True},
            {**registry, "version": True},
            {**registry, "runs": {"run-1": {**registry["runs"]["run-1"], "owner": "owner-2"}}},
            {**registry, "runs": {"run-1": {**registry["runs"]["run-1"], "domain": "production"}}},
            {**registry, "runs": {"run-1": {**registry["runs"]["run-1"], "epoch": True}}},
        ]
        for changed in registries:
            self.directory.write_json("registry.json", changed)
            with self.subTest(registry=changed), self.assertRaises(AuthorizationError):
                self.check(token)
        for payload in (b'{"version":1,"version":1}', b'NaN', b'{}', b'"\xff"'):
            self.directory.write_bytes("registry.json", payload)
            with self.subTest(payload=payload), self.assertRaises(AuthorizationError):
                self.check(token)
        self.directory.write_bytes("registry.json", original)
        (self.root / "registry.json").chmod(0o644)
        with self.assertRaises(AuthorizationError):
            self.check(token)
        (self.root / "registry.json").chmod(0o600)
        self.directory.write_bytes("key", b"x" * 32)
        with self.assertRaises(AuthorizationError):
            self.check(token)

    def test_locked_check_preserves_body_exception(self) -> None:
        token = self.token()
        with self.assertRaisesRegex(ValueError, "business failure"):
            with self.authority.locked_check(token, owner="owner-1", run_id="run-1", domain="fixture", action="status", now=101):
                raise ValueError("business failure")
        self.check(token)

    def test_locked_check_blocks_cross_process_revocation_until_commit(self) -> None:
        token = self.token()
        token_id = self.check(token)["token_id"]
        context = multiprocessing.get_context("spawn")
        started, finished = context.Event(), context.Event()
        with self.authority.locked_check(token, owner="owner-1", run_id="run-1", domain="fixture", action="finding.append", now=101):
            inode = (self.root / "lock").stat().st_ino
            process = context.Process(target=_revoke_in_process, args=(str(self.root), token_id, started, finished))
            process.start()
            try:
                self.assertTrue(started.wait(10), "child did not start")
                self.assertFalse(finished.wait(0.2), "revocation passed the commit's lock")
                (self.root.parent / "committed").write_bytes(b"authorized commit")
            except BaseException:
                process.terminate()
                process.join(10)
                raise
        try:
            self.assertTrue(finished.wait(10), "revocation did not finish after commit")
            process.join(10)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.terminate()
                process.join(10)
        self.assertEqual((self.root.parent / "committed").read_bytes(), b"authorized commit")
        self.assertEqual((self.root / "lock").stat().st_ino, inode)
        with self.assertRaises(AuthorizationError):
            self.check(token)


if __name__ == "__main__":
    unittest.main()
