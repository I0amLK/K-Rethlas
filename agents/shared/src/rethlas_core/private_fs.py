"""Small, fail-closed filesystem primitives for controller-owned state."""
from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import json
import math
import os
import secrets
import stat
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator


class PrivateFilesystemError(ValueError):
    """A path, file, or JSON value is not safe for private state."""


def _json_value(value: Any) -> None:
    if type(value) is str:
        try:
            value.encode("utf-8")
        except UnicodeError as exc:
            raise PrivateFilesystemError("JSON strings must be valid UTF-8") from exc
        return
    if value is None or type(value) in (bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _json_value(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for key, item in value.items():
            _json_value(key)
            _json_value(item)
        return
    raise PrivateFilesystemError("value is not strict JSON")


def canonical_json(value: Any) -> bytes:
    """Encode JSON deterministically, without non-finite numbers or coerced keys."""
    _json_value(value)
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def strict_json_loads(value: str | bytes) -> Any:
    """Read UTF-8 JSON, rejecting duplicate keys and non-finite numbers."""
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise PrivateFilesystemError("duplicate JSON key")
            result[key] = item
        return result

    def reject_constant(value: str) -> Any:
        raise PrivateFilesystemError(f"invalid JSON constant: {value}")

    if type(value) is bytes:
        value = value.decode("utf-8")
    if type(value) is not str:
        raise PrivateFilesystemError("JSON input must be str or bytes")
    result = json.loads(value, object_pairs_hook=object_pairs, parse_constant=reject_constant)
    _json_value(result)
    return result


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _name(name: str) -> str:
    if (
        type(name) is not str
        or name in ("", ".", "..")
        or "/" in name
        or "\\" in name
        or "\0" in name
    ):
        raise PrivateFilesystemError("expected a single safe path component")
    try:
        length = len(name.encode("utf-8"))
    except UnicodeError as exc:
        raise PrivateFilesystemError("path components must be valid UTF-8") from exc
    if length > 255:
        raise PrivateFilesystemError("path component is too long")
    return name


def _open(path: str | Path, flags: int, mode: int = 0o600, *, dir_fd: int | None = None) -> int:
    try:
        return os.open(path, flags | os.O_CLOEXEC | os.O_NOFOLLOW, mode, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise PrivateFilesystemError("symlinks and non-directory ancestors are forbidden") from exc
        raise


def _directory(info: os.stat_result) -> None:
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise PrivateFilesystemError("private directories must be owned by this uid with mode 0700")


def _regular(info: os.stat_result) -> None:
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o600
        or info.st_nlink != 1
    ):
        raise PrivateFilesystemError("private files must be single-link, uid-owned regular files with mode 0600")


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _write_all(fd: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(fd, remaining)
        if not written:
            raise OSError("short write to private file")
        remaining = remaining[written:]


def _rename_noreplace(source: str, destination: str, source_fd: int, destination_fd: int) -> None:
    """Use Linux's atomic no-overwrite rename; never fall back to check + rename."""
    try:
        rename = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise PrivateFilesystemError("atomic no-replace rename is unavailable") from exc
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    # RENAME_NOREPLACE = 1; unsupported kernels/filesystems must fail closed.
    if rename(source_fd, os.fsencode(source), destination_fd, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        if error in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
            raise PrivateFilesystemError("atomic no-replace rename is unavailable")
        raise OSError(error, os.strerror(error), destination)


class PrivateDirectory:
    """A private directory addressed with no-follow, directory-relative operations.

    Existing permissions are validated, never repaired. Locks have stable inodes;
    callers must reserve lock filenames for ``lock()`` across all instances, not
    ordinary writes. An instance also remembers its locks to detect replacement.
    Lock acquisition retries nonblocking flock for at most five seconds.
    """

    def __init__(self, root: Path, create: bool = False):
        path = Path(root)
        if not path.is_absolute():
            path = Path.cwd() / path
        if ".." in path.parts:
            raise PrivateFilesystemError("parent traversal is forbidden in a private root")
        self.root = path
        self._lock_inodes: dict[str, tuple[int, int]] = {}
        fd = self._walk(create=create)
        try:
            self._identity = _identity(os.fstat(fd))
        finally:
            os.close(fd)

    def _walk(self, *, create: bool = False) -> int:
        fd = _open("/", os.O_RDONLY | os.O_DIRECTORY)
        try:
            for component in self.root.parts[1:]:
                try:
                    child = _open(component, os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
                except FileNotFoundError:
                    if not create:
                        raise
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=fd)
                        os.fsync(fd)
                    except FileExistsError:
                        pass
                    child = _open(component, os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
                os.close(fd)
                fd = child
            _directory(os.fstat(fd))
            return fd
        except BaseException:
            os.close(fd)
            raise

    @contextmanager
    def _fd(self) -> Iterator[int]:
        fd = self._walk()
        try:
            if _identity(os.fstat(fd)) != self._identity:
                raise PrivateFilesystemError("private directory was replaced")
            yield fd
        finally:
            os.close(fd)

    def child(self, name: str, create: bool = False) -> PrivateDirectory:
        name = _name(name)
        with self._fd() as parent:
            if create:
                try:
                    os.mkdir(name, mode=0o700, dir_fd=parent)
                    os.fsync(parent)
                except FileExistsError:
                    pass
            fd = _open(name, os.O_RDONLY | os.O_DIRECTORY, dir_fd=parent)
            try:
                _directory(os.fstat(fd))
                result = object.__new__(PrivateDirectory)
                result.root = self.root / name
                result._identity = _identity(os.fstat(fd))
                result._lock_inodes = {}
                return result
            finally:
                os.close(fd)

    def read_bytes(self, name: str, *, max_bytes: int | None = None) -> bytes:
        name = _name(name)
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
            raise PrivateFilesystemError("max_bytes must be a nonnegative integer or None")
        with self._fd() as directory:
            fd = _open(name, os.O_RDONLY | os.O_NONBLOCK, dir_fd=directory)
            try:
                info = os.fstat(fd)
                _regular(info)
                if max_bytes is not None and info.st_size > max_bytes:
                    raise PrivateFilesystemError("private file exceeds max_bytes")
                chunks = []
                total = 0
                while True:
                    count = 65536 if max_bytes is None else min(65536, max_bytes - total + 1)
                    chunk = os.read(fd, count)
                    if not chunk:
                        break
                    total += len(chunk)
                    if max_bytes is not None and total > max_bytes:
                        raise PrivateFilesystemError("private file exceeds max_bytes")
                    chunks.append(chunk)
                return b"".join(chunks)
            finally:
                os.close(fd)

    def read_json(self, name: str, *, max_bytes: int | None = None) -> Any:
        return strict_json_loads(self.read_bytes(name, max_bytes=max_bytes))

    def sync_file(self, name: str) -> None:
        """Fsync an existing safe file and its directory; caller holds its store lock.

        No contents are read or written, and no object is created or replaced.
        Barrier failures propagate rather than imply a durable prior commit.
        """
        name = _name(name)
        with self._fd() as directory:
            fd = _open(name, os.O_RDONLY | os.O_NONBLOCK, dir_fd=directory)
            try:
                _regular(os.fstat(fd))
                os.fsync(fd)
                os.fsync(directory)
            finally:
                os.close(fd)

    def read_prefix(self, name: str, *, max_bytes: int) -> bytes:
        """Read at most a bounded prefix of a safe file, including oversized files.

        Unlike read_bytes(max_bytes=...), this explicitly does not promise a
        complete object. Used only to inspect a journal's first intent before
        quarantining an oversized tail; callers must never parse it as a snapshot.
        """
        name = _name(name)
        if type(max_bytes) is not int or max_bytes < 0:
            raise PrivateFilesystemError("max_bytes must be a nonnegative integer")
        with self._fd() as directory:
            fd = _open(name, os.O_RDONLY | os.O_NONBLOCK, dir_fd=directory)
            try:
                _regular(os.fstat(fd))
                chunks, remaining = [], max_bytes
                while remaining:
                    chunk = os.read(fd, min(65536, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                return b"".join(chunks)
            finally:
                os.close(fd)

    def entries(self, *, max_entries: int) -> list[tuple[str, bool, int]]:
        """List safe direct children, failing when the exclusive count bound is hit.

        Directory sizes are reported as zero. Hidden files (including .tmp-*)
        count normally; this does not recurse. The caller holds its store lock.
        """
        if type(max_entries) is not int or max_entries <= 0:
            raise PrivateFilesystemError("max_entries must be a positive integer")
        result = []
        with self._fd() as directory, os.scandir(directory) as children:
            for entry in children:
                if len(result) + 1 >= max_entries:
                    raise PrivateFilesystemError("private directory reaches max_entries")
                name = _name(entry.name)
                info = os.stat(name, dir_fd=directory, follow_symlinks=False)
                is_directory = stat.S_ISDIR(info.st_mode)
                if is_directory:
                    _directory(info)
                else:
                    _regular(info)
                result.append((name, is_directory, 0 if is_directory else info.st_size))
        return result

    def quarantine_to(self, name: str, destination: PrivateDirectory, new_name: str) -> None:
        """Atomically move one safe file without overwrite; caller holds the store lock.

        Both directories must be on the same filesystem. No file contents are
        read, no additional locks are acquired, and unsupported rename fails shut.
        """
        name, new_name = _name(name), _name(new_name)
        if not isinstance(destination, PrivateDirectory):
            raise PrivateFilesystemError("destination must be a PrivateDirectory")
        if name in self._lock_inodes or new_name in destination._lock_inodes:
            raise PrivateFilesystemError("lock files cannot be moved by quarantine")
        with self._fd() as source_fd, destination._fd() as destination_fd:
            if os.fstat(source_fd).st_dev != os.fstat(destination_fd).st_dev:
                raise PrivateFilesystemError("quarantine requires the same filesystem")
            _regular(os.stat(name, dir_fd=source_fd, follow_symlinks=False))
            try:
                target = os.stat(new_name, dir_fd=destination_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                _regular(target)
                raise FileExistsError(new_name)
            _rename_noreplace(name, new_name, source_fd, destination_fd)
            try:
                os.fsync(source_fd)
            finally:
                os.fsync(destination_fd)

    def write_bytes(
        self, name: str, data: bytes, exclusive: bool = False, *,
        before_replace: Callable[[], None] | None = None,
    ) -> None:
        name = _name(name)
        if before_replace is not None and (exclusive or not callable(before_replace)):
            raise PrivateFilesystemError("before_replace must be callable and cannot guard exclusive writes")
        if name in self._lock_inodes:
            raise PrivateFilesystemError("lock files cannot be replaced by ordinary writes")
        if type(data) is not bytes:
            raise PrivateFilesystemError("private file content must be bytes")
        with self._fd() as directory:
            try:
                _regular(os.stat(name, dir_fd=directory, follow_symlinks=False))
            except FileNotFoundError:
                pass
            else:
                if exclusive:
                    raise FileExistsError(name)
            temporary = f".tmp-{secrets.token_hex(16)}"
            fd = _open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, dir_fd=directory)
            temporary_exists = True
            try:
                os.fchmod(fd, 0o600)
                _regular(os.fstat(fd))
                _write_all(fd, data)
                os.fsync(fd)
                if exclusive:
                    # link() commits a complete file without overwriting an
                    # existing destination, unlike a precheck plus rename().
                    os.link(temporary, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
                    os.unlink(temporary, dir_fd=directory)
                else:
                    # Recheck the destination before replacing it: never overwrite
                    # a symlink, hardlink, special file, or unsafe existing file.
                    try:
                        _regular(os.stat(name, dir_fd=directory, follow_symlinks=False))
                    except FileNotFoundError:
                        pass
                    if before_replace is not None:
                        before_replace()
                    os.replace(temporary, name, src_dir_fd=directory, dst_dir_fd=directory)
                temporary_exists = False
                os.fsync(directory)
            finally:
                os.close(fd)
                if temporary_exists:
                    os.unlink(temporary, dir_fd=directory)

    def write_json(
        self, name: str, value: Any, *, before_replace: Callable[[], None] | None = None,
    ) -> None:
        self.write_bytes(name, canonical_json(value), before_replace=before_replace)

    @contextmanager
    def lock(self, name: str = "lock") -> Iterator[None]:
        name = _name(name)
        with self._fd() as directory:
            try:
                fd = _open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NONBLOCK, dir_fd=directory)
                created = True
            except FileExistsError:
                fd = _open(name, os.O_RDWR | os.O_NONBLOCK, dir_fd=directory)
                created = False
            try:
                if created:
                    os.fchmod(fd, 0o600)
                info = os.fstat(fd)
                _regular(info)
                identity = _identity(info)
                if self._lock_inodes.setdefault(name, identity) != identity:
                    raise PrivateFilesystemError("lock inode was replaced")
                if created:
                    os.fsync(fd)
                    os.fsync(directory)
                deadline = time.monotonic() + 5
                while True:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except OSError as exc:
                        if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EINTR):
                            raise
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise PrivateFilesystemError("private state lock deadline exceeded") from exc
                        time.sleep(min(0.01, remaining))
                current = os.stat(name, dir_fd=directory, follow_symlinks=False)
                _regular(current)
                if _identity(current) != _identity(os.fstat(fd)):
                    raise PrivateFilesystemError("lock inode was replaced")
                yield
            finally:
                os.close(fd)  # Closing releases flock without unlinking the lock.
