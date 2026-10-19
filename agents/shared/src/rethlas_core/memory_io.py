"""Bounded, no-follow inputs shared by controller-only memory entrypoints."""
from __future__ import annotations

import os
from pathlib import Path
import stat

from .private_fs import PrivateDirectory


def read_input(path: Path, maximum: int, *, single_link: bool = False,
               stable: bool = False) -> bytes:
    # Operator-selected untrusted file: no symlink ancestors, special files,
    # unbounded reads, or Python execution from its containing directory.
    path = path.absolute()
    if ".." in path.parts or any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("memory input must be a non-symlink path")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for component in path.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        leaf = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                       dir_fd=descriptor)
    finally:
        os.close(descriptor)
    descriptor = leaf
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise ValueError("memory input must be a bounded regular file")
        if single_link and info.st_nlink != 1:
            raise ValueError("memory input must be single-link")
        chunks, total = [], 0
        while True:
            chunk = os.read(descriptor, min(65536, maximum - total + 1))
            if not chunk:
                data = b"".join(chunks)
                if stable:
                    after = os.fstat(descriptor)
                    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
                    if (len(data) != info.st_size
                            or any(getattr(info, field) != getattr(after, field) for field in fields)):
                        raise ValueError("memory input changed while reading")
                return data
            total += len(chunk)
            if total > maximum:
                raise ValueError("memory input grew past its limit")
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def read_token(path: Path) -> str:
    # The private-directory primitive also checks parent ownership/mode, file
    # ownership, 0600, regularity and single-link identity before bounded reading.
    token = PrivateDirectory(path.absolute().parent).read_bytes(path.name, max_bytes=16384).decode("ascii")
    if not token or any(item.isspace() for item in token):
        raise ValueError("invalid memory credential file")
    return token
