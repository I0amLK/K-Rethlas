"""Offline, controller-only legacy migration; never run inside a model sandbox."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import signal
import sys
import threading
from typing import Iterator

from .legacy_migration import apply, dry_run, prepare
from .memory_io import read_token
from .memory_store import MemoryCancelled, MemoryConflict, MemoryIndeterminate, MemoryStore
from .private_fs import canonical_json


class _Parser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> None:
        # argparse's ordinary diagnostic includes untrusted argv values. Keep
        # failures structured and value-free, including unknown option values.
        raise ValueError("invalid migration arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="migrate-memory", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("dry-run", help="inspect synthetic sources without writes")
    command.add_argument("--spec", type=Path, required=True)
    command = commands.add_parser("prepare", help="freeze a fixture-store request in a private bundle")
    command.add_argument("--spec", type=Path, required=True)
    command.add_argument("--bundle", type=Path, required=True)
    command.add_argument("--store-root", type=Path, required=True)
    command.add_argument("--token-file", type=Path, required=True)
    command.add_argument("--run-id", required=True)
    command = commands.add_parser("apply", help="commit only an operator-pinned frozen request")
    command.add_argument("--bundle", type=Path, required=True)
    command.add_argument("--expected-bundle-sha256", required=True)
    command.add_argument("--store-root", type=Path, required=True)
    command.add_argument("--token-file", type=Path, required=True)
    command.add_argument("--run-id", required=True)
    return parser


@contextmanager
def _cancel_signals(cancel_event: threading.Event) -> Iterator[None]:
    previous = {}

    def cancel(signum, frame) -> None:
        # No I/O, exit, recovery, or commit-result interpretation in a handler.
        cancel_event.set()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, cancel)
        yield
    finally:
        for signum, handler in reversed(tuple(previous.items())):
            signal.signal(signum, handler)


def _emit(value: dict) -> None:
    print(canonical_json(value).decode("utf-8"))


def main(argv: list[str] | None = None, *, repository_root: Path) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] not in ("dry-run", "prepare", "apply", "-h", "--help"):
        arguments.insert(0, "dry-run")
    try:
        args = _parser().parse_args(arguments)
        cancel_event = threading.Event()
        with _cancel_signals(cancel_event):
            if args.command == "dry-run":
                # Do not open a store, credential, lock or bundle on this branch.
                result = dry_run(args.spec, cancel_event=cancel_event)
            else:
                token = read_token(args.token_file)
                store = MemoryStore(args.store_root)
                if args.command == "prepare":
                    # The thin repository entrypoint supplies its fixed root;
                    # evidence protection must not depend on the operator's cwd.
                    result = prepare(args.spec, args.bundle, store, token, run_id=args.run_id,
                                     cancel_event=cancel_event,
                                     forbidden_roots=(args.token_file.absolute().parent,
                                                      Path(repository_root).absolute() / "docs" / "evidence"))
                else:
                    # No status/export/rebase/recovery and no source recapture.
                    # In particular a commit-only credential is sufficient here.
                    result = apply(args.bundle, args.expected_bundle_sha256, store, token,
                                   run_id=args.run_id, cancel_event=cancel_event)
            # The core/store owns the commit point. A signal after its successful
            # return must not turn a durable success into a cancellation claim.
            _emit(result)
        return 0
    except MemoryConflict as exc:
        _emit({"status": "conflict", "error": type(exc).__name__})
        return 3
    except MemoryIndeterminate as exc:
        _emit({"status": "indeterminate", "error": type(exc).__name__,
               "next": "reopen-and-replay-exact-bundle-with-same-pin-and-run"})
        return 4
    except MemoryCancelled as exc:
        _emit({"status": "cancelled", "error": type(exc).__name__,
               "next": "pending-may-remain-use-explicit-authorized-recovery"})
        return 5
    except Exception as exc:
        # Includes filesystem, strict JSON, migration/capacity/quarantine and
        # authorization failures. Never print exception text, paths or payloads.
        _emit({"status": "rejected", "error": type(exc).__name__})
        return 2

