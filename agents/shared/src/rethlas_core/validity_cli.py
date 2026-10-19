"""Operator-driven fixture validity CLI, not a worker filesystem boundary."""
from __future__ import annotations

import argparse
from pathlib import Path

from .memory_io import read_input, read_token
from .memory_store import MemoryCancelled, MemoryConflict, MemoryIndeterminate
from .private_fs import PrivateDirectory, canonical_json, strict_json_loads
from .validity import ValidityConflict, ValidityIndeterminate
from .validity_service import CurrentValidityService
from .worker import WorkerCancelled, WorkerTimeout


REQUEST_BYTES = 524288
_ROLES = ("validity_writer", "validity_withdrawer", "validity_reader", "validity_verifier", "validity_publisher")


class _Parser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)

    def error(self, message):
        # Ordinary argparse diagnostics echo untrusted option values and paths.
        raise ValueError("invalid validity arguments")


def _parser():
    parser = _Parser(prog="validity-memory", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="initialize a new fixture-only owned service")
    init.add_argument("--owner", required=True)
    init.add_argument("--domain", choices=("fixture",), default="fixture")
    issue = commands.add_parser("issue", help="trusted operator issuance, never a model action")
    issue.add_argument("--role", choices=_ROLES, required=True)
    issue.add_argument("--use-id")
    issue.add_argument("--action", action="append", required=True)
    issue.add_argument("--token-file", type=Path, required=True)
    register = commands.add_parser("register-use", help="operator-approved exact immutable input disclosure")
    register.add_argument("--request", type=Path, required=True)
    recover = commands.add_parser("recover", help="explicit original control or graph recovery")
    recover.add_argument("--kind", choices=("control", "graph"), required=True)
    recover.add_argument("--apply", action="store_true")
    recover.add_argument("--cancel", action="store_true")
    recover.add_argument("--request", type=Path)
    recover.add_argument("--token-file", type=Path)
    recover.add_argument("--run-id")
    commands.add_parser("audit", help="trusted operator historical audit, not current-use permission")
    for name in ("events", "review-queue"):
        command = commands.add_parser(name, help="trusted operator bounded current-checkpoint page")
        command.add_argument("--cursor")
    for name in ("commit", "withdraw", "read", "lookup", "probe", "publish", "status"):
        command = commands.add_parser(name)
        command.add_argument("--token-file", type=Path, required=True)
        command.add_argument("--run-id", required=True)
        if name in ("commit", "withdraw", "lookup", "probe", "publish"):
            command.add_argument("--request", type=Path, required=True)
    for command in commands.choices.values():
        command.add_argument("--root", type=Path, required=True)
    return parser


def _path(value: Path) -> Path:
    result = value.absolute()
    if ".." in result.parts or any(path.is_symlink() for path in (result, *result.parents)):
        raise ValueError("private paths must not traverse parents or symlinks")
    return result


def _request(path: Path):
    return strict_json_loads(read_input(path, REQUEST_BYTES, single_link=True, stable=True))


def _emit(value):
    print(canonical_json(value).decode("utf-8"))


def main(argv=None, *, repository_root: Path) -> int:
    try:
        args = _parser().parse_args(argv)
        root, repository = _path(args.root), _path(repository_root)
        if root == repository or root in repository.parents or repository in root.parents:
            raise ValueError("private service roots must be outside the repository")
        if args.command == "init":
            service = CurrentValidityService.initialize(root, owner=args.owner, domain=args.domain)
            result = {"status": "initialized", "service_id": service.config["service_id"]}
        else:
            service = CurrentValidityService(root)
            if args.command == "issue":
                path = _path(args.token_file)
                if path == root or root in path.parents or path == repository or repository in path.parents:
                    raise ValueError("credential output must be outside service and repository")
                directory = PrivateDirectory(path.parent)
                # Reserve the operator's destination before minting, never print
                # tokens, and never replace an existing credential or lock file.
                with directory.lock():
                    if any(name == path.name for name, _, _ in directory.entries(max_entries=256)):
                        raise ValueError("credential destination already exists")
                    grant = service.issue(role=args.role, actions=args.action, use_id=args.use_id,
                                          ttl_seconds=300)
                    directory.write_bytes(path.name, grant["token"].encode("ascii"), exclusive=True)
                result = {"status": "issued", "run_id": grant["run_id"]}
            elif args.command == "register-use":
                result = service.register_use(_request(args.request))
            elif args.command == "recover":
                if (args.token_file is None) != (args.run_id is None):
                    raise ValueError("recovery credentials require both token file and run ID")
                if args.cancel and not args.apply:
                    raise ValueError("cancellation requires explicit recovery apply")
                token = read_token(args.token_file) if args.token_file is not None else None
                result = service.recover(kind=args.kind, apply=args.apply, cancel=args.cancel,
                                         request=_request(args.request) if args.request is not None else None,
                                         token=token, run_id=args.run_id)
            elif args.command == "audit":
                result = service.audit()
            elif args.command in ("events", "review-queue"):
                method = service.events if args.command == "events" else service.review_queue
                result = method(cursor=args.cursor)
            else:
                token = read_token(args.token_file)
                method = {"commit": service.commit, "withdraw": service.withdraw, "read": service.read,
                          "lookup": service.lookup, "probe": service.probe, "publish": service.publish,
                          "status": service.status}[args.command]
                if args.command in ("read", "status"):
                    result = method(token, run_id=args.run_id)
                else:
                    result = method(token, _request(args.request), run_id=args.run_id)
        _emit(result)
        return 0
    except (ValidityConflict, MemoryConflict) as exc:
        _emit({"status": "conflict", "error": type(exc).__name__})
        return 3
    except (ValidityIndeterminate, MemoryIndeterminate) as exc:
        _emit({"status": "indeterminate", "error": type(exc).__name__})
        return 4
    except (MemoryCancelled, WorkerCancelled, WorkerTimeout) as exc:
        _emit({"status": "stopped", "error": type(exc).__name__})
        return 5
    except Exception as exc:
        # Includes strict JSON, authentication, capacity, unsafe path and backend
        # failures. Do not echo exception text, argv, reason, body or hidden IDs.
        _emit({"status": "rejected", "error": type(exc).__name__})
        return 2
