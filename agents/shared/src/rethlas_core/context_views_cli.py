"""Operator-driven offline CLI; this process is not a worker security boundary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .authorization import AuthorizationError
from .context_views import ContextConflict, ContextIndeterminate, META_BYTES, REQUEST_BYTES
from .context_workspace import ContextWorkspace
from .memory_io import read_input, read_token
from .memory_store import MemoryStore, MemoryConflict, MemoryIndeterminate, MemoryCancelled
from .private_fs import PrivateDirectory, strict_json_loads
from .worker import WorkerCancelled, WorkerTimeout, WorkerUnavailable


def main(argv=None, *, repository_root: Path) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="trusted operator capture; synthetic fixture only")
    init.add_argument("--source-root", required=True, type=Path)
    init.add_argument("--source-token-file", required=True, type=Path)
    init.add_argument("--source-run-id", required=True)
    init.add_argument("--spec", required=True, type=Path)
    issue = commands.add_parser("issue", help="trusted operator issuance, never a model action")
    issue.add_argument("--kind", required=True, choices=("branch", "join", "view"))
    issue.add_argument("--branch-id")
    issue.add_argument("--action", required=True, action="append")
    issue.add_argument("--token-file", required=True, type=Path)
    build = commands.add_parser("build-view", help="trusted operator exact declassification")
    build.add_argument("--request", required=True, type=Path)
    recover = commands.add_parser("recover", help="trusted operator exact recovery; no automatic reopening")
    recover.add_argument("--kind", required=True, choices=("branch", "join", "view"))
    recover.add_argument("--branch-id")
    recover.add_argument("--request", type=Path)
    recover.add_argument("--apply", action="store_true")
    for name in ("branch-read", "branch-commit", "seal", "join-prepare", "join-read", "view-read", "probe"):
        command = commands.add_parser(name)
        command.add_argument("--token-file", required=True, type=Path)
        command.add_argument("--run-id", required=True)
        if name in ("branch-commit", "seal", "join-prepare"):
            command.add_argument("--request", required=True, type=Path)
        if name == "branch-read":
            command.add_argument("--head")
    for command in commands.choices.values():
        command.add_argument("--workspace", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        root = args.workspace.absolute()
        repo = repository_root.absolute()
        if root == repo or root in repo.parents or repo in root.parents:
            raise ValueError("private workspaces must be outside the repository")
        def request(path, maximum=META_BYTES):
            return strict_json_loads(read_input(path, maximum, single_link=True, stable=True))
        if args.command == "init":
            workspace = ContextWorkspace.initialize(root, MemoryStore(args.source_root), read_token(args.source_token_file),
                                                     run_id=args.source_run_id, spec=request(args.spec))
            result = {"status": "initialized", "workspace_id": workspace.config["workspace_id"],
                      "snapshot_sha256": workspace.config["snapshot_sha256"]}
        else:
            workspace = ContextWorkspace(root)
            if args.command == "issue":
                path = args.token_file.absolute()
                if path == root or root in path.parents or path.parent == repo or repo in path.parents:
                    raise ValueError("credential output must be outside workspace/repository")
                directory = PrivateDirectory(path.parent)
                with directory.lock():
                    if any(name == path.name for name, _, _ in directory.entries(max_entries=256)):
                        raise ValueError("credential file already exists")
                    grant = workspace.issue(kind=args.kind, branch_id=args.branch_id, actions=args.action)
                    directory.write_bytes(path.name, grant["token"].encode("ascii"))
                result = {"status": "issued", "run_id": grant["run_id"]}
            elif args.command == "build-view":
                result = workspace.build_view(request(args.request))
            elif args.command == "recover":
                if args.kind == "branch":
                    if args.request is not None or args.branch_id is None:
                        raise ValueError("branch recovery requires only branch ID")
                    result = workspace.recover_branch(args.branch_id, apply=args.apply)
                else:
                    if args.request is None or args.branch_id is not None:
                        raise ValueError("publication recovery requires only exact original request")
                    result = workspace.recover_publication(kind=args.kind, request=request(args.request), apply=args.apply)
            else:
                token = read_token(args.token_file)
                if args.command in ("branch-commit", "seal", "join-prepare"):
                    method = {"branch-commit": workspace.branch_commit, "seal": workspace.seal,
                              "join-prepare": workspace.join_prepare}[args.command]
                    result = method(token, request(args.request, REQUEST_BYTES if args.command == "branch-commit" else META_BYTES),
                                    run_id=args.run_id)
                elif args.command == "branch-read":
                    result = workspace.branch_read(token, run_id=args.run_id, head=args.head)
                else:
                    method = {"join-read": workspace.join_read, "view-read": workspace.view_read, "probe": workspace.probe}[args.command]
                    result = method(token, run_id=args.run_id)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
        return 0
    except (ContextConflict, MemoryConflict) as exc:
        print(json.dumps({"status": "conflict", "error": type(exc).__name__}))
        return 3
    except (ContextIndeterminate, MemoryIndeterminate) as exc:
        print(json.dumps({"status": "indeterminate", "error": type(exc).__name__}))
        return 4
    except (MemoryCancelled, WorkerCancelled, WorkerTimeout) as exc:
        print(json.dumps({"status": "stopped", "error": type(exc).__name__}))
        return 5
    except (AuthorizationError, ValueError, OSError, WorkerUnavailable, TypeError, KeyError, RecursionError) as exc:
        print(json.dumps({"status": "rejected", "error": type(exc).__name__}))
        return 2
