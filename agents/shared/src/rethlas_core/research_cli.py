"""Research project CLI. Configuration and credentials are trusted operator inputs."""
from __future__ import annotations

import argparse
from pathlib import Path

from .memory_io import read_input
from .private_fs import PrivateDirectory, canonical_json, strict_json_loads
from .project_client import connect, grant, configured_text
from .project_service import ProjectService
from .validity import ValidityConflict, ValidityIndeterminate
from .memory_store import MemoryConflict, MemoryIndeterminate
from .provider import ProviderIndeterminate
from .retrieval_index import RetrievalIndexStale


class _Parser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)

    def error(self, message):
        # argparse normally repeats untrusted argument values and private paths.
        raise ValueError("invalid research arguments")


def parser():
    value = _Parser(prog="rethlas-research")
    commands = value.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", allow_abbrev=False)
    start.add_argument("project", type=Path)
    start.add_argument("--question-file", type=Path)
    for purpose in ("generation", "review"):
        start.add_argument("--" + purpose + "-model")
        start.add_argument("--" + purpose + "-effort")
    start.add_argument("--continue", dest="continue_pending", action="store_true")
    start.add_argument("--renew", action="store_true")
    start.add_argument("--new-round", action="store_true")
    start.add_argument("--refresh-stage", action="store_true")
    init = commands.add_parser("init", allow_abbrev=False)
    init.add_argument("--root", type=Path, required=True)
    init.add_argument("--owner", required=True)
    issue = commands.add_parser("issue", allow_abbrev=False)
    issue.add_argument("--root", type=Path, required=True)
    issue.add_argument("--role", choices=("project_reader", "project_writer", "project_withdrawer", "research_runner"), required=True)
    issue.add_argument("--token-file", type=Path, required=True)
    for name in ("status", "commit", "withdraw", "index", "search", "get", "expand", "append", "run", "revise", "review", "research-status", "research-history", "resume", "approve-online", "approve-text", "online-usage"):
        cmd = commands.add_parser(name, allow_abbrev=False)
        cmd.add_argument("--config", type=Path, required=True)
        if name in ("commit", "withdraw", "get", "expand", "append", "run", "revise", "review", "approve-online", "approve-text"):
            cmd.add_argument("--request", type=Path, required=True)
        if name == "search":
            cmd.add_argument("--query", required=True)
            cmd.add_argument("--top-k", type=int, default=10)
        if name in ("research-status", "resume"):
            cmd.add_argument("--request-id", required=True)
        if name == "research-history":
            cmd.add_argument("--limit", type=int, default=32)
    return value


def _path(value):
    path = value.absolute()
    if ".." in path.parts or any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("private paths must not traverse parents or symlinks")
    return path


def _request(path):
    value = strict_json_loads(read_input(path, 524288, single_link=True, stable=True))
    if type(value) is not dict:
        raise ValueError("research request must be an object")
    return value


def main(argv=None):
    try:
        args = parser().parse_args(argv)
        if args.command == "start":
            from .research_start import start_project
            result = start_project(args)
        elif args.command in ("init", "issue"):
            root = _path(args.root)
            repository = _path(Path(__file__).resolve().parents[4])
            if root == repository or repository in root.parents or root in repository.parents:
                raise ValueError("private project root must be outside the repository")
            if args.command == "init":
                service = ProjectService.initialize(root, owner=args.owner)
                result = {"status": "initialized", "service_id": service.config["service_id"], "domain": "research"}
            else:
                service = ProjectService(root)
                path = _path(args.token_file)
                if path == root or root in path.parents or path == repository or repository in path.parents:
                    raise ValueError("credentials must be outside project and repository")
                directory = PrivateDirectory(path.parent)
                with directory.lock():
                    if any(n == path.name for n, _, _ in directory.entries(max_entries=256)):
                        raise ValueError("credential output already exists")
                    issued = service.issue(role=args.role)
                    directory.write_bytes(path.name, issued["token"].encode("ascii"), exclusive=True)
                result = {"status": "issued", "run_id": issued["run_id"]}
        else:
            config, service, retrieval, memory = connect(args.config,
                providers=args.command in {"index", "search", "run", "revise"})
            reader = memory.reader
            if args.command == "status":
                result = service.status(reader["token"], run_id=reader["run_id"])
            elif args.command in ("commit", "withdraw"):
                role = "writer" if args.command == "commit" else "runner"
                selected = grant(config, role, required=True)
                result = getattr(service, args.command)(selected["token"], _request(args.request), run_id=selected["run_id"])
            elif args.command == "index":
                result = memory.rebuild_index()
            elif args.command == "search":
                result = retrieval.search(reader["token"], args.query, run_id=reader["run_id"], top_k=args.top_k)
            elif args.command == "get":
                result = memory.get_revision(_request(args.request))
            elif args.command == "expand":
                result = memory.expand(**_request(args.request))
            elif args.command == "append":
                result = memory.append(**_request(args.request))
            elif args.command in ("run", "revise", "review", "research-status", "research-history", "resume"):
                from .research import ResearchPipeline
                selected = grant(config, "runner", required=True)
                offline = args.command in ("research-status", "research-history")
                if offline:
                    pipeline = ResearchPipeline(service, retrieval, None)
                elif config["version"] == "research-client-1":
                    pipeline = ResearchPipeline(service, retrieval, configured_text(service, config))
                else:
                    review_only = args.command == "review"
                    if args.command == "resume":
                        review_only = ResearchPipeline(service, retrieval, None).status(selected,
                            request_id=args.request_id)["kind"] == "review"
                    pipeline = ResearchPipeline(service, retrieval,
                        generation_provider=None if review_only else configured_text(service, config, "generation"),
                        review_provider=configured_text(service, config, "review"))
                if args.command == "research-history":
                    result = pipeline.history(selected, limit=args.limit)
                elif args.command in ("research-status", "resume"):
                    method = pipeline.status if args.command == "research-status" else pipeline.resume
                    result = method(selected, request_id=args.request_id)
                else:
                    result = getattr(pipeline, args.command)(selected, **_request(args.request))
            elif args.command == "approve-online":
                from .online_round import OnlineRound
                result = OnlineRound.initialize(service, **_request(args.request)).usage()
            elif args.command == "approve-text":
                from .online_round import OnlineRound
                result = OnlineRound.approve_text(service, **_request(args.request))
            else:
                from .online_round import OnlineRound
                result = OnlineRound(service).usage()
        print(canonical_json(result).decode())
        return 0
    except (ValidityConflict, MemoryConflict, RetrievalIndexStale) as exc:
        print(canonical_json({"status": "conflict", "error": type(exc).__name__}).decode())
        return 3
    except (ValidityIndeterminate, MemoryIndeterminate, ProviderIndeterminate) as exc:
        print(canonical_json({"status": "indeterminate", "error": type(exc).__name__}).decode())
        return 4
    except Exception as exc:
        print(canonical_json({"status": "rejected", "error": type(exc).__name__}).decode())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
