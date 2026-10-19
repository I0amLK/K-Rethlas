"""Controller-only CLI. Never run this entrypoint inside a model sandbox."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import uuid

from .authorization import Authority, AuthorizationError
from .contracts import validate_math_memory, validate_policy
from .memory_io import read_input as _memory_input, read_token as _memory_token
from .private_fs import PrivateDirectory, canonical_json, strict_json_loads
from .publication import FixturePublication
from .verification import validate_verification_output


def read_json(path: Path):
    return strict_json_loads(path.read_bytes())


def initialize(root: Path) -> None:
    PrivateDirectory(root, create=True)
    Authority(root / "authority", create=True)
    FixturePublication(root / "fixture-publications", create=True)


def _memory_command(args) -> int:
    from .memory_store import MemoryConflict, MemoryIndeterminate, MemoryStore
    if args.command == "memory-init":
        store = MemoryStore.initialize(args.store_root, owner=args.owner, domain=args.domain)
        result = {"status": "initialized", "store_id": store.config["store_id"],
                  "format_version": store.config["format_version"]}
    else:
        store = MemoryStore(args.store_root)
        if args.command == "memory-issue":
            target = PrivateDirectory(args.token_file.absolute().parent)
            if any(name == args.token_file.name for name, _, _ in target.entries(max_entries=1025)):
                raise ValueError("token target already exists")
            token = store.issue(run_id=args.run_id, actions=args.actions, ttl_seconds=args.ttl)
            # No stdout/argv token. This operator credential export retains the
            # existing no-overwrite primitive; it is not a store object commit.
            target.write_bytes(args.token_file.name, token.encode("ascii"), exclusive=True)
            result = {"status": "credential-written"}
        else:
            token = _memory_token(args.token_file)
            try:
                if args.command == "memory-status":
                    result = store.status(token, run_id=args.run_id)
                elif args.command == "memory-commit":
                    payload = strict_json_loads(_memory_input(args.request, store.limits["request_bytes"]))
                    result = store.commit(token, payload, run_id=args.run_id)
                elif args.command == "memory-export":
                    result = store.export(token, run_id=args.run_id, head=args.head)
                elif args.command == "memory-recover":
                    result = store.recover(token, run_id=args.run_id, apply=args.apply)
                else:
                    result = store.rebuild_projection(token, run_id=args.run_id)
            except MemoryConflict:
                print('{"status":"conflict"}')
                return 3
            except MemoryIndeterminate:
                print('{"status":"indeterminate","next":"reopen-and-replay-exact-request"}')
                return 4
    print(canonical_json(result).decode())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-policy", "validate-math", "validate-report"):
        command = commands.add_parser(name)
        command.add_argument("path", type=Path)
    command = commands.add_parser("init")
    command.add_argument("--private-root", type=Path, required=True)
    command = commands.add_parser("issue")
    command.add_argument("--private-root", type=Path, required=True)
    command.add_argument("--owner", required=True)
    command.add_argument("--run-id", required=True)
    command.add_argument("--domain", choices=("fixture", "production"), required=True)
    command.add_argument("--token-file", type=Path, required=True)
    command.add_argument("--ttl", type=float, default=300)
    command = commands.add_parser("serve")
    command.add_argument("--private-root", type=Path, required=True)
    command.add_argument("--driver", choices=("disabled", "fixture"), default="disabled")
    command.add_argument("--host", choices=("127.0.0.1",), default="127.0.0.1")
    command.add_argument("--port", type=int, default=8091)
    command = commands.add_parser("fixture-demo")
    command.add_argument("--private-root", type=Path, required=True)
    command = commands.add_parser("status")
    command.add_argument("--endpoint", default="http://127.0.0.1:8091/verify")
    command.add_argument("--token-file", type=Path, required=True)
    command.add_argument("--statement", type=Path)
    command.add_argument("--proof", type=Path)
    commands.add_parser("sandbox-probe")
    command = commands.add_parser("site-sync")
    command.add_argument("--results", type=Path, required=True)
    command.add_argument("--content", type=Path, required=True)
    command = commands.add_parser("launch")
    command.add_argument("--generation-root", type=Path, required=True)
    command.add_argument("--problem", default="data/example.md")
    command.add_argument("--dry-run", action="store_true")
    command = commands.add_parser("memory-init")
    command.add_argument("--store-root", type=Path, required=True)
    command.add_argument("--owner", required=True)
    command.add_argument("--domain", choices=("fixture", "production"), required=True)
    for name in ("memory-issue", "memory-status", "memory-commit", "memory-export",
                 "memory-recover", "memory-rebuild-projection"):
        command = commands.add_parser(name)
        command.add_argument("--store-root", type=Path, required=True)
        command.add_argument("--run-id", required=True)
        command.add_argument("--token-file", type=Path, required=True)
        if name == "memory-issue":
            command.add_argument("--actions", nargs="+", required=True,
                                 choices=("memory.read", "memory.commit", "memory.maintain"))
            command.add_argument("--ttl", type=float, default=300)
        elif name == "memory-commit":
            command.add_argument("--request", type=Path, required=True)
        elif name == "memory-export":
            command.add_argument("--head")
        elif name == "memory-recover":
            command.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        if args.command.startswith("memory-"):
            return _memory_command(args)
        if args.command.startswith("validate-"):
            validator = {"validate-policy": validate_policy, "validate-math": validate_math_memory,
                         "validate-report": validate_verification_output}[args.command]
            result = validator(read_json(args.path))
            print(canonical_json(result).decode())
            return 0 if result["valid"] else 1
        if args.command == "init":
            initialize(args.private_root)
            print('{"status":"initialized","domain":"fixture-only-publication"}')
        elif args.command == "issue":
            # Human/controller action only; never an HTTP/MCP tool. Refuse to
            # overwrite any existing credential. The value never reaches stdout.
            authority = Authority(args.private_root / "authority")
            token_dir = PrivateDirectory(args.token_file.parent)
            if args.token_file.exists() or args.token_file.is_symlink():
                raise ValueError("token target already exists")
            authority.register_run(owner=args.owner, run_id=args.run_id, domain=args.domain)
            token = authority.issue(owner=args.owner, run_id=args.run_id, domain=args.domain,
                                    role="generation", actions=["verify", "status"], ttl_seconds=args.ttl)
            token_dir.write_bytes(args.token_file.name, token.encode("ascii"), exclusive=True)
            print('{"status":"credential-written"}')
        elif args.command == "serve":
            from .api import create_app
            import uvicorn
            uvicorn.run(create_app(args.private_root, driver=args.driver), host=args.host,
                        port=args.port, workers=1, access_log=False, log_level="warning")
        elif args.command == "fixture-demo":
            from .controller import Controller
            initialize(args.private_root)
            controller = Controller(args.private_root, driver="fixture")
            run_id = "fixture-" + uuid.uuid4().hex
            controller.authority.register_run(owner="local", run_id=run_id, domain="fixture")
            token = controller.authority.issue(owner="local", run_id=run_id, domain="fixture",
                role="generation", actions=["verify", "status"])
            result = controller.verify(token, {"owner": "local", "run_id": run_id, "domain": "fixture",
                "request_id": uuid.uuid4().hex, "statement": "Synthetic fixture statement.",
                "proof": "Synthetic fixture body; not a mathematical proof."})
            print(canonical_json(result).decode())
        elif args.command == "status":
            from .client import local_endpoint, read_local_credential
            import httpx
            endpoint = local_endpoint(args.endpoint).removesuffix("/verify") + "/status"
            payload = {}
            for key in ("statement", "proof"):
                path = getattr(args, key)
                if path is not None:
                    payload[key] = path.read_text(encoding="utf-8")
            with httpx.Client(trust_env=False, follow_redirects=False, timeout=10) as client:
                response = client.post(endpoint, json=payload, headers={
                    "Authorization": "Bearer " + read_local_credential(args.token_file)})
                response.raise_for_status()
                result = strict_json_loads(response.content)
            print(canonical_json(result).decode())
            return 0 if result.get("production_completed") is True else 1
        elif args.command == "sandbox-probe":
            from .worker import SandboxWorker
            result = SandboxWorker().probe()
            print(canonical_json(result).decode())
            return 0 if result["status"] == "available" else 2
        elif args.command == "site-sync":
            from .site_export import export_drafts
            print(canonical_json(export_drafts(args.results, args.content)).decode())
        elif args.command == "launch":
            from .launcher import inspect_run
            result = inspect_run(args.generation_root, args.problem)
            print(canonical_json(result).decode())
            if args.dry_run:
                return 0
            print("Live generation is unavailable until an isolated provider broker is implemented. "
                  "No Codex process was started.", file=sys.stderr)
            return 69
        return 0
    except (OSError, ValueError, RecursionError, AuthorizationError) as exc:
        # Do not dump payloads, tokens, traceback locals, or provider responses.
        print("Controller operation rejected (" + type(exc).__name__ + ").", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
