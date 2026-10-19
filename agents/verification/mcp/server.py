from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from rethlas_core.private_fs import strict_json_loads
from rethlas_core.verification import (
    coverage_errors,
    normalize_findings,
    validate_verification_output,
)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - dependency managed by requirements
    FastMCP = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_ROOT = REPO_ROOT / "memory"
RESULTS_ROOT = REPO_ROOT / "results"

THEOREM_SEARCH_URL = "https://leansearch.net/thm/search"
THEOREM_SEARCH_TASK = (
    "Given a math statement, retrieve useful references, such as theorems, "
    "lemmas, and definitions, that are useful for solving the given problem."
)

CHANNEL_FILES: Dict[str, str] = {
    "statement_checks": "statement_checks.jsonl",
    "reference_checks": "reference_checks.jsonl",
    "verification_reports": "verification_reports.jsonl",
    "failed_checks": "failed_checks.jsonl",
    "events": "events.jsonl",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_run_id(raw: str) -> str:
    cleaned = re.sub(r"\s+", "_", str(raw).strip())
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or "run"


def _run_dir(run_id: str) -> Path:
    return MEMORY_ROOT / sanitize_run_id(run_id)


def _channel_path(run_id: str, channel: str) -> Path:
    if channel not in CHANNEL_FILES:
        allowed = ", ".join(sorted(CHANNEL_FILES))
        raise ValueError(f"Unknown channel '{channel}'. Allowed channels: {allowed}")
    return _run_dir(run_id) / CHANNEL_FILES[channel]


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = strict_json_loads(line)
            if not isinstance(payload, dict):
                raise ValueError("verification JSONL entries must be objects")
            yield payload


def search_arxiv_theorems(
    query: str,
    num_results: int = 10,
    endpoint: str = THEOREM_SEARCH_URL,
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    raise PermissionError(
        "External theorem search is disabled until a trusted, run-bound egress broker is configured"
    )


def memory_init(run_id: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    sanitized_run_id = sanitize_run_id(run_id)
    run_dir = _run_dir(sanitized_run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    created_files: Dict[str, str] = {}
    for channel, filename in CHANNEL_FILES.items():
        channel_path = run_dir / filename
        channel_path.touch(exist_ok=True)
        created_files[channel] = str(channel_path)

    meta_path = run_dir / "meta.json"
    existing_meta: Dict[str, Any] = {}
    if meta_path.exists() and meta_path.stat().st_size > 0:
        with meta_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
            if isinstance(loaded, dict):
                existing_meta = loaded

    merged_meta: Dict[str, Any] = {
        "run_id": sanitized_run_id,
        "created_at_utc": existing_meta.get("created_at_utc", _utc_now()),
        "updated_at_utc": _utc_now(),
    }
    merged_meta.update(existing_meta)
    if meta:
        merged_meta.update(meta)

    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(merged_meta, handle, indent=2, ensure_ascii=False)

    return {
        "run_id": sanitized_run_id,
        "memory_dir": str(run_dir),
        "meta_path": str(meta_path),
        "channels": created_files,
    }


def memory_append(run_id: str, channel: str, record: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("record must be a JSON object")

    resolved_run_id = sanitize_run_id(run_id)
    memory_init(resolved_run_id)

    entry = {
        "timestamp_utc": _utc_now(),
        "channel": channel,
        "record": record,
    }
    target = _channel_path(resolved_run_id, channel)
    _append_jsonl(target, entry)

    if channel != "events":
        _append_jsonl(
            _channel_path(resolved_run_id, "events"),
            {
                "timestamp_utc": _utc_now(),
                "event_type": "memory_append",
                "channel": channel,
            },
        )

    return {
        "status": "ok",
        "run_id": resolved_run_id,
        "channel": channel,
        "path": str(target),
        "entry": entry,
    }


def memory_query(
    run_id: str,
    channel: str,
    filters: Optional[Dict[str, Any]] = None,
    contains: Optional[str] = None,
    limit: int = 100,
    reverse: bool = True,
) -> Dict[str, Any]:
    if limit <= 0:
        raise ValueError("limit must be > 0")

    resolved_run_id = sanitize_run_id(run_id)
    path = _channel_path(resolved_run_id, channel)
    items = list(_iter_jsonl(path))

    if filters:
        filtered: List[Dict[str, Any]] = []
        for item in items:
            if all(item.get(key) == value for key, value in filters.items()):
                filtered.append(item)
        items = filtered

    if contains:
        needle = contains.lower()
        items = [item for item in items if needle in json.dumps(item, ensure_ascii=False).lower()]

    if reverse:
        items = list(reversed(items))

    items = items[:limit]
    return {
        "run_id": resolved_run_id,
        "channel": channel,
        "count": len(items),
        "items": items,
    }


def write_verification_output(run_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    validation = validate_verification_output(payload)
    if not validation["valid"]:
        raise ValueError("verification output validation failed: " + "; ".join(validation["errors"]))

    # This worker-side file is a candidate, not a trusted receipt. The controller
    # independently validates its private, append-only registered findings.
    registered = []
    for channel in ("statement_checks", "reference_checks"):
        registered.extend(_iter_jsonl(_channel_path(run_id, channel)))
    errors = coverage_errors(payload, normalize_findings(registered))
    if errors:
        raise ValueError("verification output omits locally recorded findings")
    resolved_run_id = sanitize_run_id(run_id)
    output_dir = RESULTS_ROOT / resolved_run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "verification.candidate.json"

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    memory_init(resolved_run_id)
    memory_append(
        resolved_run_id,
        "verification_reports",
        {
            "event_type": "final_report_written",
            "output_path": str(output_path),
            "verdict": payload.get("verdict"),
        },
    )

    return {
        "status": "candidate_written",
        "production_completed": False,
        "run_id": resolved_run_id,
        "output_path": str(output_path),
    }


def build_mcp_app(managed_config: Optional[Path] = None) -> Optional[Any]:
    if managed_config is not None:
        from rethlas_core.project_mcp import build_mcp_app as managed_app
        return managed_app(managed_config)
    from rethlas_core.mcp_gate import require_managed_worker
    require_managed_worker()
    if FastMCP is None:
        return None

    app = FastMCP("verification_agent")

    @app.tool(name="search_arxiv_theorems")
    def _tool_search_arxiv_theorems(query: str, num_results: int = 10) -> Dict[str, Any]:
        return search_arxiv_theorems(query=query, num_results=num_results)

    @app.tool(name="memory_init")
    def _tool_memory_init(run_id: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return memory_init(run_id=run_id, meta=meta)

    @app.tool(name="memory_append")
    def _tool_memory_append(run_id: str, channel: str, record: Dict[str, Any]) -> Dict[str, Any]:
        return memory_append(run_id=run_id, channel=channel, record=record)

    @app.tool(name="memory_query")
    def _tool_memory_query(
        run_id: str,
        channel: str,
        filters: Optional[Dict[str, Any]] = None,
        contains: Optional[str] = None,
        limit: int = 100,
        reverse: bool = True,
    ) -> Dict[str, Any]:
        return memory_query(
            run_id=run_id,
            channel=channel,
            filters=filters,
            contains=contains,
            limit=limit,
            reverse=reverse,
        )

    @app.tool(name="validate_verification_output")
    def _tool_validate_verification_output(payload: Dict[str, Any]) -> Dict[str, Any]:
        return validate_verification_output(payload=payload)

    @app.tool(name="write_verification_output")
    def _tool_write_verification_output(run_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return write_verification_output(run_id=run_id, payload=payload)

    return app


def main(argv=None) -> None:
    from rethlas_core.project_mcp import _Parser
    from rethlas_core.private_fs import canonical_json
    import sys
    try:
        parser = _Parser(prog="rethlas-verification-mcp")
        parser.add_argument("--managed-config", type=Path)
        args = parser.parse_args(argv)
    except Exception as exc:
        print(canonical_json({"status": "rejected", "error": type(exc).__name__}).decode(), file=sys.stderr)
        raise SystemExit(2) from None
    if args.managed_config is not None:
        from rethlas_core.project_mcp import main as managed_main
        raise SystemExit(managed_main(["--config", str(args.managed_config)]))
    from rethlas_core.mcp_gate import require_managed_worker
    require_managed_worker()
    app = build_mcp_app()
    if app is None:
        raise SystemExit("Install the shared runtime's MCP dependencies first.")
    app.run()


if __name__ == "__main__":
    main()
