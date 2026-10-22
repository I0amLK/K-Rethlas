from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

AGENTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENTS_ROOT))
from memory_graph import MemoryGraph

try:
    from jsonschema import Draft202012Validator
except ImportError:  # pragma: no cover - dependency managed by requirements
    Draft202012Validator = None  # type: ignore[assignment]

try:
    from fastmcp import FastMCP
except ImportError:  # pragma: no cover - dependency managed by requirements
    FastMCP = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_ROOT = REPO_ROOT / "memory"
RESULTS_ROOT = REPO_ROOT / "results"
SCHEMA_PATH = REPO_ROOT / "schemas" / "verification_output.schema.json"

THEOREM_SEARCH_URL = "https://leansearch.net/thm/search"
THEOREM_SEARCH_TASK = (
    "Given a math statement, retrieve useful references, such as theorems, "
    "lemmas, and definitions, that are useful for solving the given problem."
)

CHANNELS = (
    "statement_checks",
    "reference_checks",
    "verification_reports",
    "failed_checks",
    "events",
)


def _memory_store() -> MemoryGraph:
    return MemoryGraph(MEMORY_ROOT, api_lock_path=AGENTS_ROOT / ".embedding-api.lock")


def sanitize_run_id(raw: str) -> str:
    cleaned = re.sub(r"\s+", "_", str(raw).strip())
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned or "run"


def _run_dir(run_id: str) -> Path:
    return MEMORY_ROOT / sanitize_run_id(run_id)


def _validate_channel(channel: str) -> None:
    if channel not in CHANNELS:
        allowed = ", ".join(sorted(CHANNELS))
        raise ValueError(f"Unknown channel '{channel}'. Allowed channels: {allowed}")


def search_arxiv_theorems(
    query: str,
    num_results: int = 10,
    endpoint: str = THEOREM_SEARCH_URL,
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    if not query.strip():
        raise ValueError("query must be non-empty")
    if num_results <= 0:
        raise ValueError("num_results must be > 0")

    payload = {
        "query": query,
        "task": THEOREM_SEARCH_TASK,
        "num_results": num_results,
    }

    response = requests.post(endpoint, json=payload, timeout=timeout_seconds)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError("The theorem endpoint must return a JSON list")

    normalized: List[Dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "title": str(item.get("title", "")),
                "theorem": str(item.get("theorem", "")),
                "arxiv_id": str(item.get("arxiv_id", "")),
                "theorem_id": str(item.get("theorem_id", "")),
            }
        )

    return {
        "query": query,
        "count": len(normalized),
        "results": normalized,
        "endpoint": endpoint,
    }


def memory_init(run_id: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    resolved_run_id = sanitize_run_id(run_id)
    result = _memory_store().init(resolved_run_id, {"run_id": resolved_run_id, **(meta or {})})
    result.pop("scope")
    return {"run_id": resolved_run_id, **result, "channels": list(CHANNELS)}


def memory_append(
    run_id: str,
    channel: str,
    record: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _validate_channel(channel)
    resolved_run_id = sanitize_run_id(run_id)
    node = _memory_store().append(resolved_run_id, channel, record, metadata=metadata)
    return {
        "status": "ok",
        "run_id": resolved_run_id,
        "channel": channel,
        "path": str(_memory_store().database_path(resolved_run_id)),
        "entry": {
            "timestamp_utc": node["time"],
            "channel": node["channel"],
            "record": node["content"],
        },
        "node_id": node["id"],
    }


def memory_query(
    run_id: str,
    channel: str,
    filters: Optional[Dict[str, Any]] = None,
    contains: Optional[str] = None,
    limit: int = 100,
    reverse: bool = True,
) -> Dict[str, Any]:
    _validate_channel(channel)
    resolved_run_id = sanitize_run_id(run_id)
    result = _memory_store().query(
        resolved_run_id, channel, filters=filters, contains=contains, limit=limit, reverse=reverse,
    )
    result.pop("scope")
    return {"run_id": resolved_run_id, **result}


def memory_search(
    run_id: str,
    query: str,
    channels: Optional[List[str]] = None,
    limit_per_channel: int = 10,
    top_k: int = 10,
) -> Dict[str, Any]:
    resolved_run_id = sanitize_run_id(run_id)
    search_channels = [name for name in CHANNELS if name != "events"] if channels is None else channels
    for channel in search_channels:
        _validate_channel(channel)
    result = _memory_store().search(
        resolved_run_id, query, channels=search_channels,
        top_k=top_k, limit_per_channel=limit_per_channel,
    )
    result.pop("scope")
    return {"run_id": resolved_run_id, **result}


def memory_get(
    run_id: str,
    node_ids: List[str],
    include_dependencies: bool = True,
) -> Dict[str, Any]:
    resolved_run_id = sanitize_run_id(run_id)
    result = _memory_store().get(resolved_run_id, node_ids, include_dependencies=include_dependencies)
    result.pop("scope")
    return {"run_id": resolved_run_id, **result}


def _load_schema() -> Dict[str, Any]:
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema not found: {SCHEMA_PATH}")
    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(schema, dict):
        raise ValueError("schema must be a JSON object")
    return schema


def validate_verification_output(payload: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []

    try:
        schema = _load_schema()
    except Exception as exc:
        return {"valid": False, "errors": [f"schema load failed: {exc}"]}

    if Draft202012Validator is None:
        errors.append("jsonschema dependency is missing; cannot validate schema")
    else:
        validator = Draft202012Validator(schema)
        for error in validator.iter_errors(payload):
            path = ".".join(str(part) for part in error.path)
            if path:
                errors.append(f"schema error at '{path}': {error.message}")
            else:
                errors.append(f"schema error: {error.message}")

    report = payload.get("verification_report")
    verdict = payload.get("verdict")
    repair_hints = payload.get("repair_hints")

    critical_errors = []
    gaps = []
    if isinstance(report, dict):
        if isinstance(report.get("critical_errors"), list):
            critical_errors = report["critical_errors"]
        if isinstance(report.get("gaps"), list):
            gaps = report["gaps"]

    has_any_finding = len(critical_errors) + len(gaps) > 0

    if verdict == "correct":
        if has_any_finding:
            errors.append("verdict='correct' is invalid when critical_errors or gaps are non-empty")
        if repair_hints != "":
            errors.append("repair_hints must be empty when verdict='correct'")
    elif verdict == "wrong":
        if not has_any_finding:
            errors.append("verdict='wrong' requires at least one critical error or gap")
        if not isinstance(repair_hints, str) or not repair_hints.strip():
            errors.append("repair_hints must be non-empty when verdict='wrong'")
    else:
        errors.append("verdict must be 'correct' or 'wrong'")

    return {"valid": len(errors) == 0, "errors": errors}


def write_verification_output(run_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    validation = validate_verification_output(payload)
    if not validation["valid"]:
        raise ValueError("verification output validation failed: " + "; ".join(validation["errors"]))

    resolved_run_id = sanitize_run_id(run_id)
    output_dir = RESULTS_ROOT / resolved_run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "verification.json"

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
        "status": "ok",
        "run_id": resolved_run_id,
        "output_path": str(output_path),
    }


def build_mcp_app() -> Optional[Any]:
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
    def _tool_memory_append(
        run_id: str,
        channel: str,
        record: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return memory_append(run_id=run_id, channel=channel, record=record, metadata=metadata)

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

    @app.tool(name="memory_search")
    def _tool_memory_search(
        run_id: str,
        query: str,
        channels: Optional[List[str]] = None,
        limit_per_channel: int = 10,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        return memory_search(
            run_id=run_id, query=query, channels=channels,
            limit_per_channel=limit_per_channel, top_k=top_k,
        )

    @app.tool(name="memory_get")
    def _tool_memory_get(
        run_id: str,
        node_ids: List[str],
        include_dependencies: bool = True,
    ) -> Dict[str, Any]:
        return memory_get(run_id=run_id, node_ids=node_ids, include_dependencies=include_dependencies)

    @app.tool(name="validate_verification_output")
    def _tool_validate_verification_output(payload: Dict[str, Any]) -> Dict[str, Any]:
        return validate_verification_output(payload=payload)

    @app.tool(name="write_verification_output")
    def _tool_write_verification_output(run_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return write_verification_output(run_id=run_id, payload=payload)

    return app


APP = build_mcp_app()


def main() -> None:
    if APP is None:
        raise SystemExit(
            "fastmcp is not installed. Install dependencies from mcp/requirements.txt first."
        )
    APP.run()


if __name__ == "__main__":
    main()
