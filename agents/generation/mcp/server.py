from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

AGENTS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AGENTS_ROOT))
from memory_graph import MemoryGraph

try:
    from fastmcp import FastMCP
except ImportError:  # pragma: no cover - dependency should be installed via requirements
    FastMCP = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_ROOT = REPO_ROOT / "memory"

THEOREM_SEARCH_URL = "https://leansearch.net/thm/search"
THEOREM_SEARCH_TASK = (
    "Given a math statement, retrieve useful references, such as theorems, "
    "lemmas, and definitions, that are useful for solving the given problem."
)

VERIFY_PROOF_URL = "http://127.0.0.1:8091/verify"

CHANNELS = (
    "immediate_conclusions",
    "toy_examples",
    "counterexamples",
    "big_decisions",
    "subgoals",
    "proof_steps",
    "failed_paths",
    "verification_reports",
    "branch_states",
    "events",
)


def _memory_store() -> MemoryGraph:
    return MemoryGraph(MEMORY_ROOT, api_lock_path=AGENTS_ROOT / ".embedding-api.lock")


def _sanitize_problem_component(raw: str) -> str:
    cleaned = re.sub(r"\s+", "_", raw.strip())
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("._")
    return cleaned


def sanitize_problem_id(raw: str) -> str:
    """Return a safe problem id while preserving relative path components."""
    normalized = raw.strip().replace("\\", "/")
    parts: List[str] = []
    for part in normalized.split("/"):
        stripped = part.strip()
        if stripped in {"", "."}:
            continue
        if stripped == "..":
            raise ValueError("problem_id must not contain '..' path components")
        cleaned = _sanitize_problem_component(stripped)
        if cleaned:
            parts.append(cleaned)
    return "/".join(parts) or "problem"

def build_problem_id(source: str, identifier: str) -> str:
    return sanitize_problem_id(f"{source}_{identifier}")


def _resolve_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _problem_dir(problem_id: str) -> Path:
    sanitized_problem_id = sanitize_problem_id(problem_id)
    problem_dir = (MEMORY_ROOT / sanitized_problem_id).resolve()
    memory_root = MEMORY_ROOT.resolve()
    if not problem_dir.is_relative_to(memory_root):
        raise ValueError("problem_id resolves outside memory root")
    return problem_dir


def _validate_channel(channel: str) -> None:
    if channel not in CHANNELS:
        allowed = ", ".join(sorted(CHANNELS))
        raise ValueError(f"Unknown channel '{channel}'. Allowed channels: {allowed}")


def _append_result(problem_id: str, node: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "ok",
        "problem_id": problem_id,
        "channel": node["channel"],
        "path": str(_memory_store().database_path(problem_id)),
        "entry": {
            "timestamp_utc": node["time"],
            "channel": node["channel"],
            "record": node["content"],
        },
        "node_id": node["id"],
    }


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


def verify_proof_service(
    statement: str,
    proof: str,
    endpoint: str = VERIFY_PROOF_URL,
    timeout_seconds: int = 3600,
) -> Dict[str, Any]:
    if not statement.strip():
        raise ValueError("statement must be non-empty")
    if not isinstance(proof, str):
        raise ValueError("proof must be markdown text")
    if not proof.strip():
        raise ValueError("proof markdown must be non-empty")

    payload = {
        "statement": statement,
        "proof": proof,
    }

    response = requests.post(endpoint, json=payload, timeout=timeout_seconds)
    response.raise_for_status()

    try:
        body = response.json()
    except ValueError as exc:
        raise ValueError("verification service returned non-JSON response") from exc

    if not isinstance(body, dict):
        raise ValueError("verification service must return a JSON object")

    return {
        "statement": statement,
        "verification_report": body.get("verification_report", {}),
        "verdict": body.get("verdict"),
        "repair_hints": body.get("repair_hints"),
        "endpoint": endpoint,
    }


def memory_init(
    problem_id: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_problem_id = sanitize_problem_id(problem_id)
    result = _memory_store().init(resolved_problem_id, {"problem_id": resolved_problem_id, **(meta or {})})
    result.pop("scope")
    return {"problem_id": resolved_problem_id, **result, "channels": list(CHANNELS)}


def memory_append(
    problem_id: str,
    channel: str,
    record: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _validate_channel(channel)
    resolved_problem_id = sanitize_problem_id(problem_id)
    node = _memory_store().append(resolved_problem_id, channel, record, metadata=metadata)
    return _append_result(resolved_problem_id, node)


def memory_search(
    problem_id: str,
    query: str,
    channels: Optional[List[str]] = None,
    limit_per_channel: int = 10,
    top_k: int = 10,
) -> Dict[str, Any]:
    resolved_problem_id = sanitize_problem_id(problem_id)
    search_channels = [name for name in CHANNELS if name != "events"] if channels is None else channels
    for channel in search_channels:
        _validate_channel(channel)
    result = _memory_store().search(
        resolved_problem_id, query, channels=search_channels,
        top_k=top_k, limit_per_channel=limit_per_channel,
    )
    result.pop("scope")
    return {"problem_id": resolved_problem_id, **result}


def memory_get(
    problem_id: str,
    node_ids: List[str],
    include_dependencies: bool = True,
) -> Dict[str, Any]:
    resolved_problem_id = sanitize_problem_id(problem_id)
    result = _memory_store().get(resolved_problem_id, node_ids, include_dependencies=include_dependencies)
    result.pop("scope")
    return {"problem_id": resolved_problem_id, **result}


def branch_update(
    problem_id: str,
    branch_id: str,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    resolved_problem_id = sanitize_problem_id(problem_id)
    node = _memory_store().branch_update(resolved_problem_id, branch_id, state)
    return _append_result(resolved_problem_id, node)


def build_mcp_app() -> Optional[Any]:
    if FastMCP is None:
        return None

    app = FastMCP("reasoning-agent")

    @app.tool(name="search_arxiv_theorems")
    def _tool_search_arxiv_theorems(
        query: str,
        num_results: int = 10,
    ) -> Dict[str, Any]:
        return search_arxiv_theorems(query=query, num_results=num_results)

    @app.tool(name="verify_proof_service")
    def _tool_verify_proof_service(
        statement: str,
        proof: str,
    ) -> Dict[str, Any]:
        return verify_proof_service(statement=statement, proof=proof)

    @app.tool(name="memory_init")
    def _tool_memory_init(
        problem_id: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return memory_init(problem_id=problem_id, meta=meta)

    @app.tool(name="memory_append")
    def _tool_memory_append(
        problem_id: str,
        channel: str,
        record: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return memory_append(problem_id=problem_id, channel=channel, record=record, metadata=metadata)

    @app.tool(name="memory_search")
    def _tool_memory_search(
        problem_id: str,
        query: str,
        channels: Optional[List[str]] = None,
        limit_per_channel: int = 10,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        return memory_search(
            problem_id=problem_id,
            query=query,
            channels=channels,
            limit_per_channel=limit_per_channel,
            top_k=top_k,
        )

    @app.tool(name="memory_get")
    def _tool_memory_get(
        problem_id: str,
        node_ids: List[str],
        include_dependencies: bool = True,
    ) -> Dict[str, Any]:
        return memory_get(
            problem_id=problem_id, node_ids=node_ids, include_dependencies=include_dependencies,
        )

    @app.tool(name="branch_update")
    def _tool_branch_update(
        problem_id: str,
        branch_id: str,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        return branch_update(problem_id=problem_id, branch_id=branch_id, state=state)

    return app


APP = build_mcp_app()


def main() -> None:
    if APP is None:
        raise SystemExit(
            "fastmcp is not installed. Install requirements from mcp/requirements.txt first."
        )
    APP.run()


if __name__ == "__main__":
    main()
