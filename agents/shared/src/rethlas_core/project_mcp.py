"""Operator-configured stdio MCP over managed project memory; never legacy writes."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .private_fs import canonical_json
from .managed_memory import ManagedMemory
from .project_client import connect, configured_embedding, configured_text, grant
from .retrieval import RetrievalService


MAX_TOOL_RESULT_BYTES = 16777216


def _safe(operation):
    try:
        result = operation()
        if len(canonical_json(result)) > MAX_TOOL_RESULT_BYTES:
            raise ValueError("managed MCP result exceeds byte capacity")
        return result
    except Exception as exc:
        from .memory_store import MemoryIndeterminate
        from .provider import ProviderIndeterminate
        from .validity import ValidityIndeterminate
        # Paths, input text and provider/credential exceptions are not tool output.
        status = "indeterminate" if isinstance(exc, (MemoryIndeterminate, ProviderIndeterminate, ValidityIndeterminate)) else "rejected"
        return {"status": status, "error": type(exc).__name__}


def build_mcp_app(config_path):
    # Reject absent/invalid operator configuration before optional transport
    # imports can emit dependency diagnostics onto the CLI error channel.
    config, service, retrieval, memory = connect(Path(config_path), providers=False)
    from mcp.server.fastmcp import FastMCP

    def searchable_memory():
        # Offline tools do not initialize live providers. Search still uses the
        # configured backend or rejects; provider failure is not a lexical switch.
        current = RetrievalService(service, provider=configured_embedding(service, config))
        return ManagedMemory(service, current, reader=memory.reader, writer=memory.writer, indexer=memory.indexer)

    def research_pipeline(*, search, review_only=False):
        from .research import ResearchPipeline
        current = searchable_memory().retrieval if search else retrieval
        if config["version"] == "research-client-1":
            return ResearchPipeline(service, current, configured_text(service, config))
        return ResearchPipeline(service, current,
            generation_provider=None if review_only else configured_text(service, config, "generation"),
            review_provider=configured_text(service, config, "review"))

    app = FastMCP("rethlas-managed-project", instructions=(
        "One authorized project visibility scope. Tools use the committed JSON Graph and Markdown. "
        "Vector search is a rebuildable projection, not mathematical authority. "
        "Current input eligibility is not formal or human mathematical verification. "
        "No legacy JSONL writes, credential issuance, external theorem search or arbitrary paths are available."
    ))

    @app.tool(name="memory_init")
    def memory_init(problem_id: str, meta: dict | None = None) -> dict:
        """Select an exact problem identity; optional metadata is a candidate event."""
        return _safe(lambda: memory.init(problem_id, meta))

    @app.tool(name="memory_append")
    def memory_append(problem_id: str, channel: str, record: dict,
                      request_id: str | None = None) -> dict:
        """Append candidate memory to the sole graph authority; never write JSONL."""
        return _safe(lambda: memory.append(problem_id, channel, record, request_id=request_id))

    @app.tool(name="memory_search")
    def memory_search(problem_id: str, query: str, channels: list[str] | None = None,
                      limit_per_channel: int = 10) -> dict:
        """Search managed records for this exact problem; report retrieval mode."""
        return _safe(lambda: searchable_memory().search(problem_id, query, channels, limit_per_channel))

    @app.tool(name="project_search")
    def project_search(query: str, top_k: int = 10, filters: dict | None = None) -> dict:
        """Search typed mathematical knowledge in this authorized project only."""
        return _safe(lambda: searchable_memory().retrieval.search(memory.reader["token"], query,
            run_id=memory.reader["run_id"], top_k=top_k, filters=filters))

    @app.tool(name="branch_update")
    def branch_update(problem_id: str, branch_id: str, state: dict) -> dict:
        """Record branch progress, not a join or implicit shared graph merge."""
        return _safe(lambda: memory.branch_update(problem_id, branch_id, state))

    @app.tool(name="get_revision")
    def get_revision(ref: dict) -> dict:
        """Read an exact committed revision with its complete necessary context."""
        return _safe(lambda: memory.get_revision(ref))

    @app.tool(name="expand")
    def expand(refs: list[dict], relation_types: list[str] | None = None, depth: int = 1) -> dict:
        """Bounded typed navigation; not a substitute for complete premise closure."""
        return _safe(lambda: memory.expand(refs, relation_types, depth))

    @app.tool(name="research_run")
    def research_run(request_id: str, question: str, selected_refs: list[dict] | None = None,
                     reference_refs: list[dict] | None = None, question_only: bool = False) -> dict:
        """Generate and freshly review candidate research using approved configuration."""
        def run():
            selected = grant(config, "runner", required=True)
            pipeline = research_pipeline(search=selected_refs is None and not question_only)
            arguments = {"request_id": request_id, "question": question, "selected_refs": selected_refs}
            if reference_refs is not None:
                arguments["reference_refs"] = reference_refs
            if question_only:
                arguments["question_only"] = True
            return pipeline.run(selected, **arguments)
        return _safe(run)

    @app.tool(name="research_revise")
    def research_revise(request_id: str, parent_request_id: str, instructions: str) -> dict:
        """Explicitly draft a new candidate from prior review; never automatically loop."""
        def revise():
            selected = grant(config, "runner", required=True)
            pipeline = research_pipeline(search=True)
            return pipeline.revise(selected, request_id=request_id, parent_request_id=parent_request_id,
                                   instructions=instructions)
        return _safe(revise)

    @app.tool(name="research_review")
    def research_review(request_id: str, proof_ref: dict, statement: str) -> dict:
        """Explicit fresh-context review of an exact proof; no generation request."""
        def review():
            selected = grant(config, "runner", required=True)
            pipeline = research_pipeline(search=False, review_only=True)
            return pipeline.review(selected, request_id=request_id, proof_ref=proof_ref, statement=statement)
        return _safe(review)

    @app.tool(name="research_status")
    def research_status(request_id: str) -> dict:
        """Read historical research outcome with a freshly computed current state."""
        def status():
            from .research import ResearchPipeline
            selected = grant(config, "runner", required=True)
            pipeline = ResearchPipeline(service, retrieval, None)
            return pipeline.status(selected, request_id=request_id)
        return _safe(status)

    @app.tool(name="research_history")
    def research_history(limit: int = 32) -> dict:
        """List a bounded read-only projection of this project's research journal."""
        def history():
            from .research import ResearchPipeline
            return ResearchPipeline(service, retrieval, None).history(grant(config, "runner", required=True), limit=limit)
        return _safe(history)

    return app


class _Parser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)

    def error(self, message):
        raise ValueError("invalid managed MCP arguments")


def main(argv=None):
    try:
        parser = _Parser(prog="rethlas-project-mcp")
        parser.add_argument("--config", type=Path, required=True)
        args = parser.parse_args(argv)
        app = build_mcp_app(args.config)
        app.run(transport="stdio")
        return 0
    except Exception as exc:
        print(canonical_json({"status": "rejected", "error": type(exc).__name__}).decode(), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
