"""Fail-closed legacy MCP startup until managed live workers are implemented."""
from .worker import WorkerUnavailable


def require_managed_worker() -> None:
    # An environment variable or model-supplied nonce is not isolation evidence.
    # M1 deliberately provides no unmanaged-model opt-in or bare-process fallback.
    raise WorkerUnavailable(
        "Live model MCP startup is disabled in M1. Use controller fixture/validation "
        "commands; an isolated provider broker is required for live research."
    )
