"""Trusted test-only stdio host; not a selectable production mock provider."""
from pathlib import Path
import sys
from unittest.mock import patch

from rethlas_core.private_fs import canonical_json, strict_json_loads
from rethlas_core.project_mcp import build_mcp_app
from rethlas_core.provider import TextProvider


MODEL = "xiaoliu/gpt-6.1-sol:max"


def send(endpoint, payload, headers, **limits):
    request = strict_json_loads(payload)
    if "proof_markdown" in request["messages"][0]["content"]:
        content = {"proof_markdown": "Synthetic mock draft: x > 0 and 1 > 0 imply x + 1 > 0.",
                   "dependency_revision_ids": ["math.a.1"]}
    else:
        content = {"verdict": "wrong", "verification_report": {"summary": "Synthetic incomplete argument, not mathematical acceptance.",
            "critical_errors": [], "gaps": [{"location": "synthetic draft", "issue": "Synthetic review requests an explicit ordered-field axiom."}]},
            "repair_hints": "State the ordered-field axiom. This is a mock review, not mathematical acceptance."}
    return canonical_json({"id": "synthetic-stdio-response", "object": "chat.completion", "created": 1,
        "model": MODEL, "choices": [{"index": 0, "message": {"role": "assistant", "content": canonical_json(content).decode()},
                                    "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}})


def main():
    provider = TextProvider(mode="mock", endpoint="https://example.invalid/v1/chat/completions",
        model=MODEL, admit=lambda messages, purpose: None, transport=send)
    with patch("rethlas_core.project_mcp.configured_text", return_value=provider):
        build_mcp_app(Path(sys.argv[1])).run(transport="stdio")


if __name__ == "__main__":
    main()
