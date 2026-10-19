"""Fixed synthetic process driver. No models, tools, credentials, or network."""
from __future__ import annotations

import sys

from .private_fs import canonical_json, strict_json_loads


def main() -> None:
    request = strict_json_loads(sys.stdin.buffer.read(262145))
    if not isinstance(request, dict) or set(request) != {"statement", "proof"}:
        raise ValueError("invalid fixture input")
    if "[fixture:timeout]" in request["proof"]:
        import subprocess
        child = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(30)"])
        child.wait()
    # Deterministic sentinel for exercising a legitimate negative business result.
    finding = {"location": "synthetic fixture", "issue": "Synthetic repair requested"}
    rejected = "[fixture:wrong]" in request["proof"]
    output = {
        "verification_report": {
            "summary": "Synthetic protocol fixture; no mathematical review performed.",
            "critical_errors": [], "gaps": [finding] if rejected else [],
        },
        "verdict": "wrong" if rejected else "correct",
        "repair_hints": "Remove the synthetic fixture error." if rejected else "",
    }
    records = [{"record": {"critical_errors": [], "gaps": [finding]}}] if rejected else []
    sys.stdout.buffer.write(canonical_json({"output": output, "findings": records}))


if __name__ == "__main__":
    main()
