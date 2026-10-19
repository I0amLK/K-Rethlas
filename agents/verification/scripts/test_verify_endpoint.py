#!/usr/bin/env python3
"""Manual synthetic local verifier check. Requires an operator-issued capability."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from rethlas_core.client import verify_via_local_service
from rethlas_core.private_fs import canonical_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8091/verify")
    args = parser.parse_args()
    try:
        result = verify_via_local_service(
            "Synthetic endpoint-test statement.", "Synthetic body; no mathematical review.",
            endpoint=args.endpoint, token_file=args.token_file, owner=args.owner,
            run_id=args.run_id, domain="fixture", request_id="manual-fixture-check",
        )
        print(canonical_json(result).decode())
    except (OSError, ValueError):
        print("Local verifier check rejected; no fallback was attempted.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
