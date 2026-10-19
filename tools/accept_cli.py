#!/usr/bin/env python3
"""Run bounded local CLI/socket acceptance and preserve redacted raw evidence.

Use the already-installed Rethlas runtime; this tool never installs dependencies,
starts a model/old service, or contacts anything beyond its own loopback fixture.
Evidence is append-only per run, including failed first attempts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]


def exclusive_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path,
                        default=ROOT / "docs/evidence/m1-cli-v1")
    args = parser.parse_args()
    evidence = args.evidence_dir.absolute()
    allowed = ROOT / "docs/evidence/m1-cli-v1"
    if evidence != allowed or any(part.is_symlink() for part in (evidence, *evidence.parents)):
        parser.error("evidence must be the non-symlink docs/evidence/m1-cli-v1 directory")
    evidence.mkdir(exist_ok=True)
    number = 1
    while (evidence / f"run-{number:03d}-raw.json").exists() or (evidence / f"run-{number:03d}-report.json").exists():
        number += 1
    test_path = ROOT / "tests/test_cli.py"
    spec = importlib.util.spec_from_file_location("rethlas_cli_acceptance", test_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    stream = io.StringIO()
    started = time.monotonic()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    elapsed = round(time.monotonic() - started, 3)
    # Apply redaction after the full run, including capabilities that became
    # known after a command had returned. No authority/receipt keys are read.
    transcript = module.redact(stream.getvalue())
    raw = json.loads(module.redact(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runner_command": [sys.executable, *sys.orig_argv[1:]],
        "runner_cwd": os.getcwd(), "python": sys.version,
        "commands": module.COMMAND_RECORDS, "unittest_output": transcript,
        "exit_code": 0 if result.wasSuccessful() else 1,
        "elapsed_seconds": elapsed,
    }, ensure_ascii=False)))
    failures = [{"test": test.id(), "traceback": module.redact(detail)}
                for test, detail in result.failures + result.errors]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "M1 local synthetic independent CLI, wrappers, and owned loopback fixture socket",
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "successful": result.wasSuccessful(), "elapsed_seconds": elapsed,
        "command_count": len(module.COMMAND_RECORDS),
        "observations": module.OBSERVATIONS, "failure_details": failures,
        "files_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (test_path, Path(__file__))},
        "limitations": [
            "Fixture verdict=correct tests protocol behavior only; production_completed remains false and no mathematical review is claimed.",
            "sandbox-probe blocked is recorded as a target-host limitation, not isolation success or permission to fall back.",
            "No live model, Codex, old service, OpenViking, external network, package installation, real credential, or actual site content/theme is used.",
            "This focused suite does not replace the coordinator's HTTP/publication/security/full-suite acceptance.",
        ],
        "redaction": "Only synthetic local capability files are read; capability values are omitted/redacted and authority/receipt key files are never read by the harness.",
        "prior_runs_retained": number - 1,
    }
    exclusive_json(evidence / f"run-{number:03d}-raw.json", raw)
    exclusive_json(evidence / f"run-{number:03d}-report.json", report)
    print(transcript, end="")
    print(json.dumps({"tests_run": result.testsRun, "successful": result.wasSuccessful(),
                      "raw": str(evidence / f"run-{number:03d}-raw.json"),
                      "report": str(evidence / f"run-{number:03d}-report.json")}, ensure_ascii=False))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
