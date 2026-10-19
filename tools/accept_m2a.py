#!/usr/bin/env python3
"""Operator-authorized offline M2a acceptance; preserves each failed/passed run.

This tool EXECUTES the installed Rethlas code, synthetic CLI/socket tests and
owned child processes from https://github.com/frenzymath/Rethlas plus local
changes. Review and explicitly authorize that execution first. It is not a way
around an execution denial. No installation, network provider or old service is
used. Merely writing this tool is not evidence that it has run.
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
import re
import signal
import subprocess
import sys
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/m2a-runtime"


def redact(text: str) -> str:
    return re.sub(r"[A-Za-z0-9_-]{80,}\.[A-Za-z0-9_-]{40,}", "[REDACTED synthetic capability]", text)


def save(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def save_json(path: Path, value) -> None:
    save(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def sources() -> dict:
    paths = []
    for directory in ("agents/shared/src/rethlas_core", "tests", "tools"):
        paths.extend(path for path in (ROOT / directory).rglob("*")
                     if path.is_file() and "__pycache__" not in path.parts and path.suffix in (".py", ".json"))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))}


def history() -> dict:
    paths = list((ROOT / "docs/evidence").rglob("*"))
    paths += [ROOT / "docs" / name for name in ("implementation-status.graph.json", "implementation-status.md",
              "acceptance.md", "security-model.md", "memory-backend-decision.md", "provenance.md")]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file() and not any(
                part.startswith("m2a-") for part in path.relative_to(ROOT).parts)}


def execute(directory: Path, name: str, command: list[str], timeout: int) -> dict:
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT.parent, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        out, err = process.communicate(timeout=timeout)
        code = process.returncode
        status = "passed" if code == 0 else "failed"
    except subprocess.TimeoutExpired:
        code, status = None, "timeout"
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            out, err = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = process.communicate(timeout=5)
    finally:
        # The leader may have exited while a descendant is still alive. Clean
        # only the process group this invocation created, not any old services.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    stdout = redact(out.decode("utf-8", errors="replace"))
    stderr = redact(err.decode("utf-8", errors="replace"))
    save(directory / (name + ".stdout.log"), stdout)
    save(directory / (name + ".stderr.log"), stderr)
    combined = stdout + "\n" + stderr
    match = re.search(r"Ran ([0-9]+) tests? in ([0-9.]+)s", combined)
    return {"name": name, "command": command, "cwd": str(ROOT.parent), "exit_code": code,
            "result": status, "elapsed_seconds": round(time.monotonic() - started, 3),
            "tests": int(match.group(1)) if match else None,
            "unittest_seconds": float(match.group(2)) if match else None,
            "raw_stdout": name + ".stdout.log", "raw_stderr": name + ".stderr.log",
            "capabilities_redacted": True}


def independent_cli(directory: Path) -> dict:
    test_path = ROOT / "tests/test_memory_cli.py"
    spec = importlib.util.spec_from_file_location("m2a_cli_acceptance", test_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    stream = io.StringIO()
    started = time.monotonic()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(module))
    transcript = redact(stream.getvalue())
    save(directory / "cli-tests.log", transcript)
    records = json.loads(redact(json.dumps(module.COMMAND_RECORDS, ensure_ascii=False)))
    save_json(directory / "cli-commands.json", records)
    return {"name": "independent_real_memory_cli", "tests": result.testsRun,
            "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
            "result": "passed" if result.wasSuccessful() and not result.skipped else "failed",
            "exit_code": 0 if result.wasSuccessful() and not result.skipped else 1,
            "elapsed_seconds": round(time.monotonic() - started, 3), "commands": len(records),
            "raw_log": "cli-tests.log", "command_records": "cli-commands.json"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()  # Fixed local suite/paths; no request-selected commands.
    if Path(sys.executable).absolute() != (ROOT / ".venv/bin/python").absolute() or not sys.flags.isolated:
        parser.error("use the installed dedicated .venv/bin/python -I")
    if any(path.is_symlink() for path in (EVIDENCE, *EVIDENCE.parents)):
        parser.error("evidence directory must not contain symlinks")
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        directory = EVIDENCE / f"run-{number:03d}"
        try:
            directory.mkdir()
            break
        except FileExistsError:
            number += 1
    before, frozen = sources(), history()
    save_json(directory / "source-before.json", before)
    steps = []
    commands = [
        ("suite", [sys.executable, "-I", "-B", "-m", "unittest", "discover", "-v", "-s=" + str(ROOT / "tests")], 600),
        ("ledger-m1", [sys.executable, "-I", "-B", str(ROOT / "tools/validate_implementation.py"), "--root", str(ROOT)], 30),
        ("ledger-m2a", [sys.executable, "-I", "-B", str(ROOT / "tools/validate_implementation.py"), "--root", str(ROOT), "--milestone", "M2a"], 30),
        ("static-roadmap", [sys.executable, "-I", "-B", str(ROOT.parent / "tools/validate_graphs.py"),
             "--analysis", str(ROOT.parent / "analysis"), "--repo", "Rethlas=" + str(ROOT),
             "--repo", "Re-CTM=" + str(ROOT.parent / "Re-CTM")], 120),
        ("roadmap-regressions", [sys.executable, "-I", "-B", str(ROOT.parent / "tests/test_validate_graphs.py"),
             "--validator", str(ROOT.parent / "tools/validate_graphs.py")], 300),
        ("offline-experiment-regressions", [sys.executable, "-I", "-B",
             str(ROOT.parent / "tests/test_openviking_benchmark.py")], 60),
    ]
    for name, command, timeout in commands:
        steps.append(execute(directory, name, command, timeout))
    steps.append(independent_cli(directory))
    after, historical_after = sources(), history()
    save_json(directory / "source-after.json", after)
    changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
    historical_changed = sorted(name for name in set(frozen) | set(historical_after)
                                if frozen.get(name) != historical_after.get(name))
    successful = all(step["result"] == "passed" for step in steps) and not changed and not historical_changed
    # A skipped target-host boundary test does not count as accepted.
    suite_text = (directory / "suite.stderr.log").read_text(encoding="utf-8")
    if "skipped=" in suite_text or " ... skipped " in suite_text:
        successful = False
    report = {"recorded_at": datetime.now(timezone.utc).isoformat(), "milestone": "M2a",
              "result": "passed" if successful else "failed", "exit_code": 0 if successful else 1,
              "runner_command": [sys.executable, *sys.orig_argv[1:]], "steps": steps,
              "source_changed_during_run": changed, "m1_history_changed": historical_changed,
              "prior_runs_retained": number - 1, "external_model_requests": 0,
              "human_review": "pending", "mathematical_correctness_proved": False,
              "limitations": ["SIGKILL and injected fsync errors are not arbitrary power-loss certification.",
                              "The ID-only projection is not BGE retrieval or T10 completion.",
                              "Passing local automation does not complete the original fifteen roadmap tasks or human review."]}
    save_json(directory / "report.json", report)
    print(json.dumps({"result": report["result"], "directory": str(directory)}, ensure_ascii=False))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
