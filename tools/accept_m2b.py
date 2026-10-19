#!/usr/bin/env python3
"""Explicitly authorized, synthetic-only offline M2b acceptance.

This runner EXECUTES installed project code, the full local suite, real synthetic
migration CLI processes, existing local socket/isolation checks and fixed legacy
validators. Writing it does not authorize execution or provide runtime evidence.
It installs nothing, calls no provider, touches no old service, preserves failed
runs and never updates milestone documents or declares human/math acceptance.
Use the existing dedicated .venv/bin/python -I -B after separate authorization.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import traceback
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/m2b-runtime"
CACHE_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
HISTORICAL_DOCS = (
    "implementation-status.graph.json", "implementation-status.md", "acceptance.md",
    "security-model.md", "memory-backend-decision.md", "provenance.md",
)
ROADMAP_FILES = (
    "graph.schema.json", "rethlas.graph.json", "rethlas.md", "re-ctm.graph.json",
    "re-ctm.md", "upgrade-todo.graph.json", "upgrade-todo.md",
)
PARENT_SOURCES = (
    "tools/validate_graphs.py", "tests/test_validate_graphs.py",
    "tests/test_openviking_benchmark.py", "tools/benchmark_openviking.py",
    "tools/openviking_eval_bridge.py", "tools/openviking_lifecycle_probe.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact(text: str) -> str:
    # Capabilities are canonical JSON payloads plus a 32-byte base64url HMAC.
    # Generic long dotted identifiers also match unittest IDs: do not erase them.
    text = re.sub(r"(?<![A-Za-z0-9_.-])eyJ[A-Za-z0-9_-]{77,}\.[A-Za-z0-9_-]{43}(?![A-Za-z0-9_.-])",
                  "[REDACTED synthetic capability]", text)
    text = re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1[REDACTED]", text)
    return re.sub(r'''(?i)(["']?(?:token|capability|secret|api[_-]?key)["']?\s*[:=]\s*["'])([^"'\r\n]+)''',
                  r"\1[REDACTED]", text)


def redact_value(value):
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, dict):
        return {key: "[REDACTED]" if key.lower() in {"token", "capability", "secret", "api_key", "api-key"}
                else redact_value(item) for key, item in value.items()}
    return value


def save(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def save_json(path: Path, value) -> None:
    save(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


def regular_files(directory: Path, *, ignore_cache: bool = True) -> list[Path]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("snapshot root must be an existing nonsymlink directory")
    result = []
    for path in directory.rglob("*"):
        relative = path.relative_to(directory)
        if ignore_cache and (any(part in CACHE_PARTS for part in relative.parts) or path.suffix in {".pyc", ".pyo"}):
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("snapshot refuses symlinks: " + str(path))
        if stat.S_ISREG(mode):
            result.append(path)
        elif not stat.S_ISDIR(mode):
            raise ValueError("snapshot refuses nonregular entries: " + str(path))
    return result


def digest(path: Path) -> str:
    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("snapshot input must be a regular nonsymlink file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources() -> dict[str, str]:
    # All noncache regular files, not only .py/.json: package metadata, schemas,
    # Markdown/text/JSONL/binary fixtures and future fixture suffixes are bound.
    paths = []
    for directory in ("agents/shared", "tests", "tools"):
        paths.extend(regular_files(ROOT / directory))
    result = {str(path.relative_to(ROOT)): digest(path) for path in sorted(set(paths))}
    for relative in PARENT_SOURCES:
        result["../" + relative] = digest(ROOT.parent / relative)
    return result


def history() -> dict[str, str]:
    # Freeze all existing M1/M2a docs and evidence, including failed/blocked runs.
    # The new m2b-* evidence (including this run) is deliberately outside history.
    paths = regular_files(ROOT / "docs/evidence", ignore_cache=False)
    paths += regular_files(ROOT / "docs/milestones/m2a", ignore_cache=False)
    paths += [ROOT / "docs" / name for name in HISTORICAL_DOCS]
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(set(paths))
            if not any(part.startswith("m2b-") for part in path.relative_to(ROOT).parts)}


def roadmap() -> dict[str, str]:
    return {"../analysis/" + name: digest(ROOT.parent / "analysis" / name) for name in ROADMAP_FILES}


def changed(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))


def summary(text: str) -> dict:
    matches = list(re.finditer(r"Ran ([0-9]+) tests? in ([0-9.]+)s", text))
    match = matches[-1] if matches else None
    skipped = sum(int(value) for value in re.findall(r"\bskipped=([0-9]+)", text))
    if " ... skipped " in text and not skipped:
        skipped = 1  # A skip is still disqualifying if no standard summary survived.
    return {"tests": int(match.group(1)) if match else None,
            "unittest_seconds": float(match.group(2)) if match else None,
            "failures": sum(int(value) for value in re.findall(r"\bfailures=([0-9]+)", text)),
            "errors": sum(int(value) for value in re.findall(r"\berrors=([0-9]+)", text)),
            "skipped": skipped,
            "warning_lines": sum("Warning:" in line for line in text.splitlines())}


def execute(directory: Path, name: str, command: list[str], timeout: int,
            *, expected_tests: int | None = None, requires_tests: bool = False) -> dict:
    started, recorded = time.monotonic(), utc_now()
    process = None
    out, err = b"", b""
    code, result, exception = None, "failed", None
    cleanup_errors = []
    try:
        process = subprocess.Popen(command, cwd=ROOT.parent, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        out, err = process.communicate(timeout=timeout)
        code = process.returncode
        result = "passed" if code == 0 else "failed"
    except subprocess.TimeoutExpired as exc:
        out, err = exc.output or out, exc.stderr or err
        result = "timeout"
    except (KeyboardInterrupt, SystemExit):
        result = "interrupted"
        exception = redact(traceback.format_exc())
    except Exception:
        exception = redact(traceback.format_exc())
    finally:
        if process is not None:
            # Only the Popen-created group, even if its leader already exited.
            # No pkill, process discovery, port cleanup or pre-existing services.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except OSError as exc:
                cleanup_errors.append(type(exc).__name__)
            try:
                out, err = process.communicate(timeout=2)
            except subprocess.TimeoutExpired as exc:
                out, err = exc.output or out, exc.stderr or err
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    cleanup_errors.append(type(exc).__name__)
            try:
                out, err = process.communicate(timeout=5)
            except subprocess.TimeoutExpired as exc:
                out, err = exc.output or out, exc.stderr or err
                cleanup_errors.append("pipe_drain_timeout")
            finally:
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
            code = process.returncode
    stdout = redact(out.decode("utf-8", errors="replace"))
    stderr = redact(err.decode("utf-8", errors="replace"))
    save(directory / (name + ".stdout.log"), stdout)
    save(directory / (name + ".stderr.log"), stderr)
    details = summary(stdout + "\n" + stderr)
    if (cleanup_errors or details["failures"] or details["errors"] or details["skipped"]
            or (requires_tests and not details["tests"])
            or (expected_tests is not None and details["tests"] != expected_tests)):
        if result == "passed":
            result = "failed"
    return {"name": name, "command": command, "cwd": str(ROOT.parent), "started_at": recorded,
            "finished_at": utc_now(), "elapsed_seconds": round(time.monotonic() - started, 3),
            "exit_code": code, "result": result, "timeout_seconds": timeout,
            "expected_tests": expected_tests, **details,
            "exception": exception, "cleanup_errors": cleanup_errors,
            "raw_stdout": name + ".stdout.log", "raw_stderr": name + ".stderr.log",
            "capabilities_redacted": True}


def independent_cli(directory: Path) -> int:
    """Runs only in an isolated, timed, owned child process, never in main()."""
    test_path = ROOT / "tests/test_legacy_migration_cli.py"
    stream = io.StringIO()
    started, recorded = time.monotonic(), utc_now()
    module, test_result, error = None, None, None
    records = []
    try:
        spec = importlib.util.spec_from_file_location("m2b_legacy_migration_cli_acceptance", test_path)
        if spec is None or spec.loader is None:
            raise ValueError("migration CLI test loader unavailable")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        with redirect_stdout(stream), redirect_stderr(stream):
            spec.loader.exec_module(module)
            test_result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
                unittest.defaultTestLoader.loadTestsFromModule(module))
        records = module.COMMAND_RECORDS
        if not isinstance(records, list) or not records or not all(isinstance(item, dict) for item in records):
            raise ValueError("migration CLI COMMAND_RECORDS must be a nonempty list of objects")
    except BaseException:
        error = redact(traceback.format_exc())
        stream.write(error)
        if module is not None and isinstance(getattr(module, "COMMAND_RECORDS", None), list):
            records = module.COMMAND_RECORDS
    successful = (error is None and test_result is not None and test_result.testsRun > 0
                  and test_result.wasSuccessful() and not test_result.skipped)
    # These files are retained even on import/assertion errors. An externally
    # killed worker may not reach here; main records that absence as a failure.
    save(directory / "cli-tests.log", redact(stream.getvalue()))
    save_json(directory / "cli-commands.json", redact_value(records))
    report = {"name": "independent_real_legacy_migration_cli", "test_path": str(test_path),
              "started_at": recorded, "finished_at": utc_now(),
              "elapsed_seconds": round(time.monotonic() - started, 3),
              "result": "passed" if successful else "failed", "exit_code": 0 if successful else 1,
              "tests": test_result.testsRun if test_result is not None else None,
              "failures": len(test_result.failures) if test_result is not None else None,
              "errors": len(test_result.errors) if test_result is not None else None,
              "skipped": len(test_result.skipped) if test_result is not None else None,
              "commands": len(records), "exception": error, "overlaps_full_suite": True,
              "raw_log": "cli-tests.log", "command_records": "cli-commands.json"}
    save_json(directory / "cli-report.json", report)
    print(json.dumps({"result": report["result"], "tests": report["tests"], "commands": len(records)}))
    return report["exit_code"]


def require_safe_directory(path: Path) -> None:
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("evidence directories must not contain symlinks")


def interrupt(signum, frame):
    raise KeyboardInterrupt("runner received termination signal")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--_cli-worker-directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (Path(sys.executable).absolute() != (ROOT / ".venv/bin/python").absolute()
            or not sys.flags.isolated or not sys.dont_write_bytecode):
        parser.error("use the existing dedicated .venv/bin/python -I -B")
    try:
        require_safe_directory(EVIDENCE)
        if args._cli_worker_directory is not None:
            directory = args._cli_worker_directory.absolute()
            require_safe_directory(directory)
            if (directory.parent != EVIDENCE or not re.fullmatch(r"run-[0-9]{3,}", directory.name)
                    or not directory.is_dir()):
                raise ValueError("internal CLI worker requires an existing numbered M2b run directory")
            return independent_cli(directory)
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        number = 1
        while True:
            directory = EVIDENCE / f"run-{number:03d}"
            try:
                directory.mkdir()
                break
            except FileExistsError:
                number += 1
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    signal.signal(signal.SIGTERM, interrupt)
    started, recorded = time.monotonic(), utc_now()
    snapshots = {"source": sources, "history": history, "roadmap": roadmap}
    before, after, steps, snapshot_errors = {}, {}, [], []
    runner_error = None
    try:
        for name, snapshot in snapshots.items():
            before[name] = snapshot()
            save_json(directory / (name + "-before.json"), before[name])
        commands = [
            ("suite", [sys.executable, "-I", "-B", "-m", "unittest", "discover", "-v", "-s=" + str(ROOT / "tests")], 600, None, True),
            ("ledger-m1", [sys.executable, "-I", "-B", str(ROOT / "tools/validate_implementation.py"), "--root", str(ROOT)], 30, None, False),
            ("ledger-m2a", [sys.executable, "-I", "-B", str(ROOT / "tools/validate_implementation.py"), "--root", str(ROOT), "--milestone", "M2a"], 30, None, False),
            ("ledger-m2b", [sys.executable, "-I", "-B", str(ROOT / "tools/validate_implementation.py"), "--root", str(ROOT), "--milestone", "M2b"], 30, None, False),
            ("static-roadmap", [sys.executable, "-I", "-B", str(ROOT.parent / "tools/validate_graphs.py"),
                "--analysis", str(ROOT.parent / "analysis"), "--repo", "Rethlas=" + str(ROOT),
                "--repo", "Re-CTM=" + str(ROOT.parent / "Re-CTM")], 120, None, False),
            ("roadmap-regressions", [sys.executable, "-I", "-B", str(ROOT.parent / "tests/test_validate_graphs.py"),
                "--validator", str(ROOT.parent / "tools/validate_graphs.py")], 300, 63, True),
            ("offline-experiment-regressions", [sys.executable, "-I", "-B",
                str(ROOT.parent / "tests/test_openviking_benchmark.py")], 60, 19, True),
            ("independent-legacy-migration-cli", [sys.executable, "-I", "-B", str(Path(__file__).resolve()),
                "--_cli-worker-directory", str(directory)], 600, None, False),
        ]
        for name, command, timeout, expected_tests, requires_tests in commands:
            step = execute(directory, name, command, timeout, expected_tests=expected_tests,
                           requires_tests=requires_tests)
            steps.append(step)
            if name == "independent-legacy-migration-cli":
                try:
                    cli_report = json.loads((directory / "cli-report.json").read_text(encoding="utf-8"))
                    step["cli_report"] = "cli-report.json"
                    step["tests"] = cli_report["tests"]
                    step["failures"] = cli_report["failures"]
                    step["errors"] = cli_report["errors"]
                    step["skipped"] = cli_report["skipped"]
                    step["commands"] = cli_report["commands"]
                    step["overlaps_full_suite"] = True
                    if cli_report["result"] != "passed" or not step["tests"] or not step["commands"]:
                        step["result"] = "failed"
                except (OSError, ValueError, KeyError, TypeError):
                    step["result"] = "failed"
                    step["cli_archive_error"] = redact(traceback.format_exc())
            if step["result"] == "interrupted":
                break
    except BaseException:
        runner_error = redact(traceback.format_exc())
    finally:
        # Preserve a terminal report even after child startup/import/snapshot
        # failures or a handled interruption. SIGKILL/power loss cannot be caught.
        for name, snapshot in snapshots.items():
            try:
                after[name] = snapshot()
                save_json(directory / (name + "-after.json"), after[name])
            except BaseException:
                snapshot_errors.append({"snapshot": name + "-after", "exception": redact(traceback.format_exc())})
        differences = {name: changed(before[name], after[name]) if name in before and name in after else None
                       for name in snapshots}
        successful = (runner_error is None and not snapshot_errors and len(steps) == 8
                      and all(step["result"] == "passed" for step in steps)
                      and all(value == [] for value in differences.values()))
        report = {"milestone": "M2b", "started_at": recorded, "finished_at": utc_now(),
                  "elapsed_seconds": round(time.monotonic() - started, 3),
                  "result": "passed" if successful else "failed", "exit_code": 0 if successful else 1,
                  "runner_command": [sys.executable, *sys.orig_argv[1:]], "steps": steps,
                  "runner_exception": runner_error, "snapshot_errors": snapshot_errors,
                  "source_changed_during_run": differences["source"],
                  "m1_m2a_history_changed": differences["history"],
                  "original_roadmap_changed": differences["roadmap"],
                  "prior_run_numbers_preserved": number - 1, "external_model_requests": 0,
                  "synthetic_only": True, "human_review": "pending",
                  "mathematical_correctness_proved": False, "documentation_updated_automatically": False,
                  "limitations": [
                      "A local automation pass does not accept human review, mathematics or all fifteen roadmap tasks.",
                      "Independent migration CLI tests overlap the full suite; their counts are not added together.",
                      "The historical M2a 219-test pass does not validate changed M2b source.",
                      "SIGKILL and injected failures are not arbitrary power-loss, NFS or multi-writer certification.",
                      "Failure reports survive handled errors; SIGKILL, power loss or a full evidence disk can prevent finalization.",
                  ]}
        save_json(directory / "report.json", redact_value(report))
    print(json.dumps({"result": report["result"], "directory": str(directory)}, ensure_ascii=False))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
