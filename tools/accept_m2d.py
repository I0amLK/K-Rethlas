#!/usr/bin/env python3
"""Separately authorized local synthetic M2d acceptance; never update ledgers.

Runs the installed project, real validity CLI, spawn/SIGKILL and bubblewrap tests.
No dependencies, providers, production data or preexisting services are touched.
Every run is new. Missing summaries, warnings, skips and source drift block pass.
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
EVIDENCE = ROOT / "docs/evidence/m2d-runtime"
CACHE = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
HISTORY_DOCS = ("implementation-status.graph.json", "implementation-status.md", "acceptance.md",
                "security-model.md", "memory-backend-decision.md", "provenance.md")
ROADMAP = ("graph.schema.json", "rethlas.graph.json", "rethlas.md", "re-ctm.graph.json",
           "re-ctm.md", "upgrade-todo.graph.json", "upgrade-todo.md")
PARENT_SOURCES = ("tools/validate_graphs.py", "tests/test_validate_graphs.py", "tests/test_openviking_benchmark.py",
                  "tools/benchmark_openviking.py", "tools/openviking_eval_bridge.py", "tools/openviking_lifecycle_probe.py")
HISTORY_SHA = "d0822adaf899a944bd1f9d3df6800cdb3bfa403ad532787610a375c844b35f4a"
HISTORY_COUNT = 223
MILESTONES = ("M1", "M2a", "M2b", "M2c", "M2d")
SUITE_TIMEOUT = 600
CLI_TIMEOUT = 600


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def redact(text):
    text = re.sub(r"(?<![A-Za-z0-9_.-])eyJ[A-Za-z0-9_-]{77,}\.[A-Za-z0-9_-]{43}(?![A-Za-z0-9_.-])",
                  "[REDACTED synthetic capability]", text)
    text = re.sub(r"(?i)(bearer\s+)[^\s\"']+", r"\1[REDACTED]", text)
    return re.sub(r'''(?i)(["']?(?:token|capability|secret|api[_-]?key)["']?\s*[:=]\s*["'])([^"'\r\n]+)''',
                  r"\1[REDACTED]", text)


def redact_value(value):
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    if isinstance(value, dict):
        return {key: "[REDACTED]" if key.lower() in {"token", "capability", "secret", "api_key", "api-key"}
                else redact_value(v) for key, v in value.items()}
    return value


def save(path, text):
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def save_json(path, value):
    save(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))


def regular_files(directory, *, ignore_cache=True):
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("snapshot root must be a nonsymlink directory")
    result = []
    for path in directory.rglob("*"):
        relative = path.relative_to(directory)
        if ignore_cache and (any(part in CACHE for part in relative.parts) or path.suffix in {".pyc", ".pyo"}):
            continue
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            result.append(path)
        elif not stat.S_ISDIR(mode):
            raise ValueError("snapshot refuses symlinks and special entries")
    return result


def digest(path):
    if any(p.is_symlink() for p in (path, *path.parents)) or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("snapshot input must be nonsymlink regular")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    paths = []
    for name in ("agents/shared", "tests", "tools"):
        paths.extend(regular_files(ROOT / name))
    paths.append(ROOT / "agents/generation/.agents/skills/recursive-proving/SKILL.md")
    result = {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}
    result.update({"../" + name: digest(ROOT.parent / name) for name in PARENT_SOURCES})
    return result


def history():
    paths = [ROOT / "docs" / name for name in HISTORY_DOCS]
    for name in ("m2a", "m2b", "m2c"):
        paths += regular_files(ROOT / "docs/milestones" / name, ignore_cache=False)
    paths += regular_files(ROOT / "docs/evidence", ignore_cache=False)
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))
            if not any(part.startswith("m2d-") for part in p.relative_to(ROOT).parts)}


def roadmap():
    return {"../analysis/" + name: digest(ROOT.parent / "analysis" / name) for name in ROADMAP}


def changed(before, after):
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))


def summary(text):
    matches = list(re.finditer(r"Ran ([0-9]+) tests? in ([0-9.]+)s", text))
    match = matches[-1] if matches else None
    skips = sum(int(x) for x in re.findall(r"\bskipped=([0-9]+)", text))
    if " ... skipped " in text and not skips:
        skips = 1
    return {"tests": int(match.group(1)) if match else None,
            "unittest_seconds": float(match.group(2)) if match else None,
            "failures": sum(int(x) for x in re.findall(r"\bfailures=([0-9]+)", text)),
            "errors": sum(int(x) for x in re.findall(r"\berrors=([0-9]+)", text)), "skipped": skips,
            "warning_lines": sum("Warning:" in line for line in text.splitlines())}


def execute(directory, name, command, timeout, expected_tests=None, requires_tests=False):
    started, recorded = time.monotonic(), utc_now()
    process, code, exception = None, None, None
    out, err, result, cleanup_errors = b"", b"", "failed", []
    try:
        process = subprocess.Popen(command, cwd=ROOT.parent, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, start_new_session=True, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
        out, err = process.communicate(timeout=timeout)
        code = process.returncode
        result = "passed" if code == 0 else "failed"
    except subprocess.TimeoutExpired as exc:
        out, err, result = exc.output or out, exc.stderr or err, "timeout"
    except (KeyboardInterrupt, SystemExit):
        result, exception = "interrupted", redact(traceback.format_exc())
    except Exception:
        exception = redact(traceback.format_exc())
    finally:
        if process is not None:
            for sig, wait in ((signal.SIGTERM, 2), (signal.SIGKILL, 5)):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
                except OSError as exc:
                    cleanup_errors.append(type(exc).__name__)
                try:
                    out, err = process.communicate(timeout=wait)
                except subprocess.TimeoutExpired as exc:
                    out, err = exc.output or out, exc.stderr or err
                    if sig == signal.SIGKILL:
                        cleanup_errors.append("pipe_drain_timeout")
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            code = process.returncode
    stdout, stderr = redact(out.decode("utf-8", errors="replace")), redact(err.decode("utf-8", errors="replace"))
    save(directory / (name + ".stdout.log"), stdout)
    save(directory / (name + ".stderr.log"), stderr)
    details = summary(stdout + "\n" + stderr)
    if (cleanup_errors or details["failures"] or details["errors"] or details["skipped"] or details["warning_lines"]
            or requires_tests and not details["tests"]
            or expected_tests is not None and details["tests"] != expected_tests):
        if result == "passed":
            result = "failed"
    return {"name": name, "command": command, "cwd": str(ROOT.parent), "started_at": recorded,
            "finished_at": utc_now(), "elapsed_seconds": round(time.monotonic() - started, 3),
            "exit_code": code, "result": result, "timeout_seconds": timeout, "expected_tests": expected_tests,
            **details, "exception": exception, "cleanup_errors": cleanup_errors,
            "raw_stdout": name + ".stdout.log", "raw_stderr": name + ".stderr.log", "capabilities_redacted": True}


def independent_cli(directory):
    stream = io.StringIO()
    started, recorded = time.monotonic(), utc_now()
    module, test_result, error, records = None, None, None, []
    try:
        # Fixed trusted test modules; no caller-selected plugin or sys.path change.
        for name in ("test_memory_store", "test_context_views", "test_validity", "test_validity_cli"):
            spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / (name + ".py"))
            loaded = importlib.util.module_from_spec(spec)
            sys.modules[name] = loaded
            with redirect_stdout(stream), redirect_stderr(stream):
                spec.loader.exec_module(loaded)
            if name == "test_validity_cli":
                module = loaded
        with redirect_stdout(stream), redirect_stderr(stream):
            test_result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
        records = module.COMMAND_RECORDS
        if not isinstance(records, list) or not records or not all(isinstance(x, dict) for x in records):
            raise ValueError("validity CLI records must be nonempty objects")
    except BaseException:
        error = redact(traceback.format_exc())
        stream.write(error)
        if module is not None:
            records = getattr(module, "COMMAND_RECORDS", [])
    details = summary(stream.getvalue())
    success = (error is None and test_result is not None and test_result.testsRun > 0
               and test_result.wasSuccessful() and not test_result.skipped and not details["warning_lines"]
               and details["tests"] == test_result.testsRun and not details["failures"] and not details["errors"])
    save(directory / "cli-tests.log", redact(stream.getvalue()))
    save_json(directory / "cli-commands.json", redact_value(records))
    report = {"name": "independent_real_validity_cli", "test_path": str(ROOT / "tests/test_validity_cli.py"),
              "started_at": recorded, "finished_at": utc_now(), "elapsed_seconds": round(time.monotonic() - started, 3),
              "result": "passed" if success else "failed", "exit_code": 0 if success else 1,
              "tests": test_result.testsRun if test_result else None, "failures": len(test_result.failures) if test_result else None,
              "errors": len(test_result.errors) if test_result else None, "skipped": len(test_result.skipped) if test_result else None,
              "warning_lines": details["warning_lines"], "commands": len(records), "exception": error,
              "overlaps_full_suite": True, "raw_log": "cli-tests.log", "command_records": "cli-commands.json"}
    save_json(directory / "cli-report.json", report)
    print(json.dumps({"result": report["result"], "tests": report["tests"], "commands": len(records)}))
    return report["exit_code"]


def safe_directory(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("evidence directory must not contain symlinks")


def interrupt(signum, frame):
    raise KeyboardInterrupt("runner terminated")


def commands(directory):
    python = [sys.executable, "-I", "-B"]
    result = [("suite", python + ["-m", "unittest", "discover", "-v", "-s=" + str(ROOT / "tests")], SUITE_TIMEOUT, None, True)]
    for milestone in MILESTONES:
        result.append(("ledger-" + milestone.lower(), python + [str(ROOT / "tools/validate_implementation.py"),
            "--root", str(ROOT), "--milestone", milestone], 30, None, False))
    result.extend([
        ("static-roadmap", python + [str(ROOT.parent / "tools/validate_graphs.py"), "--analysis", str(ROOT.parent / "analysis"),
            "--repo", "Rethlas=" + str(ROOT), "--repo", "Re-CTM=" + str(ROOT.parent / "Re-CTM")], 120, None, False),
        ("roadmap-regressions", python + [str(ROOT.parent / "tests/test_validate_graphs.py"), "--validator",
            str(ROOT.parent / "tools/validate_graphs.py")], 300, 63, True),
        ("offline-experiment-regressions", python + [str(ROOT.parent / "tests/test_openviking_benchmark.py")], 60, 19, True),
        ("independent-validity-cli", python + [str(Path(__file__).resolve()), "--_cli-worker-directory", str(directory)], CLI_TIMEOUT, None, False),
    ])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--_cli-worker-directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if Path(sys.executable).absolute() != (ROOT / ".venv/bin/python").absolute() or not sys.flags.isolated or not sys.dont_write_bytecode:
        parser.error("use the existing dedicated .venv/bin/python -I -B")
    try:
        safe_directory(EVIDENCE)
        if args._cli_worker_directory is not None:
            directory = args._cli_worker_directory.absolute()
            safe_directory(directory)
            if directory.parent != EVIDENCE or not re.fullmatch(r"run-[0-9]{3,}", directory.name) or not directory.is_dir():
                raise ValueError("CLI worker requires an existing numbered M2d directory")
            return independent_cli(directory)
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        number = 1
        while True:
            directory = EVIDENCE / ("run-%03d" % number)
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
    before, after, steps, snapshot_errors, runner_error = {}, {}, [], [], None
    try:
        for name, function in snapshots.items():
            before[name] = function()
            save_json(directory / (name + "-before.json"), before[name])
        encoded = json.dumps(before["history"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        if len(before["history"]) != HISTORY_COUNT or hashlib.sha256(encoded).hexdigest() != HISTORY_SHA:
            raise ValueError("historical baseline differs before execution")
        baseline = json.loads((ROOT / "docs/evidence/m2d-history-baseline.json").read_bytes())
        if baseline.get("history_sha256") != before["history"] or baseline.get("history_map_sha256") != HISTORY_SHA:
            raise ValueError("saved preimplementation baseline differs before execution")
        for name, command, timeout, expected, required in commands(directory):
            step = execute(directory, name, command, timeout, expected, required)
            steps.append(step)
            if name == "independent-validity-cli":
                try:
                    report = json.loads((directory / "cli-report.json").read_bytes())
                    step.update({key: report[key] for key in ("tests", "failures", "errors", "skipped", "warning_lines", "commands")})
                    step.update(cli_report="cli-report.json", overlaps_full_suite=True)
                    if (report["result"] != "passed" or not step["tests"] or not step["commands"]
                            or any(step[key] for key in ("failures", "errors", "skipped", "warning_lines"))):
                        step["result"] = "failed"
                except (OSError, ValueError, KeyError, TypeError):
                    step.update(result="failed", cli_archive_error=redact(traceback.format_exc()))
            if step["result"] == "interrupted":
                break
    except BaseException:
        runner_error = redact(traceback.format_exc())
    finally:
        for name, function in snapshots.items():
            try:
                after[name] = function()
                save_json(directory / (name + "-after.json"), after[name])
            except BaseException:
                snapshot_errors.append({"snapshot": name + "-after", "exception": redact(traceback.format_exc())})
        differences = {name: changed(before[name], after[name]) if name in before and name in after else None for name in snapshots}
        success = (runner_error is None and not snapshot_errors and len(steps) == 10
                   and all(s["result"] == "passed" for s in steps) and all(v == [] for v in differences.values()))
        report = {"milestone": "M2d", "started_at": recorded, "finished_at": utc_now(),
                  "elapsed_seconds": round(time.monotonic() - started, 3), "result": "passed" if success else "failed",
                  "exit_code": 0 if success else 1, "runner_command": [sys.executable, *sys.orig_argv[1:]], "steps": steps,
                  "runner_exception": runner_error, "snapshot_errors": snapshot_errors,
                  "source_changed_during_run": differences["source"], "m1_m2a_m2b_m2c_history_changed": differences["history"],
                  "original_roadmap_changed": differences["roadmap"], "prior_run_numbers_preserved": number - 1,
                  "external_model_requests": 0, "synthetic_only": True, "human_review": "pending",
                  "mathematical_correctness_proved": False, "documentation_updated_automatically": False,
                  "limitations": ["CLI overlaps suite; counts are not added.",
                      "Synthetic current-input eligibility is not independent mathematical review.",
                      "M2c's 390-test pass does not cover new M2d code.",
                      "No arbitrary power-loss/NFS/multi-host or coordinated-authority-rollback certification.",
                      "SIGKILL, power loss or a full evidence disk may prevent final reporting."]}
        save_json(directory / "report.json", redact_value(report))
    print(json.dumps({"result": report["result"], "directory": str(directory)}, ensure_ascii=False))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
