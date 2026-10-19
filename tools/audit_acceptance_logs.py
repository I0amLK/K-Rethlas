#!/usr/bin/env python3
"""Independently audit archived acceptance logs; never execute their commands.

The original runner and reports are read-only. A recorded pass is not accepted as
warning-free when Python warnings, logging warnings or asyncio slow callbacks
appear in the raw output. Folded Rich records count once, at their header line.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys


MAX_REPORT_BYTES = 1024 * 1024
MAX_LOG_BYTES = 16 * 1024 * 1024
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
ANSI = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]")
PYTHON_WARNING = re.compile(r"\b(?:[A-Za-z_]\w*\.)*[A-Za-z_]*Warning:")
LOG_WARNING = re.compile(r"(?:^|[\s\[\]])(?:WARNING|WARN)(?=[\s:\]])")
SLOW_CALLBACK = re.compile(r"\bExecuting\s+<(?:Task|Handle|TimerHandle)\b")


class AuditError(ValueError):
    pass


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise AuditError("duplicate JSON key in evidence")
        result[key] = value
    return result


def _constant(value):
    raise AuditError("non-finite JSON value in evidence")


def _read(path, maximum):
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise AuditError("evidence path contains a symlink")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > maximum:
            raise AuditError("evidence must be a bounded single-link regular file")
        chunks, size = [], 0
        while True:
            chunk = os.read(descriptor, min(65536, maximum + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > maximum:
                raise AuditError("evidence exceeds its byte limit")
        after = os.fstat(descriptor)
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
        if identity(before) != identity(after):
            raise AuditError("evidence changed while reading")
        raw = b"".join(chunks)
        return raw, raw.decode("utf-8")
    except UnicodeError:
        raise AuditError("evidence is not exact UTF-8") from None
    finally:
        os.close(descriptor)


def _json(path):
    raw, text = _read(path, MAX_REPORT_BYTES)
    try:
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    except (ValueError, RecursionError):
        raise AuditError("invalid strict JSON evidence") from None
    if type(value) is not dict:
        raise AuditError("evidence report must be an object")
    return value, hashlib.sha256(raw).hexdigest()


def _name(value):
    if type(value) is not str or not NAME.fullmatch(value) or value in {".", ".."}:
        raise AuditError("evidence filename or step name is not a safe basename")
    return value


def audit_text(text):
    """Classify diagnostic headers without emitting log bodies or their secrets."""
    if type(text) is not str:
        raise AuditError("log text must be a string")
    events = []
    counts = {"python_warning": 0, "logging_warning": 0, "asyncio_slow_callback": 0}
    for number, original in enumerate(text.splitlines(), 1):
        line = ANSI.sub("", original)
        kinds = []
        if PYTHON_WARNING.search(line):
            kinds.append("python_warning")
        logging_warning = bool(LOG_WARNING.search(line))
        if not logging_warning and line.lstrip().startswith("{"):
            try:
                value = json.loads(line, object_pairs_hook=_pairs, parse_constant=_constant)
            except (ValueError, RecursionError):
                value = None
            if type(value) is dict:
                logging_warning = any(type(value.get(key)) is str and value[key].upper() in {"WARNING", "WARN"}
                                      for key in ("level", "levelname", "severity"))
        if logging_warning:
            kinds.append("logging_warning")
        if SLOW_CALLBACK.search(line):
            kinds.append("asyncio_slow_callback")
        if kinds:
            if len(events) >= 10000:
                raise AuditError("diagnostic event limit exceeded; no truncated clean result")
            for kind in kinds:
                counts[kind] += 1
            events.append({"header_line": number, "kinds": kinds,
                           "header_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest()})
    return {"physical_lines": len(text.splitlines()), "diagnostic_events": len(events),
            "category_counts_overlap": True, "category_counts": counts,
            "legacy_warning_lines": sum("Warning:" in line for line in text.splitlines()),
            "events": events, "raw_text_included": False}


def audit_directory(directory):
    directory = Path(directory).absolute()
    if ".." in directory.parts or any(part.is_symlink() for part in (directory, *directory.parents)) or not directory.is_dir():
        raise AuditError("input must be a nonsymlink evidence directory")
    recorded, report_hash = _json(directory / "report.json")
    followup = recorded.get("version") == "tui-warning-followup-checks-1"
    allowed_results = {"passed", "failed", "timeout", "interrupted"}
    if followup:
        allowed_results.update({"assertions_passed_pending_raw_audit", "regression_failed"})
    if type(recorded.get("result")) is not str or recorded["result"] not in allowed_results:
        raise AuditError("unknown recorded acceptance result")
    recorded_assertions_passed = (recorded["result"] == "passed" or followup
                                  and recorded["result"] == "assertions_passed_pending_raw_audit")
    steps = recorded.get("steps")
    if type(steps) is not list or not 1 <= len(steps) <= 32:
        raise AuditError("expected a bounded acceptance step list")
    names, logs, issues, child_reports = set(), set(), [], []
    if not recorded_assertions_passed or type(recorded.get("exit_code")) is not int or recorded["exit_code"] != 0:
        issues.append("recorded_top_level_not_passed")
    if followup and recorded.get("source_changed_during_run") != []:
        issues.append("recorded_followup_source_not_stable")
    for step in steps:
        if type(step) is not dict:
            raise AuditError("acceptance step must be an object")
        name = _name(step.get("name"))
        if name in names:
            raise AuditError("duplicate acceptance step")
        names.add(name)
        if (step.get("result") != "passed" or type(step.get("exit_code")) is not int or step["exit_code"] != 0
                or step.get("exception") is not None or step.get("cleanup_errors") != []
                or any(step.get(key) != 0 or type(step.get(key)) is not int
                       for key in ("failures", "errors", "skipped", "warning_lines"))):
            issues.append("recorded_step_not_clean:" + name)
        for key in ("raw_stdout", "raw_stderr"):
            logs.add(_name(step.get(key)))
        if "cli_report" in step:
            child_name = _name(step["cli_report"])
            child, child_hash = _json(directory / child_name)
            child_reports.append({"file": child_name, "sha256": child_hash})
            logs.add(_name(child.get("raw_log")))
    audited = []
    for name in sorted(logs):
        raw, text = _read(directory / name, MAX_LOG_BYTES)
        audited.append({"file": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), **audit_text(text)})
    if _json(directory / "report.json")[1] != report_hash:
        raise AuditError("acceptance report changed during audit")
    for child in child_reports:
        if _json(directory / child["file"])[1] != child["sha256"]:
            raise AuditError("child report changed during audit")
    for item in audited:
        raw, _ = _read(directory / item["file"], MAX_LOG_BYTES)
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise AuditError("raw log changed during audit")
    total = sum(item["diagnostic_events"] for item in audited)
    result = "diagnostics_found" if total else "recorded_checks_not_clean" if issues else "no_diagnostics_found"
    return {"version": "acceptance-raw-log-audit-1", "result": result,
            "exit_code": 0 if not total and not issues else 1,
            "recorded_result": recorded["result"], "recorded_result_not_rewritten": True,
            "report_sha256": report_hash, "child_reports": child_reports, "steps": len(steps), "logs": audited,
            "diagnostic_events": total, "recorded_check_issues": issues,
            "commands_executed": False, "warning_exemptions": [], "log_bodies_emitted": False,
            "limitations": ["Diagnostic headers, not Rich continuation rows, are counted.",
                            "Category counts overlap when one event is both logging WARNING and a slow callback.",
                            "No diagnostics found is a log check, not mathematical or milestone acceptance."]}


def _save(directory, value):
    directory = Path(directory).absolute()
    if ".." in directory.parts or any(part.is_symlink() for part in (directory, *directory.parents)) or not directory.parent.is_dir():
        raise AuditError("output requires an existing nonsymlink parent")
    directory.mkdir(mode=0o700)
    path = directory / "report.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_directory", type=Path, help="existing runner directory; commands are never executed")
    parser.add_argument("--output-dir", type=Path, help="new audit directory; existing output is never overwritten")
    args = parser.parse_args(argv)
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        parser.error("use the dedicated Python -I -B")
    try:
        result = audit_directory(args.evidence_directory)
        if args.output_dir is not None:
            _save(args.output_dir, result)
    except (OSError, AuditError) as exc:
        result = {"result": "invalid_evidence_or_output", "exit_code": 2,
                  "error_type": type(exc).__name__, "commands_executed": False}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
