"""Verification report/finding contracts shared by trusted local adapters.

``valid`` means contract-valid, not mathematically correct: a complete ``wrong``
report is valid. These pure functions do not read ledgers or issue authority.
"""
from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from typing import Any

from .contracts import _json_errors, _schema_errors, _validator


_KINDS = ("critical_errors", "gaps")


def _finding_errors(finding: Any, label: str) -> list[str]:
    schema = _validator("verification_output.schema.json").schema["$defs"]["finding"]
    validator = _validator("verification_output.schema.json").evolve(schema=schema)
    errors = [f"{label}: {error.message}" for error in validator.iter_errors(finding)]
    if isinstance(finding, dict):
        for key in ("location", "issue"):
            if isinstance(finding.get(key), str) and not finding[key].strip():
                errors.append(f"{label}.{key}: must be non-empty text")
    return errors


def validate_verification_output(payload: Any) -> dict:
    """Apply the unchanged three-field schema and original verdict semantics."""
    errors = _schema_errors(payload, "verification_output.schema.json")
    if not isinstance(payload, dict) or errors:
        return {"valid": False, "errors": errors}
    report = payload["verification_report"]
    for kind in _KINDS:
        for index, finding in enumerate(report[kind]):
            errors.extend(_finding_errors(finding, f"verification_report.{kind}[{index}]"))
    has_findings = bool(report["critical_errors"] or report["gaps"])
    if payload["verdict"] == "correct":
        if has_findings:
            errors.append("verdict='correct' is invalid when critical_errors or gaps are non-empty")
        if payload["repair_hints"] != "":
            errors.append("repair_hints must be empty when verdict='correct'")
    else:
        if not has_findings:
            errors.append("verdict='wrong' requires at least one critical error or gap")
        if not payload["repair_hints"].strip():
            errors.append("repair_hints must be non-empty when verdict='wrong'")
    return {"valid": not errors, "errors": errors}


def require_valid_output(payload: Any) -> None:
    """Raise ValueError for invalid reports; both valid verdicts return None."""
    result = validate_verification_output(payload)
    if not result["valid"]:
        raise ValueError("verification output validation failed: " + "; ".join(result["errors"]))


def normalize_findings(records: list[dict]) -> list[dict]:
    """Flatten statement/reference records or their JSONL ``record`` envelopes.

    Statement records contain ``critical_errors`` and/or ``gaps`` arrays;
    reference records contain one ``critical_error`` object (or null for success).
    The existing skills' known metadata is type-checked and left unmodified.
    Empty arrays are valid checks. Unknown record/events are never discarded.
    Optional envelope fields are ``timestamp_utc`` and ``channel``. Callers must
    filter non-finding channels such as ``verification_reports`` beforehand.
    Text and optional fields are preserved exactly, without trimming/coercion.
    """
    if type(records) is not list:
        raise ValueError("records must be a list of JSON objects")
    errors = _json_errors(records)
    if errors:
        raise ValueError("; ".join(errors))
    result = []
    for index, entry in enumerate(records):
        label = f"records[{index}]"
        if type(entry) is not dict:
            raise ValueError(f"{label}: must be a JSON object")
        channel = None
        record = entry
        if "record" in entry:
            if set(entry) - {"record", "timestamp_utc", "channel"}:
                raise ValueError(f"{label}: unknown envelope fields")
            record = entry["record"]
            channel = entry.get("channel")
            if "channel" in entry and channel not in ("statement_checks", "reference_checks"):
                raise ValueError(f"{label}: not a finding channel")
            if "timestamp_utc" in entry and (
                not isinstance(entry["timestamp_utc"], str) or not entry["timestamp_utc"].strip()
            ):
                raise ValueError(f"{label}: timestamp_utc must be non-empty text")
        if type(record) is not dict:
            raise ValueError(f"{label}.record: must be a JSON object")
        keys = set(record)
        if "critical_error" in keys:
            if channel == "statement_checks":
                raise ValueError(f"{label}: critical_error object belongs to reference_checks")
            text_fields = {"location", "referenced_statement", "context_expansion"}
            bool_fields = {"arxiv_match_found", "web_match_found"}
            allowed = {"critical_error"} | text_fields | bool_fields
            finding = record["critical_error"]
            groups = [("critical_errors", [] if finding is None else [finding])]
        elif keys & set(_KINDS):
            if channel == "reference_checks":
                raise ValueError(f"{label}: reference_checks requires a critical_error object")
            text_fields = {"location", "status"}
            bool_fields = set()
            allowed = set(_KINDS) | text_fields
            groups = []
            for kind in _KINDS:
                if kind in record:
                    if type(record[kind]) is not list:
                        raise ValueError(f"{label}.{kind}: must be an array")
                    groups.append((kind, record[kind]))
        else:
            raise ValueError(f"{label}: unknown finding record shape")
        if keys - allowed:
            raise ValueError(f"{label}: unknown finding record fields: {sorted(keys - allowed)}")
        for field in keys & text_fields:
            if type(record[field]) is not str:
                raise ValueError(f"{label}.{field}: metadata must be a string")
        for field in keys & bool_fields:
            if type(record[field]) is not bool:
                raise ValueError(f"{label}.{field}: metadata must be a boolean")
        for kind, findings in groups:
            for finding in findings:
                errors = _finding_errors(finding, label)
                if errors:
                    raise ValueError("; ".join(errors))
                result.append({"kind": kind, "finding": deepcopy(finding)})
    return result


def _canonical_finding(finding: dict) -> str:
    return json.dumps(finding, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def coverage_errors(output: Any, registered: list[dict]) -> list[str]:
    """Require exact same-kind report coverage of every normalized registration.

    Object key order is irrelevant; strings and optional fields match exactly.
    Duplicate registrations require duplicate report entries (multiset coverage).
    Invalid input returns errors rather than silently omitting any registration.
    """
    errors = list(validate_verification_output(output)["errors"])
    if type(registered) is not list:
        errors.append("registered findings must be a list")
        return errors
    json_errors = _json_errors(registered)
    if json_errors:
        return errors + json_errors
    report = output.get("verification_report", {}) if isinstance(output, dict) else {}
    available = {}
    for kind in _KINDS:
        findings = report.get(kind, []) if isinstance(report, dict) else []
        available[kind] = Counter(
            _canonical_finding(finding)
            for finding in findings if isinstance(finding, dict) and not _finding_errors(finding, kind)
        ) if isinstance(findings, list) else Counter()
    for index, entry in enumerate(registered):
        label = f"registered[{index}]"
        if (
            type(entry) is not dict or set(entry) != {"kind", "finding"}
            or entry["kind"] not in _KINDS
        ):
            errors.append(f"{label}: expected normalized {{kind, finding}}")
            continue
        finding_errors = _finding_errors(entry["finding"], label)
        if finding_errors:
            errors.extend(finding_errors)
            continue
        canonical = _canonical_finding(entry["finding"])
        kind = entry["kind"]
        if available[kind][canonical]:
            available[kind][canonical] -= 1
        else:
            errors.append(f"{label}: registered {kind} finding missing from report: {canonical}")
    return errors
