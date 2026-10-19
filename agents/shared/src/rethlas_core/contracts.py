"""Closed JSON interchange contracts; validation does not authorize or store data.

Review states are self-reported structural metadata, not trusted review decisions.
Evidence locators/hashes are checked for shape and references only: no locator is
opened. Body hashes cover exact UTF-8 Markdown, without name/text normalization.
Provenance date-times use the explicit RFC 3339 subset documented below, not
permissive ISO 8601 parsing or jsonschema's optional format dependencies.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import deque
from datetime import datetime
from functools import lru_cache
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from .authorization import ROLE_ACTIONS


_DATE_TIME = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})[Tt]"
    r"([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.[0-9]+)?"
    r"(?:[Zz]|[+-]([0-9]{2}):([0-9]{2}))"
)


def _is_date_time(value: Any) -> bool:
    """Check the project's RFC 3339 subset without optional dependencies.

    Require ASCII YYYY-MM-DD, T/t, HH:MM:SS, optional nonempty decimal seconds,
    and Z/z or a numeric +/-HH:MM offset (00..23 hours, 00..59 minutes).
    Gregorian years are 0001..9999; impossible dates, 24:00 and leap seconds
    are rejected. This deliberately excludes generic ISO 8601 forms. RFC 3339's
    -00:00 unknown-local-offset form is accepted as metadata, not a known local
    zone; timestamps are neither normalized nor used as trusted clock evidence.
    """
    if not isinstance(value, str):
        return True  # JSON Schema's type keyword rejects nonstrings.
    match = _DATE_TIME.fullmatch(value)
    if match is None:
        return False
    year, month, day, hour, minute, second = map(int, match.groups()[:6])
    offset_hour, offset_minute = match.groups()[6:]
    if offset_hour is not None and (int(offset_hour) > 23 or int(offset_minute) > 59):
        return False
    try:
        # Validate the wall-clock fields only; do not truncate fractional seconds
        # or convert offsets near the supported year boundaries.
        datetime(year, month, day, hour, minute, second)
    except ValueError:
        return False
    return True


# Register on a private instance, never jsonschema's process-global registry.
# Its built-in date-time checker otherwise silently disappears without the
# optional rfc3339_validator package (not part of this project's dependencies).
_FORMAT_CHECKER = FormatChecker(formats=())
_FORMAT_CHECKER.checks("date-time")(_is_date_time)


@lru_cache(maxsize=3)
def _validator(filename: str) -> Draft202012Validator:
    resource = files("rethlas_core").joinpath("schemas", filename)
    schema = json.loads(resource.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=_FORMAT_CHECKER)


def _json_errors(value: Any) -> list[str]:
    """Reject Python-only values, nonfinite numbers and cyclic container trees."""
    ancestors: set[int] = set()

    def visit(item: Any, path: str) -> None:
        if item is None or type(item) in (bool, int):
            return
        if type(item) is str:
            item.encode("utf-8")
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) not in (dict, list):
            raise ValueError(f"{path}: must contain only finite JSON values")
        if id(item) in ancestors:
            raise ValueError(f"{path}: cyclic Python container is not JSON")
        ancestors.add(id(item))
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError(f"{path}: JSON object keys must be strings")
                key.encode("utf-8")
                visit(child, f"{path}.{key}")
        else:
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        ancestors.remove(id(item))

    try:
        visit(value, "$")
    except (ValueError, UnicodeError, RecursionError) as exc:
        return [f"invalid JSON value: {exc}"]
    return []


def _schema_errors(payload: Any, filename: str) -> list[str]:
    errors = _json_errors(payload)
    if errors:
        return errors
    for error in _validator(filename).iter_errors(payload):
        path = ".".join(str(part) for part in error.path)
        prefix = f"schema error at '{path}'" if path else "schema error"
        errors.append(f"{prefix}: {error.message}")
    return errors


def _result(errors: list[str]) -> dict:
    return {"valid": not errors, "errors": errors}


def validate_policy(payload: Any) -> dict:
    """Validate the deny-by-default policy; this never loads an authority."""
    errors = _schema_errors(payload, "policy.schema.json")
    if errors:
        return _result(errors)
    for index, rule in enumerate(payload["rules"]):
        if rule["action"] not in ROLE_ACTIONS[rule["role"]]:
            errors.append(f"rules[{index}]: role '{rule['role']}' cannot perform '{rule['action']}'")
    return _result(errors)


def _relation_sha256(relation: dict, revisions: dict, evidence: dict) -> str:
    """Bind the edge, exact revision/context snapshots and evidence metadata.

    Canonical JSON is {relation, revisions: {id: snapshot}, evidence: {id: item}}.
    Revision snapshots exclude review_bindings to avoid a self-referential hash.
    Context references are included transitively; no locator is dereferenced.
    """
    pending = [relation["source"]["revision_id"], relation["target"]["revision_id"], *relation["context_revision_ids"]]
    snapshots = {}
    evidence_ids = set(relation["evidence_ids"])
    while pending:
        revision_id = pending.pop()
        if revision_id in snapshots:
            continue
        revision = revisions.get(revision_id)
        snapshots[revision_id] = None if revision is None else {
            key: value for key, value in revision.items() if key != "review_bindings"
        }
        if revision is not None:
            pending.extend(revision["context_revision_ids"])
            evidence_ids.update(revision["evidence_ids"])
    binding = {"relation": relation, "revisions": snapshots,
               "evidence": {identifier: evidence.get(identifier) for identifier in evidence_ids}}
    encoded = json.dumps(
        binding, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _index(items: list[dict], field: str, errors: list[str]) -> dict[str, dict]:
    result = {}
    for item in items:
        key = item[field]
        if key in result:
            errors.append(f"duplicate {field}: {key!r}")
        else:
            result[key] = item
    return result


def _dependency_errors(relations: list[dict]) -> list[str]:
    # Navigation, citations, reviews and research history are not mathematical
    # premises. Their cycles do not invalidate the mathematical dependency DAG.
    edges: dict[str, set[str]] = {}
    indegree: dict[str, int] = {}
    for relation in relations:
        if relation["type"] not in {"depends_on", "assumes", "in_context"}:
            continue
        source = relation["source"]["revision_id"]
        target = relation["target"]["revision_id"]
        edges.setdefault(source, set())
        indegree.setdefault(source, 0)
        indegree.setdefault(target, 0)
        if target not in edges[source]:
            edges[source].add(target)
            indegree[target] += 1
    ready = deque(node for node, count in indegree.items() if count == 0)
    visited = 0
    while ready:
        source = ready.popleft()
        visited += 1
        for target in edges.get(source, ()):
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if visited != len(indegree):
        return ["mathematical dependency subgraph (depends_on/assumes/in_context) must be a DAG"]
    return []


def validate_math_memory(payload: Any) -> dict:
    """Validate exact-revision graph structure, never promote a trust decision.

    IDs are opaque and compared exactly. Equal/similar display names do not merge
    entities. Legacy ``solved`` and research progress do not imply usable/reviewed.
    ``reviewed`` requires explicit evidence-backed Review coverage, but a caller
    can author such JSON; trusted issuance/promotion belongs to a future store.
    """
    errors = _schema_errors(payload, "math_memory.schema.json")
    if errors:
        return _result(errors)
    entities = _index(payload["entities"], "entity_id", errors)
    revisions = _index(payload["revisions"], "revision_id", errors)
    relations = _index(payload["relations"], "relation_id", errors)
    evidence = _index(payload["evidence"], "evidence_id", errors)
    for identifier in sorted(entities.keys() & revisions.keys()):
        errors.append(f"entity_id and revision_id must be separate identities: {identifier!r}")

    def check_links(item: dict, label: str) -> None:
        for revision_id in item["context_revision_ids"]:
            if revision_id not in revisions:
                errors.append(f"{label}: dangling context revision {revision_id!r}")
            elif entities.get(revisions[revision_id]["entity_id"], {}).get("type") != "DefinitionContext":
                errors.append(f"{label}: context revision {revision_id!r} must be a DefinitionContext")
        for evidence_id in item["evidence_ids"]:
            if evidence_id not in evidence:
                errors.append(f"{label}: dangling evidence {evidence_id!r}")

    for revision in revisions.values():
        label = f"revision {revision['revision_id']!r}"
        entity = entities.get(revision["entity_id"])
        if entity is None:
            errors.append(f"{label}: unknown entity_id {revision['entity_id']!r}")
        expected_hash = hashlib.sha256(revision["body"].encode("utf-8")).hexdigest()
        if revision["body_sha256"] != expected_hash:
            errors.append(f"{label}: body_sha256 does not match exact UTF-8 body")
        check_links(revision, label)
        entity_type = entity["type"] if entity else None
        if entity_type == "Review":
            if "review_kind" not in revision:
                errors.append(f"{label}: Review requires review_kind")
            if revision["assertion_state"] == "reviewed" and (
                revision.get("review_kind", "unknown") == "unknown" or not revision["evidence_ids"]
            ):
                errors.append(f"{label}: reviewed Review requires known review_kind and evidence")
        elif "review_kind" in revision or "review_bindings" in revision:
            errors.append(f"{label}: review_kind/review_bindings are only allowed on Review")
        if entity_type == "ResearchArtifact":
            if "research_progress" not in revision:
                errors.append(f"{label}: ResearchArtifact requires research_progress")
        elif "research_progress" in revision or "historical_status" in revision:
            errors.append(f"{label}: research_progress/historical_status are only allowed on ResearchArtifact")

    for relation in relations.values():
        label = f"relation {relation['relation_id']!r}"
        check_links(relation, label)
        for side in ("source", "target"):
            endpoint = relation[side]
            revision = revisions.get(endpoint["revision_id"])
            if revision is None or revision["entity_id"] != endpoint["entity_id"]:
                errors.append(f"{label}: {side} must identify an exact existing entity/revision pair")
        source_type = entities.get(relation["source"]["entity_id"], {}).get("type")
        target_type = entities.get(relation["target"]["entity_id"], {}).get("type")
        if relation["type"] == "reviews" and source_type != "Review":
            errors.append(f"{label}: reviews source must be a Review")
        if relation["type"] == "in_context" and target_type != "DefinitionContext":
            errors.append(f"{label}: in_context target must be a DefinitionContext")
        if relation["type"] == "assumes" and target_type != "Assumption":
            errors.append(f"{label}: assumes target must be an Assumption")

    # A review of a revision does not implicitly review every outgoing edge.
    # Reviewed relations need an explicit ID + digest binding covering the entire
    # relation. This guards coverage confusion, not authenticity of user JSON.
    bound_relations: set[str] = set()
    for revision in revisions.values():
        if entities.get(revision["entity_id"], {}).get("type") != "Review":
            continue
        for binding in revision.get("review_bindings", []):
            relation = relations.get(binding["relation_id"])
            if relation is None:
                errors.append(f"review {revision['revision_id']!r}: dangling relation binding {binding['relation_id']!r}")
            elif binding["sha256"] != _relation_sha256(relation, revisions, evidence):
                errors.append(f"review {revision['revision_id']!r}: relation binding sha256 mismatch for {binding['relation_id']!r}")
            elif (
                revision["assertion_state"] == "reviewed"
                and revision.get("review_kind", "unknown") != "unknown"
                and revision["evidence_ids"]
            ):
                bound_relations.add(binding["relation_id"])
    reviewed_targets: set[tuple[str, str]] = set()
    for relation in relations.values():
        if relation["assertion_state"] != "reviewed":
            continue
        label = f"relation {relation['relation_id']!r}"
        if not relation["evidence_ids"] or relation["relation_id"] not in bound_relations:
            errors.append(f"{label}: reviewed relation requires evidence and an exact Review digest binding")
            continue
        source = revisions.get(relation["source"]["revision_id"], {})
        if relation["type"] == "reviews":
            if (
                entities.get(source.get("entity_id"), {}).get("type") == "Review"
                and source.get("assertion_state") == "reviewed"
                and source.get("review_kind", "unknown") != "unknown"
                and source.get("evidence_ids")
            ):
                reviewed_targets.add((relation["target"]["entity_id"], relation["target"]["revision_id"]))
            else:
                errors.append(f"{label}: reviewed reviews source must be an evidence-backed reviewed Review")
    for revision in revisions.values():
        if revision["assertion_state"] != "reviewed":
            continue
        if entities.get(revision["entity_id"], {}).get("type") == "Review":
            continue
        if (revision["entity_id"], revision["revision_id"]) not in reviewed_targets:
            errors.append(f"revision {revision['revision_id']!r}: reviewed requires an evidence-backed exact reviews relation")
    errors.extend(_dependency_errors(payload["relations"]))
    return _result(errors)
