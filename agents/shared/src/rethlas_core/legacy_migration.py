"""Bounded legacy-source capture, not a second knowledge authority.

A package freezes one exact request for an empty, offline fixture store. Only
MemoryStore.commit publishes it. Source records are untrusted Reference bodies;
neither a legacy verdict nor this converter confers mathematical review.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import re
from typing import Any

from jsonschema import Draft202012Validator, validators

from .contracts import _FORMAT_CHECKER, validate_math_memory
from .memory_io import read_input
from .memory_store import (
    COLLECTIONS, DEFAULT_LIMITS, MemoryCancelled, MemoryConflict, MemoryStore,
    _genesis, _stored,
)
from .private_fs import (
    PrivateDirectory, PrivateFilesystemError, canonical_json, digest, strict_json_loads,
)


VERSION = "legacy-jsonl-1"
CONVERTER = "reference-wrapper-1"
PROFILES = {
    "generation-10": (
        "immediate_conclusions", "toy_examples", "counterexamples", "big_decisions",
        "subgoals", "proof_steps", "failed_paths", "verification_reports", "branch_states", "events",
    ),
    "verification-5": (
        "statement_checks", "reference_checks", "failed_checks", "verification_reports", "events",
    ),
}
LIMITS = {"sources": 16, "file_bytes": 1048576, "raw_bytes": 4194304,
          "lines": 4096, "line_bytes": 131072, "accepted": 256,
          "bundle_bytes": 16777216, "json_depth": 64, "spec_bytes": 65536,
          "references": 4096}
REASONS = (
    "blank", "invalid_utf8", "invalid_json", "duplicate_key", "nonfinite",
    "invalid_json_unicode", "too_deep", "nonobject", "unknown_envelope",
)
_META_FILES = ("snapshot.json", "map.json", "quarantine.json", "source-manifest.json",
               "target.json", "request.json")
_HASH = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)


class MigrationError(ValueError):
    """An explicit migration contract or package check failed."""


class MigrationCapacity(MigrationError):
    """A hard input/package/store budget was exceeded; no implicit splitting."""


class MigrationQuarantine(MigrationError):
    """Preparation needs an explicit allow-quarantine source option."""


@lru_cache(maxsize=12)
def _validator(kind: str):
    schema = strict_json_loads(files("rethlas_core").joinpath(
        "schemas", "legacy_migration.schema.json").read_bytes())
    Draft202012Validator.check_schema(schema)
    strict_integer = Draft202012Validator.TYPE_CHECKER.redefine(
        "integer", lambda checker, value: type(value) is int)
    validator = validators.extend(Draft202012Validator, type_checker=strict_integer)
    return validator({"$ref": "#/$defs/" + kind, "$defs": schema["$defs"]},
                     format_checker=_FORMAT_CHECKER)


def _protocol(value: Any, kind: str) -> None:
    if not _validator(kind).is_valid(value):
        raise MigrationError("invalid migration " + kind)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _cancel(event: Any) -> None:
    if event is not None and event.is_set():
        raise MemoryCancelled("migration cancelled before publication")


def _depth(raw: bytes) -> bool:
    depth, quoted, escaped = 0, False, False
    for char in raw:
        if quoted:
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                quoted = False
        elif char == 34:
            quoted = True
        elif char in (91, 123):
            depth += 1
            if depth > LIMITS["json_depth"]:
                return False
        elif char in (93, 125):
            depth -= 1
    return True


def _json(raw: bytes) -> Any:
    if not _depth(raw):
        raise MigrationError("migration metadata exceeds JSON depth")
    try:
        return strict_json_loads(raw)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise MigrationError("invalid strict migration JSON") from exc


def _source_scope(source: dict) -> None:
    if source["channel"] not in PROFILES[source["profile"]]:
        raise MigrationError("channel is not in the explicit source profile")
    scope = source["scope"]
    if source["profile"] == "generation-10":
        valid = (type(scope["problem_id"]) is str and bool(scope["problem_id"].strip())
                 and scope["source_run_id"] is None)
    else:
        valid = (scope["problem_id"] is None and scope["branch_id"] is None
                 and type(scope["source_run_id"]) is str and bool(scope["source_run_id"].strip()))
    if not valid:
        raise MigrationError("source scope does not match its profile")


def _source_key(source: dict) -> str:
    return digest({name: source[name] for name in ("profile", "scope", "channel")})


def _raw_name(source: dict) -> str:
    return "raw-" + source["alias"] + ".jsonl"


def capture(spec_path: Path, *, cancel_event: Any = None) -> tuple[dict, dict[str, bytes], tuple[Path, ...]]:
    """Read only explicitly selected sources; never discover or execute files."""
    _cancel(cancel_event)
    spec = _json(read_input(Path(spec_path), LIMITS["spec_bytes"], single_link=True, stable=True))
    _protocol(spec, "spec")
    seen_alias, seen_scope, seen_path = set(), set(), set()
    raw_sources, sources, source_paths = {}, [], []
    total = 0
    for source in sorted(spec["sources"], key=lambda item: item["alias"]):
        _cancel(cancel_event)
        _source_scope(source)
        path = Path(source["path"])
        if not path.is_absolute() or path.suffix != ".jsonl" or ".." in path.parts:
            raise MigrationError("source must be an explicit absolute JSONL path")
        key = _source_key(source)
        if source["alias"] in seen_alias or key in seen_scope or path in seen_path:
            raise MigrationError("duplicate source alias, identity or path")
        seen_alias.add(source["alias"])
        seen_scope.add(key)
        seen_path.add(path)
        raw = read_input(path, LIMITS["file_bytes"], single_link=True, stable=True)
        total += len(raw)
        if total > LIMITS["raw_bytes"]:
            raise MigrationCapacity("raw source total exceeds capacity")
        if _sha(raw) != source["expected_sha256"]:
            raise MigrationError("source snapshot hash differs")
        descriptor = {name: source[name] for name in ("alias", "profile", "channel", "scope")}
        sources.append({**descriptor, "sha256": _sha(raw), "bytes": len(raw)})
        raw_sources[source["alias"]] = raw
        source_paths.append(path)
    snapshot = {"version": VERSION, "converter_version": CONVERTER,
                "source_namespace": spec["source_namespace"],
                "import_observed_at": spec["import_observed_at"],
                "options": spec["options"], "sources": sources}
    _protocol(snapshot, "snapshot")
    return snapshot, raw_sources, tuple(source_paths)


def _physical_lines(raw: bytes):
    offset, number = 0, 1
    while offset < len(raw):
        end = raw.find(b"\n", offset)
        end = len(raw) if end < 0 else end + 1
        yield number, offset, raw[offset:end]
        offset, number = end, number + 1


def _parse_line(raw: bytes, source: dict) -> tuple[str, dict | None]:
    if not raw.strip(b" \t\r\n"):
        return "blank", None
    try:
        text = raw.decode("utf-8")
    except UnicodeError:
        return "invalid_utf8", None
    if not _depth(raw):
        return "too_deep", None
    try:
        value = strict_json_loads(text)
    except UnicodeError:
        return "invalid_json_unicode", None
    except RecursionError:
        return "too_deep", None
    except PrivateFilesystemError as exc:
        if str(exc) == "duplicate JSON key":
            reason = "duplicate_key"
        elif str(exc) == "JSON strings must be valid UTF-8":
            reason = "invalid_json_unicode"
        else:
            reason = "nonfinite"
        return reason, None
    except (ValueError, json.JSONDecodeError):
        return "invalid_json", None
    if type(value) is not dict:
        return "nonobject", None
    if (set(value) == {"timestamp_utc", "channel", "record"}
            and type(value["timestamp_utc"]) is str and value["channel"] == source["channel"]
            and type(value["record"]) is dict):
        return "accepted_record", value
    if (source["channel"] == "events" and set(value) == {"timestamp_utc", "event_type", "channel"}
            and type(value["timestamp_utc"]) is str and value["event_type"] == "memory_append"
            and value["channel"] in PROFILES[source["profile"]][:-1]):
        return "accepted_event", value
    return "unknown_envelope", None


def _legacy_ids(source: dict, envelope: dict) -> list[dict]:
    record = envelope.get("record", {})
    keys = ["id"] if "id" in record else []
    if source["profile"] == "generation-10":
        if record.get("record_type") == "decomposition_plan" and "plan_id" in record:
            keys.append("plan_id")
        if source["channel"] == "subgoals" and record.get("record_type") == "subgoal" and "subgoal_id" in record:
            keys.append("subgoal_id")
    return [{"field": field, "value": record[field]} for field in keys]


def _links(source: dict, envelope: dict):
    if source["profile"] != "generation-10" or "record" not in envelope:
        return
    record = envelope["record"]
    fields = {"subgoal_id": "subgoals", "used_examples": "toy_examples",
              "used_counterexamples": "counterexamples", "used_results": None,
              "adapted_from": None, "supports_branch_ids": None, "plan_id": None}
    for field, target_channel in fields.items():
        if field not in record:
            continue
        if field == "plan_id" and record.get("record_type") == "decomposition_plan":
            continue
        if field == "subgoal_id" and source["channel"] == "subgoals" and record.get("record_type") == "subgoal":
            continue
        value = record[field]
        if field in ("subgoal_id", "plan_id"):
            yield field, None, value, target_channel
        elif type(value) is list:
            for index, item in enumerate(value):
                yield field, index, item, target_channel
        else:
            yield field, None, value, target_channel


def _index_key(source: dict, channel: str, old_id: str) -> str:
    return digest({"profile": source["profile"], "scope": source["scope"],
                   "channel": channel, "legacy_id": old_id})


def convert_snapshot(snapshot: dict, raw_sources: dict[str, bytes], *, actor: str) -> dict:
    """Pure, deterministic conversion. No store, credential, path or clock access."""
    snapshot = _json(canonical_json(snapshot))
    _protocol(snapshot, "snapshot")
    if type(actor) is not str or not actor.strip():
        raise MigrationError("an explicit importer observation actor is required")
    sources = snapshot["sources"]
    aliases = [source["alias"] for source in sources]
    identities = [_source_key(source) for source in sources]
    if (aliases != sorted(aliases) or len(set(aliases)) != len(aliases)
            or len(set(identities)) != len(identities) or set(raw_sources) != set(aliases)):
        raise MigrationError("source descriptors must be unique, sorted and complete")
    snapshot_sha = digest(snapshot)
    rows, accepted, index, content = [], [], defaultdict(list), Counter()
    raw_total = 0
    for source in sources:
        _source_scope(source)
        raw = raw_sources[source["alias"]]
        if type(raw) is not bytes:
            raise MigrationError("frozen source must be bytes")
        if len(raw) > LIMITS["file_bytes"]:
            raise MigrationCapacity("source file exceeds capacity")
        raw_total += len(raw)
        if raw_total > LIMITS["raw_bytes"]:
            raise MigrationCapacity("raw source total exceeds capacity")
        if source["bytes"] != len(raw) or source["sha256"] != _sha(raw):
            raise MigrationError("frozen source hash or length differs")
        for number, offset, line in _physical_lines(raw):
            if len(rows) >= LIMITS["lines"] or len(line) > LIMITS["line_bytes"]:
                raise MigrationCapacity("physical line count or size exceeds capacity")
            disposition, envelope = _parse_line(line, source)
            row = {"alias": source["alias"], "file_sha256": source["sha256"], "line": number,
                   "offset": offset, "length": len(line), "raw_sha256": _sha(line),
                   "disposition": disposition, "entity_id": None, "revision_id": None,
                   "evidence_id": None, "legacy_ids": [], "references": []}
            rows.append(row)
            if envelope is None:
                continue
            if len(accepted) >= LIMITS["accepted"]:
                raise MigrationCapacity("accepted source occurrence capacity exceeded")
            seed = {"converter_version": CONVERTER, "snapshot_sha256": snapshot_sha,
                    "source_namespace": snapshot["source_namespace"],
                    **{key: source[key] for key in ("alias", "profile", "scope", "channel")},
                    "file_sha256": source["sha256"], "offset": offset}
            identity = digest(seed)
            row.update(entity_id="legacy:entity:" + identity, revision_id="legacy:revision:" + identity,
                       evidence_id="legacy:line:" + identity, legacy_ids=_legacy_ids(source, envelope))
            occurrence = {"row": row, "source": source, "envelope": envelope,
                          "line": line, "content_sha256": digest(envelope)}
            accepted.append(occurrence)
            content[occurrence["content_sha256"]] += 1
            keys = set()
            for old in row["legacy_ids"]:
                if type(old["value"]) is str and old["value"].strip():
                    keys.add(_index_key(source, source["channel"], old["value"]))
            for key in keys:
                index[key].append(occurrence)
    relations, reference_count = [], 0
    for occurrence in accepted:
        row, source = occurrence["row"], occurrence["source"]
        for field, position, value, channel in _links(source, occurrence["envelope"]):
            reference_count += 1
            if reference_count > LIMITS["references"]:
                raise MigrationCapacity("source reference index capacity exceeded")
            targets = []
            state = "unresolved"
            if type(value) is not str or not value.strip():
                state = "invalid_type"
            elif channel is not None:
                targets = index.get(_index_key(source, channel, value), [])
                state = "resolved" if len(targets) == 1 else "ambiguous" if targets else "unresolved"
            reference = {"field": field, "index": position, "value": value,
                         "target_channel": channel, "status": state,
                         "target_revision_id": targets[0]["row"]["revision_id"] if state == "resolved" else None}
            row["references"].append(reference)
            if state == "resolved":
                if len(relations) >= DEFAULT_LIMITS["entries"]:
                    raise MigrationCapacity("candidate citation capacity exceeded")
                target = targets[0]["row"]
                relations.append({"relation_id": "legacy:cites:" + digest({
                    "source": row["revision_id"], "field": field, "index": position,
                    "target": target["revision_id"]}), "type": "cites",
                    "source": {key: row[key] for key in ("entity_id", "revision_id")},
                    "target": {key: target[key] for key in ("entity_id", "revision_id")},
                    "assertion_state": "candidate", "context_revision_ids": [],
                    "evidence_ids": list(dict.fromkeys([row["evidence_id"], target["evidence_id"]]))})
    reasons = {reason: sum(row["disposition"] == reason for row in rows) for reason in REASONS}
    counts = {"source_files": len(sources), "raw_bytes": raw_total, "physical_lines": len(rows),
              "accepted_lines": len(accepted), "record_lines": sum(
                  row["disposition"] == "accepted_record" for row in rows),
              "event_lines": sum(row["disposition"] == "accepted_event" for row in rows),
              "quarantined_lines": sum(reasons.values()), "reasons": reasons,
              "duplicate_content_occurrences": sum(count - 1 for count in content.values()),
              "duplicate_id_groups": sum(len(items) > 1 for items in index.values()),
              "conflicting_id_groups": sum(len({item["content_sha256"] for item in items}) > 1
                                            for items in index.values()),
              "references": {state: sum(ref["status"] == state for row in rows for ref in row["references"])
                             for state in ("resolved", "unresolved", "ambiguous", "invalid_type")}}
    files_for_map = {source["alias"]: raw_sources[source["alias"]] for source in sources}
    cursors = {alias: 0 for alias in aliases}
    for row in rows:
        if row["offset"] != cursors[row["alias"]]:
            raise MigrationError("source byte intervals are not contiguous")
        cursors[row["alias"]] += row["length"]
    if (any(cursors[alias] != len(raw) for alias, raw in files_for_map.items())
            or counts["physical_lines"] != counts["accepted_lines"] + counts["quarantined_lines"]
            or counts["accepted_lines"] != counts["record_lines"] + counts["event_lines"]):
        raise MigrationError("source byte or line accounting differs")
    mapping = {"version": "legacy-map-1", "snapshot_sha256": snapshot_sha, "counts": counts, "lines": rows}
    quarantine = {"version": "legacy-quarantine-1", "snapshot_sha256": snapshot_sha,
                  "lines": [{key: row[key] for key in ("alias", "file_sha256", "line", "offset", "length", "raw_sha256", "disposition")}
                            for row in rows if row["disposition"] in REASONS]}
    commitment = {"version": "legacy-source-manifest-1", "snapshot_sha256": snapshot_sha,
                  "map_sha256": digest(mapping), "quarantine_sha256": digest(quarantine),
                  "raw_files": [{"name": _raw_name(source), "bytes": source["bytes"], "sha256": source["sha256"]}
                                for source in sources], "counts": counts}
    for value, kind in ((mapping, "map"), (quarantine, "quarantine"), (commitment, "source_manifest")):
        _protocol(value, kind)
    commitment_sha = digest(commitment)
    provenance = {"source": "legacy-jsonl://snapshot/" + snapshot_sha,
                  "actor": actor, "recorded_at": snapshot["import_observed_at"]}
    source_evidence = "legacy:source:" + commitment_sha
    graph = {"schema_version": "1.0", **{key: [] for key in COLLECTIONS}}
    if accepted:
        graph["evidence"].append({"evidence_id": source_evidence,
            "locator": provenance["source"] + "/source-manifest.json", "sha256": commitment_sha,
            "provenance": dict(provenance)})
    for occurrence in accepted:
        row, source = occurrence["row"], occurrence["source"]
        graph["entities"].append({"entity_id": row["entity_id"], "type": "Reference",
            "name": "Legacy source record: " + source["alias"] + "/" + source["channel"] + ":" + str(row["line"]),
            "provenance": dict(provenance)})
        graph["revisions"].append({"entity_id": row["entity_id"], "revision_id": row["revision_id"],
            "body": occurrence["line"].decode("utf-8"), "body_sha256": row["raw_sha256"],
            "provenance": dict(provenance), "assertion_state": "candidate", "validity": "needs_review",
            "context_revision_ids": [], "evidence_ids": [row["evidence_id"], source_evidence]})
        graph["evidence"].append({"evidence_id": row["evidence_id"], "sha256": row["raw_sha256"],
            "locator": (provenance["source"] + "/" + _raw_name(source) + "/" + source["sha256"]
                        + "?offset=" + str(row["offset"]) + "&length=" + str(row["length"])
                        + "&line=" + str(row["line"])), "provenance": dict(provenance)})
    graph["relations"] = [{**relation, "provenance": dict(provenance)} for relation in relations]
    if not validate_math_memory(graph)["valid"]:
        raise MigrationError("converted Reference graph is structurally invalid")
    return {"snapshot": snapshot, "snapshot_sha256": snapshot_sha, "map": mapping,
            "quarantine": quarantine, "source_manifest": commitment, "graph": graph, "counts": counts}


def _actor(target: dict) -> str:
    return "legacy-jsonl-import/" + target["owner"] + "/" + target["run_id"]


def _request(conversion: dict, target: dict) -> dict:
    identity = {"version": VERSION, "converter_version": CONVERTER,
                "snapshot_sha256": conversion["snapshot_sha256"], "target": target,
                "graph_sha256": digest(conversion["graph"])}
    return {"version": 1, "request_id": "legacy-import:" + digest(identity),
            "expected_head": target["expected_head"],
            **{key: conversion["graph"][key] for key in COLLECTIONS}}


def _budgets(conversion: dict, request: dict, limits: dict) -> dict:
    graph = conversion["graph"]
    stored_bytes = len(canonical_json(_stored(graph)))
    body_sizes = [len(item["body"].encode("utf-8")) for item in graph["revisions"]]
    budgets = {"request_bytes": len(canonical_json(request)), "graph_bytes": stored_bytes,
               "exchange_bytes": len(canonical_json(graph)),
               "hydration_bytes": stored_bytes + sum(body_sizes),
               "max_body_bytes": max(body_sizes, default=0),
               "entries": {key: len(graph[key]) for key in COLLECTIONS}}
    if (budgets["request_bytes"] > limits["request_bytes"] or stored_bytes > limits["graph_bytes"]
            or max(budgets["exchange_bytes"], budgets["hydration_bytes"]) > limits["exchange_bytes"]
            or budgets["max_body_bytes"] > limits["body_bytes"]
            or any(value > limits["entries"] for value in budgets["entries"].values())):
        raise MigrationCapacity("converted request exceeds target graph or request capacity")
    return budgets


def _summary(conversion: dict, budgets: dict, *, status: str) -> dict:
    return {"status": status, "snapshot_sha256": conversion["snapshot_sha256"],
            "counts": conversion["counts"], "budgets": budgets,
            "mathematically_reviewed": False, "live_connected": False}


def dry_run(spec_path: Path, *, cancel_event: Any = None) -> dict:
    snapshot, raw, _ = capture(spec_path, cancel_event=cancel_event)
    conversion = convert_snapshot(snapshot, raw, actor="legacy-jsonl-import/dry-run")
    request = _request(conversion, {"expected_head": "0" * 64})
    budgets = _budgets(conversion, request, DEFAULT_LIMITS)
    _cancel(cancel_event)
    return {**_summary(conversion, budgets, status="dry-run"), "target_checked": False,
            "requires_allow_quarantine": bool(conversion["counts"]["quarantined_lines"]
                                              and not snapshot["options"]["allow_quarantine"])}


def _target(store: MemoryStore, claims: dict) -> dict:
    if store.config["domain"] != "fixture":
        raise MigrationError("migration accepts only an offline fixture store")
    value = {"version": "legacy-target-1", "store_id": store.config["store_id"],
             "config_sha256": store.config_sha256, "owner": claims["owner"],
             "run_id": claims["run_id"], "domain": claims["domain"], "epoch": claims["epoch"],
             "expected_head": _genesis(store.config["store_id"])["head"], "limits": store.limits}
    _protocol(value, "target")
    return value


def _claims(store: MemoryStore, token: str, run_id: str, action: str) -> dict:
    return store.authority.check(token, owner=store.config["owner"], run_id=run_id,
                                 domain=store.config["domain"], action=action, now=store.clock())


def _empty_staging(store: MemoryStore, token: str, run_id: str) -> tuple[dict, dict]:
    status = store.status(token, run_id=run_id)
    if (status["generation"] != 0 or any(status["counts"].values())
            or status["pending"]["status"] != "empty"):
        raise MemoryConflict("preparation requires a new empty staging store")
    with store._locked(token, run_id, "memory.read") as claims:
        usage = store._inventory()
        chain, graph = store._chain()
        if chain[0][0]["generation"] != 0 or any(graph[key] for key in COLLECTIONS):
            raise MemoryConflict("staging head changed during preparation")
        genesis = _genesis(store.config["store_id"])
        allowed = {"store.json", "manifest.json", "lock", "authority/key", "authority/lock",
                   "authority/registry.json", "journal/pending.log", "graphs/" + genesis["graph_sha256"] + ".json"}
        if set(usage["sizes"]) != allowed or usage["sizes"]["journal/pending.log"] != 0:
            raise MemoryConflict("staging contains pending, orphan or previously used objects")
        return _target(store, claims), usage


def _absolute(path: Path) -> Path:
    path = Path(path).absolute()
    if ".." in path.parts or any(item.is_symlink() for item in (path, *path.parents)):
        raise MigrationError("migration paths must not contain symlinks or traversal")
    return path


def _overlap(left: Path, right: Path) -> bool:
    return left.is_relative_to(right) or right.is_relative_to(left)


def _location(bundle: Path, store: MemoryStore, sources: tuple[Path, ...] = (),
              forbidden_roots: tuple[Path, ...] = ()) -> Path:
    bundle = _absolute(bundle)
    blocked = [store.root, *(path.parent for path in sources), *forbidden_roots]
    if any(_overlap(bundle, _absolute(root)) for root in blocked):
        raise MigrationError("package overlaps a store, source or protected directory")
    if any(_overlap(_absolute(path), store.root) for path in sources):
        raise MigrationError("legacy inputs must be outside the target store")
    return bundle


def _members(conversion: dict, target: dict, request: dict, raw: dict[str, bytes]) -> dict[str, bytes]:
    result = {"snapshot.json": canonical_json(conversion["snapshot"]),
              "map.json": canonical_json(conversion["map"]),
              "quarantine.json": canonical_json(conversion["quarantine"]),
              "source-manifest.json": canonical_json(conversion["source_manifest"]),
              "target.json": canonical_json(target), "request.json": canonical_json(request)}
    for source in conversion["snapshot"]["sources"]:
        result[_raw_name(source)] = raw[source["alias"]]
    return result


def _description(members: dict[str, bytes], conversion: dict) -> dict:
    return {"version": "legacy-bundle-1", "snapshot_sha256": conversion["snapshot_sha256"],
            "files": [{"name": name, "bytes": len(raw), "sha256": _sha(raw)}
                      for name, raw in sorted(members.items())]}


def prepare(spec_path: Path, bundle_root: Path, store: MemoryStore, token: str, *, run_id: str,
            cancel_event: Any = None, forbidden_roots: tuple[Path, ...] = ()) -> dict:
    _cancel(cancel_event)
    _claims(store, token, run_id, "memory.read")
    target, usage = _empty_staging(store, token, run_id)
    snapshot, raw, source_paths = capture(spec_path, cancel_event=cancel_event)
    bundle = _location(bundle_root, store, source_paths, forbidden_roots)
    conversion = convert_snapshot(snapshot, raw, actor=_actor(target))
    if conversion["counts"]["quarantined_lines"] and not snapshot["options"]["allow_quarantine"]:
        raise MigrationQuarantine("quarantined source lines require explicit consent")
    request = _request(conversion, target)
    budgets = _budgets(conversion, request, store.limits)
    if not conversion["counts"]["accepted_lines"]:
        return _summary(conversion, budgets, status="no-op")
    unique_bodies = {item["body_sha256"]: item["body"].encode("utf-8") for item in conversion["graph"]["revisions"]}
    reserve = sum(map(len, unique_bodies.values())) + budgets["graph_bytes"] + 3 * 16384
    reserve += 2 * max(budgets["graph_bytes"], budgets["max_body_bytes"], 16384)
    if (usage["bytes"] + reserve > store.limits["store_bytes"]
            or usage["files"] + len(unique_bodies) + 4 > store.limits["files"]):
        raise MigrationCapacity("insufficient staging capacity for an atomic import")
    members = _members(conversion, target, request, raw)
    description = _description(members, conversion)
    _protocol(description, "bundle")
    if sum(map(len, members.values())) + len(canonical_json(description)) > LIMITS["bundle_bytes"]:
        raise MigrationCapacity("frozen package exceeds capacity")
    _cancel(cancel_event)
    # Recheck current capability scope, but do not create a new grant or hold a
    # package lock while acquiring authority/store locks.
    if canonical_json(_target(store, _claims(store, token, run_id, "memory.read"))) != canonical_json(target):
        raise MemoryConflict("importer scope changed during preparation")
    directory = PrivateDirectory(bundle, create=True)
    try:
        nonempty = directory.entries(max_entries=2)
    except PrivateFilesystemError as exc:
        raise MigrationError("preparation requires a new empty private package") from exc
    if nonempty:
        raise MigrationError("preparation requires a new empty private package")
    with directory.lock():
        if {name for name, _, _ in directory.entries(max_entries=3)} != {"lock"}:
            raise MigrationError("private package changed during preparation")
        for name, data in members.items():
            _cancel(cancel_event)
            directory.write_bytes(name, data)
        directory.write_json("manifest.json", description, before_replace=lambda: _cancel(cancel_event))
    return {**_summary(conversion, budgets, status="prepared"), "request_id": request["request_id"],
            "bundle_sha256": digest(description), "target_checked": True}


def _read_bundle(bundle: Path, expected: str) -> dict[str, bytes]:
    if type(expected) is not str or not _HASH.fullmatch(expected):
        raise MigrationError("an external expected bundle SHA-256 is required")
    directory = PrivateDirectory(bundle)
    entries = directory.entries(max_entries=32)
    if any(is_dir for _, is_dir, _ in entries) or "lock" not in {name for name, _, _ in entries}:
        raise MigrationError("package is absent or incompletely initialized")
    with directory.lock():
        entries = directory.entries(max_entries=32)
        if any(is_dir for _, is_dir, _ in entries) or sum(size for _, _, size in entries) > LIMITS["bundle_bytes"]:
            raise MigrationCapacity("package inventory exceeds capacity")
        if dict((name, size) for name, _, size in entries).get("lock") != 0:
            raise MigrationError("package lock must have empty contents")
        raw_description = directory.read_bytes("manifest.json", max_bytes=LIMITS["spec_bytes"])
        description = _json(raw_description)
        _protocol(description, "bundle")
        if raw_description != canonical_json(description) or _sha(raw_description) != expected:
            raise MigrationError("package pin or canonical manifest differs")
        names = [item["name"] for item in description["files"]]
        if names != sorted(set(names)) or set(names) | {"lock", "manifest.json"} != {name for name, _, _ in entries}:
            raise MigrationError("unknown, duplicate or missing package member")
        members = {}
        sizes = {name: size for name, _, size in entries}
        for item in description["files"]:
            if sizes[item["name"]] != item["bytes"]:
                raise MigrationError("package member hash or length differs")
            try:
                data = directory.read_bytes(item["name"], max_bytes=item["bytes"])
            except PrivateFilesystemError as exc:
                # A bounded read can also observe growth after the inventory.
                # Keep unsafe permissions/types/links as filesystem rejections.
                if str(exc) != "private file exceeds max_bytes":
                    raise
                raise MigrationError("package member hash or length differs") from exc
            if len(data) != item["bytes"] or _sha(data) != item["sha256"]:
                raise MigrationError("package member hash or length differs")
            members[item["name"]] = data
        members["manifest.json"] = raw_description
        return members


def apply(bundle_root: Path, expected_bundle_sha256: str, store: MemoryStore, token: str, *,
          run_id: str, cancel_event: Any = None) -> dict:
    _cancel(cancel_event)
    bundle = _location(bundle_root, store)
    claims = _claims(store, token, run_id, "memory.commit")
    members = _read_bundle(bundle, expected_bundle_sha256)
    if not set(_META_FILES).issubset(members):
        raise MigrationError("package lacks required metadata")
    snapshot, target = _json(members["snapshot.json"]), _json(members["target.json"])
    _protocol(snapshot, "snapshot")
    _protocol(target, "target")
    if canonical_json(target) != canonical_json(_target(store, claims)):
        raise MemoryConflict("frozen target store, configuration or importer scope differs")
    raw_names = {_raw_name(source) for source in snapshot["sources"]}
    if set(members) != set(_META_FILES) | raw_names | {"manifest.json"}:
        raise MigrationError("package members do not match its source snapshot")
    raw = {source["alias"]: members[_raw_name(source)] for source in snapshot["sources"]}
    conversion = convert_snapshot(snapshot, raw, actor=_actor(target))
    if (not conversion["counts"]["accepted_lines"] or
            (conversion["counts"]["quarantined_lines"] and not snapshot["options"]["allow_quarantine"])):
        raise MigrationError("package does not authorize a nonempty accepted subset")
    request = _request(conversion, target)
    _budgets(conversion, request, store.limits)
    reconstructed = _members(conversion, target, request, raw)
    if any(members[name] != data for name, data in reconstructed.items()):
        raise MigrationError("deterministic package conversion or exact request differs")
    if members["manifest.json"] != canonical_json(_description(reconstructed, conversion)):
        raise MigrationError("package description differs from its deterministic sources")
    _cancel(cancel_event)
    # No status/export/recovery or refreshed expected_head: committed exact
    # replay must still work after the first import has advanced the head.
    result = store.commit(token, request, run_id=run_id, cancel_event=cancel_event)
    return {**result, "bundle_sha256": expected_bundle_sha256}
