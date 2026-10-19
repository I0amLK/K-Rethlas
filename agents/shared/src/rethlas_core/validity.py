"""Pure, bounded current-input projections; never mathematical review authority.

The trusted service supplies committed graphs, frozen projections and append-only
withdrawal events. Nothing here reads files, follows locators, issues tokens or
changes a revision. Citation disclosure and mathematical support stay separate.
"""
from __future__ import annotations

from collections import deque
import hashlib
from typing import Any

from .context_views import (
    BODY_BYTES, GRAPH_BYTES, MAX_DEPTH, MAX_ENTRIES, MAX_REVISIONS,
    ContextCapacity, checked_graph, copied, exact_ref, indices,
    mathematical_projection, protocol as context_protocol,
)
from .private_fs import canonical_json, digest


VERSION = "validity-1"
POLICY_VERSION = "current-fixture-1"
MATH_DEPENDENCIES = frozenset({"depends_on", "assumes", "in_context"})
POLICY = {
    "version": POLICY_VERSION,
    "domain": "fixture",
    "support_relations": sorted(MATH_DEPENDENCIES),
    "revision_contexts": "support_from_support_source",
    "relation_contexts": "support_only_for_mathematical_edges_or_root_attempt",
    "citations": "disclosure_only_unless_independently_supported",
    "review_inputs": "reviews_target_and_exact_bound_relation_snapshots",
    "review_authority": False,
    "mathematical_review": False,
    "production_completed": False,
}
POLICY_SHA256 = digest(POLICY)
MAX_WITHDRAWALS = 32
MAX_TARGETS = 32
MAX_USES = 8
CONTROL_BYTES = 4194304
RESULT_BYTES = 134217728
MAX_QUEUE_ENTRIES = MAX_WITHDRAWALS * (2 * MAX_ENTRIES + MAX_USES * MAX_TARGETS)
_PROJECTION_FIELDS = frozenset({"roots", "revisions", "relations", "support", "citation_only"})


class ValidityError(ValueError):
    """A current-validity contract was rejected, without exposing body details."""


class ValidityConflict(ValidityError):
    """An immutable identity or permanent request binding differs."""


class ValidityCapacity(ValidityError):
    """An explicit finite bound was reached; no truncation or implicit retry."""


class ValidityRecoveryRequired(ValidityError):
    """An operator must inspect the exact prepared or pending transition."""


class ValidityIndeterminate(ValidityError):
    """A durable transition may have happened; never report no effect."""


def _copy(value: Any, maximum: int) -> Any:
    try:
        return copied(value, maximum)
    except ContextCapacity as exc:
        raise ValidityCapacity("validity object byte capacity reached") from exc
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValidityError("invalid strict validity object") from exc


def _graph(value: dict) -> dict:
    try:
        return checked_graph(value, revisions=MAX_ENTRIES)
    except ContextCapacity as exc:
        raise ValidityCapacity("validity graph capacity reached") from exc
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValidityError("invalid committed mathematical graph") from exc


def _protocol(value: Any, kind: str, maximum: int = GRAPH_BYTES) -> Any:
    try:
        return context_protocol(value, kind, maximum)
    except ContextCapacity as exc:
        raise ValidityCapacity("validity projection byte capacity reached") from exc
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValidityError("invalid exact mathematical projection") from exc


def _withdrawals(value: list, revision_index: dict | None = None) -> list[dict]:
    value = _copy(value, CONTROL_BYTES)
    if type(value) is not list or len(value) > MAX_WITHDRAWALS:
        raise ValidityCapacity("withdrawal event capacity reached")
    seen_events = set()
    for event in value:
        if (type(event) is not dict or not {"event_id", "targets"} <= set(event)
                or type(event["event_id"]) is not str or not event["event_id"].strip()
                or len(event["event_id"].encode("utf-8")) > 4096):
            raise ValidityError("invalid withdrawal event identity")
        if event["event_id"] in seen_events:
            raise ValidityConflict("duplicate withdrawal event identity")
        seen_events.add(event["event_id"])
        targets = _protocol(event["targets"], "refs", CONTROL_BYTES)
        if not targets or len(targets) > MAX_TARGETS:
            raise ValidityCapacity("withdrawal target capacity reached")
        seen_targets = set()
        for target in targets:
            key = target["revision_id"]
            if key in seen_targets:
                raise ValidityConflict("duplicate exact withdrawal target")
            seen_targets.add(key)
            if revision_index is not None:
                revision = revision_index.get(key)
                if revision is None or target != exact_ref(revision):
                    raise ValidityConflict("withdrawal target is not the exact committed revision")
    return value


def _support_adjacency(revisions: dict, relations: list[dict], *, proof_pair=None) -> dict[str, set[str]]:
    result = {key: set(revision["context_revision_ids"]) for key, revision in revisions.items()}
    for relation in relations:
        source = relation["source"]["revision_id"]
        target = relation["target"]["revision_id"]
        if relation["type"] in MATH_DEPENDENCIES:
            result[source].add(target)
            result[source].update(relation["context_revision_ids"])
        elif relation["type"] == "attempts_to_prove" and (source, target) == proof_pair:
            # Both roots already belong to the use. This does not prove the Claim
            # or introduce a global invalidation edge from proof to Claim.
            result[source].update(relation["context_revision_ids"])
    return result


def _support_ids(revisions: dict, relations: list[dict], roots: dict) -> set[str]:
    pair = (roots["proof"]["revision_id"], roots["claim"]["revision_id"])
    adjacency = _support_adjacency(revisions, relations, proof_pair=pair)
    pending = deque((root["revision_id"], 0) for root in roots.values())
    visited = set()
    while pending:
        key, depth = pending.popleft()
        if key in visited:
            continue
        if key not in revisions:
            raise ValidityError("support requires an undisclosed revision")
        if depth > MAX_DEPTH or len(visited) >= MAX_REVISIONS:
            raise ValidityCapacity("support closure capacity reached")
        visited.add(key)
        pending.extend((other, depth + 1) for other in sorted(adjacency[key] - visited))
    return visited


def project_use(graph: dict, *, roots: dict, dependencies: list, references: list) -> dict:
    """Disclose the complete minimal view and independently classify its support."""
    try:
        projection = mathematical_projection(graph, roots=roots, dependencies=dependencies, references=references)
    except ContextCapacity as exc:
        raise ValidityCapacity("registered use closure capacity reached") from exc
    except (ValueError, TypeError, RecursionError) as exc:
        raise ValidityError("invalid registered mathematical use") from exc
    revisions = {item["revision_id"]: item for item in projection["revisions"]}
    support = _support_ids(revisions, projection["relations"], projection["roots"])
    projection["support"] = [exact_ref(revisions[key]) for key in sorted(support)]
    projection["citation_only"] = [exact_ref(revisions[key]) for key in sorted(set(revisions) - support)]
    return _copy(projection, GRAPH_BYTES)


def _projection(value: dict) -> dict:
    """Check a frozen pure projection without trusting its claimed partition."""
    value = _copy(value, GRAPH_BYTES)
    if type(value) is not dict or set(value) != _PROJECTION_FIELDS:
        raise ValidityError("invalid frozen use fields")
    roots = value["roots"]
    if type(roots) is not dict or set(roots) != {"problem", "claim", "proof"}:
        raise ValidityError("invalid fixed mathematical roots")
    _protocol(list(roots.values()), "refs")
    revisions = value["revisions"]
    relations = value["relations"]
    if type(revisions) is not list or not 3 <= len(revisions) <= MAX_REVISIONS:
        raise ValidityCapacity("frozen use revision capacity reached")
    if type(relations) is not list or len(relations) > MAX_ENTRIES:
        raise ValidityCapacity("frozen use relation capacity reached")
    by_id = {}
    for revision in revisions:
        _protocol(revision, "dto_revision")
        key = revision["revision_id"]
        if key in by_id:
            raise ValidityConflict("duplicate frozen revision identity")
        if len(revision["body"].encode("utf-8")) > BODY_BYTES:
            raise ValidityCapacity("frozen body capacity reached")
        if hashlib.sha256(revision["body"].encode("utf-8")).hexdigest() != revision["body_sha256"]:
            raise ValidityConflict("frozen body digest differs")
        by_id[key] = revision
    for name, kind in (("problem", "Problem"), ("claim", "Claim"), ("proof", "ProofArtifact")):
        revision = by_id.get(roots[name]["revision_id"])
        if revision is None or roots[name] != exact_ref(revision) or revision["type"] != kind:
            raise ValidityConflict("fixed mathematical root differs")
    relation_ids = set()
    for relation in relations:
        _protocol(relation, "dto_relation")
        key = relation["relation_id"]
        if key in relation_ids:
            raise ValidityConflict("duplicate frozen relation identity")
        relation_ids.add(key)
        for side in ("source", "target"):
            endpoint = relation[side]
            revision = by_id.get(endpoint["revision_id"])
            if revision is None or revision["entity_id"] != endpoint["entity_id"]:
                raise ValidityConflict("frozen relation endpoint differs")
        for context in relation["context_revision_ids"]:
            if context not in by_id or by_id[context]["type"] != "DefinitionContext":
                raise ValidityConflict("frozen relation context differs")
    for revision in revisions:
        for context in revision["context_revision_ids"]:
            if context not in by_id or by_id[context]["type"] != "DefinitionContext":
                raise ValidityConflict("frozen revision context differs")
    pair = (roots["proof"]["revision_id"], roots["claim"]["revision_id"])
    if not any(r["type"] == "attempts_to_prove" and (r["source"]["revision_id"], r["target"]["revision_id"]) == pair
               for r in relations):
        raise ValidityError("frozen proof does not target the exact claim")
    outgoing: dict[str, list[dict]] = {}
    for relation in relations:
        source = relation["source"]["revision_id"]
        target = relation["target"]["revision_id"]
        if relation["type"] == "attempts_to_prove" and (source, target) != pair:
            raise ValidityError("unsupported frozen proof attempt")
        if relation["type"] == "in_context" and by_id[target]["type"] != "DefinitionContext":
            raise ValidityError("frozen mathematical context has the wrong type")
        if relation["type"] == "assumes" and by_id[target]["type"] != "Assumption":
            raise ValidityError("frozen mathematical assumption has the wrong type")
        outgoing.setdefault(source, []).append(relation)
    pending = deque((root["revision_id"], 0) for root in roots.values())
    disclosed = set()
    while pending:
        key, depth = pending.popleft()
        if key in disclosed:
            continue
        if depth > MAX_DEPTH:
            raise ValidityCapacity("frozen disclosure depth capacity reached")
        disclosed.add(key)
        required = set(by_id[key]["context_revision_ids"])
        for relation in outgoing.get(key, ()):
            required.add(relation["target"]["revision_id"])
            required.update(relation["context_revision_ids"])
        pending.extend((target, depth + 1) for target in sorted(required - disclosed))
    if disclosed != set(by_id):
        raise ValidityConflict("frozen use contains undeclared extra disclosure")
    support = _support_ids(by_id, relations, roots)
    for field, expected in (("support", support), ("citation_only", set(by_id) - support)):
        _protocol(value[field], "refs")
        if value[field] != [exact_ref(by_id[key]) for key in sorted(expected)]:
            raise ValidityConflict("frozen mathematical use partition differs")
    return value


def _taint(revisions: dict, adjacency: dict[str, set[str]], withdrawn: set[str]) -> dict[str, set[str]]:
    reverse = {key: set() for key in revisions}
    for source, targets in adjacency.items():
        for target in targets:
            reverse[target].add(source)
    blockers = {key: set() for key in revisions}
    for target in sorted(withdrawn):
        pending, visited = deque([target]), set()
        while pending:
            key = pending.popleft()
            if key in visited:
                continue
            if len(visited) >= MAX_ENTRIES:
                raise ValidityCapacity("withdrawal propagation capacity reached")
            visited.add(key)
            blockers[key].add(target)
            pending.extend(sorted(reverse[key] - visited))
    return blockers


def _review_inputs(graph: dict, index: dict) -> dict[str, set[str]]:
    revisions = index["revisions"]
    reviews = {key: set() for key, item in revisions.items()
               if index["entities"][item["entity_id"]]["type"] == "Review"}
    for relation in graph["relations"]:
        if relation["type"] == "reviews":
            reviews[relation["source"]["revision_id"]].update(
                [relation["target"]["revision_id"], *relation["context_revision_ids"]])
    for review, inputs in reviews.items():
        pending = []
        for binding in revisions[review].get("review_bindings", []):
            relation = index["relations"][binding["relation_id"]]
            pending.extend([relation["source"]["revision_id"], relation["target"]["revision_id"],
                            *relation["context_revision_ids"]])
        visited = set()
        while pending:
            key = pending.pop()
            if key in visited:
                continue
            if len(visited) >= MAX_ENTRIES:
                raise ValidityCapacity("review input closure capacity reached")
            visited.add(key)
            # The source's own immutable snapshot is hash-bound but is not a
            # new premise about itself; an explicit reviews self-edge is kept.
            if key != review:
                inputs.add(key)
            pending.extend(revisions[key]["context_revision_ids"])
    return reviews


def _states(graph: dict, index: dict, withdrawn: set[str]) -> dict:
    revisions = index["revisions"]
    blockers = _taint(revisions, _support_adjacency(revisions, graph["relations"]), withdrawn)
    review_inputs = _review_inputs(graph, index)
    review_blockers = {key: set() for key in review_inputs}
    reverse_reviews: dict[str, set[str]] = {}
    for review, inputs in review_inputs.items():
        for key in inputs:
            reverse_reviews.setdefault(key, set()).add(review)
    pending = deque(key for key, blocked in blockers.items() if blocked)
    while pending:
        key = pending.popleft()
        incoming = blockers[key] | review_blockers.get(key, set())
        for review in sorted(reverse_reviews.get(key, ())):
            extra = incoming - review_blockers[review]
            if extra:
                review_blockers[review].update(extra)
                pending.append(review)
    result, size = {}, 0
    for key in sorted(revisions):
        value = {"directly_withdrawn": key in withdrawn, "math_blocked": bool(blockers[key]),
                 "blockers": [exact_ref(revisions[target]) for target in sorted(blockers[key])],
                 "review_input_obsolete": bool(review_blockers.get(key)),
                 "review_blockers": [exact_ref(revisions[target]) for target in sorted(review_blockers.get(key, ()))]}
        size += len(canonical_json(value)) + len(key.encode("utf-8")) + 8
        if size > RESULT_BYTES:
            raise ValidityCapacity("validity result byte capacity reached")
        result[key] = value
    return result


def validity_states(graph: dict, withdrawals: list) -> dict:
    """Reverse only fixed mathematical dependencies; Review inputs are one-way."""
    graph = _graph(graph)
    index = indices(graph)
    events = _withdrawals(withdrawals, index["revisions"])
    withdrawn = {target["revision_id"] for event in events for target in event["targets"]}
    return _states(graph, index, withdrawn)


def use_overlay(projection: dict, withdrawals: list) -> dict:
    """Current semantic overlay without global head/sequence or private reasons.

    Support was closed when the service registered the use. Withdrawal anywhere
    in that exact closure blocks the use; citations remain disclosed historical
    inputs and cannot be promoted by a lookup or by the worker's own assertions.
    """
    projection = _projection(projection)
    events = _withdrawals(withdrawals)
    disclosed = {r["revision_id"]: r for r in projection["revisions"]}
    withdrawn = {}
    for event in events:
        for target in event["targets"]:
            key = target["revision_id"]
            old = withdrawn.get(key)
            if old is not None and old != target:
                raise ValidityConflict("withdrawal identity has different exact content")
            if key in disclosed and target != exact_ref(disclosed[key]):
                raise ValidityConflict("withdrawal differs from frozen exact input")
            withdrawn[key] = target
    support = [{**target, "withdrawn": target["revision_id"] in withdrawn} for target in projection["support"]]
    citations = [{**target, "withdrawn": target["revision_id"] in withdrawn,
                  "use": "historical_citation_only", "not_authorized_as_premise": True}
                 for target in projection["citation_only"]]
    blockers = [dict(target) for target in projection["support"] if target["revision_id"] in withdrawn]
    return {"policy_hash": POLICY_SHA256, "fixture_input_eligible": not blockers,
            "support": support, "citation_only": citations, "blockers": blockers}


def review_queue(graph: dict, withdrawals: list, uses: dict) -> list:
    """Deterministic notifications, not a second authority or automatic re-review."""
    graph = _graph(graph)
    index = indices(graph)
    events = _withdrawals(withdrawals, index["revisions"])
    if type(uses) is not dict or len(uses) > MAX_USES:
        raise ValidityCapacity("registered use capacity reached")
    projections = {}
    for use_id, projection in uses.items():
        if type(use_id) is not str or not use_id.strip() or len(use_id.encode("utf-8")) > 4096:
            raise ValidityError("invalid fixed use identity")
        checked = _projection(projection)
        for revision in checked["revisions"]:
            committed = index["revisions"].get(revision["revision_id"])
            if committed is None or exact_ref(committed) != exact_ref(revision):
                raise ValidityConflict("registered queue input differs from committed revision")
        projections[use_id] = checked
    entries, size = {}, 0

    def add(event_id, kind, revision, use_id=None):
        nonlocal size
        fields = {"withdrawal_event": event_id, "kind": kind, "revision": revision, "use_id": use_id}
        key = digest(fields)
        value = {"key": key, **fields}
        if key not in entries:
            size += len(canonical_json(value)) + 1
            if len(entries) >= MAX_QUEUE_ENTRIES or size > RESULT_BYTES:
                raise ValidityCapacity("derived review queue capacity reached")
            entries[key] = value

    for event in events:
        withdrawn = {target["revision_id"] for target in event["targets"]}
        states = _states(graph, index, withdrawn)
        for key, state in states.items():
            revision = exact_ref(index["revisions"][key])
            if state["directly_withdrawn"]:
                add(event["event_id"], "direct_withdrawal", revision)
            elif state["math_blocked"]:
                add(event["event_id"], "math_dependency_blocked", revision)
            if state["review_input_obsolete"]:
                add(event["event_id"], "review_input_obsolete", revision)
        for use_id, projection in sorted(projections.items()):
            for target in projection["support"]:
                if target["revision_id"] in withdrawn:
                    add(event["event_id"], "math_dependency_blocked", target, use_id)
            for target in projection["citation_only"]:
                if target["revision_id"] in withdrawn:
                    add(event["event_id"], "citation_refresh", target, use_id)
    return [entries[key] for key in sorted(entries)]
