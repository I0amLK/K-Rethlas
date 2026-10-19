"""Pure, bounded projections of exact mathematical revisions, never a review.

Only the workspace controller supplies authenticated frozen graphs to these
functions. They do not read paths, resolve locators, sign tokens, or publish facts.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator

from .contracts import validate_math_memory
from .memory_store import COLLECTIONS, MemoryConflict, MemoryStoreError
from .private_fs import canonical_json, digest, strict_json_loads


VERSION = "context-1"
POLICY = "minimal-math-1"
META_BYTES = 65536
REQUEST_BYTES = 524288
GRAPH_BYTES = 8388608
BODY_BYTES = 131072
MAX_REVISIONS = 256
MAX_ENTRIES = 1024
MAX_DEPTH = 64
MATH_TYPES = frozenset({"Problem", "Claim", "ProofArtifact", "DefinitionContext", "Assumption", "Reference"})
DEPENDENCIES = frozenset({"in_context", "assumes", "depends_on", "cites"})


class ContextError(ValueError):
    """A context contract or restricted view was rejected, without body details."""


class ContextConflict(ContextError):
    """An immutable identity or a permanently bound request differs."""


class ContextCapacity(ContextError):
    """An explicit finite bound was reached; no truncation or implicit retry."""


class ContextRecoveryRequired(ContextError):
    """Explicit operator inspection/recovery is required."""


class ContextIndeterminate(ContextError):
    """A durable transition may have happened; inspect the exact original intent."""


@lru_cache(maxsize=32)
def _validator(kind: str) -> Draft202012Validator:
    schema = strict_json_loads(files("rethlas_core").joinpath("schemas", "context_views.schema.json").read_bytes())
    Draft202012Validator.check_schema(schema)
    if kind not in schema["$defs"]:
        raise ContextError("unknown controller protocol")
    return Draft202012Validator({"$ref": "#/$defs/" + kind, "$defs": schema["$defs"]})


def copied(value: Any, maximum: int = META_BYTES) -> Any:
    raw = canonical_json(value)
    if len(raw) > maximum:
        raise ContextCapacity("context byte capacity reached")
    return strict_json_loads(raw)


def protocol(value: Any, kind: str, maximum: int = META_BYTES) -> Any:
    value = copied(value, maximum)
    if not _validator(kind).is_valid(value):
        raise ContextError("invalid " + kind + " protocol")
    # jsonschema's integer permits 1.0; persisted counters require strict ints.
    if kind == "state" and any(type(n) is not int for n in value["counts"].values()):
        raise ContextError("context counters must be strict integers")
    if kind == "seal_receipt" and type(value["closed_epoch"]) is not int:
        raise ContextError("context epoch must be a strict integer")
    return value


def checked_graph(value: dict, *, revisions: int = MAX_REVISIONS) -> dict:
    graph = copied(value, GRAPH_BYTES)
    if type(graph) is not dict or set(graph) != {"schema_version", *COLLECTIONS}:
        raise ContextError("invalid frozen graph")
    for name in COLLECTIONS:
        if type(graph[name]) is not list or len(graph[name]) > (revisions if name == "revisions" else MAX_ENTRIES):
            raise ContextCapacity("context graph entry capacity reached")
    if not validate_math_memory(graph)["valid"]:
        raise ContextError("invalid frozen mathematical graph")
    for revision in graph["revisions"]:
        if len(revision["body"].encode("utf-8")) > BODY_BYTES:
            raise ContextCapacity("context body capacity reached")
        if revision["assertion_state"] != "candidate" or revision["validity"] != "needs_review":
            raise ContextError("context cannot promote mathematical validity")
    if any(r["assertion_state"] != "candidate" for r in graph["relations"]):
        raise ContextError("context cannot promote relation validity")
    return graph


def indices(graph: dict) -> dict:
    return {name: {item[field]: item for item in graph[name]} for name, field in COLLECTIONS.items()}


def exact_ref(revision: dict) -> dict:
    return {key: revision[key] for key in ("entity_id", "revision_id", "body_sha256")}


def refs(items: list[dict], index: dict) -> dict:
    result = {}
    for item in items:
        revision_id = item["revision_id"]
        if revision_id in result or revision_id not in index or item != exact_ref(index[revision_id]):
            raise ContextError("duplicate or mismatching exact revision reference")
        result[revision_id] = item
    return result


def active_map(items: list[dict], graph: dict, *, complete: bool = True) -> dict:
    index = indices(graph)
    result = {}
    for item in items:
        entity, revision = item["entity_id"], item["revision_id"]
        if (entity in result or revision not in index["revisions"]
                or index["revisions"][revision]["entity_id"] != entity):
            raise ContextConflict("invalid active revision selection")
        result[entity] = revision
    if complete and set(result) != {r["entity_id"] for r in graph["revisions"]}:
        raise ContextConflict("active selection must explicitly cover each represented entity")
    return result


def closure(graph: dict, roots: set[str], *, proof_pair: tuple[str, str] | None = None) -> tuple[set[str], set[str]]:
    """Traverse every outgoing edge; restricted proof views reject unsupported ones."""
    index = indices(graph)
    outgoing: dict[str, list[dict]] = {}
    for relation in graph["relations"]:
        outgoing.setdefault(relation["source"]["revision_id"], []).append(relation)
    pending = deque((key, 0) for key in sorted(roots))
    visited: set[str] = set()
    selected_relations: set[str] = set()
    while pending:
        key, depth = pending.popleft()
        if key in visited:
            continue
        if depth > MAX_DEPTH or len(visited) >= MAX_REVISIONS:
            raise ContextCapacity("dependency closure capacity reached")
        if key not in index["revisions"]:
            raise ContextError("required revision is missing")
        visited.add(key)
        revision = index["revisions"][key]
        required = set(revision["context_revision_ids"])
        for relation in outgoing.get(key, ()):
            if proof_pair is not None:
                is_proof = (relation["type"] == "attempts_to_prove"
                            and (key, relation["target"]["revision_id"]) == proof_pair)
                if relation["type"] not in DEPENDENCIES and not is_proof:
                    raise ContextError("unsupported outgoing relation in minimal mathematical view")
            selected_relations.add(relation["relation_id"])
            required.add(relation["target"]["revision_id"])
            required.update(relation["context_revision_ids"])
        pending.extend((key, depth + 1) for key in sorted(required - visited))
    return visited, selected_relations


def subgraph(graph: dict, selected: set[str], relations: set[str]) -> dict:
    index = indices(graph)
    revisions = [index["revisions"][key] for key in sorted(selected)]
    edges = [index["relations"][key] for key in sorted(relations)]
    entity_ids = {item["entity_id"] for item in revisions}
    evidence_ids = {key for item in [*revisions, *edges] for key in item["evidence_ids"]}
    result = {"schema_version": "1.0", "entities": [index["entities"][k] for k in sorted(entity_ids)],
              "revisions": revisions, "relations": edges,
              "evidence": [index["evidence"][k] for k in sorted(evidence_ids)]}
    return checked_graph(result)


def capture_graph(source: dict, spec: dict) -> dict:
    spec = protocol(spec, "capture")
    source = checked_graph(source, revisions=MAX_ENTRIES)
    allowed = refs(spec["public"], indices(source)["revisions"])
    selected, edges = closure(source, set(allowed))
    if selected != set(allowed):
        raise ContextError("public allowlist omits a required dependency")
    result = subgraph(source, selected, edges)
    if not result["revisions"]:
        raise ContextError("context baseline must contain an explicit revision")
    active_map(spec["base_active"], result)
    return result


def check_branch_request(current: dict, request: dict) -> dict:
    """Append-only outgoing structure; no retroactive premise edits, even on self-writes."""
    from .memory_store import _protocol

    request = copied(request, REQUEST_BYTES)
    _protocol(request, "request")
    index = indices(current)
    new_revisions = {r["revision_id"] for r in request["revisions"]} - set(index["revisions"])
    for relation in request["relations"]:
        old = index["relations"].get(relation["relation_id"])
        if old is not None:
            if old != relation:
                raise MemoryConflict("immutable branch relation differs")
        elif relation["source"]["revision_id"] not in new_revisions:
            raise ContextConflict("new edges require a revision created in the same request")
    return request


def union_graph(graphs: list[dict]) -> dict:
    merged: dict[str, dict] = {name: {} for name in COLLECTIONS}
    adjacency: dict[str, list[dict]] = {}
    for value in graphs:
        graph = checked_graph(value, revisions=MAX_ENTRIES)
        outgoing = {r["revision_id"]: [] for r in graph["revisions"]}
        for relation in graph["relations"]:
            outgoing[relation["source"]["revision_id"]].append(relation)
        for revision_id, edges in outgoing.items():
            edges.sort(key=lambda r: r["relation_id"])
            if revision_id in adjacency and adjacency[revision_id] != edges:
                raise ContextConflict("equal revision identities have different frozen outgoing structure")
            adjacency[revision_id] = edges
        for name, identity in COLLECTIONS.items():
            for item in graph[name]:
                key = item[identity]
                old = merged[name].get(key)
                if old is not None and old != item:
                    raise ContextConflict("different immutable records share an identity")
                merged[name][key] = item
                if len(merged[name]) > MAX_ENTRIES:
                    raise ContextCapacity("join union entry capacity reached")
    result = {"schema_version": "1.0", **{name: [items[k] for k in sorted(items)] for name, items in merged.items()}}
    return checked_graph(result, revisions=MAX_ENTRIES)


def join_graph(baseline: dict, branches: list[dict], proposals: list[list[dict]], selection: dict) -> dict:
    """Explicit complete choices, including explicit omission (null), before closure."""
    selection = protocol(selection, "selection")
    if not branches or len(branches) != len(proposals) or len(branches) > 4:
        raise ContextError("join requires its fixed sealed branch set")
    graph = union_graph([baseline, *branches])
    proposed: dict[str, set[str]] = {}
    for branch, choices in zip(branches, proposals):
        for entity, revision in active_map(choices, branch).items():
            proposed.setdefault(entity, set()).add(revision)
    chosen = {}
    for item in selection["choices"]:
        entity, revision = item["entity_id"], item["revision_id"]
        if entity in chosen or entity not in proposed or (revision is not None and revision not in proposed[entity]):
            raise ContextConflict("join choice is missing, duplicate, or not proposed")
        chosen[entity] = revision
    if set(chosen) != set(proposed):
        raise ContextConflict("join requires explicit selection or omission for every entity")
    roots = {r for r in chosen.values() if r is not None}
    selected, edges = closure(graph, roots)
    historical = refs(selection["historical"], indices(graph)["revisions"])
    if set(historical) != selected - roots:
        raise ContextConflict("historical exact dependencies must be explicitly selected")
    return subgraph(graph, selected, edges)


def join_delta(baseline: dict, selected: dict, binding: dict) -> dict | None:
    before = indices(baseline)
    delta = {name: [record for record in selected[name] if record[identity] not in before[name]]
             for name, identity in COLLECTIONS.items()}
    if not any(delta.values()):
        return None
    request = {"version": 1, "request_id": "context-join:" + digest(binding),
               "expected_head": binding["source"]["head"], **delta}
    from .memory_store import _protocol

    _protocol(request, "request")
    return copied(request, REQUEST_BYTES)


def _mathematical_projection(graph: dict, roots: dict, dependencies: list, references: list) -> dict:
    index = indices(graph)
    selected_roots = refs(list(roots.values()), index["revisions"])
    for name, kind in (("problem", "Problem"), ("claim", "Claim"), ("proof", "ProofArtifact")):
        if index["entities"][roots[name]["entity_id"]]["type"] != kind:
            raise ContextError("proof manifest root has the wrong mathematical type")
    proof, claim = (roots[name]["revision_id"] for name in ("proof", "claim"))
    if not any(r["type"] == "attempts_to_prove" and r["source"]["revision_id"] == proof
               and r["target"]["revision_id"] == claim for r in graph["relations"]):
        raise ContextError("proof must target the exact claim")
    selected, edges = closure(graph, set(selected_roots), proof_pair=(proof, claim))
    allowed = refs(dependencies, index["revisions"])
    if set(allowed) != selected - set(selected_roots):
        raise ContextError("declared dependency set differs from complete required closure")
    approved = refs(references, index["revisions"])
    actual_references = set()
    revisions = []
    for key in sorted(selected):
        revision = index["revisions"][key]
        entity = index["entities"][revision["entity_id"]]
        if entity["type"] not in MATH_TYPES:
            raise ContextError("minimal mathematical view forbids this dependency type")
        if (key.startswith("legacy:revision:")
                or any(item["provenance"]["source"].startswith("legacy-jsonl://snapshot/")
                       for item in (entity, revision))):
            raise ContextError("legacy history wrappers are not mathematical sources")
        if entity["type"] == "Reference":
            actual_references.add(key)
        revisions.append({**exact_ref(revision), "type": entity["type"],
                          **{k: revision[k] for k in ("body", "context_revision_ids", "assertion_state", "validity")}})
    if set(approved) != actual_references:
        raise ContextError("Reference bodies require exact explicit approval")
    relations = [{k: index["relations"][key][k] for k in
                  ("relation_id", "type", "source", "target", "context_revision_ids", "assertion_state")}
                 for key in sorted(edges)]
    return {"roots": roots, "revisions": revisions, "relations": relations}


def mathematical_projection(graph: dict, *, roots: dict, dependencies: list, references: list) -> dict:
    """Exact minimal disclosure without inventing a workspace/join identity.

    New controllers bind the returned pure projection in their own protocol. The
    complete source may have 1024 revisions; the disclosed closure is still 256.
    """
    graph = checked_graph(graph, revisions=MAX_ENTRIES)
    roots = copied(roots)
    if type(roots) is not dict or set(roots) != {"problem", "claim", "proof"}:
        raise ContextError("invalid mathematical roots")
    protocol(list(roots.values()), "refs")
    dependencies = protocol(dependencies, "refs", GRAPH_BYTES)
    references = protocol(references, "refs", GRAPH_BYTES)
    return copied(_mathematical_projection(graph, roots, dependencies, references), GRAPH_BYTES)


def minimal_view(graph: dict, manifest: dict) -> dict:
    manifest = protocol(manifest, "proof_manifest")
    graph = checked_graph(graph)
    projection = _mathematical_projection(graph, manifest["roots"], manifest["dependencies"], manifest["references"])
    dto = {"version": "context-dto-1", "policy_version": POLICY, "manifest_sha256": digest(manifest), **projection}
    return protocol(dto, "dto", GRAPH_BYTES)
