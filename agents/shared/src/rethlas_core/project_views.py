"""Pure project-wide revision disclosure and current dependency checks.

Callers must authenticate the entire project before supplying a committed graph.
Navigation limits never truncate the complete mathematical input closure.
"""
from __future__ import annotations

from collections import deque

from .context_views import checked_graph, exact_ref, indices, refs, subgraph
from .private_fs import digest
from .project_protocol import protocol
from .validity import MATH_DEPENDENCIES, ValidityError, ValidityCapacity, validity_states


PROJECT_POLICY = {
    "version": "research-current-1", "domain": "research",
    "visibility": "whole-owned-project", "support": sorted(MATH_DEPENDENCIES),
    "citations": "disclosure-not-premise", "review_is_proof": False,
}
PROJECT_POLICY_SHA256 = digest(PROJECT_POLICY)


def project_context(graph, withdrawals, selected, reference_refs=None):
    graph = checked_graph(graph, revisions=1024)
    index = indices(graph)
    roots = refs(protocol(selected, "refs"), index["revisions"])
    citations = refs(protocol([] if reference_refs is None else reference_refs, "refs"), index["revisions"])
    all_roots = {**citations, **roots}
    if not all_roots or len(all_roots) > 16:
        raise ValidityError("context requires one to sixteen exact roots")
    outgoing = {}
    for edge in graph["relations"]:
        outgoing.setdefault(edge["source"]["revision_id"], []).append(edge)
    support, disclosed, edges = set(), set(), set()
    pending = deque([(key, True, 0) for key in sorted(roots)] +
                    [(key, False, 0) for key in sorted(citations.keys() - roots.keys())])
    seen = set()
    while pending:
        key, premise, depth = pending.popleft()
        if (key, premise) in seen:
            continue
        if depth > 64 or key not in index["revisions"]:
            raise ValidityError("required mathematical dependency is unavailable")
        if key not in disclosed and len(disclosed) >= 256:
            raise ValidityCapacity("complete mathematical context exceeds capacity")
        seen.add((key, premise))
        disclosed.add(key)
        if premise:
            support.add(key)
        revision = index["revisions"][key]
        pending.extend((other, premise, depth + 1) for other in revision["context_revision_ids"])
        for edge in outgoing.get(key, ()):
            edges.add(edge["relation_id"])
            # A proof's explicit attempted Claim is part of its input, not proof
            # of that Claim or a global reverse invalidation relationship.
            mathematical = edge["type"] in MATH_DEPENDENCIES or (
                edge["type"] == "attempts_to_prove" and key in roots)
            needed = premise and mathematical
            pending.append((edge["target"]["revision_id"], needed, depth + 1))
            pending.extend((other, needed, depth + 1) for other in edge["context_revision_ids"])
    frozen = subgraph(graph, disclosed, edges)
    states = validity_states(graph, withdrawals)
    support_refs = [exact_ref(index["revisions"][key]) for key in sorted(support)]
    citations = [exact_ref(index["revisions"][key]) for key in sorted(disclosed - support)]
    blocker_ids = {ref["revision_id"] for ref in support_refs if states[ref["revision_id"]]["directly_withdrawn"]}
    for ref in support_refs:
        state = states[ref["revision_id"]]
        blocker_ids.update(item["revision_id"] for item in state["blockers"])
        blocker_ids.update(item["revision_id"] for item in state["review_blockers"])
    blocked = [exact_ref(index["revisions"][key]) for key in sorted(blocker_ids)]
    overlay = {"policy_hash": PROJECT_POLICY_SHA256, "input_eligible": not blocked,
               "blockers": blocked,
               "support": [{**ref, "withdrawn": states[ref["revision_id"]]["directly_withdrawn"]} for ref in support_refs],
               "citation_only": [{**ref, "withdrawn": states[ref["revision_id"]]["directly_withdrawn"],
                   "not_authorized_as_premise": True} for ref in citations]}
    types = {entity["entity_id"]: entity["type"] for entity in frozen["entities"]}
    dto = {"version": "research-context-1", "roots": list(all_roots.values()),
           "revisions": [{**{k: revision[k] for k in ("entity_id", "revision_id", "body", "body_sha256",
               "context_revision_ids", "evidence_ids", "assertion_state", "validity")},
               "type": types[revision["entity_id"]],
               "use": "support" if revision["revision_id"] in support else "historical_citation_only"}
               for revision in frozen["revisions"]],
           "relations": [{k: edge[k] for k in ("relation_id", "type", "source", "target", "context_revision_ids", "evidence_ids")}
                         for edge in frozen["relations"]],
           "evidence": [{k: item[k] for k in ("evidence_id", "locator", "sha256")} for item in frozen["evidence"]]}
    return {"dto": dto, "dto_sha256": digest(dto), "overlay": overlay,
            "overlay_sha256": digest(overlay), "assertion_state": "candidate", "validity": "needs_review",
            "mathematical_correctness_proved": False, "production_completed": False}
