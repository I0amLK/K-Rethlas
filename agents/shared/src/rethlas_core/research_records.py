"""Host-built candidate graph envelopes for user-facing memory and research APIs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib

from .private_fs import canonical_json, digest
from .validity import ValidityError


CHANNELS = ("immediate_conclusions", "toy_examples", "counterexamples", "big_decisions", "subgoals",
            "proof_steps", "failed_paths", "verification_reports", "branch_states", "events")


def provenance(source="research:managed", actor="trusted-host"):
    return {"source": source, "actor": actor, "recorded_at": datetime.now(timezone.utc).isoformat()}


def candidate(entity_id, revision_id, kind, name, body, *, contexts=None, evidence_ids=None, origin=None):
    if type(body) is not str or not body.strip():
        raise ValidityError("candidate text must be nonempty")
    prov = provenance() if origin is None else origin
    revision = {"entity_id": entity_id, "revision_id": revision_id, "body": body,
                "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(), "provenance": prov,
                "assertion_state": "candidate", "validity": "needs_review",
                "context_revision_ids": [] if contexts is None else contexts,
                "evidence_ids": [] if evidence_ids is None else evidence_ids}
    if kind == "ResearchArtifact":
        revision["research_progress"] = "in_progress"
    if kind == "Review":
        revision["review_kind"] = "unknown"
    return {"entity_id": entity_id, "type": kind, "name": name, "provenance": prov}, revision


def relation(kind, source, target, *, contexts=None):
    fields = {"type": kind, "source": {k: source[k] for k in ("entity_id", "revision_id")},
              "target": {k: target[k] for k in ("entity_id", "revision_id")},
              "context_revision_ids": [] if contexts is None else contexts}
    return {"relation_id": "rel." + digest(fields), **fields, "provenance": provenance(),
            "assertion_state": "candidate", "evidence_ids": []}


def memory_record(problem_id, channel, record, *, request_id):
    if type(problem_id) is not str or not problem_id.strip() or len(problem_id.encode()) > 4096:
        raise ValidityError("invalid managed problem identity")
    if channel not in CHANNELS or type(record) is not dict:
        raise ValidityError("unknown managed memory channel or record")
    # Unicode and category path remain exact data, never an ASCII-cleaned path.
    envelope = {"version": "managed-record-1", "problem_id": problem_id, "channel": channel, "record": record}
    body = canonical_json(envelope).decode("utf-8")
    identity = digest({"problem_id": problem_id, "channel": channel, "request_id": request_id})
    kind = "Reference" if channel == "events" and any(k in record for k in ("source", "url", "reference")) else "ResearchArtifact"
    entity, revision = candidate("record." + identity, "record." + identity + ".0", kind,
                                 problem_id + ":" + channel, body,
                                 origin=provenance("managed-memory:" + identity))
    return {"entities": [entity], "revisions": [revision], "relations": [], "evidence": []}
