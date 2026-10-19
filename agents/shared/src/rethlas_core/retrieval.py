"""Authorized project retrieval. Vectors nominate IDs; the graph decides use.

The backend is selected at trusted startup, never by an MCP request. Offline
cache misses may explicitly fall back to lexical retrieval, never implicit HTTP.
"""
from __future__ import annotations

from copy import deepcopy

from .context_views import exact_ref
from .private_fs import canonical_json, strict_json_loads
from .project_views import project_context
from .validity import ValidityError, ValidityCapacity, ValidityConflict, validity_states
from .validity_service import _guard
from .retrieval_index import chunk_revisions, make_index, validate_index, rank, lexical_rank
from .embedding import EmbeddingCacheMiss


INDEX_BYTES = 33554432


class RetrievalService:
    def __init__(self, service, *, provider=None, lexical_fallback=True):
        self.service = service
        self.provider = provider
        self.lexical_fallback = lexical_fallback

    def _presend(self, token, run_id, action, binding, cancel_event):
        with self.service.snapshot(token, run_id=run_id, action=action, cancel_event=cancel_event) as (state, _, _, _):
            if self.service.binding(state) != binding:
                raise ValidityConflict("project graph changed before embedding dispatch")

    def build_index(self, token, *, run_id, cancel_event=None):
        with self.service.snapshot(token, run_id=run_id, action="project.index", cancel_event=cancel_event) as (state, graph, _, _):
            binding = self.service.binding(state)
            chunks = chunk_revisions(graph["revisions"])
        vectors, mode, missing = None, "lexical", False
        if self.provider is not None and chunks:
            try:
                vectors = []
                for offset in range(0, len(chunks), 64):
                    options = {"kind": "documents", "cancel_event": cancel_event}
                    if self.provider.mode == "bge_live":
                        options["before_send"] = lambda: self._presend(token, run_id, "project.index", binding, cancel_event)
                    vectors.extend(self.provider.embed([c["text"] for c in chunks[offset:offset + 64]], **options))
                mode = self.provider.mode
            except EmbeddingCacheMiss:
                if not self.lexical_fallback:
                    raise
                vectors, missing = None, True
        value = make_index(chunks, vectors, binding=binding, mode=mode)
        raw = canonical_json(value)
        if len(raw) > INDEX_BYTES:
            raise ValidityCapacity("retrieval index capacity reached")
        with self.service.snapshot(token, run_id=run_id, action="project.index", cancel_event=cancel_event) as (state, _, _, claims):
            if self.service.binding(state) != binding:
                raise ValidityConflict("graph changed while building its disposable index")
            self.service._reserve(len(raw))
            self.service.parts["index"].write_bytes("current.json", raw,
                before_replace=lambda: _guard(claims, self.service.clock, cancel_event))
        return {"status": "index_built", "mode": mode, "chunks": len(chunks), "binding": binding,
                "cache_miss_lexical_fallback": missing, "knowledge_authority": False}

    def _read(self, binding):
        raw = self.service.parts["index"].read_bytes("current.json", max_bytes=INDEX_BYTES)
        value = validate_index(strict_json_loads(raw), binding=binding)
        if raw != canonical_json(value):
            raise ValidityError("retrieval index is not canonical")
        return value

    def search(self, token, query, *, run_id, top_k=10, filters=None, cancel_event=None):
        if type(query) is not str or not query.strip() or len(query.encode("utf-8")) > 8192:
            raise ValidityError("query must be bounded nonempty text")
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValidityError("top_k must be a strict integer from one to twenty")
        filters = {} if filters is None else filters
        if type(filters) is not dict or set(filters) - {"types", "problem_id", "channels"}:
            raise ValidityError("unknown retrieval filter")
        for name in ("types", "channels"):
            if name in filters and (type(filters[name]) is not list or not filters[name]
                    or len(filters[name]) > 16 or any(type(v) is not str for v in filters[name])):
                raise ValidityError("retrieval filter values must be bounded string lists")
        types = {"Problem", "DefinitionContext", "Assumption", "Claim", "ProofArtifact", "Review", "Reference", "Counterexample", "Failure", "ResearchArtifact"}
        from .research_records import CHANNELS
        if ("types" in filters and not set(filters["types"]) <= types
                or "channels" in filters and not set(filters["channels"]) <= set(CHANNELS)):
            raise ValidityError("unknown retrieval type or channel")
        if "problem_id" in filters and (type(filters["problem_id"]) is not str or not filters["problem_id"].strip()
                or len(filters["problem_id"].encode("utf-8")) > 4096):
            raise ValidityError("problem filter must be exact bounded nonempty text")
        with self.service.snapshot(token, run_id=run_id, action="project.search", cancel_event=cancel_event) as (state, _, _, _):
            binding = self.service.binding(state)
            index = self._read(binding)
        mode, fallback, query_vector = "lexical", None, None
        if index["mode"] != "lexical":
            if self.provider is None:
                if not self.lexical_fallback:
                    raise ValidityError("index requires its configured embedding provider")
                fallback = "provider_unavailable"
            else:
                try:
                    options = {"kind": "queries", "cancel_event": cancel_event}
                    if self.provider.mode == "bge_live":
                        options["before_send"] = lambda: self._presend(token, run_id, "project.search", binding, cancel_event)
                    query_vector = self.provider.embed([query], **options)[0]
                    mode = self.provider.mode
                except EmbeddingCacheMiss:
                    if not self.lexical_fallback:
                        raise
                    fallback = "cache_miss"
        hits = (rank(index, query_vector, top_k=1024) if query_vector is not None
                else lexical_rank(index["chunks"], query, top_k=1024))
        # Reauthenticate and recheck the graph AND withdrawals after embedding.
        with self.service.snapshot(token, run_id=run_id, action="project.search", cancel_event=cancel_event) as (state, graph, withdrawals, _):
            if self.service.binding(state) != binding:
                raise ValidityConflict("retrieval index checkpoint is stale; rebuild explicitly")
            states = validity_states(graph, withdrawals)
            revisions = {r["revision_id"]: r for r in graph["revisions"]}
            entities = {e["entity_id"]: e for e in graph["entities"]}
            results, seen = [], set()
            for hit in hits:
                key = hit["revision_id"]
                if key in seen:
                    continue
                seen.add(key)
                revision = revisions.get(key)
                if revision is None or exact_ref(revision) != {k: hit[k] for k in ("entity_id", "revision_id", "body_sha256")}:
                    raise ValidityError("index does not match the exact committed revision")
                if states[key]["math_blocked"] or states[key]["review_input_obsolete"]:
                    continue
                entity = entities[revision["entity_id"]]
                if "types" in filters and entity["type"] not in filters["types"]:
                    continue
                metadata = self._metadata(revision)
                if "problem_id" in filters and metadata.get("problem_id") != filters["problem_id"]:
                    continue
                if "channels" in filters and metadata.get("channel") not in filters["channels"]:
                    continue
                context = project_context(graph, withdrawals, [exact_ref(revision)])
                if not context["overlay"]["input_eligible"]:
                    continue
                related = self._related(graph, key, states)
                results.append({"ref": exact_ref(revision), "type": entity["type"], "name": entity["name"],
                    "score": hit["score"], "match": {k: hit[k] for k in ("chunk_id", "byte_start", "byte_end", "text_sha256")},
                    "metadata": metadata, "context": context, "related": related,
                    "verification_kind": revision.get("review_kind", "unknown"), "current": states[key]})
                if len(results) >= top_k:
                    break
            result = {"status": "retrieved", "retrieval_mode": mode, "fallback_reason": fallback,
                      "binding": binding, "checked_at": self.service._checked_at(state), "results": results,
                      "count": len(results), "knowledge_authority": False, "mathematical_correctness_proved": False}
            if len(canonical_json(result)) > 8388608:
                raise ValidityCapacity("retrieval result exceeds byte capacity, no truncation")
            return result

    @staticmethod
    def _metadata(revision):
        # Managed channel records carry a strict host-created JSON envelope. This
        # is filtering metadata, never an authorization boundary or review claim.
        try:
            value = strict_json_loads(revision["body"])
        except ValueError:
            return {}
        if (type(value) is dict and value.get("version") == "managed-record-1"
                and type(value.get("problem_id")) is str and type(value.get("channel")) is str):
            return {key: value[key] for key in ("problem_id", "channel")}
        return {}

    @staticmethod
    def _related(graph, key, states):
        types = {e["entity_id"]: e["type"] for e in graph["entities"]}
        revisions = {r["revision_id"]: r for r in graph["revisions"]}
        related = []
        for edge in graph["relations"]:
            if edge["target"]["revision_id"] != key:
                continue
            source = revisions[edge["source"]["revision_id"]]
            if types[source["entity_id"]] in {"Counterexample", "Failure", "Review"}:
                related.append({"ref": exact_ref(source), "type": types[source["entity_id"]],
                                "relation": deepcopy(edge), "body": source["body"], "current": states[source["revision_id"]],
                                "not_authorized_as_premise": True})
        if len(related) > 64:
            raise ValidityCapacity("related evidence capacity reached")
        return related
