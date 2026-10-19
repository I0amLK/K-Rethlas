"""Trusted MCP/CLI adapter; managed writes never fall back to legacy JSONL."""
from __future__ import annotations

from .context_views import exact_ref
from .private_fs import digest
from .research_records import CHANNELS, memory_record
from .validity import ValidityError, ValidityConflict


class ManagedMemory:
    def __init__(self, service, retrieval, *, reader, writer=None, indexer=None):
        self.service, self.retrieval = service, retrieval
        self.reader, self.writer, self.indexer = reader, writer, indexer

    def init(self, problem_id, meta=None):
        if type(problem_id) is not str or not problem_id.strip():
            raise ValidityError("problem identity must be exact nonempty text")
        if meta is not None:
            if self.writer is None:
                raise ValidityError("this adapter is read-only")
            return self.append(problem_id, "events", {"event": "memory_init", "meta": meta})
        return {"status": "managed", "problem_id": problem_id, "channels": list(CHANNELS),
                "visibility_scope": "whole_project", "legacy_jsonl_writable": False}

    def append(self, problem_id, channel, record, *, request_id=None):
        if self.writer is None:
            raise ValidityError("this adapter is read-only")
        request_id = request_id or "record." + digest({"problem": problem_id, "channel": channel, "record": record})
        token, run_id = self.writer["token"], self.writer["run_id"]
        # Preserve exact original graph request on retry, including its parent
        # and real first-recorded timestamp; never manufacture an origin date.
        with self.service._locked(token, run_id, "project.commit") as (_, state, claims, resource):
            key = digest({"service_id": self.service.config["service_id"], "request_id": request_id})
            old = state["requests"].get(key)
            if old is not None:
                operation = self.service._operation(old)
                request = operation["request"]
                existing = request["revisions"][0]["body"] if request["revisions"] else None
                from .private_fs import canonical_json
                wanted = canonical_json({"version": "managed-record-1", "problem_id": problem_id,
                                         "channel": channel, "record": record}).decode("utf-8")
                if existing != wanted or operation["identity"] != self.service._identity(claims, resource):
                    raise ValidityConflict("managed memory request is permanently bound")
            else:
                request = {"version": 1, "request_id": request_id,
                           "expected_head": state["accepted_graph"]["head"],
                           **memory_record(problem_id, channel, record, request_id=request_id)}
        result = self.service.commit(token, request, run_id=run_id)
        ref = exact_ref(request["revisions"][0])
        return {"status": "committed_candidate", "ref": ref, "problem_id": problem_id, "channel": channel,
                "commit": result, "assertion_state": "candidate", "validity": "needs_review"}

    def search(self, problem_id, query, channels=None, limit_per_channel=10):
        if channels is not None and (type(channels) is not list or not set(channels) <= set(CHANNELS)):
            raise ValidityError("unknown memory channels")
        return self.retrieval.search(self.reader["token"], query, run_id=self.reader["run_id"],
            top_k=limit_per_channel, filters={"problem_id": problem_id, **({"channels": channels} if channels else {})})

    def branch_update(self, problem_id, branch_id, state):
        # A progress record is not a join. Actual branch merges remain explicit
        # ContextWorkspace seal/barrier/join plus managed candidate commit.
        return self.append(problem_id, "branch_states", {"branch_id": branch_id, "state": state,
                                                        "kind": "progress_record_not_merge"})

    def rebuild_index(self):
        if self.indexer is None:
            raise ValidityError("index maintenance requires operator-configured authority")
        return self.retrieval.build_index(self.indexer["token"], run_id=self.indexer["run_id"])

    def get_revision(self, ref):
        return self.service.get_revision(self.reader["token"], ref, run_id=self.reader["run_id"])

    def expand(self, refs, relation_types=None, depth=1):
        return self.service.expand(self.reader["token"], refs, run_id=self.reader["run_id"], relation_types=relation_types, depth=depth)
