"""Single-host, controller-only Markdown/graph commits; no model or network calls.

Only the validated manifest parent chain confers visibility. Receipts prepared
before that point bind request IDs but are not commits. This format deliberately
has no rollback, import, review promotion, branch, revocation or search operation.
Historical exports and metadata checkpoints do not confer current-use validity.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import hashlib
from importlib.resources import files
import math
from pathlib import Path
import re
import secrets
import time
from typing import Any, Callable, Iterator

from jsonschema import Draft202012Validator

from .authorization import Authority, AuthorizationError
from .contracts import validate_math_memory
from .private_fs import (
    PrivateDirectory, PrivateFilesystemError, canonical_json, digest, strict_json_loads,
)


FORMAT = "m2a-1"
GRAPH_FORMAT = "m2a-graph-1"
COLLECTIONS = {"entities": "entity_id", "revisions": "revision_id",
               "relations": "relation_id", "evidence": "evidence_id"}
DEFAULT_LIMITS = {
    "request_bytes": 524288, "body_bytes": 131072, "graph_bytes": 2097152,
    "exchange_bytes": 8388608, "entries": 1024, "history": 256,
    "receipts": 256, "journal_bytes": 16384, "quarantine_files": 32,
    "store_bytes": 268435456, "files": 8192,
}
_HASH = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_SCOPE = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z", re.ASCII)
_TEMP = re.compile(r"\.tmp-[0-9a-f]{32}\Z", re.ASCII)
_SMALL = 16384
_RECEIPT_BYTES = 16384
_AUTHORITY_BYTES = 1048576
_DIRECTORIES = ("authority", "bodies", "graphs", "receipts", "journal", "quarantine", "projection")


class MemoryStoreError(ValueError):
    """A private store operation was rejected."""


class MemoryConflict(MemoryStoreError):
    """A request binding, immutable ID or expected head conflicts."""


class MemoryCapacity(MemoryStoreError):
    """An explicit storage bound was reached; nothing is silently discarded."""


class MemoryCorruption(MemoryStoreError):
    """Committed state cannot be validated; never fall back to an older head."""


class MemoryRecoveryRequired(MemoryStoreError):
    """A pending slot needs explicit, authorized inspection/consumption."""


class MemoryCancelled(MemoryStoreError):
    """Cancellation was observed before the commit point."""


class MemoryIndeterminate(MemoryStoreError):
    """The head may be durable; reopen and replay the exact request to decide."""


@lru_cache(maxsize=10)
def _validator(kind: str) -> Draft202012Validator:
    schema = strict_json_loads(files("rethlas_core").joinpath(
        "schemas", "memory_protocol.schema.json").read_bytes())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator({"$ref": "#/$defs/" + kind, "$defs": schema["$defs"]})


def _protocol(value: Any, kind: str) -> None:
    if not _validator(kind).is_valid(value):
        raise MemoryStoreError("invalid " + kind + " protocol")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _empty() -> dict:
    return {"schema_version": "1.0", **{key: [] for key in COLLECTIONS}}


def _stored(graph: dict) -> dict:
    return {
        "format_version": GRAPH_FORMAT, "math_schema_version": graph["schema_version"],
        "entities": graph["entities"], "relations": graph["relations"], "evidence": graph["evidence"],
        "revisions": [{key: value for key, value in item.items() if key != "body"}
                      for item in graph["revisions"]],
    }


def _head(fields: dict) -> dict:
    return {**fields, "head": digest(fields)}


def _genesis(store_id: str) -> dict:
    return _head({"format_version": FORMAT, "store_id": store_id, "generation": 0,
                  "graph_sha256": digest(_stored(_empty())), "receipt_key": None,
                  "receipt_sha256": None, "parent_head": None})


def _result_manifest(receipt: dict) -> dict:
    return _head({"format_version": FORMAT, "store_id": receipt["binding"]["store_id"],
                  "generation": receipt["generation"], "graph_sha256": receipt["graph_sha256"],
                  "receipt_key": receipt["request_key"], "receipt_sha256": digest(receipt),
                  "parent_head": receipt["parent_manifest"]["head"]})


def _check_candidate(graph: dict) -> None:
    for revision in graph["revisions"]:
        if revision["assertion_state"] != "candidate" or revision["validity"] != "needs_review":
            raise MemoryConflict("memory commits cannot confer or change review validity")
    if any(item["assertion_state"] != "candidate" for item in graph["relations"]):
        raise MemoryConflict("memory commits cannot confer or change relation review")


def _append_only(parent: dict, child: dict) -> None:
    for collection, identity in COLLECTIONS.items():
        after = {item[identity]: item for item in child[collection]}
        for item in parent[collection]:
            if item[identity] not in after or canonical_json(after[item[identity]]) != canonical_json(item):
                raise MemoryConflict("an existing immutable record changed or disappeared")


class MemoryStore:
    """A private owner/domain store with an independent capability authority.

    Trusted callers may inject a clock or override _checkpoint for synthetic
    fault tests. Neither is selected by a request, CLI argument or environment.
    Lock order is always authority -> store, held throughout local operations.
    """

    def __init__(self, root: Path, *, clock: Callable[[], float] = time.time):
        self.directory = PrivateDirectory(Path(root))
        self.root = self.directory.root
        self.clock = clock
        self.config = self.directory.read_json("store.json", max_bytes=_SMALL)
        _protocol(self.config, "store")
        if any(type(value) is not int for value in self.config["limits"].values()):
            raise MemoryStoreError("limits must be strict integers")
        if type(self.config["operation_version"]) is not int:
            raise MemoryStoreError("unsupported operation version")
        self.limits = dict(self.config["limits"])
        self.config_sha256 = digest(self.config)
        names = {name: is_dir for name, is_dir, _ in
                 self.directory.entries(max_entries=self.limits["files"] + 1)}
        if names.get("lock") is not False or any(names.get(name) is not True for name in _DIRECTORIES):
            raise MemoryStoreError("store is absent or incompletely initialized")
        if names.get("manifest.json") is not False:
            raise MemoryStoreError("store manifest is absent or incompletely initialized")
        self.parts = {name: self.directory.child(name) for name in _DIRECTORIES}
        authority_names = {name: is_dir for name, is_dir, _ in
                           self.parts["authority"].entries(max_entries=self.limits["files"] + 1)}
        if any(authority_names.get(name) is not False for name in ("lock", "key", "registry.json")):
            raise MemoryStoreError("authority is absent or incompletely initialized")
        # Authority's constructor cannot create a missing lock after these checks.
        self.authority = Authority(self.root / "authority")

    @classmethod
    def initialize(cls, root: Path, *, owner: str, domain: str,
                   limits: dict | None = None) -> MemoryStore:
        if type(owner) is not str or not _SCOPE.fullmatch(owner) or owner in (".", ".."):
            raise MemoryStoreError("invalid store owner")
        if domain not in ("fixture", "production", "research"):
            raise MemoryStoreError("invalid store domain")
        selected = dict(DEFAULT_LIMITS)
        if limits is not None:
            if type(limits) is not dict or set(limits) - set(selected):
                raise MemoryStoreError("unknown store limit")
            selected.update(limits)
        config = {"format_version": FORMAT, "operation_version": 1,
                  "store_id": secrets.token_hex(32), "owner": owner,
                  "domain": domain, "limits": selected}
        _protocol(config, "store")
        if any(type(value) is not int for value in selected.values()):
            raise MemoryStoreError("limits must be strict integers")
        directory = PrivateDirectory(Path(root), create=True)
        # Initialization is operator-only and never repairs an old/partial root.
        try:
            nonempty = directory.entries(max_entries=2)
        except PrivateFilesystemError as exc:
            raise MemoryConflict("initialization requires a new empty private root") from exc
        if nonempty:
            raise MemoryConflict("initialization requires a new empty private root")
        with directory.lock():
            if {name for name, _, _ in directory.entries(max_entries=3)} != {"lock"}:
                raise MemoryConflict("private root changed during initialization")
            parts = {name: directory.child(name, create=True) for name in _DIRECTORIES}
            Authority(directory.root / "authority", create=True)
            graph = canonical_json(_stored(_empty()))
            parts["graphs"].write_bytes(_sha(graph) + ".json", graph)
            parts["journal"].write_bytes("pending.log", b"")
            directory.write_json("manifest.json", _genesis(config["store_id"]))
            # store.json is last: interrupted initialization is never an empty store.
            directory.write_json("store.json", config)
        return cls(directory.root)

    def _checkpoint(self, stage: str) -> None:
        """Test subclass hook, never an environment/CLI failpoint."""

    def _guard(self, claims: dict, cancel_event: Any = None) -> None:
        now = self.clock()
        if type(now) not in (int, float) or not math.isfinite(now):
            raise AuthorizationError("invalid trusted clock")
        if not claims["issued_at"] <= now < claims["expires_at"]:
            raise AuthorizationError("memory capability expired or is not yet valid")
        if cancel_event is not None and cancel_event.is_set():
            raise MemoryCancelled("memory commit cancelled")

    @contextmanager
    def _locked(self, token: str, run_id: str, action: str,
                cancel_event: Any = None) -> Iterator[dict]:
        with self.authority.locked_check(token, owner=self.config["owner"], run_id=run_id,
                                        domain=self.config["domain"], action=action,
                                        now=self.clock()) as claims:
            with self.directory.lock():
                if self.directory.read_bytes("store.json", max_bytes=_SMALL) != canonical_json(self.config):
                    raise MemoryCorruption("store identity or configuration changed")
                self._guard(claims, cancel_event)
                yield claims

    def _inventory(self) -> dict:
        sizes: dict[str, int] = {}
        count = 0

        def walk(directory: PrivateDirectory, prefix: str, depth: int) -> None:
            nonlocal count
            if depth > 1:
                raise MemoryCorruption("unexpected directory nesting")
            remaining = self.limits["files"] - count
            try:
                children = directory.entries(max_entries=remaining + 2)
            except PrivateFilesystemError as exc:
                if str(exc) == "private directory reaches max_entries":
                    raise MemoryCapacity("store file count reached") from exc
                raise
            for name, is_dir, size in children:
                count += 1
                if count > self.limits["files"]:
                    raise MemoryCapacity("store file count reached")
                path = prefix + name
                if is_dir:
                    if prefix or name not in _DIRECTORIES:
                        raise MemoryCorruption("unexpected store directory")
                    walk(directory.child(name), path + "/", depth + 1)
                else:
                    sizes[path] = size
        walk(self.directory, "", 0)
        total = sum(sizes.values())
        if total > self.limits["store_bytes"]:
            raise MemoryCapacity("store byte capacity reached")
        receipts = [name for name in sizes if re.fullmatch(r"receipts/[0-9a-f]{64}\.json", name)]
        quarantine = [name for name in sizes if name.startswith("quarantine/")]
        if len(receipts) > self.limits["receipts"] or len(quarantine) > self.limits["quarantine_files"]:
            raise MemoryCapacity("receipt or quarantine capacity reached")
        # Unknown files cannot be silently ignored as if they were authoritative.
        for path in sizes:
            folder, _, name = path.rpartition("/")
            if _TEMP.fullmatch(name):
                continue
            valid = (
                (not folder and name in ("store.json", "manifest.json", "lock"))
                or (folder == "authority" and name in ("key", "registry.json", "lock"))
                or (folder == "bodies" and re.fullmatch(r"[0-9a-f]{64}\.md", name))
                or (folder in ("graphs", "receipts") and re.fullmatch(r"[0-9a-f]{64}\.json", name))
                or (folder == "journal" and name == "pending.log")
                or (folder == "projection" and name == "ids.json")
                or (folder == "quarantine" and re.fullmatch(r"[0-9a-f]{64}\.log", name))
            )
            if not valid:
                raise MemoryCorruption("unexpected store file")
        return {"bytes": total, "files": count, "sizes": sizes,
                "receipts": receipts, "quarantine": quarantine}

    def _capacity(self, data: bytes) -> None:
        usage = self._inventory()
        # Conservatively reserve a full temp plus a full final object. Existing
        # old versions, orphans and SIGKILL temporaries all remain in usage.
        if (usage["bytes"] + 2 * len(data) > self.limits["store_bytes"]
                or usage["files"] + 2 > self.limits["files"]):
            raise MemoryCapacity("insufficient capacity for atomic write")

    def _write(self, directory: PrivateDirectory, name: str, data: bytes,
               *, before_replace: Callable[[], None] | None = None) -> None:
        self._capacity(data)
        directory.write_bytes(name, data, before_replace=before_replace)

    def _immutable(self, directory: PrivateDirectory, name: str, data: bytes, limit: int) -> None:
        if len(data) > limit:
            raise MemoryCapacity("immutable object exceeds its size bound")
        try:
            existing = directory.read_bytes(name, max_bytes=limit)
        except FileNotFoundError:
            self._write(directory, name, data)
        else:
            if existing != data:
                raise MemoryConflict("an immutable object or prepared request differs")
            # A previous replace may have succeeded before its directory fsync
            # failed. Exact bytes alone are not a completed durability barrier.
            directory.sync_file(name)

    def issue(self, *, run_id: str, actions: list[str], ttl_seconds: float = 300) -> str:
        """Trusted operator only; never an HTTP/MCP method or model capability."""
        with self.authority._registry() as registry:
            with self.directory.lock():
                self._inventory()
                if self.directory.read_bytes("store.json", max_bytes=_SMALL) != canonical_json(self.config):
                    raise MemoryCorruption("store configuration changed")
                if type(run_id) is not str or not _SCOPE.fullmatch(run_id) or run_id in (".", ".."):
                    raise MemoryStoreError("invalid run identifier")
                run = registry["runs"].get(run_id)
                if run is None:
                    if len(registry["runs"]) >= 256:
                        raise MemoryCapacity("authority run capacity reached")
                    registry["runs"][run_id] = {"owner": self.config["owner"],
                        "domain": self.config["domain"], "state": "open", "epoch": 0}
                elif (run["owner"] != self.config["owner"] or run["domain"] != self.config["domain"]
                      or run["state"] != "open"):
                    raise AuthorizationError("run is not open for the store")
                if len(registry["tokens"]) >= 256:
                    raise MemoryCapacity("authority token capacity reached")

                def reserve(proposed: dict) -> None:
                    data = canonical_json(proposed)
                    if len(data) > _AUTHORITY_BYTES:
                        raise MemoryCapacity("authority registry capacity reached")
                    self._capacity(data)

                return self.authority._issue_locked(registry, owner=self.config["owner"],
                    run_id=run_id, domain=self.config["domain"], role="memory", actions=actions,
                    ttl_seconds=ttl_seconds, now=self.clock(), before_save=reserve)

    def _validate_graph(self, graph: dict) -> None:
        if any(len(graph[key]) > self.limits["entries"] for key in COLLECTIONS):
            raise MemoryCapacity("graph entry capacity reached")
        if len(canonical_json(graph)) > self.limits["exchange_bytes"]:
            raise MemoryCapacity("hydrated graph capacity reached")
        # Reads account for stored metadata plus exact raw bodies as they hydrate;
        # JSON escaping/format headers make this differ from exchange JSON size.
        # Enforce both budgets before publication, not only during later reads.
        hydration_bytes = len(canonical_json(_stored(graph))) + sum(
            len(item.get("body", "").encode("utf-8")) for item in graph["revisions"]
            if type(item.get("body")) is str)
        if hydration_bytes > self.limits["exchange_bytes"]:
            raise MemoryCapacity("hydration input capacity reached")
        result = validate_math_memory(graph)
        if not result["valid"]:
            raise MemoryStoreError("invalid mathematical graph")
        _check_candidate(graph)
        for item in graph["revisions"]:
            if len(item["body"].encode("utf-8")) > self.limits["body_bytes"]:
                raise MemoryCapacity("Markdown body capacity reached")

    def _graph(self, sha256: str) -> dict:
        raw = self.parts["graphs"].read_bytes(sha256 + ".json", max_bytes=self.limits["graph_bytes"])
        if _sha(raw) != sha256:
            raise MemoryCorruption("graph hash differs")
        stored = strict_json_loads(raw)
        _protocol(stored, "graph")
        if raw != canonical_json(stored):
            raise MemoryCorruption("graph is not canonical")
        graph = {"schema_version": stored["math_schema_version"],
                 **{key: stored[key] for key in ("entities", "relations", "evidence")}, "revisions": []}
        total = len(raw)
        for revision in stored["revisions"]:
            body = self.parts["bodies"].read_bytes(revision["body_sha256"] + ".md",
                                                   max_bytes=self.limits["body_bytes"])
            total += len(body)
            if total > self.limits["exchange_bytes"]:
                raise MemoryCapacity("hydrated bodies capacity reached")
            if _sha(body) != revision["body_sha256"]:
                raise MemoryCorruption("Markdown hash differs")
            graph["revisions"].append({**revision, "body": body.decode("utf-8")})
        self._validate_graph(graph)
        return graph

    def _manifest(self, value: dict) -> None:
        _protocol(value, "manifest")
        fields = {key: item for key, item in value.items() if key != "head"}
        if (type(value["generation"]) is not int or value["generation"] > self.limits["history"]
                or value["store_id"] != self.config["store_id"] or digest(fields) != value["head"]):
            raise MemoryCorruption("manifest identity or hash differs")

    def _receipt(self, key: str) -> dict:
        if type(key) is not str or not _HASH.fullmatch(key):
            raise MemoryCorruption("invalid internal receipt key")
        raw = self.parts["receipts"].read_bytes(key + ".json", max_bytes=_RECEIPT_BYTES)
        value = strict_json_loads(raw)
        _protocol(value, "receipt")
        binding = value["binding"]
        self._manifest(value["parent_manifest"])
        if (raw != canonical_json(value) or value["request_key"] != key
                or type(value["generation"]) is not int or type(binding["epoch"]) is not int
                or type(binding["operation_version"]) is not int
                or binding["store_id"] != self.config["store_id"]
                or binding["owner"] != self.config["owner"] or binding["domain"] != self.config["domain"]
                or binding["config_sha256"] != self.config_sha256
                or key != digest({"store_id": binding["store_id"], "request_id": binding["request_id"]})
                or binding["expected_head"] != value["parent_manifest"]["head"]
                or value["generation"] != value["parent_manifest"]["generation"] + 1):
            raise MemoryCorruption("prepared receipt binding differs")
        return value

    def _chain(self) -> tuple[list[tuple[dict, dict | None]], dict]:
        try:
            raw = self.directory.read_bytes("manifest.json", max_bytes=_SMALL)
            current = strict_json_loads(raw)
            self._manifest(current)
            if raw != canonical_json(current):
                raise MemoryCorruption("manifest is not canonical")
            graph = self._graph(current["graph_sha256"])
            newest_graph = graph
            chain: list[tuple[dict, dict | None]] = []
            seen = set()
            while current["generation"]:
                if len(chain) >= self.limits["history"] or current["head"] in seen:
                    raise MemoryCorruption("committed history is cyclic or exceeds capacity")
                seen.add(current["head"])
                key = current["receipt_key"]
                if type(key) is not str or not _HASH.fullmatch(key):
                    raise MemoryCorruption("committed receipt is absent")
                receipt = self._receipt(key)
                if canonical_json(_result_manifest(receipt)) != canonical_json(current):
                    raise MemoryCorruption("receipt does not bind committed manifest")
                chain.append((current, receipt))
                parent = receipt["parent_manifest"]
                parent_graph = self._graph(parent["graph_sha256"])
                _append_only(parent_graph, graph)
                graph, current = parent_graph, parent
            if current != _genesis(self.config["store_id"]) or graph != _empty():
                raise MemoryCorruption("invalid genesis")
            chain.append((current, None))
            return chain, newest_graph
        except MemoryCorruption:
            raise
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
            raise MemoryCorruption("committed chain cannot be validated") from exc

    def _intent_binding(self, raw: bytes) -> None:
        # Inspect only a bounded prefix, even when a corrupt tail exceeds the
        # whole-slot cap. A verifiable first frame cannot be forgotten with it.
        line, separator, _ = raw.partition(b"\n")
        if not separator:
            return
        try:
            value = strict_json_loads(line)
            _protocol(value, "frame")
            valid = (line == canonical_json(value) and type(value["index"]) is int
                     and value["index"] == 0 and value["kind"] == "INTENT"
                     and value["previous_sha256"] is None
                     and digest({key: item for key, item in value.items() if key != "sha256"}) == value["sha256"])
        except (ValueError, TypeError, KeyError, RecursionError):
            return
        if valid:
            try:
                self._receipt(value["request_key"])
            except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
                raise MemoryCorruption("pending intent lost its permanent request binding") from exc

    def _pending(self, chain: list[tuple[dict, dict | None]], usage: dict) -> dict:
        size = usage["sizes"].get("journal/pending.log")
        if size is None:
            return {"status": "missing", "bytes": 0, "frames": []}
        if size > self.limits["journal_bytes"]:
            self._intent_binding(self.parts["journal"].read_prefix(
                "pending.log", max_bytes=self.limits["journal_bytes"]))
            return {"status": "corrupt", "bytes": size, "frames": [], "reason": "journal_size"}
        raw = self.parts["journal"].read_bytes("pending.log", max_bytes=self.limits["journal_bytes"])
        if not raw:
            return {"status": "empty", "bytes": 0, "frames": []}
        self._intent_binding(raw)
        try:
            lines = raw.splitlines(keepends=True)
            if not lines or not lines[0].endswith(b"\n"):
                raise MemoryStoreError("invalid journal framing")
            frames = []
            for index, line in enumerate(lines):
                if index >= 3 or not line.endswith(b"\n"):
                    raise MemoryStoreError("invalid journal framing")
                value = strict_json_loads(line[:-1])
                _protocol(value, "frame")
                if (line != canonical_json(value) + b"\n" or type(value["index"]) is not int
                        or value["index"] != index
                        or value["kind"] != ("INTENT", "PREPARED", "HEAD_OBSERVED")[index]
                        or value["previous_sha256"] != (frames[-1]["sha256"] if frames else None)
                        or digest({key: item for key, item in value.items() if key != "sha256"}) != value["sha256"]):
                    raise MemoryStoreError("invalid journal checksum or order")
                if frames and any(value[key] != frames[0][key] for key in
                                  ("request_key", "request_digest", "parent_head", "result_head")):
                    raise MemoryStoreError("journal mixes transactions")
                frames.append(value)
            receipt = self._receipt(frames[0]["request_key"])
            if (frames[0]["request_digest"] != receipt["binding"]["request_digest"]
                    or frames[0]["parent_head"] != receipt["parent_manifest"]["head"]
                    or frames[0]["result_head"] != _result_manifest(receipt)["head"]):
                raise MemoryStoreError("journal binding differs")
            committed = any(manifest["head"] == frames[0]["result_head"] for manifest, _ in chain)
            if len(frames) == 3 and not committed:
                raise MemoryStoreError("journal claimed an unreachable head")
            return {"status": "committed_pending" if committed else "uncommitted_pending",
                    "bytes": len(raw), "sha256": _sha(raw), "frames": frames}
        except MemoryCorruption:
            raise
        except (OSError, ValueError, TypeError, KeyError, RecursionError):
            return {"status": "corrupt", "bytes": len(raw), "sha256": _sha(raw),
                    "frames": [], "reason": "invalid_tail_or_binding"}

    def _journal(self, frames: list[dict], kind: str, receipt: dict) -> None:
        fields = {"format_version": FORMAT, "index": len(frames), "kind": kind,
                  "request_key": receipt["request_key"],
                  "request_digest": receipt["binding"]["request_digest"],
                  "parent_head": receipt["parent_manifest"]["head"],
                  "result_head": _result_manifest(receipt)["head"],
                  "previous_sha256": frames[-1]["sha256"] if frames else None}
        frame = {**fields, "sha256": digest(fields)}
        _protocol(frame, "frame")
        proposed = frames + [frame]
        raw = b"".join(canonical_json(item) + b"\n" for item in proposed)
        if len(proposed) > 3 or len(raw) > self.limits["journal_bytes"]:
            raise MemoryCapacity("pending journal capacity reached")
        self._write(self.parts["journal"], "pending.log", raw)
        frames.append(frame)

    def _merge(self, graph: dict, request: dict) -> dict:
        merged = {"schema_version": "1.0"}
        additions = 0
        for collection, identity in COLLECTIONS.items():
            records = {item[identity]: item for item in graph[collection]}
            seen = set()
            if len(request[collection]) > self.limits["entries"]:
                raise MemoryCapacity("request entry capacity reached")
            for item in request[collection]:
                key = item.get(identity)
                if type(key) is not str or not key.strip() or key in seen:
                    raise MemoryStoreError("missing or duplicate incremental identity")
                seen.add(key)
                if key in records:
                    if canonical_json(records[key]) != canonical_json(item):
                        raise MemoryConflict("an existing immutable identity differs")
                else:
                    records[key] = item
                    additions += 1
            merged[collection] = list(records.values())
        if not additions:
            raise MemoryConflict("request has no new immutable records")
        self._validate_graph(merged)
        if len(canonical_json(_stored(merged))) > self.limits["graph_bytes"]:
            raise MemoryCapacity("stored graph capacity reached")
        return merged

    def commit(self, token: str, request: dict, *, run_id: str, cancel_event: Any = None,
               outer_guard: Callable[[], None] | None = None) -> dict:
        # Trusted Python only: mediated callers retain their outer authority lock.
        # This guard must not acquire locks or be selected by worker input.
        if outer_guard is not None:
            if not callable(outer_guard):
                raise MemoryStoreError("outer guard must be a trusted callable")
            outer_guard()
        # Copy strict JSON once; never retain mutable caller-owned records.
        raw = canonical_json(request)
        if len(raw) > self.limits["request_bytes"]:
            raise MemoryCapacity("request byte capacity reached")
        request = strict_json_loads(raw)
        _protocol(request, "request")
        if type(request["version"]) is not int:
            raise MemoryStoreError("unsupported request version")
        with self._locked(token, run_id, "memory.commit", cancel_event) as claims:
            usage = self._inventory()
            chain, graph = self._chain()
            current = chain[0][0]
            # Even exact replay cannot turn a readable head left by an earlier
            # failed directory fsync into a success without a fresh barrier.
            try:
                self.directory.sync_file("manifest.json")
            except OSError as exc:
                raise MemoryIndeterminate("current manifest durability could not be confirmed") from exc
            key = digest({"store_id": self.config["store_id"], "request_id": request["request_id"]})
            identity = {name: claims[name] for name in ("owner", "run_id", "domain", "epoch")}
            request_digest = digest({"store_id": self.config["store_id"], **identity,
                "operation_version": 1, "config_sha256": self.config_sha256, "request": request})
            binding = {"store_id": self.config["store_id"], **identity, "operation_version": 1,
                       "config_sha256": self.config_sha256, "request_id": request["request_id"],
                       "request_digest": request_digest, "expected_head": request["expected_head"]}
            try:
                prepared = self._receipt(key)
            except FileNotFoundError:
                prepared = None
            if prepared is not None:
                if canonical_json(prepared["binding"]) != canonical_json(binding):
                    raise MemoryConflict("request ID is permanently bound to another request")
                # Precise committed replay precedes CAS and pending-slot checks.
                for manifest, receipt in chain:
                    if receipt is not None and receipt["request_key"] == key:
                        self._guard(claims, cancel_event)
                        if outer_guard is not None:
                            outer_guard()
                        return {"status": "committed", "replayed": True,
                                "head": manifest["head"], "receipt": receipt}
            pending = self._pending(chain, usage)
            if pending["status"] != "empty":
                raise MemoryRecoveryRequired("pending slot requires authorized recovery")
            if request["expected_head"] != current["head"]:
                raise MemoryConflict("expected head is not the current head")
            if current["generation"] >= self.limits["history"]:
                raise MemoryCapacity("commit history capacity reached")
            merged = self._merge(graph, request)
            stored = canonical_json(_stored(merged))
            receipt = {"format_version": FORMAT, "request_key": key, "binding": binding,
                       "parent_manifest": current, "generation": current["generation"] + 1,
                       "graph_sha256": _sha(stored)}
            _protocol(receipt, "receipt")
            if prepared is None and len(usage["receipts"]) >= self.limits["receipts"]:
                raise MemoryCapacity("prepared receipt capacity reached")
            self._immutable(self.parts["receipts"], key + ".json", canonical_json(receipt), _RECEIPT_BYTES)
            frames: list[dict] = []
            self._journal(frames, "INTENT", receipt)
            self._checkpoint("after_intent")
            for item in merged["revisions"]:
                body = item["body"].encode("utf-8")
                self._immutable(self.parts["bodies"], item["body_sha256"] + ".md",
                                body, self.limits["body_bytes"])
            self._checkpoint("after_bodies")
            self._immutable(self.parts["graphs"], _sha(stored) + ".json", stored, self.limits["graph_bytes"])
            self._checkpoint("after_graph")
            self._journal(frames, "PREPARED", receipt)
            manifest = _result_manifest(receipt)

            def final_guard() -> None:
                self._checkpoint("before_manifest")
                # Revocation/epoch/close mutations cannot pass the held authority
                # lock. Time and cancellation must still be rechecked right here.
                self._guard(claims, cancel_event)
                if outer_guard is not None:
                    outer_guard()

            try:
                self._write(self.directory, "manifest.json", canonical_json(manifest),
                            before_replace=final_guard)
                self._checkpoint("after_manifest")
                self._journal(frames, "HEAD_OBSERVED", receipt)
                self._write(self.parts["journal"], "pending.log", b"")
            except (AuthorizationError, MemoryCancelled):
                raise
            except (OSError, MemoryStoreError) as exc:
                raise MemoryIndeterminate("reopen and replay the exact request to determine commit") from exc
            return {"status": "committed", "replayed": False, "head": manifest["head"], "receipt": receipt}

    @contextmanager
    def locked_checkpoint(self, token: str, *, run_id: str,
                          cancel_event: Any = None) -> Iterator[dict]:
        """Hold authority -> store locks around a trusted metadata-only operation.

        The caller must compare this checkpoint with its own accepted graph floor;
        it is not a chain audit, pending-slot inspection, or current-use grant.
        No graph, receipt, Markdown body or projection is hydrated. While inside
        the context, do not call issue/status/export/commit or any other method
        that reacquires either non-reentrant lock. No worker-selected callback is
        accepted, and failures inside the trusted caller's body propagate intact.
        """
        with self._locked(token, run_id, "memory.read", cancel_event) as claims:
            try:
                config_raw = self.directory.read_bytes("store.json", max_bytes=_SMALL)
                config = strict_json_loads(config_raw)
                _protocol(config, "store")
                if (config_raw != canonical_json(config) or digest(config) != self.config_sha256
                        or type(config["operation_version"]) is not int
                        or any(type(value) is not int for value in config["limits"].values())):
                    raise MemoryCorruption("store checkpoint configuration differs")
                raw = self.directory.read_bytes("manifest.json", max_bytes=_SMALL)
                manifest = strict_json_loads(raw)
                self._manifest(manifest)
                if raw != canonical_json(manifest):
                    raise MemoryCorruption("store checkpoint manifest is not canonical")
                if manifest["generation"] == 0:
                    if manifest != _genesis(config["store_id"]):
                        raise MemoryCorruption("store checkpoint genesis differs")
                elif any(manifest[key] is None for key in ("receipt_key", "receipt_sha256", "parent_head")):
                    raise MemoryCorruption("store checkpoint commit metadata is incomplete")
            except MemoryCorruption:
                raise
            except (OSError, ValueError, TypeError, KeyError, RecursionError) as exc:
                raise MemoryCorruption("store checkpoint metadata cannot be validated") from exc
            try:
                self.directory.sync_file("store.json")
                self.directory.sync_file("manifest.json")
            except OSError as exc:
                raise MemoryIndeterminate("store checkpoint durability could not be confirmed") from exc
            checkpoint = {"store_id": config["store_id"], "config_sha256": self.config_sha256,
                          **{key: manifest[key] for key in ("head", "generation", "graph_sha256",
                              "receipt_key", "receipt_sha256", "parent_head")}}
            self._guard(claims, cancel_event)
            yield checkpoint
            self._guard(claims, cancel_event)

    def status(self, token: str, *, run_id: str) -> dict:
        with self._locked(token, run_id, "memory.read") as claims:
            usage = self._inventory()
            chain, graph = self._chain()
            pending = self._pending(chain, usage)
            self._guard(claims)
            return {"status": "ready" if pending["status"] == "empty" else "recovery_required",
                    "store_id": self.config["store_id"], "head": chain[0][0]["head"],
                    "generation": chain[0][0]["generation"],
                    "counts": {key: len(graph[key]) for key in COLLECTIONS},
                    "pending": {key: value for key, value in pending.items() if key != "frames"},
                    "bytes": usage["bytes"], "projection_authoritative": False,
                    "mathematically_reviewed": False}

    def export(self, token: str, *, run_id: str, head: str | None = None) -> dict:
        with self._locked(token, run_id, "memory.read") as claims:
            self._inventory()
            chain, current_graph = self._chain()
            selected = chain[0][0] if head is None else next(
                (manifest for manifest, _ in chain if manifest["head"] == head), None)
            if selected is None:
                raise MemoryConflict("requested head is not on the committed parent chain")
            graph = current_graph if selected == chain[0][0] else self._graph(selected["graph_sha256"])
            self._guard(claims)
            return {"head": selected["head"], "generation": selected["generation"],
                    "audit_only": selected != chain[0][0], "graph": graph}

    def recover(self, token: str, *, run_id: str, apply: bool = False) -> dict:
        if type(apply) is not bool:
            raise MemoryStoreError("recovery apply must be an explicit boolean")
        with self._locked(token, run_id, "memory.maintain") as claims:
            usage = self._inventory()
            chain, _ = self._chain()  # Committed corruption fails before any repair.
            pending = self._pending(chain, usage)
            committed_keys = {receipt["request_key"] for _, receipt in chain if receipt is not None}
            prepared_only = 0
            for path in usage["receipts"]:
                key = path.split("/")[1][:-5]
                self._receipt(key)  # Never erase a damaged persistent request binding.
                prepared_only += key not in committed_keys
            before = chain[0][0]["head"]
            quarantined = None
            if apply and pending["status"] != "empty":
                self._guard(claims)
                # Do not consume the recovery evidence until the reachable head
                # has a fresh file+directory durability barrier.
                try:
                    self.directory.sync_file("manifest.json")
                except OSError as exc:
                    raise MemoryIndeterminate("manifest durability could not be confirmed") from exc
                self._guard(claims)
                if pending["status"] != "missing":
                    if len(usage["quarantine"]) >= self.limits["quarantine_files"]:
                        raise MemoryCapacity("quarantine capacity reached")
                    quarantined = secrets.token_hex(32) + ".log"
                    self.parts["journal"].quarantine_to("pending.log", self.parts["quarantine"], quarantined)
                self._write(self.parts["journal"], "pending.log", b"")
            self._guard(claims)
            return {"status": "recovered" if apply else "inspected", "applied": apply,
                    "head": before, "head_changed": False, "prepared_only_receipts": prepared_only,
                    "pending_before": {key: value for key, value in pending.items() if key != "frames"},
                    "quarantined": quarantined, "external_side_effects_replayed": False}

    def rebuild_projection(self, token: str, *, run_id: str) -> dict:
        with self._locked(token, run_id, "memory.maintain") as claims:
            self._inventory()
            chain, graph = self._chain()
            manifest = chain[0][0]
            projection = {"format_version": FORMAT, "kind": "ids-only-not-retrieval",
                          "store_id": self.config["store_id"], "head": manifest["head"],
                          "generation": manifest["generation"], "revisions": [
                              {key: item[key] for key in ("entity_id", "revision_id", "body_sha256")}
                              for item in graph["revisions"]]}
            _protocol(projection, "projection")
            raw = canonical_json(projection)
            self._write(self.parts["projection"], "ids.json", raw,
                        before_replace=lambda: self._guard(claims))
            return {"status": "rebuilt", "head": manifest["head"],
                    "projection_sha256": _sha(raw), "kind": projection["kind"], "authoritative": False}
