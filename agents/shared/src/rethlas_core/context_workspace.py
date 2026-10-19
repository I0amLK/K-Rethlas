"""Finite fixture-only context workspace, owned by a trusted local controller.

Lock order: workspace authority, workspace state, one branch authority, store.
Nothing here grants a worker access to these roots or to source-memory tokens.
The source is read once by operator capture and is never committed by this code.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
from pathlib import Path
import re
import secrets
import time
from typing import Any, Callable, Iterator

from .authorization import Authority, AuthorizationError, ROLE_ACTIONS
from .context_views import (
    VERSION, META_BYTES, REQUEST_BYTES, GRAPH_BYTES, ContextError, ContextConflict,
    ContextCapacity, ContextIndeterminate, ContextRecoveryRequired, active_map,
    capture_graph, check_branch_request, checked_graph, copied, join_delta,
    join_graph, minimal_view, protocol,
)
from .memory_store import MemoryStore, MemoryCancelled, MemoryRecoveryRequired
from .private_fs import PrivateDirectory, canonical_json, digest


BUNDLE_BYTES = 16777216
WORKSPACE_BYTES = 134217728
WORKSPACE_FILES = 8192
BRANCH_BYTES = 33554432
INTERNAL_RUN = "context-internal"
_TEMP = re.compile(r"\.tmp-[0-9a-f]{32}\Z")
_ROOT_FILES = {"lock", "workspace.json", "snapshot.json", "state.json",
               "join.intent.json", "join.json", "view.intent.json", "view.json"}


def _fields(value: Any, names: set[str]) -> None:
    if type(value) is not dict or set(value) != names:
        raise ContextError("invalid context object fields")


def _branch_dir(branch_id: str) -> str:
    return digest({"branch_id": branch_id})


def _guard(claims: dict, clock: Callable[[], float], cancel_event: Any = None) -> None:
    now = clock()
    if type(now) not in (int, float) or not math.isfinite(now):
        raise AuthorizationError("invalid trusted clock")
    if not claims["issued_at"] <= now < claims["expires_at"]:
        raise AuthorizationError("context capability expired or is not yet valid")
    if cancel_event is not None and cancel_event.is_set():
        raise MemoryCancelled("context operation cancelled")


class ContextWorkspace:
    """Trusted Python/operator interface, not a worker-accessible filesystem ACL."""

    def __init__(self, root: Path, *, clock: Callable[[], float] = time.time):
        self.directory = PrivateDirectory(Path(root))
        self.root = self.directory.root
        self.clock = clock
        self.config = protocol(self.directory.read_json("workspace.json", max_bytes=META_BYTES), "workspace")
        self.config_sha256 = digest(self.config)
        self.owner = self.config["capture"]["owner"]
        self.branches = {b["branch_id"]: b for b in self.config["branches"]}
        if (len(self.branches) != len(self.config["branches"])
                or set(self.branches) != set(self.config["capture"]["branches"])):
            raise ContextError("workspace branch binding differs")
        runs = [b["run_id"] for b in self.branches.values()] + [self.config["join_run_id"], self.config["view_run_id"]]
        if len(set(runs)) != len(runs):
            raise ContextError("workspace resource runs must be distinct")
        names = {name: is_dir for name, is_dir, _ in self.directory.entries(max_entries=WORKSPACE_FILES)}
        if any(names.get(name) is not False for name in ("lock", "state.json", "snapshot.json")):
            raise ContextError("workspace initialization is incomplete")
        if any(names.get(name) is not True for name in ("authority", "branches", "seals")):
            raise ContextError("workspace private roots are incomplete")
        auth_dir = self.directory.child("authority")
        authority_entries = auth_dir.entries(max_entries=WORKSPACE_FILES)
        authority_files = {name for name, is_dir, _ in authority_entries if not is_dir}
        if (not {"lock", "key", "registry.json"} <= authority_files
                or any(is_dir or name not in {"lock", "key", "registry.json"} and not _TEMP.fullmatch(name)
                       for name, is_dir, _ in authority_entries)):
            raise ContextError("workspace authority is incomplete or has unexpected entries")
        self.authority = Authority(auth_dir.root)
        self._tokens: dict[str, tuple[str, float]] = {}

    @classmethod
    def initialize(cls, root: Path, source: MemoryStore, token: str, *, run_id: str, spec: dict) -> ContextWorkspace:
        """Operator-only authorized full hydration, followed by explicit declassification."""
        spec = protocol(spec, "capture")
        if (source.config["domain"] != "fixture" or source.config["owner"] != spec["owner"]
                or source.config["store_id"] != spec["source"]["store_id"]
                or source.config_sha256 != spec["source"]["config_sha256"]):
            raise ContextError("fixture source identity differs")
        root = Path(root).absolute()
        if root == source.root or root in source.root.parents or source.root in root.parents:
            raise ContextError("workspace and source roots must be disjoint")
        exported = source.export(token, run_id=run_id, head=spec["source"]["head"])
        baseline = capture_graph(exported["graph"], spec)
        snapshot = {"version": VERSION, "source": spec["source"], "public": spec["public"],
                    "base_active": spec["base_active"], "graph": baseline}
        protocol(snapshot, "snapshot", GRAPH_BYTES)
        directory = PrivateDirectory(root, create=True)
        if directory.entries(max_entries=WORKSPACE_FILES):
            raise ContextConflict("capture requires a new empty private workspace")
        workspace_id = secrets.token_hex(32)
        branches = []
        with directory.lock():
            if {n for n, _, _ in directory.entries(max_entries=3)} != {"lock"}:
                raise ContextConflict("workspace changed during capture")
            directory.child("authority", create=True)
            branch_root = directory.child("branches", create=True)
            directory.child("seals", create=True)
            authority = Authority(root / "authority", create=True)
            directory.write_json("snapshot.json", snapshot)
            counts = {}
            for branch_id in spec["branches"]:
                path = branch_root.child(_branch_dir(branch_id), create=True).root
                limits = dict(source.limits)
                for name in ("history", "receipts", "exchange_bytes", "request_bytes", "body_bytes", "entries", "store_bytes", "files"):
                    cap = {"history": 33, "receipts": 33, "exchange_bytes": GRAPH_BYTES,
                           "request_bytes": REQUEST_BYTES, "body_bytes": 131072, "entries": 1024,
                           "store_bytes": BRANCH_BYTES, "files": 1024}[name]
                    limits[name] = min(cap, limits[name])
                store = MemoryStore.initialize(path, owner=spec["owner"], domain="fixture", limits=limits)
                internal = store.issue(run_id=INTERNAL_RUN, actions=list(ROLE_ACTIONS["memory"]))
                initial = store.status(internal, run_id=INTERNAL_RUN)
                seed = {"version": 1, "request_id": "context-seed:" + digest({"workspace_id": workspace_id,
                        "branch_id": branch_id, "snapshot_sha256": digest(snapshot)}),
                        "expected_head": initial["head"], **{k: baseline[k] for k in ("entities", "revisions", "relations", "evidence")}}
                result = store.commit(internal, seed, run_id=INTERNAL_RUN)
                branch_run = "cb." + _branch_dir(branch_id)
                authority.register_run(spec["owner"], branch_run, "fixture")
                branches.append({"branch_id": branch_id, "run_id": branch_run, "store_id": store.config["store_id"],
                                 "config_sha256": store.config_sha256, "seed_head": result["head"]})
                counts["mint." + _branch_dir(branch_id)] = 1
            join_run, view_run = "cj." + workspace_id, "cv." + workspace_id
            authority.register_run(spec["owner"], join_run, "fixture")
            authority.register_run(spec["owner"], view_run, "fixture")
            config = {"version": VERSION, "workspace_id": workspace_id, "capture": spec,
                      "snapshot_sha256": digest(snapshot), "branches": branches,
                      "join_run_id": join_run, "view_run_id": view_run}
            protocol(config, "workspace")
            state = {"version": VERSION, "workspace_id": workspace_id, "counts": counts}
            protocol(state, "state")
            directory.write_json("state.json", state)
            # Sole bootstrap publication; a partial root is never silently repaired.
            directory.write_json("workspace.json", config)
        return cls(root)

    def _checkpoint(self, stage: str) -> None:
        """Trusted test subclass only; no request/env/CLI-selected failpoints."""

    def _inventory(self) -> dict:
        sizes = {}
        count = 0

        def walk(directory, prefix, depth):
            nonlocal count
            if depth > 4:
                raise ContextError("unexpected workspace nesting")
            for name, is_dir, size in directory.entries(max_entries=WORKSPACE_FILES - count + 1):
                count += 1
                if count > WORKSPACE_FILES:
                    raise ContextCapacity("workspace file capacity reached")
                path = prefix + name
                if not prefix:
                    if is_dir and name not in {"authority", "branches", "seals"}:
                        raise ContextError("unexpected workspace directory")
                    if not is_dir and name not in _ROOT_FILES and not _TEMP.fullmatch(name):
                        raise ContextError("unexpected workspace file")
                if prefix == "branches/" and (not is_dir or name not in {_branch_dir(k) for k in self.branches}):
                    raise ContextError("unexpected branch resource")
                if prefix == "seals/":
                    allowed = {n for branch in self.branches.values() for n in self._seal_names(branch)}
                    if is_dir or name not in allowed and not _TEMP.fullmatch(name):
                        raise ContextError("unexpected seal state file")
                if prefix == "authority/" and (is_dir or name not in {"lock", "key", "registry.json"} and not _TEMP.fullmatch(name)):
                    raise ContextError("unexpected authority state file")
                if is_dir:
                    walk(directory.child(name), path + "/", depth + 1)
                else:
                    sizes[path] = size
        walk(self.directory, "", 0)
        if sum(sizes.values()) > WORKSPACE_BYTES:
            raise ContextCapacity("workspace byte capacity reached")
        return {"bytes": sum(sizes.values()), "files": count, "sizes": sizes}

    def _reserve(self, size: int) -> None:
        usage = self._inventory()
        if usage["bytes"] + 2 * size > WORKSPACE_BYTES or usage["files"] + 2 > WORKSPACE_FILES:
            raise ContextCapacity("insufficient workspace atomic-write capacity")

    def _write(self, directory: PrivateDirectory, name: str, value: dict, *, guard=None, maximum=META_BYTES) -> None:
        raw = canonical_json(value)
        if len(raw) > maximum:
            raise ContextCapacity("context object exceeds byte capacity")
        self._reserve(len(raw))
        try:
            directory.write_bytes(name, raw, before_replace=guard)
        except OSError as exc:
            raise ContextIndeterminate("context durability uncertain; inspect original request") from exc

    def _immutable(self, directory: PrivateDirectory, name: str, value: dict, *, guard=None, maximum=META_BYTES) -> None:
        try:
            old = directory.read_bytes(name, max_bytes=maximum)
        except FileNotFoundError:
            self._write(directory, name, value, guard=guard, maximum=maximum)
        else:
            if old != canonical_json(value):
                raise ContextConflict("immutable context binding differs")
            self._sync(directory, name)

    @staticmethod
    def _sync(directory: PrivateDirectory, name: str) -> None:
        try:
            directory.sync_file(name)
        except OSError as exc:
            raise ContextIndeterminate("context durability barrier failed") from exc

    def _state(self) -> dict:
        if self.directory.read_bytes("workspace.json", max_bytes=META_BYTES) != canonical_json(self.config):
            raise ContextError("workspace identity changed")
        state = protocol(self.directory.read_json("state.json", max_bytes=META_BYTES), "state")
        if state["workspace_id"] != self.config["workspace_id"]:
            raise ContextError("workspace counter identity differs")
        self._sync(self.directory, "workspace.json")
        self._sync(self.directory, "state.json")
        return state

    def _admit(self, state: dict, key: str, maximum: int) -> None:
        used = state["counts"].get(key, 0)
        if used >= maximum:
            raise ContextCapacity("context operation budget exhausted")
        state["counts"][key] = used + 1
        self._write(self.directory, "state.json", protocol(state, "state"))
        self._checkpoint("after_admission")

    def _optional(self, directory: PrivateDirectory, name: str, maximum=META_BYTES) -> dict | None:
        try:
            raw = directory.read_bytes(name, max_bytes=maximum)
        except FileNotFoundError:
            return None
        from .private_fs import strict_json_loads
        result = strict_json_loads(raw)
        if raw != canonical_json(result):
            raise ContextError("noncanonical context object")
        return result

    def _seal_names(self, branch: dict) -> tuple[str, str]:
        key = _branch_dir(branch["branch_id"])
        return key + ".intent.json", key + ".json"

    def _require_open(self, branch: dict) -> None:
        directory = self.directory.child("seals")
        if any(self._optional(directory, name) is not None for name in self._seal_names(branch)):
            raise AuthorizationError("branch is sealing or sealed")

    def _resource(self, run_id: str, kind: str) -> dict | None:
        if kind == "branch":
            branch = next((b for b in self.branches.values() if b["run_id"] == run_id), None)
            if branch is None:
                raise AuthorizationError("capability resource differs")
            return branch
        expected = self.config["join_run_id" if kind == "join" else "view_run_id"]
        if run_id != expected:
            raise AuthorizationError("capability resource differs")
        return None

    @contextmanager
    def _locked(self, token: str, run_id: str, action: str, *, cancel_event=None) -> Iterator[tuple]:
        kind = "branch" if action.startswith("context.branch.") else "join" if action.startswith("context.join.") else "view"
        role = {"branch": "context_branch", "join": "context_join", "view": "context_verifier"}[kind]
        with self.authority._registry() as registry:
            claims = self.authority._check_locked(registry, token, owner=self.owner, run_id=run_id,
                                                  domain="fixture", action=action, now=self.clock())
            if claims["role"] != role:
                raise AuthorizationError("capability role differs")
            if len(registry["tokens"]) > 256 or len(registry["runs"]) > 6:
                raise ContextCapacity("workspace authority entry capacity reached")
            resource = self._resource(run_id, kind)
            with self.directory.lock():
                state = self._state()
                if resource is not None:
                    self._require_open(resource)
                elif kind == "join":
                    self._barrier_metadata(registry)
                _guard(claims, self.clock, cancel_event)
                yield registry, state, claims, resource

    @contextmanager
    def _operator(self):
        with self.authority._registry() as registry:
            with self.directory.lock():
                yield registry, self._state()

    def _store(self, branch: dict) -> MemoryStore:
        store = MemoryStore(self.root / "branches" / _branch_dir(branch["branch_id"]), clock=self.clock)
        if (store.config["store_id"] != branch["store_id"] or store.config_sha256 != branch["config_sha256"]
                or store.config["owner"] != self.owner or store.config["domain"] != "fixture"):
            raise ContextError("branch store identity differs")
        return store

    def _internal(self, branch: dict, state: dict, *, maintenance=False) -> tuple[MemoryStore, str]:
        store = self._store(branch)
        key = branch["branch_id"]
        cache_key = ("reserve." if maintenance else "ordinary.") + key
        cache = self._tokens.get(cache_key)
        now = self.clock()
        if cache is not None and now < cache[1] - 5:
            return store, cache[0]
        self._admit(state, ("reserve." if maintenance else "mint.") + _branch_dir(key), 64 if maintenance else 192)
        self._reserve_branch(branch, store)
        token = store.issue(run_id=INTERNAL_RUN, actions=list(ROLE_ACTIONS["memory"]), ttl_seconds=300)
        self._tokens[cache_key] = (token, now + 300)
        return store, token

    def _reserve_branch(self, branch: dict, store: MemoryStore) -> None:
        usage = self._inventory()
        prefix = "branches/" + _branch_dir(branch["branch_id"]) + "/"
        existing = sum(size for name, size in usage["sizes"].items() if name.startswith(prefix))
        # Reserve the whole remaining private-store budget, not a guessed write size.
        if usage["bytes"] - existing + store.limits["store_bytes"] + 2 * META_BYTES > WORKSPACE_BYTES:
            raise ContextCapacity("workspace cannot reserve branch write budget")
        existing_files = sum(name.startswith(prefix) for name in usage["sizes"])
        if usage["files"] - existing_files + store.limits["files"] + 4 > WORKSPACE_FILES:
            raise ContextCapacity("workspace cannot reserve branch file budget")

    def _baseline(self) -> dict:
        snapshot = self._optional(self.directory, "snapshot.json", GRAPH_BYTES)
        if snapshot is None or digest(snapshot) != self.config["snapshot_sha256"]:
            raise ContextError("frozen snapshot digest differs")
        protocol(snapshot, "snapshot", GRAPH_BYTES)
        for name in ("source", "public", "base_active"):
            if snapshot[name] != self.config["capture"][name]:
                raise ContextError("frozen capture binding differs")
        return checked_graph(snapshot["graph"])

    def issue(self, *, kind: str, actions: list[str], branch_id: str | None = None, ttl_seconds: float = 300) -> dict:
        """Trusted operator only. Returned token must be written to a private file."""
        if kind not in ("branch", "join", "view"):
            raise ContextError("unknown context resource kind")
        with self._operator() as (registry, state):
            if kind == "branch":
                if branch_id not in self.branches:
                    raise ContextError("unknown context branch")
                branch = self.branches[branch_id]
                self._require_open(branch)
                run_id = branch["run_id"]
            else:
                if branch_id is not None:
                    raise ContextError("branch ID is invalid for this resource")
                run_id = self.config["join_run_id" if kind == "join" else "view_run_id"]
                if kind == "join":
                    self._barrier(registry, state)
                elif self._load_view() is None:
                    raise ContextError("verification view is not published")
            if len(registry["tokens"]) >= 256:
                raise ContextCapacity("workspace authority token capacity reached")
            role = {"branch": "context_branch", "join": "context_join", "view": "context_verifier"}[kind]

            def reserve(proposed):
                raw = canonical_json(proposed)
                if len(raw) > 1048576:
                    raise ContextCapacity("workspace authority byte capacity reached")
                self._reserve(len(raw))
            token = self.authority._issue_locked(registry, owner=self.owner, run_id=run_id, domain="fixture",
                role=role, actions=actions, ttl_seconds=ttl_seconds, now=self.clock(), before_save=reserve)
            return {"run_id": run_id, "token": token}

    def branch_read(self, token: str, *, run_id: str, head: str | None = None, cancel_event=None) -> dict:
        with self._locked(token, run_id, "context.branch.read", cancel_event=cancel_event) as (_, state, claims, branch):
            if head is not None and not re.fullmatch(r"[0-9a-f]{64}", head):
                raise ContextError("invalid requested branch head")
            self._admit(state, "read." + _branch_dir(branch["branch_id"]), 64)
            store, internal = self._internal(branch, state)
            result = store.export(internal, run_id=INTERNAL_RUN, head=head)
            self._checkpoint("before_read_return")
            _guard(claims, self.clock, cancel_event)
            return result

    def branch_commit(self, token: str, request: dict, *, run_id: str, cancel_event=None) -> dict:
        with self._locked(token, run_id, "context.branch.commit", cancel_event=cancel_event) as (_, state, claims, branch):
            request = copied(request, REQUEST_BYTES)
            from .memory_store import _protocol
            _protocol(request, "request")
            if type(request["version"]) is not int:
                raise ContextError("commit version must be a strict integer")
            self._admit(state, "commit." + _branch_dir(branch["branch_id"]), 128)
            store, internal = self._internal(branch, state)
            current = store.export(internal, run_id=INTERNAL_RUN)["graph"]
            request = check_branch_request(current, request)
            # One wrapper/controller owns the branch; its workspace lock excludes
            # other mediated writes while this pure preflight checks tighter caps.
            index = {r["revision_id"] for r in current["revisions"]}
            if any(r["revision_id"] not in index for r in request["revisions"]) or any(
                    item[identity] not in {old[identity] for old in current[name]}
                    for name, identity in (("entities", "entity_id"), ("relations", "relation_id"), ("evidence", "evidence_id"))
                    for item in request[name]):
                checked_graph(store._merge(current, request))
            self._reserve_branch(branch, store)
            result = store.commit(internal, request, run_id=INTERNAL_RUN, cancel_event=cancel_event,
                                  outer_guard=lambda: _guard(claims, self.clock, cancel_event))
            return {k: result[k] for k in ("status", "replayed", "head")}

    def seal(self, token: str, request: dict, *, run_id: str, cancel_event=None) -> dict:
        with self._locked(token, run_id, "context.branch.seal", cancel_event=cancel_event) as (registry, state, claims, branch):
            request = protocol(request, "seal_request")
            self._admit(state, "maintain." + _branch_dir(branch["branch_id"]), 32)
            store, internal = self._internal(branch, state, maintenance=True)
            status = store.status(internal, run_id=INTERNAL_RUN)
            if status["status"] != "ready":
                raise MemoryRecoveryRequired("branch pending state blocks sealing")
            if status["head"] != request["expected_head"]:
                raise ContextConflict("seal expected head differs")
            graph = store.export(internal, run_id=INTERNAL_RUN)["graph"]
            active_map(request["active"], graph)
            intent = {"version": VERSION, "workspace_id": self.config["workspace_id"], "branch": branch,
                      "snapshot_sha256": self.config["snapshot_sha256"], "request": request}
            protocol(intent, "seal_intent")
            directory = self.directory.child("seals")
            name, _ = self._seal_names(branch)
            self._immutable(directory, name, intent, guard=lambda: _guard(claims, self.clock, cancel_event))
            self._checkpoint("after_seal_intent")
            return self._finish_seal(registry, branch, intent)

    def _finish_seal(self, registry: dict, branch: dict, intent: dict) -> dict:
        directory = self.directory.child("seals")
        intent_name, receipt_name = self._seal_names(branch)
        self._sync(directory, intent_name)
        try:
            self.authority._close_run_locked(registry, branch["run_id"])
            self._sync(self.authority._directory, "registry.json")
        except AuthorizationError as exc:
            raise ContextIndeterminate("branch closure durability uncertain") from exc
        self._checkpoint("after_seal_close")
        receipt = {"version": VERSION, "intent_sha256": digest(intent), "head": intent["request"]["expected_head"],
                   "run_id": branch["run_id"], "closed_epoch": registry["runs"][branch["run_id"]]["epoch"]}
        protocol(receipt, "seal_receipt")
        self._immutable(directory, receipt_name, receipt)
        self._checkpoint("after_seal_receipt")
        return {"status": "sealed", "head": receipt["head"], "seal_sha256": digest(receipt)}

    def _sealed(self, registry: dict, state: dict, branch: dict, *, require_receipt=True,
                maintenance=False) -> tuple[dict, dict | None, dict]:
        directory = self.directory.child("seals")
        intent_name, receipt_name = self._seal_names(branch)
        intent = self._optional(directory, intent_name)
        receipt = self._optional(directory, receipt_name)
        if intent is None or (require_receipt and receipt is None):
            raise ContextRecoveryRequired("all required branches must be durably sealed")
        protocol(intent, "seal_intent")
        if (intent["workspace_id"] != self.config["workspace_id"] or intent["branch"] != branch
                or intent["snapshot_sha256"] != self.config["snapshot_sha256"]):
            raise ContextError("seal intent resource binding differs")
        run = registry["runs"].get(branch["run_id"])
        if run is None or run["owner"] != self.owner or run["domain"] != "fixture":
            raise ContextError("seal run binding differs")
        if receipt is not None:
            protocol(receipt, "seal_receipt")
            expected = {"version": VERSION, "intent_sha256": digest(intent), "head": intent["request"]["expected_head"],
                        "run_id": branch["run_id"], "closed_epoch": run["epoch"]}
            if receipt != expected or run["state"] != "closed":
                raise ContextError("sealed receipt requires the matching closed run")
            self._sync(directory, receipt_name)
        store, internal = self._internal(branch, state, maintenance=maintenance)
        status = store.status(internal, run_id=INTERNAL_RUN)
        if status["status"] != "ready" or status["head"] != intent["request"]["expected_head"]:
            raise ContextRecoveryRequired("sealed branch chain differs or has pending state")
        graph = store.export(internal, run_id=INTERNAL_RUN)["graph"]
        active_map(intent["request"]["active"], graph)
        self._sync(directory, intent_name)
        return intent, receipt, graph

    def _barrier_metadata(self, registry: dict) -> None:
        # Complete the metadata barrier for every branch before hydrating any.
        directory = self.directory.child("seals")
        for branch in self.config["branches"]:
            intent_name, receipt_name = self._seal_names(branch)
            intent = self._optional(directory, intent_name)
            receipt = self._optional(directory, receipt_name)
            if intent is None or receipt is None:
                raise ContextRecoveryRequired("all required branches must be durably sealed")
            protocol(intent, "seal_intent")
            protocol(receipt, "seal_receipt")
            run = registry["runs"].get(branch["run_id"])
            if (run is None or run["state"] != "closed" or run["owner"] != self.owner
                    or run["domain"] != "fixture" or intent["branch"] != branch
                    or intent["workspace_id"] != self.config["workspace_id"]
                    or intent["snapshot_sha256"] != self.config["snapshot_sha256"]
                    or receipt != {"version": VERSION, "intent_sha256": digest(intent),
                        "head": intent["request"]["expected_head"], "run_id": branch["run_id"],
                        "closed_epoch": run["epoch"]}):
                raise ContextRecoveryRequired("join barrier has inconsistent closure evidence")

    def _barrier(self, registry: dict, state: dict) -> list[tuple]:
        self._barrier_metadata(registry)
        results = []
        for branch in self.config["branches"]:
            results.append(self._sealed(registry, state, branch))
        return results

    def recover_branch(self, branch_id: str, *, apply: bool = False) -> dict:
        """Operator-only recovery never reopens a run or chooses a newer seal head."""
        if type(apply) is not bool or branch_id not in self.branches:
            raise ContextError("invalid operator recovery request")
        with self._operator() as (registry, state):
            branch = self.branches[branch_id]
            self._admit(state, "maintain." + _branch_dir(branch_id), 32)
            store, internal = self._internal(branch, state, maintenance=True)
            recovery = store.recover(internal, run_id=INTERNAL_RUN, apply=apply)
            names = self._seal_names(branch)
            intent = self._optional(self.directory.child("seals"), names[0])
            if intent is None:
                if self._optional(self.directory.child("seals"), names[1]) is not None:
                    raise ContextError("seal receipt has no intent")
                return {"status": "unsealed", "store_recovery": recovery,
                        "run_state": registry["runs"][branch["run_id"]]["state"]}
            intent, receipt, _ = self._sealed(registry, state, branch, require_receipt=False, maintenance=True)
            if not apply:
                return {"status": "sealed" if receipt else "sealing", "head": intent["request"]["expected_head"],
                        "run_state": registry["runs"][branch["run_id"]]["state"]}
            return self._finish_seal(registry, branch, intent)

    def _join_object(self, registry: dict, state: dict, request: dict) -> dict:
        sealed = self._barrier(registry, state)
        baseline = self._baseline()
        selected = join_graph(baseline, [entry[2] for entry in sealed],
                              [entry[0]["request"]["active"] for entry in sealed], request["selection"])
        binding = {"version": VERSION, "workspace_id": self.config["workspace_id"],
                   "workspace_config_sha256": self.config_sha256, "source": self.config["capture"]["source"],
                   "snapshot_sha256": self.config["snapshot_sha256"],
                   "seals": [{"branch_id": item[0]["branch"]["branch_id"], "head": item[1]["head"],
                              "seal_sha256": digest(item[1])} for item in sealed],
                   "request_sha256": digest(request), "graph_sha256": digest(selected)}
        value = {"version": VERSION, "request": request, "binding": binding, "graph": selected,
                 "commit_request": join_delta(baseline, selected, binding)}
        return protocol(value, "join", BUNDLE_BYTES)

    def _load_join(self) -> dict | None:
        value = self._optional(self.directory, "join.json", BUNDLE_BYTES)
        if value is None:
            return None
        protocol(value, "join", BUNDLE_BYTES)
        binding = value["binding"]
        if (value["version"] != VERSION or binding["version"] != VERSION
                or binding["workspace_id"] != self.config["workspace_id"]
                or binding["workspace_config_sha256"] != self.config_sha256
                or binding["source"] != self.config["capture"]["source"]
                or binding["snapshot_sha256"] != self.config["snapshot_sha256"]
                or binding["request_sha256"] != digest(value["request"])
                or binding["graph_sha256"] != digest(value["graph"])
                or self._optional(self.directory, "join.intent.json", BUNDLE_BYTES) != value):
            raise ContextError("join publication binding differs")
        checked_graph(value["graph"])
        if join_delta(self._baseline(), value["graph"], binding) != value["commit_request"]:
            raise ContextError("join original request differs")
        self._sync(self.directory, "join.intent.json")
        self._sync(self.directory, "join.json")
        return value

    def join_prepare(self, token: str, request: dict, *, run_id: str, cancel_event=None) -> dict:
        with self._locked(token, run_id, "context.join.prepare", cancel_event=cancel_event) as (registry, state, claims, _):
            request = protocol(request, "join_request")
            if request["snapshot_sha256"] != self.config["snapshot_sha256"]:
                raise ContextConflict("join snapshot differs")
            self._admit(state, "join.calls", 128)
            existing = self._load_join()
            if existing is not None:
                if existing["request"] != request:
                    raise ContextConflict("workspace join is permanently bound")
                self._barrier(registry, state)
                _guard(claims, self.clock, cancel_event)
                return {"status": "joined", "replayed": True, "join_sha256": digest(existing),
                        "no_op": existing["commit_request"] is None}
            if self._optional(self.directory, "join.intent.json", BUNDLE_BYTES) is not None:
                raise ContextRecoveryRequired("prepared join requires operator recovery")
            self._admit(state, "prepare", 32)
            value = self._join_object(registry, state, request)
            guard = lambda: _guard(claims, self.clock, cancel_event)
            self._immutable(self.directory, "join.intent.json", value, guard=guard, maximum=BUNDLE_BYTES)
            self._checkpoint("after_join_intent")
            self._immutable(self.directory, "join.json", value, guard=guard, maximum=BUNDLE_BYTES)
            self._checkpoint("after_join_publish")
            return {"status": "joined", "replayed": False, "join_sha256": digest(value),
                    "no_op": value["commit_request"] is None}

    def join_read(self, token: str, *, run_id: str, cancel_event=None) -> dict:
        with self._locked(token, run_id, "context.join.read", cancel_event=cancel_event) as (registry, state, claims, _):
            self._admit(state, "read.join", 64)
            self._barrier(registry, state)
            value = self._load_join()
            if value is None:
                raise ContextError("join is not published")
            if value != self._join_object(registry, state, value["request"]):
                raise ContextError("join differs from its sealed source records")
            _guard(claims, self.clock, cancel_event)
            return {"join_sha256": digest(value), "bundle": value, "source_committed": False}

    def build_view(self, manifest: dict) -> dict:
        """Trusted operator declassification; a verifier cannot expand its view."""
        manifest = protocol(manifest, "proof_manifest")
        with self._operator() as (registry, state):
            self._admit(state, "view.calls", 128)
            existing = self._load_view()
            if existing is not None:
                if existing["manifest"] != manifest:
                    raise ContextConflict("verification view is permanently bound")
                return {"status": "view_frozen", "replayed": True, "view_sha256": digest(existing)}
            if self._optional(self.directory, "view.intent.json", BUNDLE_BYTES) is not None:
                raise ContextRecoveryRequired("prepared view requires operator recovery")
            self._admit(state, "prepare", 32)
            self._barrier(registry, state)
            value = self._view_object(manifest)
            self._immutable(self.directory, "view.intent.json", value, maximum=BUNDLE_BYTES)
            self._checkpoint("after_view_intent")
            self._immutable(self.directory, "view.json", value, maximum=BUNDLE_BYTES)
            self._checkpoint("after_view_publish")
            return {"status": "view_frozen", "replayed": False, "view_sha256": digest(value)}

    def _view_object(self, manifest: dict) -> dict:
        joined = self._load_join()
        if joined is None or (manifest["workspace_id"] != self.config["workspace_id"]
                or manifest["snapshot_sha256"] != self.config["snapshot_sha256"]
                or manifest["join_sha256"] != digest(joined)):
            raise ContextConflict("verification manifest source binding differs")
        dto = minimal_view(joined["graph"], manifest)
        return protocol({"version": VERSION, "manifest": manifest, "dto": dto, "dto_sha256": digest(dto)}, "view", BUNDLE_BYTES)

    def _load_view(self) -> dict | None:
        value = self._optional(self.directory, "view.json", BUNDLE_BYTES)
        if value is None:
            return None
        protocol(value, "view", BUNDLE_BYTES)
        manifest = protocol(value["manifest"], "proof_manifest")
        dto = protocol(value["dto"], "dto", GRAPH_BYTES)
        if (value["version"] != VERSION or manifest["workspace_id"] != self.config["workspace_id"]
                or manifest["snapshot_sha256"] != self.config["snapshot_sha256"]
                or value["dto_sha256"] != digest(dto) or dto["manifest_sha256"] != digest(manifest)
                or dto["roots"] != manifest["roots"]
                or self._optional(self.directory, "view.intent.json", BUNDLE_BYTES) != value):
            raise ContextError("verification publication binding differs")
        # Verification reads only its immutable DTO, never the broader join graph.
        # Construction/recovery separately rebuild from the pinned join source.
        self._sync(self.directory, "view.intent.json")
        self._sync(self.directory, "view.json")
        return value

    def view_read(self, token: str, *, run_id: str, cancel_event=None) -> dict:
        with self._locked(token, run_id, "context.view.read", cancel_event=cancel_event) as (_, state, claims, _):
            self._admit(state, "read.view", 64)
            value = self._load_view()
            if value is None:
                raise ContextError("verification view is not published")
            self._checkpoint("before_read_return")
            _guard(claims, self.clock, cancel_event)
            return value["dto"]

    def probe(self, token: str, *, run_id: str, cancel_event=None) -> dict:
        """Actual fresh sandbox context observation, never an LLM/math verdict."""
        import hashlib
        from importlib.resources import files
        import tempfile

        from .private_fs import strict_json_loads
        from .worker import SandboxWorker, WorkerUnavailable

        with self._locked(token, run_id, "context.view.probe", cancel_event=cancel_event) as (_, state, claims, _):
            self._admit(state, "probe.view", 16)
            value = self._load_view()
            if value is None:
                raise ContextError("verification view is not published")
            dto = value["dto"]
            view_sha = digest(value)
            _guard(claims, self.clock, cancel_event)
        raw = canonical_json(dto)
        driver = files("rethlas_core").joinpath("context_fixture_driver.py").read_bytes()
        driver_sha = hashlib.sha256(driver).hexdigest()
        policy = {"backend": "bubblewrap", "command": ["/usr/bin/python3.12", "-I", "-B", "/input/driver.py"],
                  "network": "disabled", "input": "read-only", "work": "new-empty-private",
                  "environment": ["PATH", "LANG", "HOME"], "timeout_seconds": 5, "output_bytes": 262144}
        with tempfile.TemporaryDirectory(prefix="rethlas-context-") as temporary:
            directory = PrivateDirectory(Path(temporary))
            input_dir = directory.child("input", create=True)
            work_dir = directory.child("work", create=True)
            input_dir.write_bytes("context.json", raw)
            input_dir.write_bytes("driver.py", driver)
            # Reauthorize at dispatch without retaining locks while the child runs.
            with self._locked(token, run_id, "context.view.probe", cancel_event=cancel_event):
                pass
            output = SandboxWorker().run_bounded(input_dir.root, work_dir.root, policy["command"],
                                                  cancel_event=cancel_event)
        observed = strict_json_loads(output.stdout)
        expected = {"version": "context-observation-1", "input_files": ["context.json", "driver.py"],
                    "input_sha256": hashlib.sha256(raw).hexdigest(), "driver_sha256": driver_sha,
                    "manifest_sha256": dto["manifest_sha256"],
                    "revisions": [{**{k: r[k] for k in ("entity_id", "revision_id", "type", "body_sha256")},
                                   "body_bytes": len(r["body"].encode("utf-8"))} for r in dto["revisions"]],
                    "relations": len(dto["relations"]), "empty_work": True, "input_readonly": True,
                    "mathematical_review": False}
        if observed != expected or output.stderr:
            raise WorkerUnavailable("child context observation differs from exact approved input")
        with self._locked(token, run_id, "context.view.probe", cancel_event=cancel_event) as (_, _, claims, _):
            _guard(claims, self.clock, cancel_event)
            return {"status": "synthetic_context_observed", "view_sha256": view_sha,
                    "policy": policy, "policy_sha256": digest(policy), "observation": observed,
                    "exit_code": output.returncode, "mathematical_review": False}

    def recover_publication(self, *, kind: str, request: dict, apply: bool = False) -> dict:
        """Finish only an exact prepared operator-selected join/view publication."""
        if kind not in ("join", "view") or type(apply) is not bool:
            raise ContextError("invalid publication recovery")
        request = protocol(request, "join_request" if kind == "join" else "proof_manifest")
        with self._operator() as (registry, state):
            self._admit(state, "recover.publication", 32)
            value = self._optional(self.directory, kind + ".intent.json", BUNDLE_BYTES)
            if value is None or value["request" if kind == "join" else "manifest"] != request:
                raise ContextConflict("exact prepared publication is required")
            rebuilt = self._join_object(registry, state, request) if kind == "join" else self._view_object(request)
            if value != rebuilt:
                raise ContextError("prepared publication differs from frozen source")
            if apply:
                self._sync(self.directory, kind + ".intent.json")
                self._immutable(self.directory, kind + ".json", value, maximum=BUNDLE_BYTES)
            return {"status": "published" if apply else "prepared", "kind": kind, "sha256": digest(value)}
