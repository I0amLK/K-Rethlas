"""Owned graph/current-validity controller with a fixture-only default protocol.

Lock order is service authority -> control -> owned memory authority -> store.
The graph manifest is the only graph commit point. Independent control anchors
withdrawals and accepted graph checkpoints; it never supplies a second graph.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
from importlib.resources import files
import math
from pathlib import Path
import re
import secrets
import tempfile
import time
from typing import Any, Callable

from .authorization import Authority, AuthorizationError, ROLE_ACTIONS
from .context_views import check_branch_request, exact_ref
from .memory_store import (
    MemoryStore, MemoryCancelled, MemoryRecoveryRequired, MemoryIndeterminate,
    _protocol as memory_protocol, _result_manifest,
)
from .private_fs import PrivateDirectory, canonical_json, digest, strict_json_loads
from .validity import (
    VERSION, POLICY_SHA256, ValidityError, ValidityConflict, ValidityCapacity,
    ValidityRecoveryRequired, ValidityIndeterminate, project_use, use_overlay,
    validity_states, review_queue,
)
from .validity_control import ControlJournal, stable_anchor
from .validity_protocol import (
    META_BYTES, STATE_BYTES, REQUEST_BYTES, DTO_BYTES, USE_BYTES,
    ROOT_BYTES, CONTROL_BYTES, STORE_BYTES, ROOT_FILES, protocol, copied,
)


INTERNAL_RUN = "validity-internal"
_TEMP = re.compile(r"\.tmp-[0-9a-f]{32}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
ROLES = frozenset({"validity_writer", "validity_withdrawer", "validity_reader", "validity_verifier", "validity_publisher"})
SANDBOX_POLICY = {"backend": "bubblewrap", "command": ["/usr/bin/python3.12", "-I", "-B", "/input/driver.py"],
                  "network": "disabled", "input": "read-only", "work": "new-empty-private",
                  "environment": ["PATH", "LANG", "HOME"], "timeout_seconds": 5, "output_bytes": 262144}


def _guard(claims: dict | None, clock: Callable, cancel_event=None) -> None:
    now = clock()
    if type(now) not in (int, float) or not math.isfinite(now):
        raise AuthorizationError("invalid trusted clock")
    if claims is not None and not claims["issued_at"] <= now < claims["expires_at"]:
        raise AuthorizationError("validity capability expired or is not yet valid")
    if cancel_event is not None and cancel_event.is_set():
        raise MemoryCancelled("validity operation cancelled before acceptance")


def _floor(checkpoint: dict) -> dict:
    return {key: checkpoint[key] for key in ("head", "generation", "graph_sha256")}


def _fields(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValidityError("invalid validity object fields")


class CurrentValidityService:
    """Operator-side Python API; not a same-UID filesystem security boundary."""

    DOMAIN = "fixture"
    VERSION = VERSION
    POLICY_HASH = POLICY_SHA256
    PROTOCOL = staticmethod(protocol)
    COMMIT_ACTION = "validity.commit"
    WITHDRAW_ACTION = "validity.withdraw"
    DIRECTORIES = ("authority", "control", "memory", "uses", "operations")

    def __init__(self, root: Path, *, clock: Callable = time.time):
        self.directory = PrivateDirectory(Path(root))
        self.root = self.directory.root
        self.clock = clock
        self.config = self.PROTOCOL(self.directory.read_json("service.json", max_bytes=META_BYTES), "config")
        self.config_sha256 = digest(self.config)
        self.owner = self.config["owner"]
        if (self.config["version"] != self.VERSION or self.config["domain"] != self.DOMAIN
                or self.config["policy_hash"] != self.POLICY_HASH):
            raise ValidityError("fixed validity service policy differs")
        self.parts = {name: self.directory.child(name) for name in self.DIRECTORIES}
        names = {n: d for n, d, _ in self.parts["authority"].entries(max_entries=ROOT_FILES)}
        if any(names.get(n) is not False for n in ("key", "registry.json", "lock")):
            raise ValidityError("service authority initialization is incomplete")
        self.authority = Authority(self.parts["authority"].root)
        self._tokens = {}
        self.journal = ControlJournal(self.parts["control"], self.config,
                                      reserve=self._reserve, checkpoint=self._checkpoint,
                                      protocol=self.PROTOCOL, version=self.VERSION)

    @classmethod
    def initialize(cls, root: Path, *, owner: str, domain: str = "fixture"):
        if domain != cls.DOMAIN:
            raise ValidityError("service requires its fixed domain")
        directory = PrivateDirectory(Path(root), create=True)
        if directory.entries(max_entries=ROOT_FILES):
            raise ValidityConflict("initialization requires a new empty private root")
        with directory.lock():
            if {n for n, _, _ in directory.entries(max_entries=3)} != {"lock"}:
                raise ValidityConflict("service root changed during initialization")
            parts = {n: directory.child(n, create=True) for n in cls.DIRECTORIES}
            Authority(parts["authority"].root, create=True)
            store = MemoryStore.initialize(parts["memory"].root, owner=owner, domain=cls.DOMAIN,
                limits={"history": 64, "receipts": 64, "store_bytes": STORE_BYTES, "files": 4096})
            genesis = store.directory.read_json("manifest.json", max_bytes=META_BYTES)
            config = cls.PROTOCOL({"version": cls.VERSION, "service_id": secrets.token_hex(32), "owner": owner,
                "domain": cls.DOMAIN, "policy_hash": cls.POLICY_HASH, "store_id": store.config["store_id"],
                "store_config_sha256": store.config_sha256, "genesis": _floor(genesis)}, "config")
            state = cls.PROTOCOL({"version": cls.VERSION, "service_id": config["service_id"], "config_sha256": digest(config),
                "sequence": 0, "events": [], "requests": {}, "accepted_graph": _floor(genesis), "pending_graph": None,
                "uses": {}, "probes": {}, "publications": {}, "runs": {}, "counts": {}, "withdrawal_version": 0}, "state", STATE_BYTES)
            parts["control"].write_json("state.json", state)
            parts["control"].write_json("anchor.json", stable_anchor(config, state))
            directory.write_json("service.json", config)
        return cls(directory.root)

    def _checkpoint(self, stage: str) -> None:
        """Trusted synthetic fault subclass only; never request/env-selected."""

    def _inventory(self) -> dict:
        sizes = {}
        count = 0

        def walk(directory, prefix, depth):
            nonlocal count
            if depth > 3:
                raise ValidityError("unexpected service directory nesting")
            for name, is_dir, size in directory.entries(max_entries=ROOT_FILES - count + 2):
                count += 1
                if count > ROOT_FILES:
                    raise ValidityCapacity("service file capacity reached")
                path = prefix + name
                if not prefix:
                    if is_dir and name not in self.parts or not is_dir and name not in {"lock", "service.json"} and not _TEMP.fullmatch(name):
                        raise ValidityError("unexpected service entry")
                if prefix in ("authority/", "control/", "operations/", "uses/"):
                    allowed = {"authority/": {"key", "registry.json", "lock"},
                               "control/": {"state.json", "anchor.json", "transition.json"}}.get(prefix)
                    valid = name in allowed if allowed is not None else bool(re.fullmatch(r"[0-9a-f]{64}\.json", name))
                    if is_dir or not valid and not _TEMP.fullmatch(name):
                        raise ValidityError("unexpected control or immutable object entry")
                if is_dir:
                    walk(directory.child(name), path + "/", depth + 1)
                else:
                    sizes[path] = size
        walk(self.directory, "", 0)
        total = sum(sizes.values())
        control = sum(s for n, s in sizes.items() if not n.startswith("memory/"))
        if total > ROOT_BYTES or control > CONTROL_BYTES:
            raise ValidityCapacity("service byte capacity reached")
        return {"bytes": total, "control_bytes": control, "files": count, "sizes": sizes}

    def _reserve(self, size: int, *, reserved=False) -> None:
        usage = self._inventory()
        safety = 0 if reserved else 33554432
        if (usage["bytes"] + 2 * size + safety > ROOT_BYTES
                or usage["control_bytes"] + 2 * size + safety > CONTROL_BYTES
                or usage["files"] + (4 if reserved else 260) > ROOT_FILES):
            raise ValidityCapacity("insufficient service atomic-write capacity")

    def _immutable(self, directory, name, value, maximum, *, reserved=False):
        raw = canonical_json(value)
        if len(raw) > maximum:
            raise ValidityCapacity("immutable validity object capacity reached")
        self._reserve(len(raw), reserved=reserved)
        try:
            old = directory.read_bytes(name, max_bytes=maximum)
        except FileNotFoundError:
            try:
                directory.write_bytes(name, raw)
            except OSError as exc:
                raise ValidityIndeterminate("immutable validity object durability uncertain") from exc
        else:
            if old != raw:
                raise ValidityConflict("immutable validity object differs")
            directory.sync_file(name)

    def _state(self):
        if self.directory.read_bytes("service.json", max_bytes=META_BYTES) != canonical_json(self.config):
            raise ValidityError("service configuration changed")
        self._inventory()
        self.directory.sync_file("service.json")
        return self.journal.ready()

    @contextmanager
    def _operator(self, *, ready=True):
        with self.authority._registry() as registry:
            with self.directory.lock():
                state = self._state() if ready else self.journal.inspect()[0]
                yield registry, state

    @contextmanager
    def _locked(self, token, run_id, action, cancel_event=None):
        with self.authority._registry() as registry:
            claims = self.authority._check_locked(registry, token, owner=self.owner, run_id=run_id,
                domain=self.DOMAIN, action=action, now=self.clock())
            if claims["role"] not in ROLES or action not in ROLE_ACTIONS[claims["role"]]:
                raise AuthorizationError("validity capability role differs")
            if len(registry["runs"]) > 32 or len(registry["tokens"]) > 256:
                raise ValidityCapacity("service authority capacity reached")
            with self.directory.lock():
                state = self._state()
                resource = state["runs"].get(run_id)
                if resource is None or resource["role"] != claims["role"]:
                    raise AuthorizationError("validity run resource differs")
                _guard(claims, self.clock, cancel_event)
                yield registry, state, claims, resource["resource"]

    def _change(self, state, kind, mutate, *, key=None, claims=None, cancel_event=None, reserved=False):
        following = self.journal.change(state, kind, mutate, request_key=key, claims=claims,
            guard=lambda: _guard(claims, self.clock, cancel_event), reserved=reserved)
        state.clear()
        state.update(following)
        try:
            self._checkpoint("after_" + kind.lower())
        except Exception as exc:
            raise ValidityIndeterminate("control change accepted before interrupted return") from exc

    def _admit(self, state, name, maximum, *, claims=None, cancel_event=None, reserved=False):
        if state["counts"].get(name, 0) >= maximum:
            raise ValidityCapacity("validity operation budget exhausted")
        self._change(state, "ADMISSION", lambda n: n["counts"].update({name: n["counts"].get(name, 0) + 1}),
                     claims=claims, cancel_event=cancel_event, reserved=reserved)

    def _store(self):
        store = MemoryStore(self.parts["memory"].root, clock=self.clock)
        if (store.config["store_id"] != self.config["store_id"] or store.config_sha256 != self.config["store_config_sha256"]
                or store.config["owner"] != self.owner or store.config["domain"] != self.DOMAIN):
            raise ValidityError("owned store identity differs")
        return store

    def _internal(self, state, *, maintenance=False):
        store = self._store()
        name = "internal.reserve" if maintenance else "internal.ordinary"
        cached = self._tokens.get(name)
        now = self.clock()
        if cached is not None and now < cached[1] - 5:
            return store, cached[0]
        self._admit(state, name, 64 if maintenance else 192, reserved=maintenance)
        token = store.issue(run_id=INTERNAL_RUN, actions=list(ROLE_ACTIONS["memory"]), ttl_seconds=300)
        self._tokens[name] = token, now + 300
        return store, token

    @contextmanager
    def _owned_checkpoint(self, state, store, token):
        sequence = state["sequence"]
        try:
            with store.locked_checkpoint(token, run_id=INTERNAL_RUN) as checkpoint:
                yield checkpoint
        except (AuthorizationError, MemoryCancelled) as exc:
            if state["sequence"] != sequence:
                raise ValidityIndeterminate("control changed before checkpoint return guard failed") from exc
            raise

    @contextmanager
    def _gate(self, state, *, maintenance=False):
        if state["pending_graph"] is not None:
            raise ValidityRecoveryRequired("graph acceptance fence requires recovery")
        store, token = self._internal(state, maintenance=maintenance)
        with self._owned_checkpoint(state, store, token) as checkpoint:
            if _floor(checkpoint) != state["accepted_graph"]:
                raise ValidityError("disk graph differs from accepted exact checkpoint")
            yield store, checkpoint

    def _identity(self, claims, resource):
        if claims is None:
            return {"owner": self.owner, "run_id": "operator", "domain": self.DOMAIN, "role": "operator", "epoch": 0, "resource": resource}
        return {**{k: claims[k] for k in ("owner", "run_id", "domain", "role", "epoch")}, "resource": resource}

    def _binding(self, state, kind, request, identity, *, claims=None, cancel_event=None):
        key = digest({"service_id": self.config["service_id"], "request_id": request["request_id"]})
        operation = {"version": self.VERSION, "service_id": self.config["service_id"], "kind": kind,
                     "identity": identity, "request": request}
        old = state["requests"].get(key)
        if old is not None:
            if old["operation_sha256"] != digest(operation):
                raise ValidityConflict("request ID is permanently bound to another operation")
            return key, old
        bucket = "withdraw" if kind == "withdraw" else "ordinary"
        if sum(b["bucket"] == bucket for b in state["requests"].values()) >= (32 if bucket == "withdraw" else 192):
            raise ValidityCapacity("permanent validity request capacity reached")
        self._immutable(self.parts["operations"], digest(operation) + ".json", operation, REQUEST_BYTES + META_BYTES, reserved=bucket == "withdraw")
        binding = {"request_id": request["request_id"], "operation_sha256": digest(operation), "identity": identity,
                   "kind": kind, "bucket": bucket, "status": "bound", "result": {}}
        self._change(state, "REQUEST_BOUND", lambda n: n["requests"].update({key: binding}), key=key,
                     claims=claims, cancel_event=cancel_event, reserved=bucket == "withdraw")
        return key, state["requests"][key]

    def _operation(self, binding):
        raw = self.parts["operations"].read_bytes(binding["operation_sha256"] + ".json", max_bytes=REQUEST_BYTES + META_BYTES)
        operation = strict_json_loads(raw)
        _fields(operation, {"version", "service_id", "kind", "identity", "request"})
        if (raw != canonical_json(operation) or digest(operation) != binding["operation_sha256"]
                or operation["version"] != self.VERSION
                or operation["service_id"] != self.config["service_id"] or operation["identity"] != binding["identity"]
                or operation["kind"] != binding["kind"] or operation["request"]["request_id"] != binding["request_id"]):
            raise ValidityError("permanent operation object differs")
        return operation

    def _withdrawals(self, state):
        result = []
        for binding in state["requests"].values():
            if binding["kind"] == "withdraw" and binding["status"] == "withdrawn":
                request = self.PROTOCOL(self._operation(binding)["request"], "withdraw")
                result.append({"event_id": binding["result"]["withdrawal_id"], "targets": request["targets"]})
        if len(result) != state["withdrawal_version"]:
            raise ValidityError("withdrawal version differs from permanent events")
        return sorted(result, key=lambda r: r["event_id"])

    @staticmethod
    def _terminal(binding):
        if binding["status"] == "aborted":
            raise ValidityConflict("request was permanently terminated")

    def issue(self, *, role: str, actions: list[str], use_id: str | None = None, ttl_seconds=300) -> dict:
        if role not in ROLES:
            raise AuthorizationError("unknown validity role")
        store_role = role in ("validity_writer", "validity_withdrawer")
        if store_role and use_id is not None or not store_role and use_id is None:
            raise ValidityError("role requires its fixed resource kind")
        resource = "owned-store" if store_role else use_id
        self.PROTOCOL(resource, "scope")
        run_id = "v." + digest({"service_id": self.config["service_id"], "role": role, "resource": resource})
        with self._operator() as (registry, state):
            if not store_role and resource not in state["uses"]:
                raise ValidityError("use is not registered")
            if not actions or not set(actions) <= ROLE_ACTIONS[role]:
                raise AuthorizationError("actions exceed fixed validity role")
            self._admit(state, "service.tokens", 256)
            if run_id not in state["runs"]:
                if len(state["runs"]) >= 32:
                    raise ValidityCapacity("service fixed-run capacity reached")
                self._change(state, "RUN_BOUND", lambda n: n["runs"].update({run_id: {"role": role, "resource": resource}}))
            if run_id not in registry["runs"]:
                registry["runs"][run_id] = {"owner": self.owner, "domain": self.DOMAIN, "state": "open", "epoch": 0}
            if len(registry["tokens"]) >= 256:
                raise ValidityCapacity("service capability registry capacity reached")
            def reserve(proposed):
                raw = canonical_json(proposed)
                if len(raw) > 1048576:
                    raise ValidityCapacity("service authority byte capacity reached")
                self._reserve(len(raw))
            token = self.authority._issue_locked(registry, owner=self.owner, run_id=run_id, domain=self.DOMAIN,
                role=role, actions=actions, ttl_seconds=ttl_seconds, now=self.clock(), before_save=reserve)
            return {"run_id": run_id, "token": token}

    def commit(self, token: str, request: dict, *, run_id: str, cancel_event=None) -> dict:
        request = copied(request, REQUEST_BYTES)
        memory_protocol(request, "request")
        if type(request["version"]) is not int:
            raise ValidityError("graph request version must be a strict integer")
        with self._locked(token, run_id, self.COMMIT_ACTION, cancel_event) as (_, state, claims, resource):
            self._admit(state, "commit.calls", 128, claims=claims, cancel_event=cancel_event)
            key, binding = self._binding(state, "commit", request, self._identity(claims, resource), claims=claims, cancel_event=cancel_event)
            self._terminal(binding)
            if binding["status"] == "committed":
                with self._gate(state):
                    _guard(claims, self.clock, cancel_event)
                    return {**binding["result"], "replayed": True, "accepted_graph": state["accepted_graph"], "historical_only": True}
            if state["pending_graph"] is not None:
                raise ValidityRecoveryRequired("original graph transaction requires recovery")
            if request["expected_head"] != state["accepted_graph"]["head"]:
                raise ValidityConflict("expected graph checkpoint differs")
            with self._gate(state) as (store, _):
                pass
            store, internal = self._internal(state)
            exported = store.export(internal, run_id=INTERNAL_RUN)
            if exported["head"] != request["expected_head"]:
                raise ValidityConflict("disk graph changed before preparation")
            check_branch_request(exported["graph"], request)
            if state["counts"].get("graph.commits", 0) >= 64:
                raise ValidityCapacity("managed graph commit capacity reached")
            parent = deepcopy(state["accepted_graph"])
            def prepare(n):
                n["pending_graph"] = {"request_key": key, "operation_sha256": binding["operation_sha256"], "parent": parent}
                n["counts"]["graph.commits"] = n["counts"].get("graph.commits", 0) + 1
            self._change(state, "GRAPH_PREPARE", prepare, key=key, claims=claims, cancel_event=cancel_event)
            result = store.commit(internal, request, run_id=INTERNAL_RUN, cancel_event=cancel_event,
                                  outer_guard=lambda: _guard(claims, self.clock, cancel_event))
            try:
                return self._accept_graph(state, key, store, internal, result, claims=claims, cancel_event=cancel_event)
            except (AuthorizationError, MemoryCancelled, ValidityError, OSError) as exc:
                if isinstance(exc, ValidityIndeterminate):
                    raise
                raise ValidityIndeterminate("graph committed before control acceptance completed") from exc

    def _accept_graph(self, state, key, store, internal, result, *, claims=None, cancel_event=None):
        pending = state["pending_graph"]
        receipt = result["receipt"]
        expected = _result_manifest(receipt)
        operation = self._operation(state["requests"][key])
        if (pending is None or pending["request_key"] != key or result["head"] != expected["head"]
                or _floor(receipt["parent_manifest"]) != pending["parent"]
                or receipt["binding"]["request_id"] != operation["request"]["request_id"]
                or receipt["binding"]["run_id"] != INTERNAL_RUN):
            raise ValidityError("graph acceptance receipt differs from exact fenced request")
        with self._owned_checkpoint(state, store, internal) as checkpoint:
            if _floor(checkpoint) != _floor(expected):
                raise ValidityError("graph acceptance checkpoint differs")
            response = {"status": "committed", "head": expected["head"], "generation": expected["generation"],
                        "receipt_sha256": digest(receipt)}
            def accept(n):
                n["accepted_graph"] = _floor(expected)
                n["pending_graph"] = None
                n["requests"][key].update(status="committed", result=response)
            self._change(state, "GRAPH_ACCEPT", accept, key=key, claims=claims, cancel_event=cancel_event, reserved=True)
        return {**response, "replayed": False, "accepted_graph": state["accepted_graph"], "historical_only": False}

    def withdraw(self, token: str, request: dict, *, run_id: str, cancel_event=None):
        request = self.PROTOCOL(request, "withdraw")
        if len(request["reason"].encode("utf-8")) > 4096:
            raise ValidityCapacity("withdrawal reason byte capacity reached")
        with self._locked(token, run_id, self.WITHDRAW_ACTION, cancel_event) as (_, state, claims, resource):
            self._admit(state, "withdraw.calls", 64, claims=claims, cancel_event=cancel_event, reserved=True)
            key, binding = self._binding(state, "withdraw", request, self._identity(claims, resource), claims=claims, cancel_event=cancel_event)
            self._terminal(binding)
            if binding["status"] == "withdrawn":
                with self._gate(state, maintenance=True):
                    _guard(claims, self.clock, cancel_event)
                    return {**binding["result"], "replayed": True, "withdrawal_version": state["withdrawal_version"]}
            with self._gate(state, maintenance=True) as (store, _):
                pass
            store, internal = self._internal(state, maintenance=True)
            exported = store.export(internal, run_id=INTERNAL_RUN)
            index = {r["revision_id"]: r for r in exported["graph"]["revisions"]}
            if any(t["revision_id"] not in index or exact_ref(index[t["revision_id"]]) != t for t in request["targets"]):
                raise ValidityError("withdrawal requires exact committed targets")
            if state["withdrawal_version"] >= 32:
                raise ValidityCapacity("withdrawal event capacity reached")
            response = {"status": "withdrawn", "withdrawal_id": digest({"service_id": self.config["service_id"], "request_key": key}),
                        "targets": request["targets"]}
            with self._gate(state, maintenance=True):
                def append(n):
                    n["withdrawal_version"] += 1
                    n["requests"][key].update(status="withdrawn", result=response)
                self._change(state, "WITHDRAW", append, key=key, claims=claims, cancel_event=cancel_event, reserved=True)
            return {**response, "replayed": False, "withdrawal_version": state["withdrawal_version"]}

    def register_use(self, request: dict):
        request = protocol(request, "register", REQUEST_BYTES)
        with self._operator() as (_, state):
            self._admit(state, "operator", 64)
            key, binding = self._binding(state, "register", request, self._identity(None, request["use_id"]))
            self._terminal(binding)
            if binding["status"] == "registered":
                with self._gate(state):
                    return {**binding["result"], "replayed": True}
            if request["use_id"] in state["uses"] or len(state["uses"]) >= 8:
                raise ValidityConflict("use identity is immutable or capacity reached")
            with self._gate(state):
                if request["expected_head"] != state["accepted_graph"]["head"]:
                    raise ValidityConflict("use registration expected graph differs")
            store, internal = self._internal(state)
            exported = store.export(internal, run_id=INTERNAL_RUN)
            if exported["head"] != request["expected_head"]:
                raise ValidityConflict("use registration graph changed")
            projection = project_use(exported["graph"], roots=request["roots"], dependencies=request["dependencies"], references=request["references"])
            dto = {"version": "validity-dto-1", "policy_hash": POLICY_SHA256,
                   **{k: projection[k] for k in ("roots", "revisions", "relations")}}
            descriptor = {"version": "validity-use-1", "service_id": self.config["service_id"], "store_id": self.config["store_id"],
                "store_config_sha256": self.config["store_config_sha256"], "use_id": request["use_id"],
                "graph": deepcopy(state["accepted_graph"]), "policy_hash": POLICY_SHA256, "dto_sha256": digest(dto),
                "support": projection["support"], "citation_only": projection["citation_only"]}
            value = protocol({"dto": dto, "descriptor": descriptor}, "use", USE_BYTES)
            if len(canonical_json(dto)) > DTO_BYTES or len(canonical_json(descriptor)) > REQUEST_BYTES:
                raise ValidityCapacity("registered fixture input byte capacity reached")
            self._immutable(self.parts["uses"], digest(value) + ".json", value, USE_BYTES)
            record = {"object_sha256": digest(value), "dto_sha256": digest(dto), "descriptor_sha256": digest(descriptor),
                      "graph": deepcopy(state["accepted_graph"]), "request_key": key}
            response = {"status": "registered", "use_id": request["use_id"], "dto_sha256": digest(dto), "descriptor_sha256": digest(descriptor)}
            with self._gate(state):
                def register(n):
                    n["uses"][request["use_id"]] = record
                    n["requests"][key].update(status="registered", result=response)
                self._change(state, "USE_REGISTERED", register, key=key)
            return {**response, "replayed": False}

    def _use(self, state, use_id):
        record = state["uses"].get(use_id)
        if record is None:
            raise AuthorizationError("run has no registered fixed use")
        raw = self.parts["uses"].read_bytes(record["object_sha256"] + ".json", max_bytes=USE_BYTES)
        value = protocol(strict_json_loads(raw), "use", USE_BYTES)
        _fields(value, {"dto", "descriptor"})
        dto, descriptor = value["dto"], value["descriptor"]
        _fields(dto, {"version", "policy_hash", "roots", "revisions", "relations"})
        _fields(descriptor, {"version", "service_id", "store_id", "store_config_sha256", "use_id", "graph", "policy_hash", "dto_sha256", "support", "citation_only"})
        if (raw != canonical_json(value) or digest(value) != record["object_sha256"]
                or digest(dto) != record["dto_sha256"] or digest(descriptor) != record["descriptor_sha256"]
                or descriptor["dto_sha256"] != record["dto_sha256"] or descriptor["graph"] != record["graph"]
                or descriptor["service_id"] != self.config["service_id"] or descriptor["store_id"] != self.config["store_id"]
                or descriptor["store_config_sha256"] != self.config["store_config_sha256"] or descriptor["use_id"] != use_id
                or descriptor["policy_hash"] != POLICY_SHA256 or dto["policy_hash"] != POLICY_SHA256
                or dto["version"] != "validity-dto-1" or descriptor["version"] != "validity-use-1"):
            raise ValidityError("frozen use binding differs")
        projection = {**{k: dto[k] for k in ("roots", "revisions", "relations")},
                      **{k: descriptor[k] for k in ("support", "citation_only")}}
        overlay = protocol(use_overlay(projection, self._withdrawals(state)), "overlay", REQUEST_BYTES)
        return value, projection, overlay

    def _checked_at(self, state):
        return {"control_sequence": state["sequence"], "withdrawal_version": state["withdrawal_version"],
                "graph_head": state["accepted_graph"]["head"], "policy_hash": self.POLICY_HASH}

    def _current(self, state, use_id, overlay):
        return protocol({"use_id": use_id, "fixture_input_eligible": overlay["fixture_input_eligible"],
                "blockers": overlay["blockers"], "overlay": overlay, "overlay_sha256": digest(overlay),
                "checked_at": self._checked_at(state), "assertion_state": "candidate", "validity": "needs_review",
                "mathematical_review": False, "production_completed": False}, "current", REQUEST_BYTES + META_BYTES)

    @staticmethod
    def _eligible(overlay):
        if not overlay["fixture_input_eligible"]:
            raise ValidityError("fixed mathematical support is withdrawn")

    def read(self, token: str, *, run_id: str, cancel_event=None):
        with self._locked(token, run_id, "validity.read", cancel_event) as (_, state, claims, use_id):
            self._admit(state, "read." + digest(use_id), 64, claims=claims, cancel_event=cancel_event)
            with self._gate(state):
                value, _, overlay = self._use(state, use_id)
                self._eligible(overlay)
                self._checkpoint("before_read_return")
                _guard(claims, self.clock, cancel_event)
                return {"status": "current_fixture_input", **value, "current": self._current(state, use_id, overlay)}

    def lookup(self, token: str, request: dict, *, run_id: str, cancel_event=None):
        request = protocol(request, "lookup")
        with self._locked(token, run_id, "validity.read", cancel_event) as (_, state, claims, use_id):
            self._admit(state, "read." + digest(use_id), 64, claims=claims, cancel_event=cancel_event)
            with self._gate(state):
                value, projection, overlay = self._use(state, use_id)
                self._eligible(overlay)
                index = {r["revision_id"]: r for r in value["dto"]["revisions"]}
                if any(t["revision_id"] not in index or exact_ref(index[t["revision_id"]]) != t for t in request["refs"]):
                    raise ValidityError("lookup is outside exact registered disclosure")
                selected = {t["revision_id"] for t in request["refs"]}
                relation_ids = set()
                frontier = set(selected)
                for _ in range(request["depth"]):
                    following = set()
                    for key in frontier:
                        following.update(index[key]["context_revision_ids"])
                    for edge in value["dto"]["relations"]:
                        if edge["source"]["revision_id"] in frontier:
                            relation_ids.add(edge["relation_id"])
                            following.add(edge["target"]["revision_id"])
                            following.update(edge["context_revision_ids"])
                    frontier = following - selected
                    selected.update(following)
                    if len(selected) > 64 or len(relation_ids) > 64:
                        raise ValidityCapacity("lookup expansion capacity reached")
                support = {r["revision_id"] for r in projection["support"]}
                _guard(claims, self.clock, cancel_event)
                return {"status": "exact_registered_lookup", "revisions": [
                    {**index[k], "use": "support" if k in support else "historical_citation_only",
                     "not_authorized_as_premise": k not in support} for k in sorted(selected)],
                    "relations": [e for e in value["dto"]["relations"] if e["relation_id"] in relation_ids],
                    "current": self._current(state, use_id, overlay), "retrieval_kind": "exact_id_not_embedding"}

    def status(self, token: str, *, run_id: str, cancel_event=None):
        with self._locked(token, run_id, "validity.status", cancel_event) as (_, state, claims, use_id):
            self._admit(state, "read." + digest(use_id), 64, claims=claims, cancel_event=cancel_event)
            with self._gate(state):
                _, _, overlay = self._use(state, use_id)
                _guard(claims, self.clock, cancel_event)
                return {"status": "current_status", "current": self._current(state, use_id, overlay),
                    "accepted_probes": sorted(k for k, p in state["probes"].items() if p["use_id"] == use_id and p["status"] == "accepted"),
                    "historical_publications": [p for p in state["publications"].values() if p["use_id"] == use_id]}

    def probe(self, token: str, request: dict, *, run_id: str, cancel_event=None):
        from .worker import SandboxWorker, WorkerUnavailable
        request = protocol(request, "probe")
        driver = files("rethlas_core").joinpath("validity_fixture_driver.py").read_bytes()
        driver_sha = hashlib.sha256(driver).hexdigest()
        with self._locked(token, run_id, "validity.probe", cancel_event) as (_, state, claims, use_id):
            key, binding = self._binding(state, "probe", request, self._identity(claims, use_id), claims=claims, cancel_event=cancel_event)
            self._terminal(binding)
            if binding["status"] == "accepted":
                self._admit(state, "read." + digest(use_id), 64, claims=claims, cancel_event=cancel_event)
                with self._gate(state):
                    _, _, overlay = self._use(state, use_id)
                    _guard(claims, self.clock, cancel_event)
                    return {"status": "synthetic_input_observed", "probe": state["probes"][binding["result"]["probe_id"]],
                            "replayed": True, "current": self._current(state, use_id, overlay)}
            if binding["status"] == "started":
                raise ValidityRecoveryRequired("started fixture is historical; a new request must dispatch")
            self._admit(state, "probe." + digest(use_id), 16, claims=claims, cancel_event=cancel_event)
            with self._gate(state):
                value, _, overlay = self._use(state, use_id)
                self._eligible(overlay)
                probe_id = digest({"service_id": self.config["service_id"], "request_key": key})
                started = {"status": "started", "probe_id": probe_id, "use_id": use_id,
                    "dto_sha256": digest(value["dto"]), "descriptor_sha256": digest(value["descriptor"]),
                    "overlay_sha256": digest(overlay), "driver_sha256": driver_sha, "sandbox_policy_sha256": digest(SANDBOX_POLICY),
                    "observed_at": self._checked_at(state)}
                def start(n):
                    n["requests"][key].update(status="started", result={"probe_id": probe_id})
                    n["probes"][probe_id] = started
                self._change(state, "PROBE_STARTED", start, key=key, claims=claims, cancel_event=cancel_event)
        with tempfile.TemporaryDirectory(prefix="rethlas-validity-") as temporary:
            directory = PrivateDirectory(Path(temporary))
            input_dir, work_dir = directory.child("input", create=True), directory.child("work", create=True)
            for name, obj in (("context.json", value["dto"]), ("use.json", value["descriptor"]), ("overlay.json", overlay)):
                input_dir.write_bytes(name, canonical_json(obj))
            input_dir.write_bytes("driver.py", driver)
            with self._locked(token, run_id, "validity.probe", cancel_event) as (_, state, claims, resource):
                with self._gate(state):
                    _, _, current_overlay = self._use(state, resource)
                    self._eligible(current_overlay)
                    if digest(current_overlay) != started["overlay_sha256"]:
                        raise ValidityConflict("fixture overlay changed before dispatch")
                    _guard(claims, self.clock, cancel_event)
            self._checkpoint("before_probe_dispatch")
            output = SandboxWorker().run_bounded(input_dir.root, work_dir.root, SANDBOX_POLICY["command"], cancel_event=cancel_event)
        expected = {"version": "validity-observation-1", "input_files": ["context.json", "driver.py", "overlay.json", "use.json"],
            "dto_sha256": digest(value["dto"]), "descriptor_sha256": digest(value["descriptor"]), "overlay_sha256": digest(overlay),
            "driver_sha256": driver_sha, "revisions": [{**{k: r[k] for k in ("entity_id", "revision_id", "type", "body_sha256")},
            "body_bytes": len(r["body"].encode("utf-8"))} for r in value["dto"]["revisions"]],
            "relations": len(value["dto"]["relations"]), "empty_work": True, "input_readonly": True, "mathematical_review": False}
        if strict_json_loads(output.stdout) != expected or output.stderr or output.returncode:
            raise WorkerUnavailable("actual child input observation differs")
        self._checkpoint("before_probe_accept")
        with self._locked(token, run_id, "validity.probe", cancel_event) as (_, state, claims, resource):
            with self._gate(state):
                _, _, current_overlay = self._use(state, resource)
                self._eligible(current_overlay)
                if digest(current_overlay) != started["overlay_sha256"]:
                    raise ValidityConflict("relevant citation observation changed; dispatch a new request")
                accepted = {**started, "status": "accepted", "observation": expected,
                            "accepted_at": self._checked_at(state), "mathematical_review": False}
                def accept(n):
                    n["probes"][probe_id] = accepted
                    n["requests"][key].update(status="accepted", result={"probe_id": probe_id})
                self._change(state, "PROBE_ACCEPTED", accept, key=key, claims=claims, cancel_event=cancel_event)
                return {"status": "synthetic_input_observed", "probe": accepted, "replayed": False,
                        "current": self._current(state, resource, current_overlay)}

    def publish(self, token: str, request: dict, *, run_id: str, cancel_event=None):
        request = protocol(request, "publish")
        with self._locked(token, run_id, "validity.publish", cancel_event) as (_, state, claims, use_id):
            key, binding = self._binding(state, "publish", request, self._identity(claims, use_id), claims=claims, cancel_event=cancel_event)
            self._terminal(binding)
            if binding["status"] == "published":
                self._admit(state, "read." + digest(use_id), 64, claims=claims, cancel_event=cancel_event)
                with self._gate(state):
                    _, _, overlay = self._use(state, use_id)
                    _guard(claims, self.clock, cancel_event)
                    return {"status": "historical_fixture_publication", "receipt": state["publications"][binding["result"]["publication_id"]],
                            "replayed": True, "current": self._current(state, use_id, overlay)}
            self._admit(state, "publish." + digest(use_id), 16, claims=claims, cancel_event=cancel_event)
            with self._gate(state):
                value, _, overlay = self._use(state, use_id)
                self._eligible(overlay)
                probe = state["probes"].get(request["probe_id"])
                if (probe is None or probe["status"] != "accepted" or probe["use_id"] != use_id
                        or probe["dto_sha256"] != digest(value["dto"]) or probe["descriptor_sha256"] != digest(value["descriptor"])
                        or probe["overlay_sha256"] != digest(overlay)):
                    raise ValidityConflict("publication requires a matching current service observation")
                receipt = protocol({"version": "validity-fixture-receipt-1", "publication_id": digest({"service": self.config["service_id"], "request_key": key}),
                    "probe_id": request["probe_id"], "use_id": use_id, "dto_sha256": probe["dto_sha256"],
                    "descriptor_sha256": probe["descriptor_sha256"], "overlay_sha256": probe["overlay_sha256"],
                    "published_at": self._checked_at(state), "mathematical_review": False, "production_completed": False}, "fixture_receipt")
                def publish(n):
                    n["publications"][receipt["publication_id"]] = receipt
                    n["requests"][key].update(status="published", result={"publication_id": receipt["publication_id"]})
                self._change(state, "FIXTURE_PUBLISHED", publish, key=key, claims=claims, cancel_event=cancel_event)
                return {"status": "fixture_published", "receipt": receipt, "replayed": False,
                        "current": self._current(state, use_id, overlay)}

    def recover(self, *, kind: str, apply: bool = False, cancel: bool = False,
                request: dict | None = None, token: str | None = None, run_id: str | None = None):
        if kind not in ("control", "graph") or type(apply) is not bool or type(cancel) is not bool:
            raise ValidityError("invalid explicit recovery operation")
        if cancel and not apply:
            raise ValidityError("recovery cancellation requires explicit apply")
        with self.authority._registry() as registry:
            with self.directory.lock():
                self._inventory()
                if self.directory.read_bytes("service.json", max_bytes=META_BYTES) != canonical_json(self.config):
                    raise ValidityError("service configuration changed")
                state, _, transition, phase = self.journal.inspect()
                if state["counts"].get("operator", 0) >= 64 and phase != "pending":
                    raise ValidityCapacity("operator recovery budget exhausted")
                if kind == "control":
                    if request is not None:
                        raise ValidityError("control recovery uses its exact persisted transition")
                    result = {"status": phase, "applied": apply, "cancelled": cancel,
                              "kind": kind, "sequence": state["sequence"]}
                    if not apply:
                        if phase == "stable":
                            self._admit(state, "operator", 64, reserved=True)
                        return result
                    if phase == "stable":
                        self._admit(state, "operator", 64, reserved=True)
                        return {**result, "status": "stable", "sequence": state["sequence"]}
                    if phase == "prepared" and cancel and transition["kind"] in ("GRAPH_ACCEPT", "GRAPH_ABORT"):
                        raise ValidityConflict("historical graph reconciliation cannot be cancelled")
                    if phase == "prepared":
                        transition = self.journal.reserve_prepared_recovery(mint=not cancel and transition["kind"] in
                            ("USE_REGISTERED", "WITHDRAW", "GRAPH_PREPARE", "GRAPH_ACCEPT", "PROBE_STARTED", "PROBE_ACCEPTED", "FIXTURE_PUBLISHED"))
                    if phase == "prepared" and not cancel and transition["kind"] not in ("GRAPH_ACCEPT", "GRAPH_ABORT"):
                        claims = transition["authorization"]
                        if claims is not None:
                            if token is None or run_id != claims["run_id"]:
                                raise AuthorizationError("prepared control recovery requires original live grant")
                            current = self.authority._check_locked(registry, token, owner=self.owner, run_id=run_id,
                                domain=self.DOMAIN, action=claims["actions"][0], now=self.clock())
                            identity = ("owner", "run_id", "domain", "role", "epoch")
                            if any(current[k] != claims[k] for k in identity):
                                raise AuthorizationError("prepared control recovery identity differs")
                            _guard(claims, self.clock)
                    if phase == "prepared" and not cancel:
                        previous_sequence = state["sequence"]
                        try:
                            with self._prepared_condition(state, transition):
                                historical = transition["kind"] in ("GRAPH_ACCEPT", "GRAPH_ABORT")
                                state = self.journal.recover(apply=True, guard=lambda: _guard(None if historical else transition["authorization"], self.clock))
                        except (AuthorizationError, MemoryCancelled) as exc:
                            if state["sequence"] != previous_sequence:
                                raise ValidityIndeterminate("recovery accepted before checkpoint return guard failed") from exc
                            raise
                    else:
                        state = self.journal.recover(apply=True, cancel=cancel,
                            guard=lambda: _guard(transition["authorization"] if transition and not cancel else None, self.clock))
                    return {**result, "status": "recovered", "sequence": state["sequence"]}
                if phase != "stable":
                    raise ValidityRecoveryRequired("recover the exact control transition before graph recovery")
                self._admit(state, "operator", 64, reserved=True)
                pending = state["pending_graph"]
                if pending is None:
                    return {"status": "no_pending_graph", "applied": False}
                key = pending["request_key"]
                binding = state["requests"][key]
                operation = self._operation(binding)
                if request is None or canonical_json(request) != canonical_json(operation["request"]):
                    raise ValidityConflict("graph recovery requires the exact original request")
                store, internal = self._internal(state, maintenance=True)
                exported = store.export(internal, run_id=INTERNAL_RUN)
                if exported["head"] == pending["parent"]["head"]:
                    if not apply:
                        return {"status": "graph_prepared", "applied": False, "head": exported["head"]}
                    # Explicitly consume only the original store pending journal;
                    # this never rolls back a committed head or frees receipt IDs.
                    store.recover(internal, run_id=INTERNAL_RUN, apply=True)
                    if cancel:
                        def abort(n):
                            n["pending_graph"] = None
                            n["requests"][key].update(status="aborted", result={"status": "aborted"})
                        with self._owned_checkpoint(state, store, internal) as checkpoint:
                            if _floor(checkpoint) != pending["parent"]:
                                raise ValidityError("cannot abort a graph that has committed")
                            self._change(state, "GRAPH_ABORT", abort, key=key, reserved=True)
                        return {"status": "aborted", "applied": True}
                    if token is None or run_id is None:
                        raise AuthorizationError("continuing graph write requires live original identity")
                    claims = self.authority._check_locked(registry, token, owner=self.owner, run_id=run_id,
                        domain=self.DOMAIN, action=self.COMMIT_ACTION, now=self.clock())
                    if self._identity(claims, "owned-store") != binding["identity"]:
                        raise AuthorizationError("graph recovery writer differs")
                    result = store.commit(internal, request, run_id=INTERNAL_RUN,
                                          outer_guard=lambda: _guard(claims, self.clock))
                    return self._accept_graph(state, key, store, internal, result, claims=claims)
                receipt = store._receipt(digest({"store_id": store.config["store_id"], "request_id": request["request_id"]}))
                expected = _result_manifest(receipt)
                if exported["head"] != expected["head"] or _floor(receipt["parent_manifest"]) != pending["parent"]:
                    raise ValidityError("graph recovery head is not the original fenced result")
                if cancel:
                    raise ValidityConflict("committed graph cannot be cancelled")
                if not apply:
                    return {"status": "graph_committed_control_pending", "applied": False, "head": expected["head"]}
                # Original immutable receipt and full chain were checked by export;
                # confirm original request binding with the store's exact replay.
                result = store.commit(internal, request, run_id=INTERNAL_RUN)
                store.recover(internal, run_id=INTERNAL_RUN, apply=True)
                return self._accept_graph(state, key, store, internal, result)

    @contextmanager
    def _prepared_condition(self, state, transition):
        kind = transition["kind"]
        # A prepared graph-accept is historical reconciliation. Others that can
        # authorize current bytes must recheck the current owned floor/use.
        if kind == "GRAPH_ACCEPT":
            pending = state["pending_graph"]
            key = transition["request_key"]
            if pending is None or pending["request_key"] != key:
                raise ValidityError("prepared graph acceptance lacks its fence")
            binding = state["requests"][key]
            operation = self._operation(binding)
            store = self._store()
            token = store.issue(run_id=INTERNAL_RUN, actions=list(ROLE_ACTIONS["memory"]), ttl_seconds=30)
            receipt = store._receipt(digest({"store_id": store.config["store_id"], "request_id": operation["request"]["request_id"]}))
            expected = _result_manifest(receipt)
            exported = store.export(token, run_id=INTERNAL_RUN)
            if exported["head"] != expected["head"]:
                raise ValidityError("historical graph acceptance cannot recreate a missing commit")
            result = store.commit(token, operation["request"], run_id=INTERNAL_RUN)
            if _floor(result["receipt"]["parent_manifest"]) != pending["parent"]:
                raise ValidityError("prepared graph parent differs")
            if transition["next_state"]["accepted_graph"] != _floor(expected):
                raise ValidityError("prepared graph acceptance target differs")
            with store.locked_checkpoint(token, run_id=INTERNAL_RUN) as checkpoint:
                if _floor(checkpoint) != _floor(expected):
                    raise ValidityError("prepared graph acceptance disk checkpoint differs")
                yield
        elif kind in ("USE_REGISTERED", "WITHDRAW", "GRAPH_PREPARE", "PROBE_STARTED", "PROBE_ACCEPTED", "FIXTURE_PUBLISHED"):
            store = self._store()
            token = store.issue(run_id=INTERNAL_RUN, actions=["memory.read"], ttl_seconds=30)
            with store.locked_checkpoint(token, run_id=INTERNAL_RUN) as checkpoint:
                if state["pending_graph"] is not None or _floor(checkpoint) != state["accepted_graph"]:
                    raise ValidityError("prepared current-use conditions no longer hold")
                if kind in ("PROBE_STARTED", "PROBE_ACCEPTED", "FIXTURE_PUBLISHED"):
                    key = transition["request_key"]
                    binding = transition["next_state"]["requests"][key]
                    use_id = binding["identity"]["resource"]
                    _, _, overlay = self._use(state, use_id)
                    self._eligible(overlay)
                    record_id = binding["result"].get("probe_id") or binding["result"].get("publication_id")
                    record = transition["next_state"]["probes" if kind != "FIXTURE_PUBLISHED" else "publications"][record_id]
                    if record["overlay_sha256"] != digest(overlay):
                        raise ValidityConflict("prepared fixture observation changed")
                yield
        else:
            yield

    def audit(self):
        with self._operator() as (_, state):
            self._admit(state, "operator", 64, reserved=True)
            with self._gate(state, maintenance=True):
                pass
            store, internal = self._internal(state, maintenance=True)
            exported = store.export(internal, run_id=INTERNAL_RUN)
            withdrawals = self._withdrawals(state)
            return {"status": "operator_historical_audit", "graph": exported, "control": deepcopy(state),
                    "validity": validity_states(exported["graph"], withdrawals), "mathematical_review": False}

    def _page(self, state, entries, cursor, *, previous_sequence):
        offset = 0
        if cursor is not None:
            if type(cursor) is not str or len(cursor) > META_BYTES:
                raise ValidityError("invalid queue cursor")
            try:
                import base64
                decoded = strict_json_loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
            except (ValueError, UnicodeError) as exc:
                raise ValidityError("invalid queue cursor") from exc
            _fields(decoded, {"control_sequence", "graph_head", "offset", "entries_sha256"})
            if (decoded["control_sequence"] != previous_sequence or decoded["graph_head"] != state["accepted_graph"]["head"]
                    or decoded["entries_sha256"] != digest(entries) or type(decoded["offset"]) is not int):
                raise ValidityConflict("queue cursor is stale")
            offset = decoded["offset"]
            if not 0 <= offset < len(entries):
                raise ValidityError("queue cursor offset is out of bounds")
        following = offset + 128
        next_cursor = None
        if following < len(entries):
            import base64
            value = {"control_sequence": state["sequence"], "graph_head": state["accepted_graph"]["head"],
                     "offset": following, "entries_sha256": digest(entries)}
            next_cursor = base64.urlsafe_b64encode(canonical_json(value)).decode("ascii")
        return {"entries": entries[offset:following], "total": len(entries), "next_cursor": next_cursor,
                "checked_at": self._checked_at(state)}

    def events(self, *, cursor=None):
        with self._operator() as (_, state):
            previous_sequence = state["sequence"]
            entries = [deepcopy(e) for e in state["events"] if e["kind"] != "ADMISSION"]
            self._admit(state, "operator", 64, reserved=True)
            return self._page(state, entries, cursor, previous_sequence=previous_sequence)

    def review_queue(self, *, cursor=None):
        with self._operator() as (_, state):
            previous_sequence = state["sequence"]
            self._admit(state, "operator", 64, reserved=True)
            with self._gate(state, maintenance=True):
                pass
            store, internal = self._internal(state, maintenance=True)
            exported = store.export(internal, run_id=INTERNAL_RUN)
            projections = {key: self._use(state, key)[1] for key in state["uses"]}
            entries = review_queue(exported["graph"], self._withdrawals(state), projections)
            return self._page(state, entries, cursor, previous_sequence=previous_sequence)
