"""Single-slot durable control transitions; a graph snapshot is never this journal.

The caller holds service authority -> control lock. A pending anchor accepts one
exact next state; recovery only moves forward. Prepared transitions retain request
bindings even when cancelled. This is not a hardware anti-rollback primitive.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Callable

from .authorization import AuthorizationError
from .memory_store import MemoryCancelled
from .private_fs import PrivateDirectory, canonical_json, digest
from .validity import (
    VERSION, ValidityError, ValidityCapacity, ValidityConflict,
    ValidityRecoveryRequired, ValidityIndeterminate,
)
from .validity_protocol import STATE_BYTES, TRANSITION_BYTES, META_BYTES, protocol


def stable_anchor(config: dict, state: dict) -> dict:
    return {"version": config["version"], "service_id": config["service_id"], "config_sha256": digest(config),
            "phase": "stable", "sequence": state["sequence"], "state_sha256": digest(state),
            "transition_sha256": None, "next_sequence": None, "next_sha256": None}


class ControlJournal:
    def __init__(self, directory: PrivateDirectory, config: dict, *, reserve: Callable,
                 checkpoint: Callable, protocol: Callable = protocol, version: str = VERSION):
        if config["version"] != version:
            raise ValidityError("control protocol version differs")
        self.directory = directory
        self.config = config
        self.reserve = reserve
        self.checkpoint = checkpoint
        self.protocol = protocol
        self.version = version

    def _object(self, name: str, kind: str, maximum: int) -> dict:
        raw = self.directory.read_bytes(name, max_bytes=maximum)
        from .private_fs import strict_json_loads
        value = self.protocol(strict_json_loads(raw), kind, maximum)
        if raw != canonical_json(value):
            raise ValidityError("control object is not canonical")
        if value["service_id"] != self.config["service_id"] or value["config_sha256"] != digest(self.config):
            raise ValidityError("control identity differs")
        return value

    def _sync(self, name: str) -> None:
        try:
            self.directory.sync_file(name)
        except OSError as exc:
            raise ValidityIndeterminate("control durability barrier failed") from exc

    def _write(self, name: str, value: dict, *, maximum: int, guard=None, reserved=False) -> None:
        raw = canonical_json(value)
        if len(raw) > maximum:
            raise ValidityCapacity("control object exceeds capacity")
        self.reserve(len(raw), reserved=reserved)
        try:
            self.directory.write_bytes(name, raw, before_replace=guard)
        except (AuthorizationError, MemoryCancelled):
            # PermissionError is an OSError subclass. A trusted final guard runs
            # before replace, so its denial is not an uncertain filesystem write.
            raise
        except OSError as exc:
            raise ValidityIndeterminate("control transition durability is uncertain") from exc

    def inspect(self) -> tuple[dict, dict, dict | None, str]:
        state = self._object("state.json", "state", STATE_BYTES)
        anchor = self._object("anchor.json", "anchor", META_BYTES)
        if len(state["events"]) != state["sequence"]:
            raise ValidityError("control sequence differs from events")
        if [e["sequence"] for e in state["events"]] != list(range(1, state["sequence"] + 1)):
            raise ValidityError("control event sequence differs")
        try:
            transition = self._object("transition.json", "transition", TRANSITION_BYTES)
        except FileNotFoundError:
            transition = None
        if transition is not None:
            next_state = transition["next_state"]
            if (transition["next_sequence"] != transition["prev_sequence"] + 1
                    or transition["next_sequence"] != next_state["sequence"]
                    or transition["next_sha256"] != digest(next_state)
                    or next_state["service_id"] != self.config["service_id"]
                    or next_state["config_sha256"] != digest(self.config)):
                raise ValidityError("control transition target differs")
        if anchor["phase"] == "stable":
            if anchor != stable_anchor(self.config, state):
                raise ValidityError("control state and stable anchor differ")
            if transition is None:
                if state["sequence"] != 0:
                    raise ValidityError("completed transition is missing")
                phase = "stable"
            elif (transition["next_sequence"], transition["next_sha256"]) == (state["sequence"], digest(state)):
                phase = "stable"
            elif (transition["prev_sequence"], transition["prev_sha256"]) == (state["sequence"], digest(state)):
                self._validate_next(state, transition["next_state"])
                phase = "prepared"
            else:
                raise ValidityError("control transition is unrelated to stable anchor")
        else:
            if transition is None or anchor != self._pending_anchor(transition):
                raise ValidityError("pending anchor lacks its exact transition")
            point = state["sequence"], digest(state)
            previous = transition["prev_sequence"], transition["prev_sha256"]
            following = transition["next_sequence"], transition["next_sha256"]
            if point not in (previous, following):
                raise ValidityError("pending state is neither exact previous nor next")
            if point == previous:
                self._validate_next(state, transition["next_state"])
            phase = "pending"
        for name in ("state.json", "anchor.json"):
            self._sync(name)
        if transition is not None:
            self._sync("transition.json")
        return state, anchor, transition, phase

    def ready(self) -> dict:
        state, _, _, phase = self.inspect()
        if phase != "stable":
            raise ValidityRecoveryRequired("control transition requires explicit operator recovery")
        return state

    @staticmethod
    def _validate_next(previous: dict, following: dict) -> None:
        if (following["sequence"] != previous["sequence"] + 1
                or following["events"][:-1] != previous["events"]
                or following["withdrawal_version"] < previous["withdrawal_version"]):
            raise ValidityError("control history cannot regress")
        for key, value in previous["counts"].items():
            if following["counts"].get(key, -1) < value:
                raise ValidityError("control counters cannot regress")
        for name in ("uses", "runs", "publications"):
            if any(following[name].get(k) != v for k, v in previous[name].items()):
                raise ValidityError("immutable control record changed")
        for key, binding in previous["requests"].items():
            other = following["requests"].get(key)
            if other is None or any(other[k] != binding[k] for k in
                                    ("request_id", "operation_sha256", "identity", "kind", "bucket")):
                raise ValidityConflict("permanent control request binding differs")
            if binding["status"] not in ("bound", "started") and other != binding:
                raise ValidityConflict("terminal control request changed")
        for key, old in previous["probes"].items():
            new = following["probes"].get(key)
            if new is None or old.get("status") == "accepted" and new != old:
                raise ValidityConflict("accepted fixture observation changed")

    def _next(self, state: dict, kind: str, mutate: Callable, request_key: str | None, reserved: bool) -> dict:
        if state["sequence"] >= 512 or not reserved and state["counts"].get("ordinary.events", 0) >= 384:
            raise ValidityCapacity("control event capacity reached")
        result = deepcopy(state)
        mutate(result)
        result["sequence"] = state["sequence"] + 1
        if not reserved:
            result["counts"]["ordinary.events"] = state["counts"].get("ordinary.events", 0) + 1
        result["events"].append({"event_id": digest({"service_id": self.config["service_id"],
            "previous": digest(state), "sequence": result["sequence"], "kind": kind, "request_key": request_key}),
            "sequence": result["sequence"], "kind": kind, "request_key": request_key})
        result = self.protocol(result, "state", STATE_BYTES)
        self._validate_next(state, result)
        return result

    def _transition(self, previous: dict, following: dict, kind: str,
                    request_key: str | None, claims: dict | None) -> dict:
        return self.protocol({"version": self.version, "service_id": self.config["service_id"],
            "config_sha256": digest(self.config), "prev_sequence": previous["sequence"],
            "prev_sha256": digest(previous), "next_sequence": following["sequence"],
            "next_sha256": digest(following), "kind": kind, "request_key": request_key,
            "authorization": claims, "next_state": following}, "transition", TRANSITION_BYTES)

    def _pending_anchor(self, transition: dict) -> dict:
        return {"version": self.version, "service_id": self.config["service_id"],
            "config_sha256": digest(self.config), "phase": "pending",
            "sequence": transition["prev_sequence"], "state_sha256": transition["prev_sha256"],
            "transition_sha256": digest(transition), "next_sequence": transition["next_sequence"],
            "next_sha256": transition["next_sha256"]}

    def _finish(self, transition: dict, *, prepared: bool, guard=None, reserved=False) -> dict:
        accepted = not prepared
        try:
            if prepared:
                self.checkpoint("before_anchor_pending")
                self._write("anchor.json", self._pending_anchor(transition), maximum=META_BYTES,
                            guard=guard, reserved=reserved)
                accepted = True
                self.checkpoint("after_anchor_pending")
            self._write("state.json", transition["next_state"], maximum=STATE_BYTES, reserved=reserved)
            self.checkpoint("after_control_state")
            self._write("anchor.json", stable_anchor(self.config, transition["next_state"]),
                        maximum=META_BYTES, reserved=reserved)
            self.checkpoint("after_anchor_stable")
        except BaseException as exc:
            if accepted and isinstance(exc, Exception) and not isinstance(exc, ValidityIndeterminate):
                raise ValidityIndeterminate("accepted control transition needs exact recovery") from exc
            raise
        return transition["next_state"]

    def change(self, state: dict, kind: str, mutate: Callable, *, request_key=None,
               claims=None, guard=None, reserved=False) -> dict:
        if self.ready() != state:
            raise ValidityConflict("control checkpoint changed")
        following = self._next(state, kind, mutate, request_key, reserved)
        transition = self._transition(state, following, kind, request_key, claims)
        self._write("transition.json", transition, maximum=TRANSITION_BYTES, reserved=reserved)
        self.checkpoint("after_control_prepare")
        return self._finish(transition, prepared=True, guard=guard, reserved=reserved)

    def reserve_prepared_recovery(self, *, mint=False) -> dict:
        state, _, transition, phase = self.inspect()
        if phase != "prepared":
            raise ValidityRecoveryRequired("only an unaccepted prepared transition can reserve recovery")
        following = deepcopy(transition["next_state"])
        budgets = [("operator", 64)]
        if mint:
            budgets.append(("internal.reserve", 64))
        for name, maximum in budgets:
            used = max(state["counts"].get(name, 0), following["counts"].get(name, 0))
            if used >= maximum:
                raise ValidityCapacity("prepared recovery budget exhausted")
            following["counts"][name] = used + 1
        following = self.protocol(following, "state", STATE_BYTES)
        self._validate_next(state, following)
        transition = self._transition(state, following, transition["kind"], transition["request_key"], transition["authorization"])
        self._write("transition.json", transition, maximum=TRANSITION_BYTES, reserved=True)
        return transition

    def recover(self, *, apply: bool, cancel: bool = False, guard=None) -> dict:
        state, _, transition, phase = self.inspect()
        if phase == "stable":
            return state
        if not apply:
            raise ValidityRecoveryRequired("control transition is not stable")
        if phase == "pending":
            if cancel:
                raise ValidityConflict("accepted pending transition cannot be cancelled")
            return self._finish(transition, prepared=False, reserved=True)
        if cancel:
            if transition["kind"] in ("GRAPH_ACCEPT", "GRAPH_ABORT"):
                raise ValidityConflict("historical graph reconciliation can only finish forward")
            key = transition["request_key"]
            prepared_binding = transition["next_state"]["requests"].get(key)

            def terminate(next_state):
                if prepared_binding is not None:
                    binding = deepcopy(prepared_binding)
                    binding.update(status="aborted", result={"status": "aborted"})
                    old = next_state["requests"].get(key)
                    if old is not None and old["status"] not in ("bound", "started"):
                        raise ValidityConflict("cannot cancel a terminal historical operation")
                    next_state["requests"][key] = binding
                # Retain the original admission in either crash outcome.
                for name, value in transition["next_state"]["counts"].items():
                    next_state["counts"][name] = max(next_state["counts"].get(name, 0), value)
                next_state["counts"]["operator"] = max(next_state["counts"].get("operator", 0), 1)
                if state["pending_graph"] is not None:
                    # Control cancellation must not clear a separate graph fence.
                    next_state["pending_graph"] = state["pending_graph"]

            following = self._next(state, "PREPARED_ABORT", terminate, key, True)
            transition = self._transition(state, following, "PREPARED_ABORT", key, None)
            # One atomic replacement preserves either original prepared intent or
            # its permanently terminated binding; never unlink the old evidence.
            self._write("transition.json", transition, maximum=TRANSITION_BYTES, reserved=True)
            self.checkpoint("after_prepared_abort")
        return self._finish(transition, prepared=True, guard=guard, reserved=True)
