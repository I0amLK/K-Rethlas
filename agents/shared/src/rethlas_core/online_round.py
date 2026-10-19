"""One explicitly approved online round, not a general billing service.

Admissions live in the existing control journal before HTTP dispatch. Exact text
prompts may be approved in stages by the operator, without resetting the round.
Failures and uncertain sends are never refunded or automatically repeated.
"""
from __future__ import annotations

import hashlib
from copy import deepcopy

from .private_fs import canonical_json, digest, strict_json_loads
from .validity import ValidityError, ValidityCapacity, ValidityConflict
from .validity_protocol import META_BYTES


LIMITS = {"embedding_requests": 10, "documents": 200, "queries": 20,
          "embedding_bytes": 65536, "text_requests": 4, "text_input_bytes": 16384,
          "output_tokens": 4096}


def _prompt(messages, purpose):
    if purpose not in ("generation", "review") or type(messages) is not list or not 1 <= len(messages) <= 64:
        raise ValidityError("unknown or empty online text request")
    for message in messages:
        if (type(message) is not dict or set(message) != {"role", "content"}
                or message["role"] not in ("system", "user", "assistant")
                or type(message["content"]) is not str or not message["content"].strip()):
            raise ValidityError("invalid approved text message")
    if not any(m["role"] == "user" for m in messages):
        raise ValidityError("an explicit user message is required")
    raw = canonical_json(messages)
    if len(raw) > LIMITS["text_input_bytes"]:
        raise ValidityCapacity("approved text input limit exceeded")
    return {"purpose": purpose, "messages": strict_json_loads(raw)}


class OnlineRound:
    """Approval is a trusted operator action, never an exposed model tool."""

    def __init__(self, service):
        self.service = service
        with service._operator():
            raw = service.parts["research"].read_bytes("online-round.json", max_bytes=META_BYTES)
            self.config = strict_json_loads(raw)
            if (type(self.config) is not dict
                    or set(self.config) != {"version", "service_id", "limits", "allowed_sha256", "materials"}
                    or self.config["version"] != "online-round-1"
                    or self.config["service_id"] != service.config["service_id"]
                    or self.config["limits"] != LIMITS
                    or self.config["materials"] != "public_or_synthetic"
                    or raw != canonical_json(self.config)):
                raise ValidityError("online round approval differs")
            self._hashes(self.config["allowed_sha256"])
        self.allowed = frozenset(self.config["allowed_sha256"])

    @staticmethod
    def _hashes(hashes):
        import re
        if (type(hashes) is not list or not hashes or len(hashes) > 512
                or any(type(h) is not str or not re.fullmatch("[0-9a-f]{64}", h) for h in hashes)
                or len(set(hashes)) != len(hashes)):
            raise ValidityError("online material approval requires exact input hashes")

    @classmethod
    def initialize(cls, service, *, allowed_sha256, materials):
        cls._hashes(allowed_sha256)
        if materials != "public_or_synthetic":
            raise ValidityError("this approved round excludes private material")
        config = {"version": "online-round-1", "service_id": service.config["service_id"],
                  "limits": deepcopy(LIMITS), "allowed_sha256": sorted(allowed_sha256), "materials": materials}
        with service._operator() as (_, state):
            directory = service.parts["research"]
            try:
                old = directory.read_bytes("online-round.json", max_bytes=META_BYTES)
            except FileNotFoundError:
                if any(k.startswith("online.") for k in state["counts"]):
                    raise ValidityError("cannot recreate missing approval after online admission")
                service._immutable(directory, "online-round.json", config, META_BYTES)
            else:
                if old != canonical_json(config):
                    raise ValidityConflict("online round is immutable; no automatic reset or extension")
        return cls(service)

    @classmethod
    def approve_text(cls, service, *, messages, purpose, materials, provider=None):
        """Approve exact outgoing bytes after inspection, including a fresh review.

        This appends an approval only. It neither increases the four-request cap
        nor sends a request, and is intentionally unavailable through MCP.
        """
        approved = cls(service)
        if materials != "public_or_synthetic":
            raise ValidityError("text approval excludes private material")
        request = _prompt(messages, purpose)
        value = approved._approval_value(request, provider)
        with service._operator():
            service._immutable(service.parts["research"], digest(value) + ".json", value, META_BYTES)
        return {"status": "exact_text_approved", "purpose": purpose,
                "input_sha256": digest(request), "limits_unchanged": True, "request_sent": False}

    def _check(self, texts):
        if type(texts) is not list or not texts or any(type(t) is not str or not t for t in texts):
            raise ValidityError("online inputs must be explicit text")
        if any(hashlib.sha256(t.encode("utf-8")).hexdigest() not in self.allowed for t in texts):
            raise ValidityError("online input is not in this round's approved public material")
        return sum(len(t.encode("utf-8")) for t in texts)

    def _approval_value(self, request, provider):
        value = {"version": "online-text-approval-1", "service_id": self.service.config["service_id"],
                 "round_sha256": digest(self.config), "materials": "public_or_synthetic", **request}
        if type(provider) is dict and "version" in provider:
            from .codex_binding import validate_codex_binding
            value.update(version="online-text-approval-3", provider=validate_codex_binding(provider))
        elif provider is not None:
            if (type(provider) is not dict or set(provider) != {"provider_type", "mode", "model", "configuration_sha256"}
                    or provider["provider_type"] not in ("openai_chat", "codex_cli")
                    or provider["mode"] not in ("live", "mock") or type(provider["model"]) is not str
                    or not provider["model"] or len(provider["model"].encode("utf-8")) > 256):
                raise ValidityError("invalid exact provider approval binding")
            self._hashes([provider["configuration_sha256"]])
            value.update(version="online-text-approval-2", provider=deepcopy(provider))
        return value

    def text_approved(self, messages, purpose, *, provider=None):
        request = _prompt(messages, purpose)
        if provider is None and all(hashlib.sha256(m["content"].encode("utf-8")).hexdigest() in self.allowed for m in messages):
            return True
        value = self._approval_value(request, provider)
        with self.service._operator():
            try:
                raw = self.service.parts["research"].read_bytes(digest(value) + ".json", max_bytes=META_BYTES)
            except FileNotFoundError:
                return False
            if raw != canonical_json(value):
                raise ValidityError("exact text approval differs")
            return True

    def _charge(self, amounts, maxima, kind):
        with self.service._operator() as (_, state):
            for key, amount in amounts.items():
                if state["counts"].get(key, 0) + amount > maxima[key]:
                    raise ValidityCapacity("approved online round exhausted")
            def admit(following):
                for key, amount in amounts.items():
                    following["counts"][key] = following["counts"].get(key, 0) + amount
            self.service._change(state, "EGRESS_STARTED", admit, reserved=True)
            return {"sequence": state["sequence"], "purpose": kind, "amounts_sha256": digest(amounts)}

    def admit_embedding(self, texts, kind):
        if kind not in ("documents", "queries"):
            raise ValidityError("unknown online embedding purpose")
        size = self._check(texts)
        key = "online." + kind
        amounts = {"online.embedding_requests": 1, key: len(texts), "online.embedding_bytes": size}
        maxima = {"online.embedding_requests": LIMITS["embedding_requests"], key: LIMITS[kind],
                  "online.embedding_bytes": LIMITS["embedding_bytes"]}
        return self._charge(amounts, maxima, kind)

    def admit_text(self, messages, purpose, *, provider=None):
        if not self.text_approved(messages, purpose, provider=provider):
            raise ValidityError("exact text prompt requires explicit operator approval")
        return self._charge({"online.text_requests": 1}, {"online.text_requests": LIMITS["text_requests"]}, purpose)

    def usage(self):
        with self.service._operator() as (_, state):
            return {"limits": deepcopy(LIMITS), "used": {k[7:]: n for k, n in state["counts"].items() if k.startswith("online.")},
                    "no_automatic_retries": True, "failures_count": True}


class NativeRound:
    """Per-request native consent; only the existing journal owns attempts."""

    def __init__(self, service):
        self.service = service

    def policy(self, key, providers):
        from .codex_native_provider import LIMITS as native_limits, TRUST, validate_binding
        return {"schema": "rethlas.codex-native-policy", "version": 1,
            "service_id": self.service.config["service_id"], "request_key": key,
            "materials": "public_or_synthetic", "providers": {p: validate_binding(b) for p, b in providers.items()},
            "limits": deepcopy(native_limits), "trust": deepcopy(TRUST),
            "operator_assertion": "known extra-model extensions disabled; global model concurrency checked <=3"}

    def _exists(self, value):
        try:
            raw = self.service.parts["research"].read_bytes(digest(value) + ".json", max_bytes=META_BYTES)
        except FileNotFoundError:
            return False
        if raw != canonical_json(value):
            raise ValidityError("native approval bytes differ")
        return True

    def _approval(self, value):
        return {"schema": "rethlas.codex-native-approval", "version": 1,
                "service_id": self.service.config["service_id"], "approved": deepcopy(value)}

    def approved(self, value):
        with self.service._operator():
            return self._exists(self._approval(value))

    def approve(self, value, *, materials):
        # Trusted operator API, never registered as an MCP tool. A schema/type
        # check here prevents an old HTTP approval from authorizing native work.
        if (materials != "public_or_synthetic" or type(value) is not dict
                or value.get("schema") not in ("rethlas.codex-native-policy", "rethlas.codex-native-execution")
                or type(value.get("version")) is not int or value["version"] != 1
                or value.get("service_id") != self.service.config["service_id"]):
            raise ValidityError("invalid exact native approval")
        approved = self._approval(value)
        with self.service._operator() as (_, state):
            key = value.get("request_key")
            binding = state["requests"].get(key)
            if binding is None or binding["kind"] != "research" or binding["status"] != "bound":
                raise ValidityConflict("native approval requires the original pending request")
            progress = binding["result"]
            if not progress.get("stage", "").startswith("awaiting_") or "approval_prompt_sha256" not in progress:
                raise ValidityConflict("native approval requires an exact pending prompt")
            raw = self.service.parts["research"].read_bytes(progress["approval_prompt_sha256"] + ".json", max_bytes=META_BYTES)
            expected = {"version": "research-object-1", "service_id": self.service.config["service_id"],
                        "request_key": key, "kind": "approval_prompt", "value": value}
            if raw != canonical_json(expected) or digest(expected) != progress["approval_prompt_sha256"]:
                raise ValidityConflict("native approval differs from the actual pending input")
            self.service._immutable(self.service.parts["research"], digest(approved) + ".json", approved, META_BYTES)
        return {"status": "native_exact_approved", "request_sent": False, "approval_sha256": digest(approved)}

    def start(self, grant, key, progress, envelope, policy, selected, context, *, reference_refs=None, cancel_event=None):
        from .codex_native_provider import LIMITS as native_limits
        from .project_views import project_context
        from .validity import ValidityRecoveryRequired
        purpose = envelope["purpose"]
        started = {**progress, "stage": purpose + "_started", "execution_sha256": digest(envelope)}
        # snapshot authenticates and holds the service lock, graph checkpoint and
        # journal state through the single transition. No nested operator lock.
        with self.service.snapshot(grant["token"], run_id=grant["run_id"], action="research.run",
                                   cancel_event=cancel_event) as (state, graph, withdrawals, claims):
            binding = state["requests"][key]
            self.service._terminal(binding)
            if binding["status"] == "started":
                raise ValidityRecoveryRequired("native attempt already started; no resend")
            if (binding["status"] != "bound" or binding["result"] != progress
                    or progress.get("stage") != "awaiting_" + purpose + "_approval"):
                raise ValidityConflict("native stage changed before start")
            operation = self.service._operation(binding)
            if (binding["identity"]["run_id"] != grant["run_id"]
                    or policy != self.policy(key, operation["request"]["provider"])):
                raise ValidityConflict("native round or run identity differs")
            retained = {"version": "research-object-1", "service_id": self.service.config["service_id"],
                        "request_key": key, "kind": "native_execution", "value": envelope}
            if progress.get("native_execution_sha256") != digest(retained):
                raise ValidityConflict("native execution differs from retained stage")
            if (envelope["request_key"] != key or envelope["run_id"] != grant["run_id"]
                    or policy["request_key"] != key or policy["providers"][purpose] != envelope["provider"]
                    or purpose not in ("generation", "review")):
                raise ValidityConflict("native execution differs from round")
            if claims["expires_at"] - self.service.clock() < native_limits["minimum_ttl_seconds"]:
                raise ValidityError("native capability needs explicit renewal before dispatch")
            current = project_context(graph, withdrawals, selected, reference_refs=reference_refs)
            if (not current["overlay"]["input_eligible"] or current["dto_sha256"] != context["dto_sha256"]
                    or current["overlay_sha256"] != context["overlay_sha256"]):
                raise ValidityConflict("native mathematical input changed")
            if not self._exists(self._approval(policy)) or not self._exists(self._approval(envelope)):
                raise ValidityError("native policy and exact stage approval required")
            total, stage = "native." + key, "native." + key + "." + purpose
            if state["counts"].get(total, 0) >= 2 or state["counts"].get(stage, 0) >= 1:
                raise ValidityCapacity("native round attempts exhausted; failures are not refunded")
            def begin(following):
                following["counts"][total] = following["counts"].get(total, 0) + 1
                following["counts"][stage] = following["counts"].get(stage, 0) + 1
                following["requests"][key].update(status="started", result=deepcopy(started))
            self.service._change(state, "RESEARCH_STARTED", begin, key=key, claims=claims, cancel_event=cancel_event)
        return started

