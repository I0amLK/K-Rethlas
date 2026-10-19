"""Controller-issued, HMAC-authenticated capabilities backed by private state.

Only trusted controller code may initialize an authority, register runs, issue
capabilities, or mutate lifecycle state. Workers receive tokens, never this root.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import math
import re
import secrets
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, NoReturn

from .private_fs import PrivateDirectory, canonical_json, strict_json_loads


class AuthorizationError(PermissionError):
    """A capability or its persistent authority state was rejected."""


ROLE_ACTIONS = MappingProxyType({
    "generation": frozenset({"verify", "status"}),
    "verification": frozenset({"finding.append", "report.submit", "status"}),
    "publisher": frozenset({"publish", "status"}),
    "memory": frozenset({"memory.read", "memory.commit", "memory.maintain"}),
    "context_branch": frozenset({"context.branch.read", "context.branch.commit", "context.branch.seal"}),
    "context_join": frozenset({"context.join.read", "context.join.prepare"}),
    "context_verifier": frozenset({"context.view.read", "context.view.probe"}),
    "validity_writer": frozenset({"validity.commit"}),
    "validity_withdrawer": frozenset({"validity.withdraw"}),
    "validity_reader": frozenset({"validity.read", "validity.status"}),
    "validity_verifier": frozenset({"validity.probe", "validity.status"}),
    "validity_publisher": frozenset({"validity.publish", "validity.status"}),
    "project_reader": frozenset({"project.read", "project.search"}),
    "project_writer": frozenset({"project.read", "project.search", "project.commit", "project.index"}),
    "project_withdrawer": frozenset({"project.withdraw"}),
    "research_runner": frozenset({"project.read", "project.search", "project.commit", "project.index",
                                  "project.withdraw", "research.run", "research.status"}),
})
_DOMAINS = frozenset({"fixture", "production", "research"})
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z", re.ASCII)
_CLAIMS = frozenset({
    "version", "token_id", "owner", "run_id", "domain", "role", "epoch",
    "actions", "issued_at", "expires_at",
})
_MAX_TTL = 3600


def _reject(message: str) -> NoReturn:
    raise AuthorizationError(message)


def _identifier(value: Any) -> str:
    if type(value) is not str or not _IDENTIFIER.fullmatch(value) or value in (".", ".."):
        _reject("invalid identifier")
    return value


def _domain(value: Any) -> str:
    if type(value) is not str or value not in _DOMAINS:
        _reject("invalid domain")
    return value


def _number(value: Any) -> float:
    if type(value) not in (int, float):
        _reject("expected a finite number, not a bool")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        _reject("expected a finite number")
    if not math.isfinite(number):
        _reject("expected a finite number")
    return number


def _epoch(value: Any) -> int:
    if type(value) is not int or value < 0:
        _reject("invalid epoch")
    return value


def _now(value: int | float | None) -> float:
    return _number(time.time() if value is None else value)


def _action(value: Any) -> str:
    if type(value) is not str or not any(value in actions for actions in ROLE_ACTIONS.values()):
        _reject("unknown action")
    return value


def _validate_claims(claims: Any) -> dict[str, Any]:
    if type(claims) is not dict or claims.keys() != _CLAIMS:
        _reject("invalid capability fields")
    if type(claims["version"]) is not int or claims["version"] != 1:
        _reject("unsupported capability version")
    for name in ("token_id", "owner", "run_id"):
        _identifier(claims[name])
    _domain(claims["domain"])
    role = claims["role"]
    if type(role) is not str or role not in ROLE_ACTIONS:
        _reject("unknown role")
    _epoch(claims["epoch"])
    actions = claims["actions"]
    if type(actions) is not list or not actions or any(type(item) is not str for item in actions):
        _reject("invalid capability actions")
    if len(actions) != len(set(actions)) or not set(actions) <= ROLE_ACTIONS[role]:
        _reject("actions exceed the role")
    issued = _number(claims["issued_at"])
    expires = _number(claims["expires_at"])
    if not 0 < expires - issued <= _MAX_TTL:
        _reject("invalid capability lifetime")
    return claims


def _validate_registry(registry: Any) -> dict[str, Any]:
    if type(registry) is not dict or registry.keys() != {"version", "runs", "tokens"}:
        _reject("invalid registry fields")
    if type(registry["version"]) is not int or registry["version"] != 1:
        _reject("unsupported registry version")
    runs, tokens = registry["runs"], registry["tokens"]
    if type(runs) is not dict or type(tokens) is not dict:
        _reject("invalid registry mappings")
    for run_id, run in runs.items():
        _identifier(run_id)
        if type(run) is not dict or run.keys() != {"owner", "domain", "state", "epoch"}:
            _reject("invalid run fields")
        _identifier(run["owner"])
        _domain(run["domain"])
        if type(run["state"]) is not str or run["state"] not in ("open", "closed"):
            _reject("invalid run state")
        _epoch(run["epoch"])
    for token_id, record in tokens.items():
        _identifier(token_id)
        if type(record) is not dict or record.keys() != {"claims", "revoked"}:
            _reject("invalid token registry fields")
        claims = _validate_claims(record["claims"])
        if token_id != claims["token_id"] or type(record["revoked"]) is not bool:
            _reject("invalid token registry entry")
        run = runs.get(claims["run_id"])
        if (
            run is None
            or run["owner"] != claims["owner"]
            or run["domain"] != claims["domain"]
            or claims["epoch"] > run["epoch"]
        ):
            _reject("token registry has an inconsistent run")
    return registry


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _decode(value: str) -> bytes:
    if not value or not re.fullmatch(r"[A-Za-z0-9_-]+", value, re.ASCII):
        _reject("invalid token encoding")
    try:
        result = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
    except (ValueError, binascii.Error) as exc:
        raise AuthorizationError("invalid token encoding") from exc
    if _encode(result) != value:
        _reject("noncanonical token encoding")
    return result


class Authority:
    """Persistent local capability authority; no signing interface is exposed to workers."""

    def __init__(self, root: Path, create: bool = False):
        self.root = Path(root)
        try:
            self._directory = PrivateDirectory(self.root, create=create)
            with self._directory.lock():
                try:
                    key = self._directory.read_bytes("key", max_bytes=32)
                except FileNotFoundError:
                    key = None
                try:
                    registry = self._directory.read_json("registry.json", max_bytes=1048576)
                    registry_exists = True
                except FileNotFoundError:
                    registry, registry_exists = None, False
                if key is None and not registry_exists and create:
                    key = secrets.token_bytes(32)
                    self._directory.write_bytes("key", key, exclusive=True)
                    registry = {"version": 1, "runs": {}, "tokens": {}}
                    self._directory.write_bytes("registry.json", canonical_json(registry), exclusive=True)
                    registry_exists = True
                if key is None or not registry_exists or len(key) != 32:
                    _reject("authority is absent or incompletely initialized")
                _validate_registry(registry)
                self._key = key
        except (OSError, ValueError, TypeError, RecursionError) as exc:
            raise AuthorizationError("unsafe or invalid authority state") from exc

    @contextmanager
    def _registry(self) -> Iterator[dict[str, Any]]:
        # Translate authority-state failures, not exceptions from a caller's
        # commit inside locked_check(). The ExitStack holds flock across yield.
        with ExitStack() as stack:
            try:
                stack.enter_context(self._directory.lock())
                key = self._directory.read_bytes("key", max_bytes=32)
                if not hmac.compare_digest(key, self._key):
                    _reject("authority key changed")
                registry = _validate_registry(self._directory.read_json("registry.json", max_bytes=1048576))
            except AuthorizationError:
                raise
            except (OSError, ValueError, TypeError, RecursionError) as exc:
                raise AuthorizationError("unsafe or invalid authority state") from exc
            yield registry

    def _save(self, registry: dict[str, Any]) -> None:
        try:
            self._directory.write_json("registry.json", registry)
        except (OSError, ValueError, TypeError, RecursionError) as exc:
            raise AuthorizationError("could not persist authority state") from exc

    def register_run(self, owner: str, run_id: str, domain: str) -> None:
        _identifier(owner)
        _identifier(run_id)
        _domain(domain)
        with self._registry() as registry:
            if run_id in registry["runs"]:
                _reject("run is already registered")
            registry["runs"][run_id] = {"owner": owner, "domain": domain, "state": "open", "epoch": 0}
            self._save(registry)

    def issue(
        self, *, owner: str, run_id: str, domain: str, role: str,
        actions: list[str] | tuple[str, ...] | set[str] | frozenset[str],
        ttl_seconds: int | float = 300, now: int | float | None = None,
    ) -> str:
        with self._registry() as registry:
            return self._issue_locked(
                registry, owner=owner, run_id=run_id, domain=domain, role=role,
                actions=actions, ttl_seconds=ttl_seconds, now=now,
            )

    def _issue_locked(
        self, registry: dict[str, Any], *, owner: str, run_id: str, domain: str, role: str,
        actions: list[str] | tuple[str, ...] | set[str] | frozenset[str],
        ttl_seconds: int | float = 300, now: int | float | None = None,
        before_save: Callable[[dict[str, Any]], None] | None = None,
    ) -> str:
        """Issue once under the caller's existing authority lock.

        ``registry`` must come from this authority's active ``_registry()``.
        ``before_save`` is a trusted Python-only guard over the proposed registry,
        not a capability claim or worker/request option. It must not mutate the
        registry. Failure propagates without persisting the proposed token.
        """
        _identifier(owner)
        _identifier(run_id)
        _domain(domain)
        if type(role) is not str or role not in ROLE_ACTIONS:
            _reject("unknown role")
        if type(actions) not in (list, tuple, set, frozenset):
            _reject("actions must be an explicit collection")
        if not actions or any(type(item) is not str for item in actions):
            _reject("invalid capability actions")
        if len(actions) != len(set(actions)) or not set(actions) <= ROLE_ACTIONS[role]:
            _reject("actions exceed the role")
        ttl = _number(ttl_seconds)
        if not 0 < ttl <= _MAX_TTL:
            _reject("invalid capability TTL")
        if before_save is not None and not callable(before_save):
            _reject("before_save must be a trusted callable")
        issued = _now(now)
        run = registry["runs"].get(run_id)
        if run is None or run["state"] != "open" or run["owner"] != owner or run["domain"] != domain:
            _reject("run is not open for this owner and domain")
        token_id = secrets.token_hex(16)
        while token_id in registry["tokens"]:
            token_id = secrets.token_hex(16)
        claims = _validate_claims({
            "version": 1, "token_id": token_id, "owner": owner, "run_id": run_id,
            "domain": domain, "role": role, "epoch": run["epoch"],
            "actions": sorted(actions), "issued_at": issued, "expires_at": issued + ttl,
        })
        payload = canonical_json(claims)
        signature = hmac.new(self._key, payload, hashlib.sha256).digest()
        registry["tokens"][token_id] = {"claims": claims, "revoked": False}
        if before_save is not None:
            before_save(registry)
        self._save(registry)
        return f"{_encode(payload)}.{_encode(signature)}"

    def _claims(self, token: str, action: str, now: float, registry: dict[str, Any]) -> dict[str, Any]:
        _action(action)
        if type(token) is not str or len(token) > 16384 or token.count(".") != 1:
            _reject("malformed capability")
        payload_text, signature_text = token.split(".")
        payload, signature = _decode(payload_text), _decode(signature_text)
        expected = hmac.new(self._key, payload, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, signature):
            _reject("invalid capability signature")
        try:
            claims = _validate_claims(strict_json_loads(payload))
            if payload != canonical_json(claims):
                _reject("noncanonical capability claims")
        except AuthorizationError:
            raise
        except (ValueError, TypeError, RecursionError) as exc:
            raise AuthorizationError("malformed capability claims") from exc
        record = registry["tokens"].get(claims["token_id"])
        if record is None or record["revoked"] or canonical_json(record["claims"]) != payload:
            _reject("capability is absent, revoked, or differs from its registry")
        run = registry["runs"].get(claims["run_id"])
        if (
            run is None or run["state"] != "open" or run["owner"] != claims["owner"]
            or run["domain"] != claims["domain"] or run["epoch"] != claims["epoch"]
        ):
            _reject("capability run is closed, changed, or has a newer epoch")
        if not claims["issued_at"] <= now < claims["expires_at"]:
            _reject("capability is not yet valid or has expired")
        if action not in claims["actions"]:
            _reject("capability does not grant this action")
        return claims

    def authenticate(self, token: str, *, action: str, now: int | float | None = None) -> dict[str, Any]:
        with self._registry() as registry:
            return self._claims(token, action, _now(now), registry)

    def check(
        self, token: str, *, owner: str, run_id: str, domain: str, action: str,
        now: int | float | None = None,
    ) -> dict[str, Any]:
        with self.locked_check(token, owner=owner, run_id=run_id, domain=domain, action=action, now=now) as claims:
            return claims

    @contextmanager
    def locked_check(
        self, token: str, *, owner: str, run_id: str, domain: str, action: str,
        now: int | float | None = None,
    ) -> Iterator[dict[str, Any]]:
        with self._registry() as registry:
            yield self._check_locked(registry, token, owner=owner, run_id=run_id,
                                     domain=domain, action=action, now=now)

    def _check_locked(
        self, registry: dict[str, Any], token: str, *, owner: str, run_id: str,
        domain: str, action: str, now: int | float | None = None,
    ) -> dict[str, Any]:
        """Trusted lifecycle transaction using this authority's held _registry()."""
        _identifier(owner)
        _identifier(run_id)
        _domain(domain)
        claims = self._claims(token, action, _now(now), registry)
        if claims["owner"] != owner or claims["run_id"] != run_id or claims["domain"] != domain:
            _reject("capability scope does not match this request")
        return claims

    def revoke(self, token_id: str) -> None:
        _identifier(token_id)
        with self._registry() as registry:
            record = registry["tokens"].get(token_id)
            if record is None:
                _reject("unknown token")
            if not record["revoked"]:
                record["revoked"] = True
                self._save(registry)

    def close_run(self, run_id: str) -> None:
        _identifier(run_id)
        with self._registry() as registry:
            self._close_run_locked(registry, run_id)

    def _close_run_locked(self, registry: dict[str, Any], run_id: str) -> None:
        """Close under this authority's held registry lock, never reacquire it."""
        _identifier(run_id)
        run = registry["runs"].get(run_id)
        if run is None:
            _reject("unknown run")
        if run["state"] != "closed":
            run["state"] = "closed"
            self._save(registry)

    def bump_epoch(self, run_id: str) -> int:
        _identifier(run_id)
        with self._registry() as registry:
            run = registry["runs"].get(run_id)
            if run is None or run["state"] != "open":
                _reject("run is not open")
            run["epoch"] += 1
            self._save(registry)
            return run["epoch"]
