"""Codex-native login metadata and a fail-closed text adapter.

Authentication remains inside the installed CLI. A logged-in CLI does not prove a
model is available, or that its internal HTTP calls obey our old API round cap.
The current compatibility gate therefore never starts a live model process.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import tempfile
import time

from .private_fs import canonical_json, digest, strict_json_loads
from .provider import (RESPONSE_BYTES, ProviderConfigurationError, ProviderError, ProviderIndeterminate,
                       ProviderResponseError, ProviderTimeout, _cancelled, _messages, _model)


VERIFIED_VERSION = "0.163.0"
BLOCKERS = (
    "Codex upstream HTTP request and output-token hard caps are not verified",
    "Codex all-tool, MCP, skills and hook isolation is not yet certified",
)
HELP_FLAGS = ("--ignore-user-config", "--ephemeral", "--output-schema", "--json", "--model", "--sandbox")
RESOLUTION_BLOCKER = "Version-pinned safe Codex model/effort/provider resolution is not verified"
STATIC_CAPABILITY_SOURCE = {
    "kind": "installation_static_strings_not_account_support",
    "version": VERIFIED_VERSION,
    "binary_sha256": "399ff69c431c066e40de318fba45c2b07f45a76b19939d247ee4828fc70c4215",
}


class CodexUnavailable(ProviderConfigurationError):
    pass


def native_profile(value):
    """Validate selection intent, without resolving defaults or inspecting Codex."""
    from .codex_binding import OBSERVED_EFFORTS
    if (type(value) is not dict or type(value.get("type")) is not str or value["type"] != "codex_cli"
            or not {"type"} <= value.keys() <= {"type", "model", "reasoning_effort"}):
        raise ProviderConfigurationError("invalid native Codex selection")
    if "model" in value:
        if _model(value["model"]).lower() in {"default", "unknown", "auto"}:
            raise ProviderConfigurationError("omit model to follow Codex defaults")
    if "reasoning_effort" in value and (type(value["reasoning_effort"]) is not str
            or value["reasoning_effort"] not in OBSERVED_EFFORTS):
        raise ProviderConfigurationError("unknown Codex effort selection; compatibility is not confirmed")
    return deepcopy(value)


def effort_choices():
    from .codex_binding import OBSERVED_EFFORTS
    return [("跟随 Codex 默认（尚未确认）", ""), *(
        (value + " · 兼容未确认", value) for value in OBSERVED_EFFORTS)]


def codex_boundary():
    """Offline explanation, not a metadata subprocess or a capability grant."""
    return {"provider_type": "codex_cli", "run_enabled": False,
            "default_resolution": "blocked_missing_verified_protocol",
            "blockers": [RESOLUTION_BLOCKER, *BLOCKERS],
            "effort_vocabulary_source": deepcopy(STATIC_CAPABILITY_SOURCE),
            "per_model_effort_compatibility": "unconfirmed",
            "configuration_or_authentication_read": False, "cli_inspected": False,
            "model_requests_sent": 0}


def inspect_selections(profiles, *, check_login=False):
    """Explicit local metadata only; unknown effective settings stay unknown.

    The installed artifact contains config/read and model/list strings, but no
    reviewed version-pinned protocol or startup isolation evidence. Do not start
    those candidate interfaces, or read config/auth files to fill their gaps.
    """
    if (type(profiles) is not dict or not profiles
            or not profiles.keys() <= {"generation", "review"}):
        raise ProviderConfigurationError("select at least one native Codex purpose")
    clean = {purpose: native_profile(value) for purpose, value in profiles.items()}
    metadata = inspect_codex(check_login=check_login)
    return {"selection_sha256": digest(clean), "metadata": metadata,
            "profiles": {purpose: {"selection": value,
                "resolution_status": "blocked", "confirmation_status": "checked_unresolved",
                "resolved_model": None, "resolved_reasoning_effort": None,
                "provider_identity": None, "model_availability": "unconfirmed",
                "blockers": [RESOLUTION_BLOCKER, *metadata["blockers"]]}
                for purpose, value in clean.items()}, "model_requests_sent": 0}


def _environment():
    # The CLI, not this application, accesses its own login. No environment API
    # keys, proxies, hooks or provider overrides are forwarded by the adapter.
    names = ("HOME", "CODEX_HOME", "PATH", "LANG", "LC_ALL", "XDG_CONFIG_HOME",
             "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_RUNTIME_DIR")
    return {key: os.environ[key] for key in names if key in os.environ}


def _capture(command, *, timeout=5, maximum=65536):
    """Bounded metadata-only subprocess; never used for a model request."""
    with tempfile.TemporaryDirectory(prefix="rethlas-codex-inspect-") as cwd:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, cwd=cwd, env=_environment(), start_new_session=True)
        deadline = time.monotonic() + timeout
        output = {"stdout": bytearray(), "stderr": bytearray()}
        try:
            with selectors.DefaultSelector() as selector:
                for name in output:
                    selector.register(getattr(process, name), selectors.EVENT_READ, name)
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise CodexUnavailable("Codex metadata check timed out")
                    for key, _ in selector.select(min(remaining, 0.1)):
                        chunk = os.read(key.fd, 4096)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        output[key.data].extend(chunk)
                        if sum(map(len, output.values())) > maximum:
                            raise CodexUnavailable("Codex metadata output exceeds its byte limit")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CodexUnavailable("Codex metadata check timed out")
            return process.wait(timeout=remaining), bytes(output["stdout"]), bytes(output["stderr"])
        except subprocess.TimeoutExpired:
            raise CodexUnavailable("Codex metadata check timed out") from None
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            process.stdout.close()
            process.stderr.close()


def inspect_codex(*, check_login=False):
    """Return redacted local metadata; login status is an explicit user action.

    This does not enumerate models, contact a model, or read authentication or
    global configuration files. Raw CLI stdout/stderr never reaches the caller.
    """
    if type(check_login) is not bool:
        raise ProviderConfigurationError("login inspection must be explicit")
    executable = shutil.which("codex")
    result = {"provider_type": "codex_cli", "installed": executable is not None,
              "version": None, "login_status": "not_checked", "model_availability": "unconfirmed",
              "run_enabled": False, "blockers": list(BLOCKERS), "verified_flags": [],
              "default_resolution": "blocked_missing_verified_protocol",
              "resolved_model": None, "resolved_reasoning_effort": None,
              "effort_vocabulary_source": deepcopy(STATIC_CAPABILITY_SOURCE)}
    if executable is None:
        result.update(login_status="not_installed", blockers=["Codex CLI is not installed", *BLOCKERS])
        return result
    try:
        code, output, _ = _capture([executable, "--version"])
        match = re.fullmatch(rb"codex-cli ([0-9]+\.[0-9]+\.[0-9]+)\s*", output)
        if code == 0 and match:
            result["version"] = match.group(1).decode("ascii")
        code, output, _ = _capture([executable, "exec", "--help"])
        if code == 0:
            result["verified_flags"] = [flag for flag in HELP_FLAGS if flag.encode("ascii") in output]
        if check_login:
            code, output, errors = _capture([executable, "login", "status"], maximum=16384)
            status = (output + errors).lower()
            if code == 0 and b"logged in" in status:
                result["login_status"] = "logged_in"
            elif code != 0 and (b"not logged in" in status or b"not authenticated" in status):
                result["login_status"] = "login_required"
            else:
                result["login_status"] = "unconfirmed"
    except (OSError, CodexUnavailable):
        result["login_status"] = "unconfirmed" if check_login else "not_checked"
        result["blockers"] = ["Codex local metadata could not be confirmed", *BLOCKERS]
    if result["version"] != VERIFIED_VERSION:
        result["blockers"].insert(0, "Installed Codex version is not the inspected compatibility version")
    return result


def _schema(purpose):
    if purpose == "generation":
        properties = {"proof_markdown": {"type": "string", "minLength": 1},
                      "dependency_revision_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}}
    else:
        finding = {"type": "object", "additionalProperties": False,
                   "required": ["location", "issue"], "properties": {
                       "location": {"type": "string"}, "issue": {"type": "string"}}}
        properties = {"verdict": {"enum": ["correct", "wrong"]}, "repair_hints": {"type": "string"},
            "verification_report": {"type": "object", "additionalProperties": False,
                "required": ["summary", "critical_errors", "gaps"], "properties": {
                    "summary": {"type": "string"}, "critical_errors": {"type": "array", "items": finding},
                    "gaps": {"type": "array", "items": finding}}}}
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}


def _events(raw, maximum):
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        raise ProviderResponseError("invalid bounded Codex event output")
    texts = []
    completed = False
    usage = {}
    try:
        for line in raw.splitlines():
            value = strict_json_loads(line)
            if type(value) is not dict or type(value.get("type")) is not str:
                raise ValueError
            kind = value["type"]
            if kind in ("thread.started", "turn.started"):
                if texts or completed:
                    raise ValueError
            elif kind == "item.completed":
                item = value.get("item")
                if (type(item) is not dict or item.get("type") != "agent_message"
                        or type(item.get("text")) is not str or not item["text"].strip() or completed):
                    raise ValueError
                texts.append(item["text"])
            elif kind == "turn.completed":
                if completed or len(texts) != 1 or type(value.get("usage")) is not dict:
                    raise ValueError
                usage = value["usage"]
                if any(type(n) is not int or n < 0 for n in usage.values()):
                    raise ValueError
                completed = True
            else:
                # Tool/reasoning/error/unknown events are not silently accepted as
                # proof of pure-text behavior. Nothing executes from these bytes.
                raise ValueError
    except (ValueError, UnicodeError, RecursionError):
        raise ProviderResponseError("unsupported or invalid Codex event stream") from None
    if not completed or len(texts) != 1:
        raise ProviderResponseError("Codex did not return one complete candidate text")
    return texts[0], usage


class CodexCLIProvider:
    """Native CLI configuration, with live execution deliberately gated.

    ``transport`` is a trusted test-host injection in explicit mock mode only.
    It receives argv/stdin/cwd and owns any synthetic process it creates. Operator
    configuration and model tools cannot select an arbitrary executable.
    """
    def __init__(self, *, model=None, selection=None, resolved=None, mode="live", admit=None,
                 transport=None, timeout=30, max_response_bytes=RESPONSE_BYTES):
        if mode not in ("live", "mock") or type(mode) is not str:
            raise ProviderConfigurationError("Codex mode must be live or explicit mock")
        if (type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 120
                or type(max_response_bytes) is not int or not 0 < max_response_bytes <= RESPONSE_BYTES):
            raise ProviderConfigurationError("invalid Codex resource bounds")
        if not callable(admit) or (mode == "mock" and not callable(transport)) or (mode == "live" and transport is not None):
            raise ProviderConfigurationError("Codex needs explicit admission; injection is test-only")
        if model is not None and selection is not None:
            raise ProviderConfigurationError("use either a legacy model or a versioned selection")
        if resolved is not None and mode != "mock":
            raise ProviderConfigurationError("effective settings injection is restricted to the trusted mock host")
        self.mode, self.admit = mode, admit
        self.provider_type, self.endpoint = "codex_cli", "codex://native-login"
        self.timeout, self.max_response_bytes, self.transport = timeout, max_response_bytes, transport
        self._legacy = model is not None
        self._selection = ({"type": "codex_cli", "model": _model(model)} if self._legacy else
                           native_profile({"type": "codex_cli"} if selection is None else selection))
        self._frozen_binding = None
        self.model = self._selection.get("model")
        self.reasoning_effort = None
        if resolved is not None:
            if self._legacy:
                raise ProviderConfigurationError("legacy Codex bindings cannot carry new effective settings")
            from .codex_binding import make_codex_binding
            execution = digest({"adapter": "synthetic-frozen-codex-1", "mode": mode,
                                "timeout": timeout, "response_bytes": max_response_bytes,
                                "real_cli_execution": False})
            self._frozen_binding = make_codex_binding(mode=mode,
                selection={key: value for key, value in self._selection.items() if key != "type"},
                resolved=resolved, cli_version=VERIFIED_VERSION, execution_sha256=execution)
            self.model = self._frozen_binding["resolved"]["model"]
            self.reasoning_effort = self._frozen_binding["resolved"]["reasoning_effort"]
            self.endpoint = self._frozen_binding["resolved"]["provider"]["destination"]
        elif mode == "mock" and not self._legacy:
            raise ProviderConfigurationError("a versioned synthetic host must supply complete effective settings")

    @property
    def selection(self):
        return deepcopy(self._selection)

    def binding(self):
        if not self._legacy:
            if self._frozen_binding is None:
                raise CodexUnavailable(RESOLUTION_BLOCKER)
            from .codex_binding import validate_codex_binding
            frozen = validate_codex_binding(self._frozen_binding)
            execution = digest({"adapter": "synthetic-frozen-codex-1", "mode": self.mode,
                                "timeout": self.timeout, "response_bytes": self.max_response_bytes,
                                "real_cli_execution": False})
            if (self.model != frozen["resolved"]["model"] or self.mode != frozen["mode"]
                    or self.reasoning_effort != frozen["resolved"]["reasoning_effort"]
                    or self.endpoint != frozen["resolved"]["provider"]["destination"]
                    or execution != frozen["execution_sha256"]
                    or {key: value for key, value in self._selection.items() if key != "type"} != frozen["selection"]):
                raise CodexUnavailable("frozen effective Codex settings changed")
            return frozen
        return {"provider_type": self.provider_type, "mode": self.mode, "model": self.model,
                "configuration_sha256": digest({"provider_type": self.provider_type, "mode": self.mode,
                    "model": self.model, "endpoint": self.endpoint, "timeout": self.timeout,
                    "response_bytes": self.max_response_bytes, "compatibility_version": VERIFIED_VERSION})}

    def preflight(self):
        if self._legacy:
            return {"provider_type": self.provider_type, "model": self.model,
                    "run_enabled": self.mode == "mock", "model_availability": "unconfirmed",
                    "blockers": [] if self.mode == "mock" else list(BLOCKERS)}
        resolved = None if self._frozen_binding is None else self.binding()["resolved"]
        return {"provider_type": self.provider_type, "selection": self.selection,
                "resolved": resolved, "resolved_model": None if resolved is None else resolved["model"],
                "resolved_reasoning_effort": None if resolved is None else resolved["reasoning_effort"],
                "resolution_status": "blocked" if resolved is None else "synthetic_test_host_only",
                "run_enabled": self.mode == "mock" and resolved is not None,
                "model_availability": "unconfirmed",
                "blockers": [] if self.mode == "mock" else [RESOLUTION_BLOCKER, *BLOCKERS]}

    def complete(self, messages, *, purpose="generation", cancel_event=None, before_send=None):
        if purpose not in ("generation", "review") or type(purpose) is not str:
            raise ProviderConfigurationError("unknown Codex purpose")
        _cancelled(cancel_event)
        clean = _messages(messages)
        if self.mode == "live":
            raise CodexUnavailable("Codex execution is blocked: unverified tool isolation and upstream hard caps")
        if before_send is not None and not callable(before_send):
            raise ProviderConfigurationError("invalid Codex pre-send guard")
        frozen = self.binding()
        self.admit(strict_json_loads(canonical_json(clean)), purpose)
        _cancelled(cancel_event)
        if before_send is not None:
            before_send()
        if self.binding() != frozen:
            raise CodexUnavailable("Codex binding changed before synthetic dispatch")
        with tempfile.TemporaryDirectory(prefix="rethlas-codex-input-") as directory:
            path = Path(directory) / "response.schema.json"
            path.write_bytes(canonical_json(_schema(purpose)))
            path.chmod(0o600)
            argv = ["codex", "--no-daemon", "exec", "--ignore-user-config", "--ephemeral",
                "--sandbox", "read-only", "--skip-git-repo-check", "--color", "never", "--json",
                "--model", self.model, "--cd", directory, "--output-schema", str(path), "-"]
            try:
                settings = {} if self._legacy else {"effective_settings": deepcopy(frozen)}
                # This extra envelope is for the synthetic host, not a guessed
                # CLI effort flag. No live transport exists behind this branch.
                raw = self.transport(argv, canonical_json(clean), cwd=directory, timeout=self.timeout,
                                     max_response_bytes=self.max_response_bytes, cancel_event=cancel_event, **settings)
            except ProviderError:
                raise
            except (subprocess.TimeoutExpired, TimeoutError):
                raise ProviderTimeout("Codex test transport timed out; no automatic retry") from None
            except Exception:
                raise ProviderIndeterminate("Codex test transport failed; no automatic retry") from None
        _cancelled(cancel_event, request_started=True)
        if self.binding() != frozen:
            raise ProviderIndeterminate("Codex binding changed during synthetic dispatch")
        content, usage = _events(raw, self.max_response_bytes)
        result = {"content": content, "model": self.model, "requested_model": self.model,
                  "reported_model": None, "provider_type": "codex_cli", "mode": "mock", "usage": usage,
                  "response_sha256": hashlib.sha256(raw).hexdigest()}
        if not self._legacy:
            result.update(requested_model=self._selection.get("model"),
                          requested_reasoning_effort=self._selection.get("reasoning_effort"),
                          resolved_model=self.model, resolved_reasoning_effort=self.reasoning_effort,
                          reported_reasoning_effort=None)
        return result
