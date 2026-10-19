"""Explicit, bounded text requests; mock results are never reported as live."""
from __future__ import annotations

import hashlib
import http.client
import math
from pathlib import Path
import re
import socket
import ssl
import time
from urllib.parse import urlsplit

from .memory_io import read_token
from .private_fs import canonical_json, digest, strict_json_loads


INPUT_BYTES = 16384
RESPONSE_BYTES = 1048576
OUTPUT_TOKENS = 4096
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z", re.ASCII)


class ProviderError(RuntimeError):
    pass


class ProviderConfigurationError(ProviderError, ValueError):
    pass


class ProviderResponseError(ProviderError, ValueError):
    pass


class ProviderIndeterminate(ProviderError):
    def __init__(self, message, *, request_started=True):
        super().__init__(message)
        self.request_started = request_started
        self.uncertain = request_started


class ProviderCancelled(ProviderIndeterminate):
    pass


class ProviderTimeout(ProviderIndeterminate, TimeoutError):
    pass


class ProviderHTTPError(ProviderIndeterminate):
    def __init__(self, status_code):
        super().__init__("text provider returned an unsuccessful HTTP status")
        self.status_code = status_code


def _cancelled(cancel_event, *, request_started=False):
    if cancel_event is not None and cancel_event.is_set():
        raise ProviderCancelled("text request cancelled; no automatic retry", request_started=request_started)


def _endpoint(value):
    if (type(value) is not str or not value or chr(92) in value
            or any(ord(char) <= 32 or ord(char) == 127 for char in value)):
        raise ProviderConfigurationError("an explicit HTTPS chat completions URL is required")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ProviderConfigurationError("invalid text provider URL") from None
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment
            or not parsed.path.endswith("/v1/chat/completions") or ".." in parsed.path.split("/")
            or port is not None and not 1 <= port <= 65535):
        raise ProviderConfigurationError("an explicit HTTPS chat completions URL is required")
    return parsed


def _model(value):
    if (type(value) is not str or len(value) > 256 or not _MODEL.fullmatch(value)
            or any(part in (".", "..") for part in value.split("/"))):
        raise ProviderConfigurationError("an explicit bounded model ID is required")
    return value


def _messages(value):
    if type(value) is not list or not 1 <= len(value) <= 64:
        raise ProviderConfigurationError("messages must be a bounded nonempty list")
    for item in value:
        if (type(item) is not dict or set(item) != {"role", "content"}
                or type(item["role"]) is not str or item["role"] not in ("system", "user", "assistant")
                or type(item["content"]) is not str or not item["content"].strip()):
            raise ProviderConfigurationError("only explicit text messages without tools are supported")
    if not any(item["role"] == "user" for item in value):
        raise ProviderConfigurationError("an explicit user message is required")
    try:
        raw = canonical_json(value)
    except ValueError:
        raise ProviderConfigurationError("messages must be valid UTF-8") from None
    if len(raw) > INPUT_BYTES:
        raise ProviderConfigurationError("text request input exceeds 16 KiB")
    return strict_json_loads(raw)


def _https(endpoint, payload, headers, *, timeout, max_response_bytes, cancel_event, before_send=None):
    parsed = _endpoint(endpoint)
    deadline = time.monotonic() + timeout
    connection = http.client.HTTPSConnection(parsed.hostname, port=parsed.port,
        timeout=timeout, context=ssl.create_default_context())
    sent = False
    try:
        _cancelled(cancel_event)
        connection.connect()
        _cancelled(cancel_event)
        if before_send is not None:
            before_send()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProviderTimeout("text request deadline exceeded before send", request_started=False)
        active_socket = connection.sock
        active_socket.settimeout(remaining)
        sent = True
        connection.request("POST", parsed.path, body=payload, headers=headers)
        response = connection.getresponse()
        _cancelled(cancel_event, request_started=True)
        if response.status != 200:
            raise ProviderHTTPError(response.status)
        if response.getheader("Content-Encoding", "identity").lower() != "identity":
            raise ProviderResponseError("compressed provider responses are unsupported")
        length = response.getheader("Content-Length")
        if length is not None and (not length.isdecimal() or int(length) > max_response_bytes):
            raise ProviderResponseError("text response size limit exceeded")
        data = bytearray()
        while True:
            _cancelled(cancel_event, request_started=True)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProviderTimeout("text request deadline exceeded; outcome uncertain")
            # HTTPConnection may detach its socket for a close-delimited body.
            # The response still uses that socket, so retain the deadline there.
            active_socket.settimeout(remaining)
            chunk = response.read1(min(65536, max_response_bytes - len(data) + 1))
            if not chunk:
                return bytes(data)
            data.extend(chunk)
            if len(data) > max_response_bytes:
                raise ProviderResponseError("text response size limit exceeded")
    except (socket.timeout, TimeoutError):
        raise ProviderTimeout("text request deadline exceeded; no automatic retry", request_started=sent) from None
    except (OSError, http.client.HTTPException):
        raise ProviderIndeterminate("text transport failed; no automatic retry", request_started=sent) from None
    finally:
        connection.close()


def _response(raw, model, maximum, aliases=()):
    if type(raw) is not bytes or not raw or len(raw) > maximum:
        raise ProviderResponseError("text provider returned an invalid bounded response")
    try:
        value = strict_json_loads(raw)
    except (ValueError, UnicodeError, RecursionError):
        raise ProviderResponseError("text provider returned invalid strict JSON") from None
    required = {"id", "object", "created", "model", "choices", "usage"}
    if (type(value) is not dict or not required <= value.keys()
            or value.keys() - required - {"system_fingerprint", "service_tier"}
            or type(value["id"]) is not str or not value["id"]
            or value["object"] != "chat.completion"
            or type(value["created"]) is not int or value["created"] < 0
            or type(value["model"]) is not str or value["model"] not in (model, *aliases)):
        raise ProviderResponseError("text response envelope or model binding differs")
    choices = value["choices"]
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        raise ProviderResponseError("text response must have exactly one choice")
    choice = choices[0]
    if (not {"index", "message", "finish_reason"} <= choice.keys()
            or choice.keys() - {"index", "message", "finish_reason", "logprobs"}
            or type(choice["index"]) is not int or choice["index"] != 0
            or choice["finish_reason"] != "stop" or choice.get("logprobs") is not None):
        raise ProviderResponseError("text choice is incomplete or unsupported")
    message = choice["message"]
    if (type(message) is not dict or not {"role", "content"} <= message.keys()
            or message.keys() - {"role", "content", "refusal"}
            or message["role"] != "assistant" or type(message["content"]) is not str
            or not message["content"].strip() or message.get("refusal") is not None):
        raise ProviderResponseError("only a complete assistant text response is supported")
    usage = value["usage"]
    if (type(usage) is not dict or set(usage) != {"prompt_tokens", "completion_tokens", "total_tokens"}
            or any(type(n) is not int or n < 0 for n in usage.values())
            or usage["completion_tokens"] > OUTPUT_TOKENS
            or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]):
        raise ProviderResponseError("invalid or excessive token usage")
    result = {"content": message["content"], "model": value["model"], "usage": usage,
              "response_sha256": hashlib.sha256(raw).hexdigest()}
    if aliases:
        result.update(requested_model=model, reported_model=value["model"])
    return result


class TextProvider:
    """Trusted caller supplies configuration, admission and optional mock transport.

    transport(endpoint, payload_bytes, headers, *, timeout, max_response_bytes,
              cancel_event) -> bytes. Injection is permitted only in mock mode.
    Admission receives a detached messages list and purpose before any send.
    """

    def __init__(self, mode="live", endpoint=None, key_file=None, model=None, admit=None,
                 transport=None, timeout=30, max_response_bytes=RESPONSE_BYTES, allowed_response_models=None):
        if type(mode) is not str or mode not in ("live", "mock"):
            raise ProviderConfigurationError("provider mode must be live or mock")
        if (type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 120
                or type(max_response_bytes) is not int or not 0 < max_response_bytes <= RESPONSE_BYTES):
            raise ProviderConfigurationError("invalid text provider resource bounds")
        _endpoint(endpoint)
        self.model = _model(model)
        aliases = [] if allowed_response_models is None else allowed_response_models
        if (type(aliases) is not list or len(aliases) > 16
                or any(type(item) is not str for item in aliases) or len(set(aliases)) != len(aliases)):
            raise ProviderConfigurationError("model aliases require an explicit bounded allowlist")
        self.allowed_response_models = tuple(_model(item) for item in aliases)
        self.provider_type = "openai_chat"
        if not callable(admit):
            raise ProviderConfigurationError("explicit request admission is required")
        if mode == "mock":
            if not callable(transport) or key_file is not None:
                raise ProviderConfigurationError("mock requires explicit transport and no credential")
        elif transport is not None or key_file is None or not isinstance(key_file, (str, Path)):
            raise ProviderConfigurationError("live requires an explicit credential file and built-in transport")
        self.mode, self.endpoint, self.admit = mode, endpoint, admit
        self.key_file = Path(key_file) if key_file is not None else None
        self.transport = transport if mode == "mock" else _https
        self.timeout, self.max_response_bytes = timeout, max_response_bytes

    def binding(self):
        return {"provider_type": self.provider_type, "mode": self.mode, "model": self.model,
                "configuration_sha256": digest({"provider_type": self.provider_type, "mode": self.mode,
                    "model": self.model, "endpoint": self.endpoint, "timeout": self.timeout,
                    "response_bytes": self.max_response_bytes, "allowed_response_models": list(self.allowed_response_models),
                    "key_source": None if self.key_file is None else str(self.key_file)})}

    def complete(self, messages, *, purpose="generation", cancel_event=None, before_send=None):
        if type(purpose) is not str or purpose not in ("generation", "review"):
            raise ProviderConfigurationError("unknown text request purpose")
        _cancelled(cancel_event)
        if before_send is not None and not callable(before_send):
            raise ProviderConfigurationError("invalid trusted pre-send guard")
        clean = _messages(messages)
        payload = canonical_json({"model": self.model, "messages": clean,
                                  "max_tokens": OUTPUT_TOKENS, "stream": False})
        # Admission cannot mutate the actual outgoing request through this copy.
        self.admit(strict_json_loads(canonical_json(clean)), purpose)
        _cancelled(cancel_event)
        headers = {"Content-Type": "application/json", "Accept": "application/json",
                   "Accept-Encoding": "identity"}
        if self.mode == "live":
            try:
                credential = read_token(self.key_file)
            except (OSError, ValueError, UnicodeError):
                raise ProviderConfigurationError("text credential file is unavailable or unsafe") from None
            headers["Authorization"] = "Bearer " + credential
        _cancelled(cancel_event)
        try:
            if self.mode == "mock" and before_send is not None:
                before_send()
            try:
                options = {"timeout": self.timeout, "max_response_bytes": self.max_response_bytes,
                           "cancel_event": cancel_event}
                if self.mode == "live" and before_send is not None:
                    options["before_send"] = before_send
                raw = self.transport(self.endpoint, payload, headers, **options)
            except ProviderError:
                raise
            except (socket.timeout, TimeoutError):
                raise ProviderTimeout("text request timed out; no automatic retry") from None
            except Exception:
                raise ProviderIndeterminate("text transport failed; no automatic retry") from None
        finally:
            headers.pop("Authorization", None)
        _cancelled(cancel_event, request_started=True)
        return {**_response(raw, self.model, self.max_response_bytes, self.allowed_response_models), "mode": self.mode}


