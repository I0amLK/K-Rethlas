"""Bounded BGE-M3 adapters; authorization and durable admission belong to the caller.

Cache replay never accesses the network. Live calls require an explicit endpoint,
private credential file, text allowlist and admission callback. A trusted host may
inject a transport in tests; model requests cannot select one.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from .memory_io import read_input, read_token
from .private_fs import canonical_json, strict_json_loads


MODEL = "BAAI/bge-m3"
DIMENSIONS = 1024
MAX_BATCH = 64
MAX_TEXT_BYTES = 32768
MAX_BATCH_BYTES = 262144
CACHE_BYTES = 262144


class EmbeddingError(ValueError):
    """A bounded, value-free embedding rejection."""


class EmbeddingCacheMiss(EmbeddingError):
    """The exact approved model/input pair is absent; no live fallback occurred."""


class EmbeddingCancelled(EmbeddingError):
    """The caller cancelled before delivery; an admitted request is not refunded."""


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_key(text: str) -> str:
    return hashlib.sha256((MODEL + "\0" + text).encode("utf-8")).hexdigest()


def validate_vector(value: Any) -> list[float]:
    if type(value) is not list or len(value) != DIMENSIONS:
        raise EmbeddingError("embedding dimension mismatch")
    if any(type(item) not in (int, float) for item in value):
        raise EmbeddingError("embedding values must be finite numbers")
    try:
        vector = [float(item) for item in value]
        norm = math.hypot(*vector)
    except (OverflowError, ValueError):
        raise EmbeddingError("embedding values must be finite numbers") from None
    if not all(math.isfinite(item) for item in vector) or not math.isfinite(norm) or norm == 0:
        raise EmbeddingError("embedding vector must be finite and nonzero")
    return vector


def _cancel(event: Any) -> None:
    if event is not None and event.is_set():
        raise EmbeddingCancelled("embedding request cancelled")


def _texts(value: Any, kind: str) -> list[str]:
    if kind not in ("documents", "queries"):
        raise EmbeddingError("invalid embedding input kind")
    if type(value) is not list or not 1 <= len(value) <= MAX_BATCH:
        raise EmbeddingError("embedding input must be a bounded nonempty list")
    lengths = []
    for text in value:
        if type(text) is not str or not text.strip():
            raise EmbeddingError("embedding input must contain nonempty UTF-8 strings")
        try:
            length = len(text.encode("utf-8"))
        except UnicodeError:
            raise EmbeddingError("embedding input must contain valid UTF-8") from None
        if length > MAX_TEXT_BYTES:
            raise EmbeddingError("embedding text exceeds byte limit")
        lengths.append(length)
    if sum(lengths) > MAX_BATCH_BYTES:
        raise EmbeddingError("embedding batch exceeds byte limit")
    return list(value)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _http(request: urllib.request.Request, timeout: float, maximum: int,
          cancel_event: Any) -> bytes:
    # No environment proxy, cookies, redirect expansion or retrying SDK.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    deadline = time.monotonic() + timeout
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                raise EmbeddingError("embedding upstream returned a non-success status")
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/json":
                raise EmbeddingError("embedding response must be JSON")
            length = response.headers.get("Content-Length")
            if length is not None:
                try:
                    declared = int(length)
                except (TypeError, ValueError):
                    raise EmbeddingError("invalid embedding response length") from None
                if declared < 0 or declared > maximum:
                    raise EmbeddingError("embedding response exceeds byte limit")
            chunks, total = [], 0
            while True:
                _cancel(cancel_event)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise EmbeddingError("embedding response deadline exceeded")
                if response.isclosed():
                    break
                # One bounded read per deadline check prevents a slow sender
                # from extending the operation by repeatedly resetting a socket timeout.
                response.fp.raw._sock.settimeout(remaining)
                chunk = response.read1(min(65536, maximum - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum:
                    raise EmbeddingError("embedding response exceeds byte limit")
                chunks.append(chunk)
            if length is not None and total != declared:
                raise EmbeddingError("embedding response length mismatch")
            return b"".join(chunks)
    except urllib.error.HTTPError as exc:
        status = exc.code
        exc.close()
        raise EmbeddingError(f"embedding upstream HTTP {status}; no automatic retry") from None
    except EmbeddingError:
        raise
    except Exception:
        raise EmbeddingError("embedding transport failed; no automatic retry") from None


class EmbeddingProvider:
    model = MODEL
    dimensions = DIMENSIONS

    def __init__(self, mode: str = "bge_cache_only", *, cache_dir: Path | None = None,
                 endpoint: str | None = None, key_file: Path | None = None,
                 admit: Callable[[list[str], str], Any] | None = None,
                 allowed_sha256=(), timeout: float = 30, response_bytes: int = 4194304,
                 transport: Callable | None = None):
        if mode not in ("bge_cache_only", "bge_live"):
            raise EmbeddingError("unknown embedding mode")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 90:
            raise EmbeddingError("invalid embedding timeout")
        if type(response_bytes) is not int or not 1024 <= response_bytes <= 16777216:
            raise EmbeddingError("invalid embedding response byte limit")
        self.mode, self.timeout, self.response_bytes = mode, float(timeout), response_bytes
        self.cache_dir = Path(cache_dir).absolute() if cache_dir is not None else None
        self.endpoint, self.key_file, self.admit = endpoint, Path(key_file) if key_file is not None else None, admit
        try:
            approved = frozenset(allowed_sha256)
        except TypeError:
            raise EmbeddingError("invalid embedding text allowlist") from None
        if any(type(item) is not str or len(item) != 64 or any(c not in "0123456789abcdef" for c in item)
               for item in approved):
            raise EmbeddingError("invalid embedding text allowlist")
        self.allowed_sha256 = approved
        self._transport = transport or _http
        if transport is not None and not callable(transport):
            raise EmbeddingError("invalid trusted embedding transport")
        if mode == "bge_cache_only":
            if self.cache_dir is None or endpoint is not None or key_file is not None or admit is not None:
                raise EmbeddingError("cache-only mode requires only an explicit cache directory")
        else:
            if type(endpoint) is not str:
                raise EmbeddingError("live embedding requires an explicit HTTPS endpoint")
            try:
                parsed = urllib.parse.urlsplit(endpoint)
                port = parsed.port
            except ValueError:
                raise EmbeddingError("invalid embedding endpoint") from None
            if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
                    or parsed.password is not None or parsed.query or parsed.fragment
                    or not parsed.path or parsed.path.endswith("/")
                    or any(c.isspace() or ord(c) < 32 for c in endpoint) or "\\" in endpoint
                    or port is not None and not 1 <= port <= 65535):
                raise EmbeddingError("live embedding requires a fixed HTTPS endpoint without credentials or redirects")
            if self.key_file is None or not callable(admit) or not approved:
                raise EmbeddingError("live embedding requires credentials, admission and explicit text allowlist")

    def _cached(self, text: str) -> list[float]:
        try:
            raw = read_input(self.cache_dir / (cache_key(text) + ".json"), CACHE_BYTES,
                             single_link=True, stable=True)
            item = strict_json_loads(raw)
        except FileNotFoundError:
            raise EmbeddingCacheMiss("exact BGE model/input cache entry is absent") from None
        except Exception:
            raise EmbeddingError("embedding cache entry is unsafe or invalid") from None
        if type(item) is not dict:
            raise EmbeddingError("invalid embedding cache object")
        if set(item) == {"model", "input_sha256", "text", "vector"}:
            # Exact format written by the frozen OpenViking pilot gateway.
            expected_hash = cache_key(text)
        elif set(item) == {"version", "model", "dimensions", "input_sha256", "text", "vector"}:
            if item["version"] != "bge-cache-1" or type(item["dimensions"]) is not int or item["dimensions"] != DIMENSIONS:
                raise EmbeddingError("unsupported embedding cache format")
            expected_hash = text_sha256(text)
        else:
            raise EmbeddingError("invalid embedding cache fields")
        if item["model"] != MODEL or item["text"] != text or item["input_sha256"] != expected_hash:
            raise EmbeddingError("embedding cache model/input binding mismatch")
        return validate_vector(item["vector"])

    def embed(self, texts: list[str], *, kind: str = "documents", cancel_event: Any = None,
              before_send: Callable | None = None) -> list[list[float]]:
        inputs = _texts(texts, kind)
        if before_send is not None and not callable(before_send):
            raise EmbeddingError("invalid trusted pre-send guard")
        _cancel(cancel_event)
        if self.mode == "bge_cache_only":
            vectors = [self._cached(text) for text in inputs]
            _cancel(cancel_event)
            return vectors
        if any(text_sha256(text) not in self.allowed_sha256 for text in inputs):
            raise EmbeddingError("embedding input is outside the approved external text allowlist")
        # Admission is deliberately not refunded after credentials, timeout or
        # response validation fail. The caller must durably enforce its run cap.
        self.admit(list(inputs), kind)
        _cancel(cancel_event)
        if before_send is not None:
            before_send()
        try:
            key = read_token(self.key_file)
        except Exception:
            raise EmbeddingError("embedding credential file is unavailable or unsafe") from None
        payload = canonical_json({"model": MODEL, "input": inputs, "encoding_format": "float"})
        request = urllib.request.Request(self.endpoint, data=payload, method="POST", headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json", "Accept": "application/json"})
        _cancel(cancel_event)
        if before_send is not None:
            before_send()
        try:
            raw = self._transport(request, self.timeout, self.response_bytes, cancel_event)
        except EmbeddingError:
            raise
        except Exception:
            raise EmbeddingError("embedding transport failed; no automatic retry") from None
        _cancel(cancel_event)
        if type(raw) is not bytes or len(raw) > self.response_bytes:
            raise EmbeddingError("embedding response exceeds byte limit or is not bytes")
        try:
            result = strict_json_loads(raw)
        except Exception:
            raise EmbeddingError("embedding response is invalid strict JSON") from None
        if type(result) is not dict or result.get("model") != MODEL or type(result.get("data")) is not list:
            raise EmbeddingError("embedding response model or data mismatch")
        rows = result["data"]
        if len(rows) != len(inputs):
            raise EmbeddingError("embedding response item count mismatch")
        indexed = {}
        for row in rows:
            if (type(row) is not dict or type(row.get("index")) is not int
                    or not 0 <= row["index"] < len(inputs) or row["index"] in indexed):
                raise EmbeddingError("embedding response indices mismatch")
            indexed[row["index"]] = validate_vector(row.get("embedding"))
        _cancel(cancel_event)
        return [indexed[index] for index in range(len(inputs))]
