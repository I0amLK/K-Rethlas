"""Offline synthetic embedding adapter checks, not live BGE quality evidence."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import urllib.request

from rethlas_core.embedding import (
    DIMENSIONS, MODEL, EmbeddingCacheMiss, EmbeddingCancelled, EmbeddingError,
    EmbeddingProvider, _http, cache_key, text_sha256, validate_vector,
)
from rethlas_core.private_fs import canonical_json


def vector(axis=0):
    return [1.0 if index == axis else 0.0 for index in range(DIMENSIONS)]


def response(vectors, **fields):
    return canonical_json({"model": MODEL, "data": [{"index": index, "embedding": item}
                                                    for index, item in enumerate(vectors)], **fields})


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rethlas-embedding-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.root.chmod(0o700)
        self.key = self.root / "key"
        self.key.write_text("synthetic-test-key", encoding="ascii")
        self.key.chmod(0o600)
        self.admissions = []
        self.sent = []

    def live(self, texts, *, transport=None, **kwargs):
        def send(request, timeout, maximum, cancel_event):
            self.sent.append((request, timeout, maximum))
            return response([vector(index) for index in range(len(texts))])
        return EmbeddingProvider("bge_live", endpoint="https://embedding.invalid/v1/embeddings",
            key_file=self.key, admit=lambda values, kind: self.admissions.append((values, kind)),
            allowed_sha256=[text_sha256(text) for text in texts], transport=transport or send, **kwargs)

    def cache(self, input_text, *, canonical=False, **changes):
        item = {"model": MODEL, "input_sha256": cache_key(input_text), "text": input_text, "vector": vector()}
        if canonical:
            item.update(version="bge-cache-1", dimensions=DIMENSIONS, input_sha256=text_sha256(input_text))
        item.update(changes)
        path = self.root / (cache_key(input_text) + ".json")
        path.write_bytes(canonical_json(item))
        return path

    def test_exact_legacy_cache_replay_never_sends_or_writes(self):
        text = "# 一致连续\r\nA fixed mathematical input.\n"
        path = self.cache(text)
        before = path.read_bytes()
        provider = EmbeddingProvider(cache_dir=self.root)
        with patch("rethlas_core.embedding._http", side_effect=AssertionError("network forbidden")):
            self.assertEqual(provider.embed([text, text], kind="queries"), [vector(), vector()])
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(provider.mode, "bge_cache_only")

    def test_presend_revocation_guard_prevents_credential_read_and_dispatch(self):
        provider = self.live(["fixed"])
        def revoked():
            raise PermissionError("synthetic revoked project")
        with patch("rethlas_core.embedding.read_token", side_effect=AssertionError("must not read key")):
            with self.assertRaises(PermissionError):
                provider.embed(["fixed"], before_send=revoked)
        self.assertEqual(self.admissions, [(["fixed"], "documents")])
        self.assertEqual(self.sent, [])

    def test_canonical_cache_format(self):
        self.cache("fixed", canonical=True)
        self.assertEqual(EmbeddingProvider(cache_dir=self.root).embed(["fixed"]), [vector()])

    def test_cache_miss_is_explicit_and_no_implicit_live_fallback(self):
        provider = EmbeddingProvider(cache_dir=self.root)
        with patch("rethlas_core.embedding._http", side_effect=AssertionError("network forbidden")):
            with self.assertRaises(EmbeddingCacheMiss):
                provider.embed(["missing"])
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["key"])

    def test_cache_text_model_hash_and_unknown_fields_are_bound(self):
        for change in ({"model": "other"}, {"text": "changed"}, {"input_sha256": "0" * 64},
                       {"unknown": True}, {"vector": vector()[:-1]}, {"dimensions": 1024.0}):
            with self.subTest(change=list(change)):
                self.cache("fixed", **change)
                with self.assertRaises(EmbeddingError):
                    EmbeddingProvider(cache_dir=self.root).embed(["fixed"])

    def test_cache_strict_json_and_symlink_refused(self):
        path = self.cache("fixed")
        path.write_bytes(b'{"model":"a","model":"b"}')
        with self.assertRaises(EmbeddingError):
            EmbeddingProvider(cache_dir=self.root).embed(["fixed"])
        path.unlink()
        path.symlink_to(self.key)
        with self.assertRaises(EmbeddingError):
            EmbeddingProvider(cache_dir=self.root).embed(["fixed"])

    def test_vector_validation_rejects_bool_nonfinite_zero_and_wrong_dimensions(self):
        for value in ([], [0.0] * DIMENSIONS, [True] + vector()[1:],
                      [float("nan")] + vector()[1:], [float("inf")] + vector()[1:],
                      [10 ** 10000] + vector()[1:]):
            with self.subTest(kind=type(value[0]).__name__ if value else "empty"):
                with self.assertRaises(EmbeddingError):
                    validate_vector(value)
        self.assertEqual(validate_vector([1] + [0] * (DIMENSIONS - 1)), vector())

    def test_input_bounds_and_kind_are_checked_before_admission(self):
        provider = self.live(["fixed"])
        for texts, kind in (([], "documents"), (["fixed"] * 65, "documents"),
                            ([""], "documents"), (["\ud800"], "documents"),
                            (["x" * 32769], "documents"), (["fixed"], "other"),
                            (["x" * 32768] * 9, "documents")):
            with self.subTest(kind=kind, count=len(texts)):
                with self.assertRaises(EmbeddingError):
                    provider.embed(texts, kind=kind)
        self.assertFalse(self.admissions)
        self.assertFalse(self.sent)

    def test_live_requires_fixed_https_credential_admission_and_allowlist(self):
        for endpoint in ("http://localhost/v1/embeddings", "https://user:pass@host/v1/embeddings",
                         "https://host/v1/embeddings?key=x", "https://host/v1/embeddings#fragment",
                         "https://host", "https://host:bad/v1/embeddings", "https://host\\evil/v1/embeddings"):
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(EmbeddingError):
                    EmbeddingProvider("bge_live", endpoint=endpoint, key_file=self.key,
                        admit=lambda *_: None, allowed_sha256=[text_sha256("fixed")])
        with self.assertRaises(EmbeddingError):
            EmbeddingProvider("bge_live", endpoint="https://host/v1/embeddings", key_file=self.key,
                              allowed_sha256=[text_sha256("fixed")])
        with self.assertRaises(EmbeddingError):
            EmbeddingProvider("bge_live", endpoint="https://host/v1/embeddings", key_file=self.key,
                              admit=lambda *_: None)

    def test_live_allowlist_precedes_admission_credential_and_transport(self):
        provider = self.live(["fixed"])
        self.key.unlink()
        with self.assertRaises(EmbeddingError):
            provider.embed(["not approved"])
        self.assertFalse(self.admissions)
        self.assertFalse(self.sent)

    def test_live_admits_once_and_preserves_input_order(self):
        texts = ["中文精确正文\n", "second"]
        def transport(request, timeout, maximum, event):
            self.sent.append(request)
            return canonical_json({"model": MODEL, "data": [
                {"index": 1, "embedding": vector(1)}, {"index": 0, "embedding": vector(0)}]})
        result = self.live(texts, transport=transport).embed(texts, kind="queries")
        self.assertEqual(result, [vector(0), vector(1)])
        self.assertEqual(self.admissions, [(texts, "queries")])
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.sent[0].get_header("Authorization"), "Bearer synthetic-test-key")
        self.assertIn("中文精确正文".encode(), self.sent[0].data)

    def test_admission_denial_prevents_credential_read_and_send(self):
        provider = self.live(["fixed"])
        provider.admit = lambda *_: (_ for _ in ()).throw(ValueError("budget exhausted"))
        with patch("rethlas_core.embedding.read_token", side_effect=AssertionError("must not read key")):
            with self.assertRaisesRegex(ValueError, "budget exhausted"):
                provider.embed(["fixed"])
        self.assertFalse(self.sent)

    def test_unsafe_or_missing_credential_is_not_refunded_or_echoed(self):
        provider = self.live(["fixed"])
        self.key.chmod(0o644)
        with self.assertRaises(EmbeddingError) as error:
            provider.embed(["fixed"])
        self.assertEqual(self.admissions, [(["fixed"], "documents")])
        self.assertFalse(self.sent)
        self.assertNotIn(str(self.key), str(error.exception))
        self.assertNotIn("synthetic-test-key", str(error.exception))

    def test_bad_response_is_not_retried(self):
        bad = [b"{}", b'{"model":"a","model":"b"}', response([]),
               response([vector()], model="wrong"), response([[0] * DIMENSIONS]),
               canonical_json({"model": MODEL, "data": [{"index": True, "embedding": vector()}]}),
               canonical_json({"model": MODEL, "data": [{"index": 1, "embedding": vector()}]}),
               b"x" * 4097]
        for raw in bad:
            with self.subTest(size=len(raw)):
                calls = []
                def transport(*args):
                    calls.append(1)
                    return raw
                provider = self.live(["fixed"], transport=transport, response_bytes=4096)
                with self.assertRaises(EmbeddingError):
                    provider.embed(["fixed"])
                self.assertEqual(calls, [1])
        self.assertEqual(len(self.admissions), len(bad))

    def test_duplicate_response_indices_refused(self):
        def transport(*args):
            return canonical_json({"model": MODEL, "data": [
                {"index": 0, "embedding": vector()}, {"index": 0, "embedding": vector(1)}]})
        with self.assertRaises(EmbeddingError):
            self.live(["one", "two"], transport=transport).embed(["one", "two"])

    def test_transport_error_is_value_free_and_not_retried(self):
        calls = []
        def transport(*args):
            calls.append(1)
            raise OSError("synthetic-test-key private-body upstream diagnostic")
        with self.assertRaises(EmbeddingError) as error:
            self.live(["fixed"], transport=transport).embed(["fixed"])
        self.assertEqual(calls, [1])
        self.assertNotIn("synthetic-test-key", str(error.exception))
        self.assertNotIn("private-body", str(error.exception))

    def test_cancellation_before_admission_and_after_send(self):
        event = threading.Event()
        event.set()
        provider = self.live(["fixed"])
        with self.assertRaises(EmbeddingCancelled):
            provider.embed(["fixed"], cancel_event=event)
        self.assertFalse(self.admissions)
        event.clear()
        def transport(*args):
            event.set()
            return response([vector()])
        with self.assertRaises(EmbeddingCancelled):
            self.live(["fixed"], transport=transport).embed(["fixed"], cancel_event=event)
        self.assertEqual(len(self.admissions), 1)

    def loopback(self):
        calls = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                calls.append(self.path)
                if self.path == "/redirect":
                    self.send_response(307)
                    self.send_header("Location", "/ok")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if self.path == "/rate":
                    self.send_response(429)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                raw = response([vector()])
                self.send_response(200)
                self.send_header("Content-Type", "text/html" if self.path == "/html" else "application/json")
                self.send_header("Content-Length", str(len(raw) + (10 if self.path == "/truncated" else 0)))
                self.end_headers()
                try:
                    if self.path == "/slow":
                        for part in (raw[:1], raw[1:2], raw[2:]):
                            self.wfile.write(part)
                            self.wfile.flush()
                            time.sleep(0.15)
                    else:
                        self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        def cleanup():
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
        self.addCleanup(cleanup)
        return "http://127.0.0.1:" + str(server.server_port), calls

    def test_actual_loopback_http_transport_success_and_streamed_byte_limit(self):
        base, calls = self.loopback()
        request = urllib.request.Request(base + "/ok", data=b"{}", method="POST")
        self.assertEqual(_http(request, 2, 32768, None), response([vector()]))
        with self.assertRaises(EmbeddingError):
            _http(request, 2, 1024, None)
        self.assertEqual(calls, ["/ok", "/ok"])

    def test_actual_loopback_redirect_rate_limit_and_wrong_content_type_do_not_retry(self):
        base, calls = self.loopback()
        for endpoint in ("/redirect", "/rate", "/html", "/truncated"):
            with self.subTest(endpoint=endpoint):
                request = urllib.request.Request(base + endpoint, data=b"{}", method="POST")
                with self.assertRaises(EmbeddingError):
                    _http(request, 2, 32768, None)
        self.assertEqual(calls, ["/redirect", "/rate", "/html", "/truncated"])

    def test_actual_loopback_slow_response_obeys_total_deadline(self):
        base, calls = self.loopback()
        request = urllib.request.Request(base + "/slow", data=b"{}", method="POST")
        started = time.monotonic()
        with self.assertRaises(EmbeddingError):
            _http(request, 0.22, 32768, None)
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(calls, ["/slow"])

    def test_no_redirect_handler_or_environment_proxy_expansion(self):
        with patch("rethlas_core.embedding.urllib.request.build_opener") as build:
            build.return_value.open.side_effect = OSError("synthetic transport failure")
            request = urllib.request.Request("https://embedding.invalid/v1/embeddings", data=b"{}")
            with self.assertRaises(EmbeddingError):
                _http(request, 1, 4096, None)
        handlers = build.call_args.args
        self.assertEqual(handlers[0].proxies, {})
        self.assertIsNone(handlers[1].redirect_request(None, None, 302, "", {}, "https://other.invalid"))


if __name__ == "__main__":
    unittest.main()
