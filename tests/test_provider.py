from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import socket
import threading
import unittest
from unittest.mock import Mock, patch

from rethlas_core.private_fs import canonical_json
from rethlas_core.provider import (
    INPUT_BYTES, OUTPUT_TOKENS, RESPONSE_BYTES, ProviderCancelled,
    ProviderConfigurationError, ProviderHTTPError, ProviderIndeterminate,
    ProviderResponseError, ProviderTimeout, TextProvider,
)


MODEL = "xiaoliu/gpt-6.1-sol:max"
ENDPOINT = "https://provider.example/v1/chat/completions"
MESSAGES = [{"role": "system", "content": "Return mathematical candidate text."},
            {"role": "user", "content": "证明一个合成命题。"}]


def completion():
    return {"id": "synthetic-response-1", "object": "chat.completion", "created": 1,
            "model": MODEL, "choices": [{"index": 0,
                "message": {"role": "assistant", "content": "Synthetic candidate, not a proof certificate."},
                "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}}


class ProviderTests(unittest.TestCase):
    def provider(self, value=None, *, transport=None, admit=None, **kwargs):
        raw = canonical_json(completion() if value is None else value)
        self.sent = []
        def send(endpoint, payload, headers, **limits):
            self.sent.append((endpoint, json.loads(payload), dict(headers), limits))
            return raw
        return TextProvider(mode="mock", endpoint=ENDPOINT, model=MODEL,
            admit=admit or (lambda messages, purpose: None), transport=transport or send, **kwargs)

    def test_mock_success_is_explicit_and_bound_to_response_bytes(self):
        raw = canonical_json(completion())
        result = self.provider().complete(MESSAGES)
        self.assertEqual(result, {"content": completion()["choices"][0]["message"]["content"],
            "model": MODEL, "usage": completion()["usage"], "mode": "mock",
            "response_sha256": hashlib.sha256(raw).hexdigest()})
        endpoint, payload, headers, limits = self.sent[0]
        self.assertEqual(endpoint, ENDPOINT)
        self.assertEqual(payload, {"model": MODEL, "messages": MESSAGES,
                                  "max_tokens": 4096, "stream": False})
        self.assertNotIn("Authorization", headers)
        self.assertEqual(limits["timeout"], 30)
        self.assertEqual(limits["max_response_bytes"], RESPONSE_BYTES)

    def test_admission_precedes_transport_and_cannot_mutate_payload(self):
        observed = []
        def admit(messages, purpose):
            observed.append((copy.deepcopy(messages), purpose))
            messages[0]["content"] = "not approved outgoing bytes"
        provider = self.provider(admit=admit)
        provider.complete(MESSAGES, purpose="review")
        self.assertEqual(observed, [(MESSAGES, "review")])
        self.assertEqual(self.sent[0][1]["messages"], MESSAGES)
        self.assertEqual(MESSAGES[0]["content"], "Return mathematical candidate text.")

    def test_presend_guard_rechecks_after_admission_without_transport(self):
        order = []
        provider = self.provider(admit=lambda *args: order.append("admitted"))
        def revoked():
            order.append("guard")
            raise PermissionError("synthetic revoked input")
        with self.assertRaises(PermissionError):
            provider.complete(MESSAGES, before_send=revoked)
        self.assertEqual(order, ["admitted", "guard"])
        self.assertEqual(self.sent, [])

    def test_live_presend_guard_runs_after_tls_before_http_request(self):
        connection = FakeConnection(FakeResponse())
        with patch("rethlas_core.provider.read_token", return_value="synthetic-key"), \
             patch("rethlas_core.provider.http.client.HTTPSConnection", return_value=connection):
            provider = TextProvider(mode="live", endpoint=ENDPOINT, key_file=Path("/explicit/key"),
                                    model=MODEL, admit=lambda *args: None)
            def revoked():
                raise PermissionError("synthetic revoked grant")
            with self.assertRaises(ProviderIndeterminate):
                provider.complete(MESSAGES, before_send=revoked)
        self.assertEqual(connection.requests, [])
        self.assertTrue(connection.closed)

    def test_admission_denial_never_dispatches(self):
        def deny(messages, purpose):
            raise PermissionError("quota exhausted")
        with self.assertRaisesRegex(PermissionError, "quota exhausted"):
            self.provider(admit=deny).complete(MESSAGES)
        self.assertEqual(self.sent, [])

    def test_config_accepts_explicit_models_without_substitution(self):
        for model in (None, "", "model with spaces", "../secret", "bad\nmodel", [], True):
            with self.subTest(model=model), self.assertRaises(ProviderConfigurationError):
                TextProvider(mode="mock", endpoint=ENDPOINT, model=model, admit=lambda *a: None,
                             transport=lambda *a, **k: b"{}")
        for model in ("6.1-sol", "gpt-6.1-sol", MODEL, "gpt-5.4", "review-model", "provider/arbitrary-model:v2"):
            provider = TextProvider(mode="mock", endpoint=ENDPOINT, model=model,
                admit=lambda *a: None, transport=lambda *a, **k: b"{}")
            self.assertEqual(provider.model, model)

    def test_explicit_credential_transport_and_admission_requirements(self):
        invalid = [dict(mode="mock"), dict(mode="live"),
            dict(mode="live", key_file="/explicit/key", transport=lambda *a: None),
            dict(mode="mock", transport=lambda *a: None, key_file="/explicit/key"),
            dict(mode="unknown", transport=lambda *a: None)]
        for extra in invalid:
            with self.subTest(extra=extra), self.assertRaises(ProviderConfigurationError):
                TextProvider(endpoint=ENDPOINT, model=MODEL, admit=lambda *a: None, **extra)
        with self.assertRaises(ProviderConfigurationError):
            TextProvider(mode="mock", endpoint=ENDPOINT, model=MODEL, transport=lambda *a: None)

    def test_endpoint_cannot_redirect_auth_or_select_other_protocol(self):
        endpoints = [None, "http://provider.example/v1/chat/completions",
            "https://user:credential@provider.example/v1/chat/completions",
            ENDPOINT + "?api_key=unapproved", ENDPOINT + "#fragment",
            "https://provider.example/other", "https://provider.example:0/v1/chat/completions",
            "https://provider.example/../v1/chat/completions", ENDPOINT + "\n",
            "https://provider.example\\@other.example/v1/chat/completions"]
        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint), self.assertRaises(ProviderConfigurationError):
                TextProvider(mode="mock", endpoint=endpoint, model=MODEL,
                    admit=lambda *a: None, transport=lambda *a, **k: b"{}")

    def test_resource_bounds_are_strict(self):
        for extra in ({"timeout": 0}, {"timeout": True}, {"timeout": float("nan")},
            {"timeout": 121}, {"max_response_bytes": True}, {"max_response_bytes": 0},
            {"max_response_bytes": RESPONSE_BYTES + 1}):
            with self.subTest(extra=extra), self.assertRaises(ProviderConfigurationError):
                self.provider(**extra)

    def test_messages_reject_tools_empty_unsupported_and_unknown_fields(self):
        values = [None, {}, [], [{"role": "user", "content": ""}],
            [{"role": "tool", "content": "result"}], [{"role": "user", "content": []}],
            [{"role": "user", "content": "hello", "tool_calls": []}],
            [{"role": "assistant", "content": "no user request"}],
            [{"role": "user", "content": "hi"}] * 65,
            [{"role": "user", "content": "\ud800"}]]
        for value in values:
            with self.subTest(value=value), self.assertRaises(ProviderConfigurationError):
                self.provider().complete(value)
            self.assertEqual(self.sent, [])

    def test_input_budget_counts_utf8_and_message_envelope(self):
        with self.assertRaises(ProviderConfigurationError):
            self.provider().complete([{"role": "user", "content": "中" * (INPUT_BYTES // 3)}])
        self.assertEqual(self.sent, [])
        self.provider().complete([{"role": "user", "content": "x" * (INPUT_BYTES - 100)}])
        self.assertEqual(len(self.sent), 1)

    def test_purpose_is_closed_before_admission(self):
        admit = Mock()
        with self.assertRaises(ProviderConfigurationError):
            self.provider(admit=admit).complete(MESSAGES, purpose="publish")
        admit.assert_not_called()
        self.assertEqual(self.sent, [])

    def test_cancel_before_admission_is_definitely_unsent(self):
        event, admit = threading.Event(), Mock()
        event.set()
        with self.assertRaises(ProviderCancelled) as caught:
            self.provider(admit=admit).complete(MESSAGES, cancel_event=event)
        self.assertFalse(caught.exception.request_started)
        admit.assert_not_called()
        self.assertEqual(self.sent, [])

    def test_cancel_after_admission_does_not_dispatch_or_refund(self):
        event = threading.Event()
        admitted = []
        def admit(messages, purpose):
            admitted.append(purpose)
            event.set()
        with self.assertRaises(ProviderCancelled) as caught:
            self.provider(admit=admit).complete(MESSAGES, cancel_event=event)
        self.assertEqual(admitted, ["generation"])
        self.assertFalse(caught.exception.request_started)
        self.assertEqual(self.sent, [])

    def test_cancel_after_transport_reports_uncertainty(self):
        event = threading.Event()
        calls = []
        def send(*args, **kwargs):
            calls.append(1)
            event.set()
            return canonical_json(completion())
        with self.assertRaises(ProviderCancelled) as caught:
            self.provider(transport=send).complete(MESSAGES, cancel_event=event)
        self.assertTrue(caught.exception.uncertain)
        self.assertEqual(calls, [1])

    def test_timeout_and_transport_error_do_not_retry_or_echo_secrets(self):
        for failure in (socket.timeout("hidden transport input"), OSError("hidden credential"), RuntimeError("hidden text")):
            calls = []
            def send(*args, **kwargs):
                calls.append(1)
                raise failure
            with self.subTest(failure=type(failure).__name__), self.assertRaises(ProviderIndeterminate) as caught:
                self.provider(transport=send).complete(MESSAGES)
            self.assertTrue(caught.exception.uncertain)
            self.assertNotIn("hidden", str(caught.exception))
            self.assertEqual(calls, [1])

    def test_response_size_and_strict_json_are_enforced(self):
        invalid = [b"", "not bytes", b"x" * (RESPONSE_BYTES + 1), b"\xff", b"{", b"{\"x\":NaN}",
                   b"{\"choices\":[],\"choices\":[]}", b"[]"]
        for raw in invalid:
            with self.subTest(raw_type=type(raw).__name__), self.assertRaises(ProviderResponseError):
                self.provider(transport=lambda *a, **k: raw).complete(MESSAGES)

    def test_response_envelope_rejects_unknown_fields_and_wrong_model(self):
        mutations = [("model", "gpt-6-astra"), ("object", "chat.completion.chunk"),
                     ("created", True), ("created", -1), ("id", ""), ("unexpected", {})]
        for key, value in mutations:
            response = completion()
            response[key] = value
            with self.subTest(key=key), self.assertRaises(ProviderResponseError):
                self.provider(response).complete(MESSAGES)
        response = completion()
        del response["usage"]
        with self.assertRaises(ProviderResponseError):
            self.provider(response).complete(MESSAGES)

    def test_choices_are_single_complete_text_and_never_tools(self):
        for choices in ([], [completion()["choices"][0]] * 2, {}, [None]):
            response = completion()
            response["choices"] = choices
            with self.subTest(choices=choices), self.assertRaises(ProviderResponseError):
                self.provider(response).complete(MESSAGES)
        for key, value in (("index", True), ("index", 1), ("finish_reason", "length"),
                           ("finish_reason", "tool_calls"), ("logprobs", {}), ("unknown", None)):
            response = completion()
            response["choices"][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ProviderResponseError):
                self.provider(response).complete(MESSAGES)
        for key, value in (("role", "tool"), ("content", []), ("content", " "),
                           ("tool_calls", []), ("function_call", {}), ("refusal", "no")):
            response = completion()
            response["choices"][0]["message"][key] = value
            with self.subTest(key=key), self.assertRaises(ProviderResponseError):
                self.provider(response).complete(MESSAGES)

    def test_usage_is_finite_strict_consistent_and_bounded(self):
        usages = [{}, {"prompt_tokens": 1, "completion_tokens": True, "total_tokens": 2},
            {"prompt_tokens": -1, "completion_tokens": 1, "total_tokens": 0},
            {"prompt_tokens": 1, "completion_tokens": OUTPUT_TOKENS + 1, "total_tokens": OUTPUT_TOKENS + 2},
            {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 99},
            {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30, "extra": 0}]
        for usage in usages:
            response = completion()
            response["usage"] = usage
            with self.subTest(usage=usage), self.assertRaises(ProviderResponseError):
                self.provider(response).complete(MESSAGES)
        response = canonical_json(completion()).replace(b'"completion_tokens":10', b'"completion_tokens":1e999')
        with self.assertRaises(ProviderResponseError):
            self.provider(transport=lambda *a, **k: response).complete(MESSAGES)

    def test_live_reads_only_explicit_key_at_send_after_admission(self):
        order = []
        response = FakeResponse()
        connection = FakeConnection(response)
        with patch("rethlas_core.provider.read_token", side_effect=lambda p: order.append(("key", p)) or "fixture-only-key") as read, \
             patch("rethlas_core.provider.http.client.HTTPSConnection", return_value=connection), \
             patch.dict("os.environ", {"HTTPS_PROXY": "http://unused.invalid", "OPENAI_API_KEY": "not-selected"}):
            provider = TextProvider(mode="live", endpoint=ENDPOINT, key_file=Path("/explicit/selected/key"),
                model=MODEL, admit=lambda *a: order.append(("admit", None)))
            read.assert_not_called()
            result = provider.complete(MESSAGES)
        self.assertEqual(order, [("admit", None), ("key", Path("/explicit/selected/key"))])
        self.assertEqual(connection.requests[0][0:2], ("POST", "/v1/chat/completions"))
        self.assertEqual(connection.requests[0][3]["Authorization"], "Bearer fixture-only-key")
        self.assertEqual(len(connection.requests), 1)
        self.assertTrue(connection.closed)
        # TLS is replaced at the boundary in this unit test; no actual online call.
        self.assertEqual(result["model"], MODEL)

    def test_live_bad_key_does_not_open_connection_or_echo_value(self):
        with patch("rethlas_core.provider.read_token", side_effect=ValueError("secret path")), \
             patch("rethlas_core.provider.http.client.HTTPSConnection") as connect:
            provider = TextProvider(mode="live", endpoint=ENDPOINT, key_file="/explicit/key", model=MODEL,
                                    admit=lambda *a: None)
            with self.assertRaises(ProviderConfigurationError) as caught:
                provider.complete(MESSAGES)
            connect.assert_not_called()
            self.assertNotIn("secret", str(caught.exception))

    def test_http_redirect_and_rate_limit_never_follow_or_retry(self):
        for status in (301, 302, 307, 429, 500):
            connection = FakeConnection(FakeResponse(status=status))
            with self.subTest(status=status), \
                 patch("rethlas_core.provider.read_token", return_value="fixture-only-key"), \
                 patch("rethlas_core.provider.http.client.HTTPSConnection", return_value=connection) as connect:
                provider = TextProvider(mode="live", endpoint=ENDPOINT, key_file="/explicit/key", model=MODEL,
                                        admit=lambda *a: None)
                with self.assertRaises(ProviderHTTPError) as caught:
                    provider.complete(MESSAGES)
                self.assertEqual(caught.exception.status_code, status)
                self.assertEqual(len(connection.requests), 1)
                self.assertEqual(connect.call_count, 1)
                self.assertTrue(connection.closed)

    def test_http_timeout_and_detached_socket_keep_deadline_without_retry(self):
        connection = FakeConnection(FakeResponse())
        connection.getresponse = Mock(side_effect=socket.timeout("hidden response"))
        with patch("rethlas_core.provider.read_token", return_value="fixture-only-key"), \
             patch("rethlas_core.provider.http.client.HTTPSConnection", return_value=connection):
            provider = TextProvider(mode="live", endpoint=ENDPOINT, key_file="/explicit/key", model=MODEL,
                                    admit=lambda *a: None)
            with self.assertRaises(ProviderTimeout) as caught:
                provider.complete(MESSAGES)
            self.assertTrue(caught.exception.uncertain)
            self.assertEqual(len(connection.requests), 1)
            self.assertTrue(connection.closed)
        connection = FakeConnection(FakeResponse())
        sock = connection.sock
        def detached():
            connection.sock = None
            return connection.response
        connection.getresponse = detached
        with patch("rethlas_core.provider.read_token", return_value="fixture-only-key"), \
             patch("rethlas_core.provider.http.client.HTTPSConnection", return_value=connection):
            provider = TextProvider(mode="live", endpoint=ENDPOINT, key_file="/explicit/key", model=MODEL,
                                    admit=lambda *a: None)
            provider.complete(MESSAGES)
        self.assertGreaterEqual(sock.settimeout.call_count, 3)
        self.assertTrue(connection.closed)

    def test_http_response_declared_and_streaming_bounds(self):
        for response in (FakeResponse(headers={"Content-Length": str(RESPONSE_BYTES + 1)}),
            FakeResponse(headers={"Content-Encoding": "gzip"}),
            FakeResponse(raw=b"x" * 101)):
            connection = FakeConnection(response)
            with self.subTest(headers=response.headers), \
                 patch("rethlas_core.provider.read_token", return_value="fixture-only-key"), \
                 patch("rethlas_core.provider.http.client.HTTPSConnection", return_value=connection):
                provider = TextProvider(mode="live", endpoint=ENDPOINT, key_file="/explicit/key", model=MODEL,
                                        admit=lambda *a: None, max_response_bytes=100)
                with self.assertRaises(ProviderResponseError):
                    provider.complete(MESSAGES)
                self.assertTrue(connection.closed)


class FakeResponse:
    def __init__(self, *, raw=None, status=200, headers=None):
        self.raw = canonical_json(completion()) if raw is None else raw
        self.offset, self.status = 0, status
        self.headers = headers or {}

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def read1(self, maximum):
        chunk = self.raw[self.offset:self.offset + maximum]
        self.offset += len(chunk)
        return chunk


class FakeConnection:
    def __init__(self, response):
        self.response, self.sock = response, Mock()
        self.requests, self.closed = [], False

    def connect(self):
        pass

    def request(self, method, path, *, body, headers):
        self.requests.append((method, path, body, dict(headers)))

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


if __name__ == "__main__":
    unittest.main()
