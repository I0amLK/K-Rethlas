"""Actual loopback HTTP acceptance for a fixed synthetic driver (no model calls)."""
from __future__ import annotations

import hashlib
import http.client
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import httpx

from rethlas_core.authorization import Authority
from rethlas_core.private_fs import PrivateDirectory
from rethlas_core.publication import FixturePublication


class ActualHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.authority = Authority(self.root / "authority", create=True)
        FixturePublication(self.root / "fixture-publications", create=True)
        self.authority.register_run("local", "run", "fixture")
        self.token = self.authority.issue(owner="local", run_id="run", domain="fixture",
            role="generation", actions=["verify", "status"])
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.log = self.root / "service.log"
        self.handle = self.log.open("wb")
        code = ("from pathlib import Path; from rethlas_core.api import create_app; import uvicorn; "
                f"uvicorn.run(create_app(Path({str(self.root)!r}),driver='fixture',timeout=0.5),"
                f"host='127.0.0.1',port={self.port},access_log=False,log_level='warning')")
        self.process = subprocess.Popen([sys.executable, "-I", "-c", code], cwd="/",
            stdout=self.handle, stderr=subprocess.STDOUT, start_new_session=True)
        self.addCleanup(self.stop)
        self.client = httpx.Client(base_url=self.url, trust_env=False, timeout=5, follow_redirects=False)
        self.addCleanup(self.client.close)
        for _ in range(100):
            if self.process.poll() is not None:
                self.fail("service exited before readiness")
            try:
                if self.client.get("/health").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.03)
        else:
            self.fail("service readiness timeout")
        self.headers = {"Authorization": "Bearer " + self.token}
        self.payload = {"owner": "local", "run_id": "run", "domain": "fixture", "request_id": "req",
                        "statement": "Synthetic private statement sentinel 24016",
                        "proof": "Synthetic private proof sentinel 24016"}

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.handle.close()

    def assert_no_run(self):
        self.assertEqual(list((self.root / "fixture-publications/runs").iterdir()), [])

    def test_authentication_and_scope_before_worker(self):
        self.assertEqual(self.client.post("/verify", json=self.payload).status_code, 401)
        self.assertEqual(self.client.post("/verify", headers={"Authorization": "Bearer invalid"},
                                        json=self.payload).status_code, 403)
        self.assertEqual(self.client.post("/verify", headers=self.headers,
                                        json={**self.payload, "owner": "other"}).status_code, 403)
        self.assert_no_run()

    def test_real_success_status_and_private_logs(self):
        result = self.client.post("/verify", headers=self.headers, json=self.payload)
        self.assertEqual(result.status_code, 200, result.text)
        body = result.json()
        self.assertEqual(body["status"], "fixture_completed")
        self.assertFalse(body["production_completed"])
        self.assertEqual(body["envelope"]["binding"]["proof_sha256"],
                         hashlib.sha256(self.payload["proof"].encode()).hexdigest())
        status = self.client.post("/status", headers=self.headers,
                                  json={"statement": self.payload["statement"], "proof": "changed"})
        self.assertEqual(status.json()["status"], "stale")
        log = self.log.read_text()
        for secret in (self.token, self.payload["statement"], self.payload["proof"]):
            self.assertNotIn(secret, log)

    def test_body_limits_complete_and_chunked(self):
        response = self.client.post("/verify", headers=self.headers, content=b"x" * 140000)
        self.assertEqual(response.status_code, 413)
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        try:
            connection.request("POST", "/verify", body=iter([b"x" * 70000, b"y" * 70000]),
                headers=self.headers, encode_chunked=True)
            self.assertEqual(connection.getresponse().status, 413)
        finally:
            connection.close()
        self.assert_no_run()

    def test_duplicate_keys_invalid_constants_and_deep_json(self):
        for payload in (b'{"owner":"local","owner":"other"}', b'{"x":NaN}',
                        b"[" * 2000 + b"0" + b"]" * 2000):
            with self.subTest(payload=payload[:30]):
                response = self.client.post("/verify", headers=self.headers, content=payload)
                self.assertEqual(response.status_code, 422)
        self.assert_no_run()

    def test_no_arbitrary_commands_or_mode_upgrade(self):
        for field in ("command", "driver", "verification_kind", "production_completed"):
            response = self.client.post("/verify", headers=self.headers,
                                        json={**self.payload, field: "injected"})
            self.assertEqual(response.status_code, 422)
        self.assert_no_run()

    def test_disconnect_cancels_request_without_receipt(self):
        body = json.dumps({**self.payload, "proof": "[fixture:timeout]"}).encode()
        with socket.create_connection(("127.0.0.1", self.port), timeout=3) as connection:
            connection.sendall((f"POST /verify HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                f"Authorization: Bearer {self.token}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n\r\n").encode() + body)
            run = self.root / "fixture-publications/runs/run/state.json"
            for _ in range(100):
                if run.exists():
                    break
                time.sleep(0.005)
            self.assertTrue(run.exists())
        # The disconnect is handled promptly, not by waiting for the fixture's
        # 30-second sleep. State remains uncommitted; no success is replayable.
        deadline = time.monotonic() + 0.4
        while True:
            response = self.client.post("/status", headers=self.headers, json={})
            if response.status_code != 429 or time.monotonic() >= deadline:
                break
            time.sleep(0.005)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "unreviewed")
        manifest = json.loads((self.root / "fixture-publications/manifest.json").read_text())
        self.assertEqual(manifest["runs"], {})

    def test_concurrency_limit_timeout_and_health(self):
        payload = {**self.payload, "proof": "[fixture:timeout]"}
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self.client.post, "/verify", headers=self.headers, json=payload)
            # Wait for a real private run allocation, not a mocked worker signal.
            for _ in range(100):
                if list((self.root / "fixture-publications/runs").iterdir()):
                    break
                time.sleep(0.005)
            response = self.client.post("/verify", headers=self.headers, json=self.payload)
            self.assertEqual(response.status_code, 429)
            self.assertEqual(self.client.get("/health").status_code, 200)
            self.assertEqual(future.result().status_code, 504)


if __name__ == "__main__":
    unittest.main()
