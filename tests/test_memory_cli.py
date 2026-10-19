"""Real offline CLI interactions for M2a; credentials stay in temporary 0600 files.

These cases deliberately invoke the installed CLI in fresh processes, rather than
calling its main() or replacing MemoryStore with mocks. Discovery writes no repo
artifacts. The separate acceptance runner can archive redacted command records.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest


COMMAND_RECORDS: list[dict] = []


def request(head: str, identity: str = "定义/Ω", request_id: str = "cli-request") -> dict:
    provenance = {"source": "synthetic CLI fixture", "actor": "self-reported author",
                  "recorded_at": "2026-10-17T00:00:00Z"}
    body = "# 合成定义 Ω\n\nA CLI durability fixture, not a reviewed theorem.\n"
    return {"version": 1, "request_id": request_id, "expected_head": head,
            "entities": [{"entity_id": identity, "type": "DefinitionContext", "name": "定义 Ω",
                          "provenance": provenance}],
            "revisions": [{"entity_id": identity, "revision_id": identity + "/r1", "body": body,
                           "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
                           "provenance": provenance, "assertion_state": "candidate", "validity": "needs_review",
                           "context_revision_ids": [], "evidence_ids": []}],
            "relations": [], "evidence": []}


class MemoryCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rethlas-memory-cli-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.store = self.work / "private-store"
        self.credentials = self.work / "credentials"
        self.credentials.mkdir(mode=0o700)
        self.token = self.credentials / "full.capability"
        self.run_id = "cli-run"
        self.cli("memory-init", "--store-root", self.store, "--owner", "cli-owner", "--domain", "fixture")
        self.issue(self.token, ["memory.read", "memory.commit", "memory.maintain"])

    def cli(self, *arguments, expected=0):
        command = [sys.executable, "-I", "-B", "-m", "rethlas_core.cli", *map(str, arguments)]
        result = subprocess.run(command, cwd=self.work, stdin=subprocess.DEVNULL, capture_output=True,
                                text=True, timeout=20, env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
        # Commands contain paths, never token values or private-root key data.
        record = {"test": self.id(), "argv": command, "exit_code": result.returncode,
                  "stdout": result.stdout, "stderr": result.stderr}
        COMMAND_RECORDS.append(record)
        self.assertEqual(result.returncode, expected,
                         "CLI exit mismatch: " + json.dumps(record, ensure_ascii=False))
        self.assertNotIn("Traceback", result.stderr)
        return json.loads(result.stdout) if result.stdout.strip() else None

    def issue(self, path, actions):
        return self.cli("memory-issue", "--store-root", self.store, "--run-id", self.run_id,
                        "--token-file", path, "--actions", *actions, "--ttl", "600")

    def operation(self, name, *arguments, token=None, expected=0):
        return self.cli(name, "--store-root", self.store, "--run-id", self.run_id,
                        "--token-file", token or self.token, *arguments, expected=expected)

    def write_request(self, value):
        path = self.work / "request.json"
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def test_fresh_process_commit_export_validate_and_exact_replay(self):
        initial = self.operation("memory-status")
        payload = request(initial["head"])
        path = self.write_request(payload)
        first = self.operation("memory-commit", "--request", path)
        self.assertFalse(first["replayed"])
        exported = self.operation("memory-export")
        self.assertEqual(exported["head"], first["head"])
        self.assertEqual(exported["graph"]["revisions"][0]["body"], payload["revisions"][0]["body"])
        self.assertFalse(exported["audit_only"])
        exchange = self.work / "exchange.json"
        exchange.write_text(json.dumps(exported["graph"], ensure_ascii=False), encoding="utf-8")
        self.assertTrue(self.cli("validate-math", exchange)["valid"])
        replay = self.operation("memory-commit", "--request", path)
        self.assertTrue(replay["replayed"])
        self.assertEqual(replay["receipt"], first["receipt"])
        self.assertEqual(self.operation("memory-status")["counts"]["revisions"], 1)
        self.assertEqual(self.operation("memory-recover")["pending_before"]["status"], "empty")
        self.assertEqual(stat.S_IMODE(self.token.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.store.stat().st_mode), 0o700)
        token = self.token.read_text(encoding="ascii")
        for record in COMMAND_RECORDS:
            self.assertNotIn(token, json.dumps(record))

    def test_cas_and_bound_request_conflicts_have_explicit_exit_three(self):
        head = self.operation("memory-status")["head"]
        first = request(head)
        path = self.write_request(first)
        self.operation("memory-commit", "--request", path)
        other = request(head, "another/identity", "another-request")
        self.assertEqual(self.operation("memory-commit", "--request", self.write_request(other),
                                        expected=3)["status"], "conflict")
        changed = request(self.operation("memory-status")["head"], "different", first["request_id"])
        self.assertEqual(self.operation("memory-commit", "--request", self.write_request(changed),
                                        expected=3)["status"], "conflict")
        self.assertEqual(self.operation("memory-status")["generation"], 1)

    def test_historical_export_is_audit_only_and_orphan_head_is_rejected(self):
        head = self.operation("memory-status")["head"]
        self.operation("memory-commit", "--request", self.write_request(request(head)))
        old = self.operation("memory-export", "--head", head)
        self.assertTrue(old["audit_only"])
        self.assertEqual(old["graph"]["revisions"], [])
        self.operation("memory-export", "--head", "f" * 64, expected=3)
        self.assertEqual(self.operation("memory-status")["generation"], 1)

    def test_projection_can_be_deleted_and_rebuilt_without_changing_authority(self):
        head = self.operation("memory-status")["head"]
        self.operation("memory-commit", "--request", self.write_request(request(head)))
        before = (self.store / "manifest.json").read_bytes()
        first = self.operation("memory-rebuild-projection")
        projection = self.store / "projection/ids.json"
        saved = projection.read_bytes()
        projection.unlink()
        second = self.operation("memory-rebuild-projection")
        self.assertEqual(saved, projection.read_bytes())
        self.assertEqual(first["projection_sha256"], second["projection_sha256"])
        self.assertEqual(before, (self.store / "manifest.json").read_bytes())
        projection.write_text('{"forged":"not authority"}', encoding="utf-8")
        self.assertEqual(self.operation("memory-status")["generation"], 1)
        self.assertEqual(len(self.operation("memory-export")["graph"]["revisions"]), 1)
        self.operation("memory-rebuild-projection")
        self.assertEqual(saved, projection.read_bytes())

    def test_read_credential_cannot_commit_recover_or_rebuild(self):
        read = self.credentials / "read.capability"
        self.issue(read, ["memory.read"])
        head = self.operation("memory-status", token=read)["head"]
        before = (self.store / "manifest.json").read_bytes()
        self.operation("memory-commit", "--request", self.write_request(request(head)), token=read, expected=2)
        self.operation("memory-recover", token=read, expected=2)
        self.operation("memory-rebuild-projection", token=read, expected=2)
        self.assertEqual(before, (self.store / "manifest.json").read_bytes())
        self.assertEqual((self.store / "journal/pending.log").read_bytes(), b"")
        self.assertFalse((self.store / "projection/ids.json").exists())

    def test_bad_json_unknown_fields_oversize_and_unsafe_paths_do_not_commit(self):
        head = self.operation("memory-status")["head"]
        path = self.work / "bad.json"
        cases = ('{"version":1,"version":1}', '{"version":NaN}', '{"version":Infinity}')
        for text in cases:
            with self.subTest(text=text):
                path.write_text(text, encoding="utf-8")
                self.operation("memory-commit", "--request", path, expected=2)
        payload = request(head)
        payload["command"] = "must not run"
        self.operation("memory-commit", "--request", self.write_request(payload), expected=2)
        path.write_bytes(b"x" * (524288 + 1))
        self.operation("memory-commit", "--request", path, expected=2)
        link = self.work / "request-link.json"
        link.symlink_to(path)
        self.operation("memory-commit", "--request", link, expected=2)
        fifo = self.work / "request.fifo"
        os.mkfifo(fifo)
        self.operation("memory-commit", "--request", fifo, expected=2)
        self.assertEqual(self.operation("memory-status")["generation"], 0)

    def test_credential_symlink_hardlink_and_mode_are_rejected(self):
        link = self.credentials / "link.capability"
        link.symlink_to(self.token)
        self.operation("memory-status", token=link, expected=2)
        link.unlink()
        os.link(self.token, link)
        self.operation("memory-status", token=link, expected=2)
        link.unlink()
        self.token.chmod(0o644)
        self.operation("memory-status", expected=2)
        self.token.chmod(0o600)
        self.assertEqual(self.operation("memory-status")["generation"], 0)

    def test_init_and_issuer_refuse_to_overwrite_private_state_or_credentials(self):
        before = (self.store / "store.json").read_bytes()
        self.cli("memory-init", "--store-root", self.store, "--owner", "cli-owner", "--domain", "fixture", expected=2)
        credential = self.token.read_bytes()
        self.cli("memory-issue", "--store-root", self.store, "--run-id", self.run_id,
                 "--token-file", self.token, "--actions", "memory.read", expected=2)
        self.assertEqual(credential, self.token.read_bytes())
        self.assertEqual(before, (self.store / "store.json").read_bytes())

    def test_import_is_isolated_pure_and_cli_has_no_failpoint_option(self):
        code = """
import pathlib, sys
import rethlas_core.memory_store
assert sys.flags.isolated == 1
assert not pathlib.Path(sys.argv[1]).exists()
assert 'rethlas_core.api' not in sys.modules
assert 'rethlas_core.controller' not in sys.modules
"""
        private = self.work / "must-not-create"
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(private)],
                                cwd=self.work, env={"PATH": "/usr/bin:/bin"}, capture_output=True,
                                text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(private.exists())
        for command in ("memory-commit", "memory-recover"):
            help_result = subprocess.run([sys.executable, "-I", "-B", "-m", "rethlas_core.cli", command, "--help"],
                                         cwd=self.work, env={"PATH": "/usr/bin:/bin"}, capture_output=True,
                                         text=True, timeout=20)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertNotIn("failpoint", help_result.stdout)
            self.assertNotIn("checkpoint", help_result.stdout)
