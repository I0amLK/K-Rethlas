"""Real offline migration subprocesses with synthetic, temporary 0600 credentials.

The package must already be installed in the selected interpreter. No import-path
injection, provider, failpoint, or project-data discovery is used. The acceptance
runner may archive COMMAND_RECORDS independently of the M2a command ledger.
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


REPOSITORY = Path(__file__).absolute().parents[1]
WORKSPACE_PARENT = REPOSITORY.parent
MIGRATION_TOOL = REPOSITORY / "tools" / "migrate_memory.py"
ENVIRONMENT = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}
COMMAND_RECORDS: list[dict] = []


def json_bytes(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


class LegacyMigrationCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="rethlas-legacy-cli-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.store = self.work / "fixture-store"
        self.credentials = self.work / "credentials"
        self.credentials.mkdir(mode=0o700)
        self.sources = self.work / "sources"
        self.sources.mkdir(mode=0o700)
        self.source = self.sources / "legacy.jsonl"
        self.spec = self.work / "spec.json"
        self.bundle = self.work / "frozen-bundle"
        self.token = self.credentials / "full.capability"
        self.run_id = "legacy-cli-run"
        self.token_values: list[str] = []
        self.body_marker = "synthetic-private-body-CLI-Ω"
        self.legacy_id = "synthetic-old-step-id-CLI"
        self.row = {"timestamp_utc": "2026-10-17T00:00:00Z", "channel": "proof_steps",
                    "record": {"id": self.legacy_id, "statement": self.body_marker, "status": "solved"}}
        self.raw = json_bytes(self.row) + b"\n"
        self.write_source(self.raw)
        self._memory("memory-init", "--store-root", self.store,
                     "--owner", "synthetic-cli-owner", "--domain", "fixture")
        self.issue(self.token, ["memory.read", "memory.commit", "memory.maintain"])

    def write_source(self, raw: bytes, *, allow_quarantine=False):
        self.source.write_bytes(raw)
        self.spec_value = {
            "version": "legacy-jsonl-1", "source_namespace": "synthetic-cli-fixture",
            "import_observed_at": "2026-10-17T00:00:00Z",
            "options": {"allow_quarantine": allow_quarantine},
            "sources": [{"alias": "main", "profile": "generation-10", "channel": "proof_steps",
                         "scope": {"problem_id": "合成题 Ω", "branch_id": "fixture-branch",
                                   "source_run_id": None},
                         "path": str(self.source), "expected_sha256": hashlib.sha256(raw).hexdigest()}],
        }
        self.spec.write_bytes(json_bytes(self.spec_value))

    def _record_text(self, value: str) -> str:
        for token in self.token_values:
            value = value.replace(token, "[redacted synthetic credential]")
        return value

    def _run(self, command, *, expected=0, migration=False, export=False):
        command = list(map(str, command))
        self.assertFalse(any(token in argument for token in self.token_values for argument in command),
                         "a synthetic credential was put in argv")
        result = subprocess.run(command, cwd=WORKSPACE_PARENT, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=20, env=ENVIRONMENT)
        # Export is intentionally a body-bearing operation, not migration CLI
        # output. Its evidence record retains only public metadata and counts.
        recorded_stdout = result.stdout
        if export and result.returncode == 0:
            exported = json.loads(result.stdout)
            recorded_stdout = json_bytes({
                "head": exported["head"], "generation": exported["generation"],
                "audit_only": exported["audit_only"], "graph_redacted": True,
                "counts": {key: len(exported["graph"][key])
                           for key in ("entities", "revisions", "relations", "evidence")},
            }).decode("utf-8") + "\n"
        record = {"test": self.id(), "argv": command, "exit_code": result.returncode,
                  "stdout": self._record_text(recorded_stdout),
                  "stderr": self._record_text(result.stderr)}
        COMMAND_RECORDS.append(record)
        self.assertEqual(result.returncode, expected,
                         "CLI exit mismatch: " + json.dumps(record, ensure_ascii=False))
        self.assertNotIn("Traceback", result.stderr)
        for token in self.token_values:
            self.assertFalse(token in result.stdout or token in result.stderr,
                             "a synthetic credential appeared in process output")
        value = json.loads(result.stdout) if result.stdout.strip() else None
        if migration:
            self.assertEqual(result.stderr, "")
            self.assertEqual(result.stdout, json_bytes(value).decode("utf-8") + "\n")
            for private in (self.body_marker, self.legacy_id, str(self.work)):
                self.assertFalse(private in result.stdout, "migration output disclosed fixture input")
            if expected:
                self.assertEqual(value["status"], {2: "rejected", 3: "conflict",
                                                    4: "indeterminate", 5: "cancelled"}[expected])
                self.assertLessEqual(set(value), {"status", "error", "next"})
                self.assertRegex(value["error"], r"^[A-Za-z][A-Za-z0-9_]*$")
        return value

    def _memory(self, *arguments, expected=0):
        return self._run([sys.executable, "-I", "-B", "-m", "rethlas_core.cli", *arguments],
                         expected=expected, export=bool(arguments and arguments[0] == "memory-export"))

    def migration(self, *arguments, expected=0):
        return self._run([sys.executable, "-I", "-B", MIGRATION_TOOL, *arguments],
                         expected=expected, migration=True)

    def issue(self, path, actions, *, store=None, run_id=None):
        result = self._memory("memory-issue", "--store-root", store or self.store,
                              "--token-file", path, "--run-id", run_id or self.run_id,
                              "--actions", *actions, "--ttl", "600")
        # This reads only the synthetic credential just created by this test.
        self.token_values.append(Path(path).read_text(encoding="ascii"))
        self.assertEqual(stat.S_IMODE(Path(path).stat().st_mode), 0o600)
        return result

    def operation(self, command, *arguments, token=None, store=None, run_id=None, expected=0):
        return self._memory(command, "--store-root", store or self.store,
                            "--token-file", token or self.token, "--run-id", run_id or self.run_id,
                            *arguments, expected=expected)

    def prepare(self, *, token=None, bundle=None, expected=0):
        return self.migration("prepare", "--spec", self.spec, "--bundle", bundle or self.bundle,
                              "--store-root", self.store, "--token-file", token or self.token,
                              "--run-id", self.run_id, expected=expected)

    def apply(self, pin, *, token=None, bundle=None, store=None, run_id=None, expected=0):
        return self.migration("apply", "--bundle", bundle or self.bundle,
                              "--expected-bundle-sha256", pin, "--store-root", store or self.store,
                              "--token-file", token or self.token, "--run-id", run_id or self.run_id,
                              expected=expected)

    def tree(self, root=None):
        root = self.work if root is None else root
        result = {}
        for path in (root, *sorted(root.rglob("*"))):
            info = path.lstat()
            result[str(path.relative_to(root))] = (
                stat.S_IFMT(info.st_mode), stat.S_IMODE(info.st_mode), info.st_ino,
                info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
                hashlib.sha256(path.read_bytes()).hexdigest() if stat.S_ISREG(info.st_mode) else None,
            )
        return result

    def assert_export(self, expected_body):
        exported = self.operation("memory-export")
        graph = exported["graph"]
        self.assertFalse(exported["audit_only"])
        self.assertEqual(len(graph["entities"]), 1)
        self.assertEqual(len(graph["revisions"]), 1)
        self.assertEqual(len(graph["evidence"]), 2)
        self.assertEqual(graph["revisions"][0]["body"], expected_body)
        self.assertEqual(graph["revisions"][0]["assertion_state"], "candidate")
        self.assertEqual(graph["revisions"][0]["validity"], "needs_review")
        self.assertEqual(graph["revisions"][0]["body_sha256"],
                         hashlib.sha256(expected_body.encode("utf-8")).hexdigest())
        return exported

    def test_default_and_explicit_dry_run_are_zero_write_summary_only(self):
        before = self.tree()
        default = self.migration("--spec", self.spec)
        explicit = self.migration("dry-run", "--spec", self.spec)
        self.assertEqual(default, explicit)
        self.assertEqual(default["status"], "dry-run")
        self.assertEqual(default["counts"]["source_files"], 1)
        self.assertEqual(default["counts"]["accepted_lines"], 1)
        self.assertFalse(default["target_checked"])
        self.assertFalse(default["requires_allow_quarantine"])
        self.assertFalse(self.bundle.exists())
        self.assertEqual(before, self.tree())

    def test_dry_run_does_not_need_store_or_credential_access(self):
        # The normal synthetic credential/store paths are absent for dry-run.
        unavailable_token = self.credentials / "unavailable.capability"
        unavailable_store = self.work / "unavailable-store"
        self.token.rename(unavailable_token)
        self.store.rename(unavailable_store)
        try:
            before = self.tree()
            result = self.migration("--spec", self.spec)
            self.assertEqual(result["status"], "dry-run")
            self.assertFalse(self.token.exists())
            self.assertFalse(self.store.exists())
            self.assertEqual(before, self.tree())
        finally:
            unavailable_token.rename(self.token)
            unavailable_store.rename(self.store)

    def test_prepare_apply_export_exact_replay_and_moved_bundle(self):
        before = self.tree(self.store)
        prepared = self.prepare()
        self.assertEqual(prepared["status"], "prepared")
        self.assertRegex(prepared["bundle_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(prepared["snapshot_sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(prepared["request_id"])
        self.assertEqual(prepared["counts"]["accepted_lines"], 1)
        self.assertEqual(before, self.tree(self.store))
        self.assertEqual(stat.S_IMODE(self.bundle.stat().st_mode), 0o700)
        for member in self.bundle.iterdir():
            self.assertEqual(stat.S_IMODE(member.stat().st_mode), 0o600)
        pin = prepared["bundle_sha256"]
        committed = self.apply(pin)
        self.assertEqual(committed["status"], "committed")
        self.assertFalse(committed["replayed"])
        self.assertEqual(committed["bundle_sha256"], pin)
        self.assertEqual(self.assert_export(self.raw.decode("utf-8"))["head"], committed["head"])
        replayed = self.apply(pin)
        self.assertTrue(replayed["replayed"])
        self.assertEqual(replayed["receipt"], committed["receipt"])
        moved = self.work / "moved-frozen-bundle"
        self.bundle.rename(moved)
        after_move = self.apply(pin, bundle=moved)
        self.assertTrue(after_move["replayed"])
        self.assertEqual(after_move["head"], committed["head"])
        self.assertEqual(after_move["receipt"], committed["receipt"])
        self.assertEqual(self.operation("memory-status")["generation"], 1)

    def test_apply_uses_frozen_bytes_after_source_and_spec_change(self):
        pin = self.prepare()["bundle_sha256"]
        changed = {**self.row, "record": {"id": "changed", "statement": "must not be recaptured"}}
        self.source.write_bytes(json_bytes(changed) + b"\n")
        self.spec.write_bytes(b"not JSON and no longer a migration spec")
        self.apply(pin)
        self.assert_export(self.raw.decode("utf-8"))
        self.source.unlink()
        self.spec.unlink()
        self.assertTrue(self.apply(pin)["replayed"])

    def test_required_pin_and_wrong_pin_reject_without_store_writes(self):
        pin = self.prepare()["bundle_sha256"]
        before = self.tree(self.store)
        for bad in ("0" * 64 if pin != "0" * 64 else "1" * 64, "not-a-sha256"):
            with self.subTest(pin_shape=len(bad)):
                self.assertEqual(self.apply(bad, expected=2)["error"], "MigrationError")
        missing = self.migration("apply", "--bundle", self.bundle, "--store-root", self.store,
                                 "--token-file", self.token, "--run-id", self.run_id, expected=2)
        self.assertEqual(missing["error"], "ValueError")
        self.assertEqual(before, self.tree(self.store))
        self.apply(pin)

    def test_tampered_raw_request_manifest_and_unknown_member_reject(self):
        pin = self.prepare()["bundle_sha256"]
        before = self.tree(self.store)
        for name in ("raw-main.jsonl", "request.json", "manifest.json"):
            with self.subTest(member=name):
                member = self.bundle / name
                saved = member.read_bytes()
                try:
                    member.write_bytes(saved + b" ")
                    self.assertEqual(self.apply(pin, expected=2)["error"], "MigrationError")
                    self.assertEqual(before, self.tree(self.store))
                finally:
                    member.write_bytes(saved)
        extra = self.bundle / "untrusted.json"
        extra.write_bytes(b"{}")
        extra.chmod(0o600)
        self.assertEqual(self.apply(pin, expected=2)["error"], "MigrationError")
        extra.unlink()
        self.assertEqual(before, self.tree(self.store))
        self.apply(pin)

    def test_wrong_run_and_store_cannot_retarget_the_frozen_request(self):
        pin = self.prepare()["bundle_sha256"]
        self.assertEqual(self.apply(pin, run_id="other-run", expected=2)["error"], "AuthorizationError")
        other_run_token = self.credentials / "other-run.capability"
        self.issue(other_run_token, ["memory.commit"], run_id="other-run")
        self.assertEqual(self.apply(pin, token=other_run_token, run_id="other-run", expected=3)["error"],
                         "MemoryConflict")
        other_store = self.work / "other-fixture-store"
        self._memory("memory-init", "--store-root", other_store,
                     "--owner", "synthetic-cli-owner", "--domain", "fixture")
        other_token = self.credentials / "other-store.capability"
        self.issue(other_token, ["memory.read", "memory.commit"], store=other_store)
        before = self.tree(other_store)
        self.assertEqual(self.apply(pin, store=other_store, token=other_token, expected=3)["error"],
                         "MemoryConflict")
        self.assertEqual(before, self.tree(other_store))
        self.assertEqual(self.operation("memory-status")["generation"], 0)
        self.apply(pin)

    def test_read_only_can_prepare_but_cannot_apply(self):
        read = self.credentials / "read.capability"
        self.issue(read, ["memory.read"])
        pin = self.prepare(token=read)["bundle_sha256"]
        before = self.tree(self.store)
        self.assertEqual(self.apply(pin, token=read, expected=2)["error"], "AuthorizationError")
        self.assertEqual(before, self.tree(self.store))
        self.apply(pin)

    def test_commit_only_apply_and_replay_do_not_require_read_permission(self):
        pin = self.prepare()["bundle_sha256"]
        commit = self.credentials / "commit.capability"
        self.issue(commit, ["memory.commit"])
        self.operation("memory-status", token=commit, expected=2)
        self.operation("memory-export", token=commit, expected=2)
        first = self.apply(pin, token=commit)
        second = self.apply(pin, token=commit)
        self.assertFalse(first["replayed"])
        self.assertTrue(second["replayed"])
        self.assertEqual(first["receipt"], second["receipt"])
        self.assert_export(self.raw.decode("utf-8"))

    def test_unsafe_source_paths_and_oversize_are_rejected_without_writes(self):
        before = self.tree(self.store)
        symlink = self.sources / "link.jsonl"
        symlink.symlink_to(self.source)
        hardlink = self.sources / "hardlink.jsonl"
        os.link(self.source, hardlink)
        try:
            for path in (symlink, hardlink):
                with self.subTest(kind=path.name):
                    value = json.loads(json.dumps(self.spec_value))
                    value["sources"][0]["path"] = str(path)
                    self.spec.write_bytes(json_bytes(value))
                    self.migration("--spec", self.spec, expected=2)
        finally:
            hardlink.unlink()
            symlink.unlink()
        ancestor = self.work / "source-link"
        ancestor.symlink_to(self.sources, target_is_directory=True)
        fifo = self.sources / "source.fifo.jsonl"
        os.mkfifo(fifo)
        directory = self.sources / "directory.jsonl"
        directory.mkdir(mode=0o700)
        traversal = str(self.sources / ".." / self.sources.name / self.source.name)
        for path in (ancestor / self.source.name, fifo, directory, traversal, "relative.jsonl"):
            with self.subTest(path_kind=type(path).__name__):
                value = json.loads(json.dumps(self.spec_value))
                value["sources"][0]["path"] = str(path)
                self.spec.write_bytes(json_bytes(value))
                self.migration("--spec", self.spec, expected=2)
        self.write_source(b"x" * (1048576 + 1))
        self.migration("--spec", self.spec, expected=2)
        self.assertEqual(before, self.tree(self.store))
        self.assertFalse(self.bundle.exists())

    def test_unsafe_spec_symlink_hardlink_fifo_and_oversize_reject(self):
        before = self.tree(self.store)
        link = self.work / "linked-spec.json"
        link.symlink_to(self.spec)
        self.migration("--spec", link, expected=2)
        link.unlink()
        os.link(self.spec, link)
        self.migration("--spec", link, expected=2)
        link.unlink()
        fifo = self.work / "spec.fifo"
        os.mkfifo(fifo)
        self.migration("--spec", fifo, expected=2)
        self.spec.write_bytes(b"x" * (65536 + 1))
        self.migration("--spec", self.spec, expected=2)
        self.assertEqual(before, self.tree(self.store))
        self.assertFalse(self.bundle.exists())

    def test_unsafe_credentials_reject_without_committing_or_disclosing(self):
        pin = self.prepare()["bundle_sha256"]
        before = self.tree(self.store)
        link = self.credentials / "linked.capability"
        link.symlink_to(self.token)
        self.apply(pin, token=link, expected=2)
        link.unlink()
        os.link(self.token, link)
        self.apply(pin, token=link, expected=2)
        link.unlink()
        self.token.chmod(0o644)
        try:
            self.apply(pin, expected=2)
        finally:
            self.token.chmod(0o600)
        self.credentials.chmod(0o755)
        try:
            self.apply(pin, expected=2)
        finally:
            self.credentials.chmod(0o700)
        parent_link = self.work / "credential-parent-link"
        parent_link.symlink_to(self.credentials, target_is_directory=True)
        self.apply(pin, token=parent_link / self.token.name, expected=2)
        fifo = self.credentials / "fifo.capability"
        os.mkfifo(fifo, mode=0o600)
        self.apply(pin, token=fifo, expected=2)
        invalid = self.credentials / "invalid.capability"
        for raw in (b"", b"credential with whitespace", b"\xff", b"x" * (16384 + 1)):
            with self.subTest(credential_bytes=len(raw)):
                invalid.write_bytes(raw)
                invalid.chmod(0o600)
                self.apply(pin, token=invalid, expected=2)
        self.assertEqual(before, self.tree(self.store))
        self.apply(pin)

    def test_bundle_cannot_overlap_credentials_sources_or_store(self):
        before = self.tree(self.store)
        for protected in (self.credentials, self.sources, self.store):
            with self.subTest(protected=protected.name):
                bundle = protected / "must-not-create-bundle"
                result = self.prepare(bundle=bundle, expected=2)
                self.assertEqual(result["error"], "MigrationError")
                self.assertFalse(bundle.exists())
        self.assertEqual(before, self.tree(self.store))

    def test_bad_lines_default_reject_and_explicit_boolean_allow_quarantine(self):
        raw = self.raw + b'{"record":}\n\n'
        self.write_source(raw)
        inspected = self.migration("--spec", self.spec)
        self.assertTrue(inspected["requires_allow_quarantine"])
        self.assertEqual(inspected["counts"]["accepted_lines"], 1)
        self.assertEqual(inspected["counts"]["quarantined_lines"], 2)
        self.assertEqual(inspected["counts"]["reasons"]["invalid_json"], 1)
        self.assertEqual(inspected["counts"]["reasons"]["blank"], 1)
        before = self.tree(self.store)
        self.assertEqual(self.prepare(expected=2)["error"], "MigrationQuarantine")
        self.assertFalse(self.bundle.exists())
        self.write_source(raw, allow_quarantine=1)
        self.prepare(expected=2)
        self.assertFalse(self.bundle.exists())
        self.write_source(raw, allow_quarantine=True)
        prepared = self.prepare()
        self.assertEqual(prepared["counts"]["quarantined_lines"], 2)
        self.assertEqual((self.bundle / "raw-main.jsonl").read_bytes(), raw)
        self.assertEqual(before, self.tree(self.store))
        self.apply(prepared["bundle_sha256"])
        self.assert_export(self.raw.decode("utf-8"))

    def test_zero_accepted_lines_are_no_op_without_bundle_or_commit(self):
        for raw, allowed in ((b"", False), (b"\n", True)):
            with self.subTest(raw_bytes=len(raw)):
                self.write_source(raw, allow_quarantine=allowed)
                before = self.tree()
                inspected = self.migration("--spec", self.spec)
                self.assertEqual(inspected["counts"]["accepted_lines"], 0)
                result = self.prepare()
                self.assertEqual(result["status"], "no-op")
                self.assertNotIn("bundle_sha256", result)
                self.assertFalse(self.bundle.exists())
                self.assertEqual(before, self.tree())
        self.assertEqual(self.operation("memory-status")["generation"], 0)

    def test_competing_frozen_requests_conflict_without_rebase_or_recovery(self):
        first_pin = self.prepare()["bundle_sha256"]
        second_bundle = self.work / "second-frozen-bundle"
        self.write_source(json_bytes({**self.row, "record": {"id": "second", "statement": "synthetic two"}})
                          + b"\n")
        second_pin = self.prepare(bundle=second_bundle)["bundle_sha256"]
        first = self.apply(first_pin)
        before = self.tree(self.store)
        self.assertEqual(self.apply(second_pin, bundle=second_bundle, expected=3)["error"], "MemoryConflict")
        self.assertEqual(before, self.tree(self.store))
        self.assertEqual(self.operation("memory-status")["head"], first["head"])
        self.assertTrue(self.apply(first_pin)["replayed"])
        third_bundle = self.work / "must-not-rebase-preparation"
        self.assertEqual(self.prepare(bundle=third_bundle, expected=3)["error"], "MemoryConflict")
        self.assertFalse(third_bundle.exists())

    def test_cli_has_no_failpoint_or_checkpoint_and_unknown_options_are_sanitized(self):
        for command in ((), ("dry-run",), ("prepare",), ("apply",)):
            help_result = subprocess.run([sys.executable, "-I", "-B", str(MIGRATION_TOOL),
                                          *command, "--help"], cwd=WORKSPACE_PARENT,
                                         stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                         timeout=20, env=ENVIRONMENT)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertNotIn("failpoint", help_result.stdout)
            self.assertNotIn("checkpoint", help_result.stdout)
            self.assertNotIn("--token ", help_result.stdout)
        before = self.tree()
        result = self.migration("dry-run", "--spec", self.spec,
                                "--failpoint", "before-manifest", expected=2)
        self.assertEqual(result, {"status": "rejected", "error": "ValueError"})
        self.assertEqual(before, self.tree())
        records = json.dumps(COMMAND_RECORDS, ensure_ascii=False)
        for token in self.token_values:
            self.assertFalse(token in records, "a synthetic credential appeared in command records")
