"""Fixture protocol tests: not mathematical or production-review acceptance."""
from __future__ import annotations

from copy import deepcopy
import tempfile
import threading
from pathlib import Path
import unittest
from unittest.mock import patch

from rethlas_core.authorization import Authority, AuthorizationError
from rethlas_core.controller import Controller
from rethlas_core.private_fs import PrivateDirectory, digest
from rethlas_core.publication import FixturePublication, PublicationConflict, text_hash
from rethlas_core.worker import WorkerCancelled, WorkerUnavailable


GOOD = {"verification_report": {"summary": "fixture", "critical_errors": [], "gaps": []},
        "verdict": "correct", "repair_hints": ""}
GAP = {"location": "line 1", "issue": "synthetic missing step"}


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = FixturePublication(self.root / "publication", create=True)
        self.statement, self.proof = "Synthetic theorem.", "Synthetic proof."
        self.binding = {"owner": "local", "run_id": "run-1", "domain": "fixture", "epoch": 0,
            "request_id": "request-1", "statement_sha256": text_hash(self.statement),
            "proof_sha256": text_hash(self.proof), "dependencies_sha256": digest([]),
            "config_sha256": digest({"driver": "fixture"})}

    def begin(self):
        return self.store.begin(self.binding, self.statement, self.proof)

    def assert_committed_run_is_sealed(self, result):
        directory = self.root / "publication"
        replacement = deepcopy(GOOD)
        replacement["verification_report"]["summary"] = "replacement fixture report"

        def snapshot():
            return {str(path.relative_to(directory)): path.read_bytes()
                    for path in directory.rglob("*") if path.is_file()}

        before = snapshot()
        operations = (
            ("append", lambda: self.store.append_findings("run-1", [{"gaps": [GAP]}])),
            ("replace receipt", lambda: self.store.complete_fixture("run-1", replacement, self.binding)),
            ("repeat completion", lambda: self.store.complete_fixture("run-1", GOOD, self.binding)),
        )
        for name, operation in operations:
            with self.subTest(operation=name):
                with self.assertRaises(PublicationConflict):
                    operation()
                self.assertEqual(snapshot(), before)
                self.assertEqual(self.store.status("run-1"), result)
        self.assertEqual(self.begin(), result)
        self.assertEqual(snapshot(), before)

    def test_fixture_receipt_is_not_production(self):
        self.begin()
        result = self.store.complete_fixture("run-1", GOOD, self.binding)
        self.assertEqual(result["status"], "fixture_completed")
        self.assertFalse(result["production_completed"])
        self.assertEqual(result["envelope"]["verification_kind"], "fixture")
        self.assertEqual(self.begin(), result)
        reopened = FixturePublication(self.root / "publication")
        self.assertEqual(reopened.status("run-1"), result)
        self.assert_committed_run_is_sealed(result)

    def test_omission_and_malformed_findings_rejected(self):
        self.begin()
        self.store.append_findings("run-1", [{"gaps": [GAP]}])
        with self.assertRaises(ValueError):
            self.store.complete_fixture("run-1", GOOD, self.binding)
        with self.assertRaises(ValueError):
            self.store.append_findings("run-1", [{"gaps": "not an array"}])
        wrong = deepcopy(GOOD)
        wrong["verification_report"]["gaps"] = [GAP]
        wrong.update(verdict="wrong", repair_hints="Repair synthetic gap.")
        result = self.store.complete_fixture("run-1", wrong, self.binding)
        self.assertEqual(result["status"], "fixture_rejected")

    def test_worker_file_deletion_does_not_remove_registration(self):
        self.begin()
        draft = self.root / "worker-findings.jsonl"
        draft.write_text("synthetic worker record")
        self.store.append_findings("run-1", [{"gaps": [GAP]}])
        draft.unlink()
        with self.assertRaises(ValueError):
            self.store.complete_fixture("run-1", GOOD, self.binding)

    def test_all_binding_changes_rejected(self):
        self.begin()
        for key, value in (("proof_sha256", "0" * 64), ("statement_sha256", "1" * 64),
                           ("dependencies_sha256", "2" * 64), ("config_sha256", "3" * 64),
                           ("request_id", "another"), ("run_id", "other"), ("epoch", 1)):
            with self.subTest(key=key), self.assertRaises((ValueError, FileNotFoundError)):
                binding = {**self.binding, key: value}
                self.store.complete_fixture("run-1", GOOD, binding)

    def test_stale_resume_and_forged_old_file(self):
        (self.root / "blueprint_verified.md").write_text("forged")
        self.assertEqual(self.store.status("run-1")["status"], "unreviewed")
        self.begin()
        self.store.complete_fixture("run-1", GOOD, self.binding)
        self.assertEqual(self.store.status("run-1", "modified", self.proof)["status"], "stale")
        self.assertEqual(self.store.status("run-1", self.statement, "modified")["status"], "stale")
        with self.assertRaises(PublicationConflict):
            self.store.begin({**self.binding, "config_sha256": "0" * 64}, self.statement, self.proof)

    def test_frozen_bytes_not_mutable_source(self):
        source = self.root / "draft.md"
        source.write_text(self.proof)
        self.begin()
        source.write_text("replacement")
        result = self.store.complete_fixture("run-1", GOOD, self.binding)
        self.assertEqual(result["envelope"]["binding"]["proof_sha256"], text_hash(self.proof))
        object_path = self.root / "publication/objects" / (text_hash(self.proof) + ".md")
        self.assertEqual(object_path.read_text(), self.proof)

    def test_tamper_receipt_or_frozen_object(self):
        self.begin()
        result = self.store.complete_fixture("run-1", GOOD, self.binding)
        receipt = self.root / "publication/receipts" / (result["receipt"]["id"] + ".json")
        content = receipt.read_bytes()
        receipt.write_bytes(content.replace(b'fixture_completed', b'production_solved'))
        with self.assertRaises(ValueError):
            self.store.status("run-1")
        receipt.write_bytes(content)
        frozen = self.root / "publication/objects" / (text_hash(self.proof) + ".md")
        frozen.write_text("tampered")
        with self.assertRaises(ValueError):
            self.store.status("run-1")

    def test_commit_rechecks_expiry_and_cancellation(self):
        self.begin()
        with self.assertRaises(AuthorizationError):
            self.store.complete_fixture("run-1", GOOD, self.binding,
                before_commit=lambda: (_ for _ in ()).throw(AuthorizationError("expired")))
        self.assertEqual(self.store.status("run-1")["status"], "unreviewed")

    def test_manifest_is_completion_authority_after_interrupted_state_update(self):
        self.begin()
        run = self.store._run("run-1")
        original = run.write_json

        def fail_final_state(name, payload):
            if name == "state.json" and payload["state"] == "completed":
                raise OSError("injected final state write failure")
            return original(name, payload)

        with patch.object(self.store, "_run", return_value=run), patch.object(run, "write_json", side_effect=fail_final_state):
            with self.assertRaises(OSError):
                self.store.complete_fixture("run-1", GOOD, self.binding)
        result = self.store.status("run-1")
        self.assertEqual(result["status"], "fixture_completed")
        self.assertFalse(result["production_completed"])
        self.assertEqual(run.read_json("state.json")["state"], "started")
        self.assert_committed_run_is_sealed(result)

    def test_committed_run_with_missing_state_is_not_recreated(self):
        self.begin()
        result = self.store.complete_fixture("run-1", GOOD, self.binding)
        directory = self.root / "publication"
        state_path = directory / "runs/run-1/state.json"
        state_path.unlink()
        before = {str(path.relative_to(directory)): path.read_bytes()
                  for path in directory.rglob("*") if path.is_file()}
        replacement = deepcopy(GOOD)
        replacement["verification_report"]["summary"] = "replacement fixture report"
        for name, operation in (
            ("begin", self.begin),
            ("begin changed binding", lambda: self.store.begin(
                {**self.binding, "config_sha256": "0" * 64}, self.statement, self.proof)),
            ("append", lambda: self.store.append_findings("run-1", [{"gaps": [GAP]}])),
            ("complete", lambda: self.store.complete_fixture("run-1", replacement, self.binding)),
        ):
            with self.subTest(operation=name):
                expected = (FileNotFoundError, PublicationConflict) if name.startswith("begin") else PublicationConflict
                with self.assertRaises(expected):
                    operation()
                self.assertFalse(state_path.exists())
                self.assertEqual({str(path.relative_to(directory)): path.read_bytes()
                                  for path in directory.rglob("*") if path.is_file()}, before)
        self.assertEqual(self.store.directory.read_json("manifest.json")["runs"]["run-1"], result["receipt"]["id"])
        with self.assertRaises(FileNotFoundError):
            self.store.status("run-1")

    def test_committed_run_with_missing_directory_is_not_recreated(self):
        self.begin()
        result = self.store.complete_fixture("run-1", GOOD, self.binding)
        run_path = self.root / "publication/runs/run-1"
        (run_path / "state.json").unlink()
        run_path.rmdir()
        with self.assertRaises(FileNotFoundError):
            self.begin()
        self.assertFalse(run_path.exists())
        self.assertEqual(self.store.directory.read_json("manifest.json")["runs"]["run-1"], result["receipt"]["id"])

    def test_production_domain_cannot_use_fixture_store(self):
        with self.assertRaises(ValueError):
            self.store.begin({**self.binding, "domain": "production"}, self.statement, self.proof)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        Authority(self.root / "authority", create=True)
        FixturePublication(self.root / "fixture-publications", create=True)
        self.controller = Controller(self.root, driver="fixture")
        self.controller.authority.register_run("local", "run", "fixture")
        self.token = self.controller.authority.issue(owner="local", run_id="run", domain="fixture",
            role="generation", actions=["verify", "status"])
        self.payload = {"owner": "local", "run_id": "run", "domain": "fixture", "request_id": "req",
                        "statement": "Synthetic statement", "proof": "Synthetic proof"}

    def test_real_fixed_process_and_replay(self):
        result = self.controller.verify(self.token, self.payload)
        self.assertEqual(result["status"], "fixture_completed")
        self.assertFalse(result["production_completed"])
        self.assertEqual(self.controller.verify(self.token, self.payload), result)

    def test_wrong_is_valid_business_result(self):
        result = self.controller.verify(self.token, {**self.payload, "proof": "[fixture:wrong]"})
        self.assertEqual(result["verdict"], "wrong")
        self.assertEqual(result["status"], "fixture_rejected")

    def test_unauthorized_no_driver_or_write(self):
        with patch("rethlas_core.controller.run_fixture_process") as driver:
            with self.assertRaises(AuthorizationError):
                self.controller.verify("invalid", self.payload)
            driver.assert_not_called()
        self.assertEqual(list((self.root / "fixture-publications/runs").iterdir()), [])

    def test_live_disabled_without_fallback(self):
        disabled = Controller(self.root, driver="disabled")
        with patch("rethlas_core.controller.run_fixture_process") as driver:
            with self.assertRaises(WorkerUnavailable):
                disabled.verify(self.token, self.payload)
            driver.assert_not_called()

    def test_cancelled_request_does_not_start_or_commit(self):
        cancelled = threading.Event()
        cancelled.set()
        with patch("rethlas_core.controller.run_fixture_process") as driver:
            with self.assertRaises(WorkerCancelled):
                self.controller.verify(self.token, self.payload, cancel_event=cancelled)
            driver.assert_not_called()
        self.assertEqual(list((self.root / "fixture-publications/runs").iterdir()), [])
        cancelled.clear()
        from rethlas_core.private_fs import canonical_json

        def finish_then_cancel(*args, **kwargs):
            cancelled.set()
            return canonical_json({"output": GOOD, "findings": []})

        with patch("rethlas_core.controller.run_fixture_process", side_effect=finish_then_cancel):
            with self.assertRaises(WorkerCancelled):
                self.controller.verify(self.token, self.payload, cancel_event=cancelled)
        self.assertEqual(self.controller.publications.status("run")["status"], "unreviewed")

    def test_current_epoch_and_configuration_rechecked_on_status(self):
        self.controller.verify(self.token, self.payload)
        self.controller.authority.bump_epoch("run")
        replacement = self.controller.authority.issue(owner="local", run_id="run", domain="fixture",
            role="generation", actions=["verify", "status"])
        self.assertEqual(self.controller.status(replacement)["status"], "stale")
        changed = Controller(self.root, driver="disabled")
        self.assertEqual(changed.status(replacement)["status"], "stale")

    def test_no_requested_command_or_driver(self):
        for key in ("command", "driver", "verification_kind", "production_completed"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.controller.verify(self.token, {**self.payload, key: "anything"})


if __name__ == "__main__":
    unittest.main()
