"""Synthetic-only raw-log audit regressions; never run a provider or acceptance command."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


TOOL = Path(__file__).absolute().parents[1] / "tools/audit_acceptance_logs.py"
SPEC = importlib.util.spec_from_file_location("acceptance_raw_log_audit", TOOL)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class DiagnosticClassificationTests(unittest.TestCase):
    def test_python_warning_classes_and_plain_warning_are_counted(self):
        for category in ("Warning", "ResourceWarning", "DeprecationWarning", "UserWarning", "SyntheticWarning"):
            with self.subTest(category=category):
                value = audit.audit_text("synthetic.py:12: " + category + ": synthetic diagnostic\n")
                self.assertEqual(value["diagnostic_events"], 1)
                self.assertEqual(value["category_counts"]["python_warning"], 1)

    def test_logging_warning_and_warn_formats_are_not_legacy_false_negatives(self):
        for text in ("WARNING:asyncio:synthetic\n", "2026-10-18 WARNING synthetic\n",
                     "[10/18/26 12:17:35] WARNING synthetic base_events.py:1982\n",
                     "                    WARNING synthetic\n", "[WARN] synthetic\n",
                     '{"level":"warning","message":"synthetic"}\n',
                     '{"levelname":"WARNING","message":"synthetic"}\n',
                     '{"severity":"WARN","message":"synthetic"}\n'):
            with self.subTest(text=text):
                value = audit.audit_text(text)
                self.assertEqual(value["legacy_warning_lines"], 0)
                self.assertEqual(value["diagnostic_events"], 1)
                self.assertEqual(value["category_counts"]["logging_warning"], 1)

    def test_unprefixed_asyncio_diagnostic_blocks_even_with_unittest_ok(self):
        text = "Executing <Task finished name='synthetic'> took 0.371 seconds\nRan 1 test in 0.5s\n\nOK\n"
        value = audit.audit_text(text)
        self.assertEqual(value["legacy_warning_lines"], 0)
        self.assertEqual(value["diagnostic_events"], 1)
        self.assertEqual(value["events"][0]["kinds"], ["asyncio_slow_callback"])

    def test_rich_folding_is_one_event_not_continuation_rows(self):
        text = ("[10/18/26 12:17:35] WARNING Executing <Task pending base_events.py:1982\n"
                "                             name='synthetic'\n"
                "                             coro=<MessagePump._process_messages()\n"
                "                             took 0.203 seconds\n")
        value = audit.audit_text(text)
        self.assertEqual(value["physical_lines"], 4)
        self.assertEqual(value["diagnostic_events"], 1)
        self.assertEqual(value["category_counts"]["logging_warning"], 1)
        self.assertEqual(value["category_counts"]["asyncio_slow_callback"], 1)
        self.assertTrue(value["category_counts_overlap"])

    def test_ansi_headers_and_handle_variants_are_detected_without_emitting_text(self):
        for kind in ("Task", "Handle", "TimerHandle"):
            with self.subTest(kind=kind):
                text = "\x1b[33mWARNING\x1b[0m Executing <" + kind + " synthetic-private-value> took 0.201 seconds\n"
                value = audit.audit_text(text)
                self.assertEqual(value["diagnostic_events"], 1)
                self.assertEqual(value["events"][0]["header_line"], 1)
                self.assertNotIn("synthetic-private-value", json.dumps(value))
                self.assertFalse(value["raw_text_included"])

    def test_clean_test_names_info_and_warning_configuration_are_not_diagnostics(self):
        text = ('test_warning_or_skip_prevents_execute_success ... ok\n'
                'INFO warning_lines=0; NotAWarningClass\n'
                '{"warning_lines":0,"result":"passed"}\n'
                'Ran 1 test in 0.001s\n\nOK\n')
        self.assertEqual(audit.audit_text(text)["diagnostic_events"], 0)


class EvidenceAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="raw-log-audit-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.evidence = self.root / "evidence"
        self.evidence.mkdir()
        self.report = {"result": "passed", "exit_code": 0, "steps": [{
            "name": "synthetic-step", "result": "passed", "exit_code": 0,
            "failures": 0, "errors": 0, "skipped": 0, "warning_lines": 0,
            "exception": None, "cleanup_errors": [],
            "raw_stdout": "step.stdout.log", "raw_stderr": "step.stderr.log"}]}
        (self.evidence / "step.stdout.log").write_text("", encoding="utf-8")
        (self.evidence / "step.stderr.log").write_text("Ran 1 test in 0.001s\n\nOK\n", encoding="utf-8")
        self.save_report()

    def save_report(self):
        (self.evidence / "report.json").write_text(json.dumps(self.report), encoding="utf-8")

    def test_recorded_pass_with_logging_warning_is_rejected_without_mutating_inputs(self):
        (self.evidence / "step.stderr.log").write_text("WARNING synthetic\nRan 1 test in 0.001s\nOK\n", encoding="utf-8")
        before = {path.name: path.read_bytes() for path in self.evidence.iterdir()}
        value = audit.audit_directory(self.evidence)
        self.assertEqual(value["recorded_result"], "passed")
        self.assertEqual(value["result"], "diagnostics_found")
        self.assertEqual(value["exit_code"], 1)
        self.assertEqual(value["diagnostic_events"], 1)
        self.assertEqual(value["warning_exemptions"], [])
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.evidence.iterdir()})

    def test_clean_raw_output_does_not_rewrite_recorded_failure_or_cleanup(self):
        self.assertEqual(audit.audit_directory(self.evidence)["result"], "no_diagnostics_found")
        self.report["steps"][0]["cleanup_errors"] = ["synthetic_cleanup_error"]
        self.save_report()
        value = audit.audit_directory(self.evidence)
        self.assertEqual(value["result"], "recorded_checks_not_clean")
        self.assertEqual(value["exit_code"], 1)

    def test_followup_assertion_result_is_supported_without_promoting_acceptance(self):
        self.report.update(version="tui-warning-followup-checks-1",
                           result="assertions_passed_pending_raw_audit", source_changed_during_run=[])
        self.save_report()
        value = audit.audit_directory(self.evidence)
        self.assertEqual(value["recorded_result"], "assertions_passed_pending_raw_audit")
        self.assertEqual(value["result"], "no_diagnostics_found")
        (self.evidence / "step.stderr.log").write_text("Executing <Task synthetic> took 0.2 seconds\n", encoding="utf-8")
        self.assertEqual(audit.audit_directory(self.evidence)["result"], "diagnostics_found")
        (self.evidence / "step.stderr.log").write_text("", encoding="utf-8")
        self.report["source_changed_during_run"] = ["synthetic.py"]
        self.save_report()
        self.assertIn("recorded_followup_source_not_stable", audit.audit_directory(self.evidence)["recorded_check_issues"])
        self.report["source_changed_during_run"] = []
        self.report["result"] = "regression_failed"
        self.save_report()
        self.assertEqual(audit.audit_directory(self.evidence)["result"], "recorded_checks_not_clean")

    def test_followup_status_without_recognized_version_is_rejected(self):
        self.report["result"] = "assertions_passed_pending_raw_audit"
        self.save_report()
        with self.assertRaises(audit.AuditError):
            audit.audit_directory(self.evidence)

    def test_missing_logs_and_unsafe_names_fail_closed(self):
        for name in ("../outside.log", str(self.root / "outside.log"), "missing.log"):
            with self.subTest(name=name):
                self.report["steps"][0]["raw_stderr"] = name
                self.save_report()
                with self.assertRaises((OSError, audit.AuditError)):
                    audit.audit_directory(self.evidence)

    def test_symlink_hardlink_special_and_oversized_evidence_are_refused(self):
        path = self.evidence / "step.stderr.log"
        path.unlink()
        other = self.root / "other.log"
        other.write_text("synthetic", encoding="utf-8")
        path.symlink_to(other)
        with self.assertRaises(audit.AuditError):
            audit.audit_directory(self.evidence)
        path.unlink()
        os.link(other, path)
        with self.assertRaises(audit.AuditError):
            audit.audit_directory(self.evidence)
        path.unlink()
        os.mkfifo(path)
        with self.assertRaises(audit.AuditError):
            audit.audit_directory(self.evidence)
        path.unlink()
        with path.open("wb") as handle:
            handle.truncate(audit.MAX_LOG_BYTES + 1)
        with self.assertRaises(audit.AuditError):
            audit.audit_directory(self.evidence)

    def test_duplicate_keys_and_non_utf8_are_refused(self):
        path = self.evidence / "report.json"
        for raw in (b'{"result":"passed","result":"failed"}', b'{"result":NaN}', b"\xff"):
            with self.subTest(raw=raw):
                path.write_bytes(raw)
                with self.assertRaises(audit.AuditError):
                    audit.audit_directory(self.evidence)

    def test_child_cli_log_is_audited_in_addition_to_outer_stdout(self):
        self.report["steps"][0]["cli_report"] = "cli-report.json"
        self.save_report()
        (self.evidence / "cli-report.json").write_text('{"raw_log":"cli-tests.log"}', encoding="utf-8")
        (self.evidence / "cli-tests.log").write_text("Executing <Task pending synthetic> took 0.2 seconds\n", encoding="utf-8")
        value = audit.audit_directory(self.evidence)
        self.assertEqual(value["result"], "diagnostics_found")
        self.assertEqual(len(value["logs"]), 3)

    def test_report_command_is_never_executed(self):
        self.report["steps"][0]["command"] = ["synthetic-must-not-execute"]
        self.save_report()
        with patch.object(subprocess, "Popen", side_effect=AssertionError("audit is data only")):
            value = audit.audit_directory(self.evidence)
        self.assertFalse(value["commands_executed"])

    def test_new_private_output_is_exclusive(self):
        target = self.root / "audit"
        audit._save(target, {"synthetic": True})
        self.assertEqual(target.stat().st_mode & 0o777, 0o700)
        self.assertEqual((target / "report.json").stat().st_mode & 0o777, 0o600)
        with self.assertRaises(FileExistsError):
            audit._save(target, {})

    def test_actual_audit_cli_exit_is_nonzero_for_diagnostics(self):
        (self.evidence / "step.stderr.log").write_text("WARNING synthetic\n", encoding="utf-8")
        output = self.root / "cli-output"
        process = subprocess.run([sys.executable, "-I", "-B", str(TOOL), str(self.evidence),
                                  "--output-dir", str(output)], cwd=self.root, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=False)
        self.assertEqual(process.returncode, 1, process.stderr.decode())
        value = json.loads(process.stdout)
        self.assertEqual(value["result"], "diagnostics_found")
        self.assertEqual(value, json.loads((output / "report.json").read_bytes()))
        self.assertEqual(process.stderr, b"")


if __name__ == "__main__":
    unittest.main()
