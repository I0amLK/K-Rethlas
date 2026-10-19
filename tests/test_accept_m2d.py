"""M2d runner definitions/privacy; importing never starts acceptance."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_accept_m2b import synthetic_capability


SPEC = importlib.util.spec_from_file_location("m2d_acceptance_privacy", Path(__file__).absolute().parents[1] / "tools/accept_m2d.py")
acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceptance)


class M2dAcceptanceTests(unittest.TestCase):
    def test_capability_bearer_and_named_secret_redacted(self):
        token = synthetic_capability()
        for text in (token, json.dumps({"nested": token}), "Bearer synthetic-credential", "secret='synthetic-secret'"):
            with self.subTest(kind=text[:3]):
                output = acceptance.redact(text)
                self.assertIn("[REDACTED", output)
                self.assertNotIn(token, output)

    def test_long_dotted_test_identity_and_nonzero_command_preserved(self):
        name = "test_validity_cli.ValidityCliTests.test_actual_current_input_probe_publish_then_withdrawal_rejects_old_use"
        value = {"test": name, "argv": ["python", "--token-file", "synthetic.capability"],
                 "exit_code": 2, "nested": {"token": synthetic_capability()}}
        result = acceptance.redact_value(value)
        self.assertEqual(result["test"], name)
        self.assertEqual(result["argv"], value["argv"])
        self.assertEqual(result["exit_code"], 2)
        self.assertEqual(result["nested"]["token"], "[REDACTED]")

    def test_summary_keeps_skips_errors_warning_and_missing_summary(self):
        result = acceptance.summary("Ran 3 tests in 0.231s\nFAILED (failures=1, errors=1, skipped=1)\nResourceWarning: fixture")
        self.assertEqual((result["tests"], result["failures"], result["errors"], result["skipped"], result["warning_lines"]), (3, 1, 1, 1, 1))
        self.assertIsNone(acceptance.summary("import ok")["tests"])
        self.assertEqual(acceptance.summary("test_x ... skipped reason")["skipped"], 1)

    def test_fixed_ten_groups_timeouts_and_no_predicted_suite_count(self):
        commands = acceptance.commands(Path("/synthetic/run-001"))
        self.assertEqual([entry[0] for entry in commands], ["suite", "ledger-m1", "ledger-m2a", "ledger-m2b", "ledger-m2c", "ledger-m2d",
            "static-roadmap", "roadmap-regressions", "offline-experiment-regressions", "independent-validity-cli"])
        self.assertEqual((commands[0][2], commands[-1][2]), (600, 600))
        self.assertIsNone(commands[0][3])
        self.assertTrue(commands[0][4])
        self.assertEqual((commands[-3][3], commands[-2][3]), (63, 19))
        for _, command, _, _, _ in commands:
            self.assertEqual(command[1:3], ["-I", "-B"])
            self.assertFalse(any("PYTHONPATH" in item or "--dangerously" in item for item in command))

    def test_warning_or_skip_prevents_execute_success(self):
        class Process:
            pid = 12345
            returncode = 0
            stdin = stdout = stderr = None

            def __init__(self, output):
                self.output = output

            def communicate(self, timeout):
                return b"", self.output

        for output in (b"Ran 1 test in 0.001s\nOK\nResourceWarning: synthetic\n",
                       b"Ran 1 test in 0.001s\nOK (skipped=1)\n", b"import only\n"):
            with self.subTest(output=output), tempfile.TemporaryDirectory() as name:
                with patch.object(acceptance.subprocess, "Popen", return_value=Process(output)), \
                     patch.object(acceptance.os, "killpg", side_effect=ProcessLookupError):
                    result = acceptance.execute(Path(name), "synthetic-step", ["synthetic"], 20, requires_tests=True)
                self.assertEqual(result["result"], "failed")
                self.assertEqual(result["exit_code"], 0)

    def test_preimplementation_baseline_is_fixed_and_history_excludes_only_m2d(self):
        self.assertEqual(acceptance.HISTORY_COUNT, 223)
        self.assertEqual(acceptance.HISTORY_SHA, "d0822adaf899a944bd1f9d3df6800cdb3bfa403ad532787610a375c844b35f4a")
        self.assertEqual(acceptance.EVIDENCE, acceptance.ROOT / "docs/evidence/m2d-runtime")
        source = Path(SPEC.origin).read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "history")
        text = ast.get_source_segment(source, function)
        self.assertIn('("m2a", "m2b", "m2c")', text)
        self.assertIn('part.startswith("m2d-")', text)
        self.assertEqual(len(acceptance.PARENT_SOURCES), 6)

    def test_cli_archival_hides_body_but_keeps_actual_observation(self):
        from test_validity_cli import archive_output
        value = {"graph": {"revisions": [{"body": "synthetic private narrative"}]},
                 "observation": {"dto_sha256": "a" * 64, "empty_work": True}, "token": synthetic_capability()}
        result = archive_output(value)
        self.assertNotIn("synthetic private narrative", json.dumps(result))
        self.assertEqual(result["observation"], value["observation"])
        self.assertEqual(result["token"], "[REDACTED]")


if __name__ == "__main__":
    unittest.main()
