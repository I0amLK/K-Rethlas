"""Runner metadata/privacy only; importing definitions never starts acceptance."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

from test_accept_m2b import synthetic_capability


SPEC = importlib.util.spec_from_file_location("m2c_acceptance_privacy", Path(__file__).absolute().parents[1] / "tools/accept_m2c.py")
acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceptance)


class M2cAcceptanceTests(unittest.TestCase):
    def test_capability_bearer_and_named_secret_redacted(self):
        token = synthetic_capability()
        for text in (token, json.dumps({"nested": token}), "Bearer synthetic-credential", "secret='synthetic-secret'"):
            with self.subTest(kind=text[:3]):
                output = acceptance.redact(text)
                self.assertIn("[REDACTED", output)
                self.assertNotIn(token, output)

    def test_dotted_test_identity_and_commands_preserved(self):
        name = "test_context_views_cli.ContextCliTests.test_actual_conflict_resolution_join_and_view_isolated_probe"
        value = {"test": name, "argv": ["python", "--token-file", "synthetic.capability"],
                 "exit_code": 2, "nested": {"token": synthetic_capability()}}
        result = acceptance.redact_value(value)
        self.assertEqual(result["test"], name)
        self.assertEqual(result["argv"], value["argv"])
        self.assertEqual(result["exit_code"], 2)
        self.assertEqual(result["nested"]["token"], "[REDACTED]")

    def test_skips_errors_and_missing_summary_not_success(self):
        value = acceptance.summary("Ran 3 tests in 0.231s\nFAILED (failures=1, errors=1, skipped=1)")
        self.assertEqual((value["tests"], value["failures"], value["errors"], value["skipped"]), (3, 1, 1, 1))
        self.assertIsNone(acceptance.summary("import ok")["tests"])
        self.assertEqual(acceptance.summary("test_x ... skipped reason")["skipped"], 1)

    def test_cli_archival_hides_body_but_keeps_actual_observation(self):
        from test_context_views_cli import archive_output
        value = {"graph": {"revisions": [{"body": "synthetic private narrative"}]},
                 "observation": {"input_sha256": "a" * 64, "empty_work": True}, "token": synthetic_capability()}
        result = archive_output(value)
        self.assertNotIn("synthetic private narrative", json.dumps(result))
        self.assertEqual(result["observation"], value["observation"])
        self.assertEqual(result["token"], "[REDACTED]")
