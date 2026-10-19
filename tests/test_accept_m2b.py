"""Synthetic acceptance-log privacy fixtures; no acceptance run on import."""
from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "m2b_acceptance_privacy", Path(__file__).absolute().parents[1] / "tools/accept_m2b.py")
acceptance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(acceptance)


def synthetic_capability():
    # No usable grant or secret: only the canonical JSON/base64url/HMAC shape.
    claims = {"actions": ["memory.commit"], "domain": "fixture", "owner": "synthetic-owner",
              "run_id": "synthetic-run", "token_id": "a" * 32, "version": 1}
    payload = json.dumps(claims, sort_keys=True, separators=(",", ":")).encode()
    encode = lambda value: base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")
    return encode(payload) + "." + encode(b"\0" * 32)


class M2bAcceptancePrivacyTests(unittest.TestCase):
    def test_capability_shape_redacted_in_plain_and_quoted_logs(self):
        token = synthetic_capability()
        self.assertEqual(len(token.split(".")[1]), 43)
        for text in (token, "fixture " + token + "\n", json.dumps({"value": token}),
                     "(" + token + ")"):
            with self.subTest(wrapper=text[:1]):
                result = acceptance.redact(text)
                self.assertNotIn(token, result)
                self.assertIn("[REDACTED synthetic capability]", result)

    def test_long_dotted_test_identifiers_are_not_erased(self):
        names = [
            "test_legacy_migration_cli.LegacyMigrationCliTests.test_tampered_raw_request_manifest_and_unknown_member_reject",
            "m2b_legacy_migration_cli_acceptance.LegacyMigrationCliTests.test_prepare_apply_export_exact_replay_and_moved_bundle",
            "ordinary_package_name_with_underscores.another_long_component_with_underscores.method_name",
        ]
        for name in names:
            with self.subTest(module=name.split(".")[0]):
                self.assertEqual(acceptance.redact(name), name)

    def test_bearer_and_named_secret_values_remain_redacted(self):
        for text in ('Bearer unusable-synthetic-value', '{"api_key":"synthetic-key"}',
                     "secret='synthetic-secret'", 'token = "not-a-capability"'):
            with self.subTest(kind=text.split(":", 1)[0]):
                self.assertIn("[REDACTED]", acceptance.redact(text))
        self.assertEqual(acceptance.redact("a test result: ok"), "a test result: ok")

    def test_structured_redaction_preserves_command_record_fields(self):
        token = synthetic_capability()
        value = {"test": "test_legacy_migration_cli.LegacyMigrationCliTests.test_example",
                 "argv": ["python", "--token-file", "synthetic.capability"], "exit_code": 2,
                 "stdout": "", "stderr": "fixture " + token, "nested": {"token": token}}
        result = acceptance.redact_value(value)
        self.assertEqual(result["test"], value["test"])
        self.assertEqual(result["argv"], value["argv"])
        self.assertEqual(result["exit_code"], 2)
        self.assertNotIn(token, json.dumps(result))
        self.assertEqual(result["nested"]["token"], "[REDACTED]")
