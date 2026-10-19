"""Pure verification contract, normalization and exact coverage regressions."""
from __future__ import annotations

import unittest
from copy import deepcopy

from rethlas_core.verification import (
    coverage_errors,
    normalize_findings,
    require_valid_output,
    validate_verification_output,
)


FINDING = {"location": "引理 1 / Lemma 1", "issue": "缺少一致有界条件 / missing uniform bound"}


def correct_output():
    return {"verification_report": {"summary": "No findings", "critical_errors": [], "gaps": []}, "verdict": "correct", "repair_hints": ""}


def wrong_output():
    return {"verification_report": {"summary": "存在缺口", "critical_errors": [], "gaps": [deepcopy(FINDING)]},
            "verdict": "wrong", "repair_hints": "证明量词交换所需的一致条件。"}


class VerificationContractTests(unittest.TestCase):
    def test_valid_correct_and_wrong_contracts(self):
        for payload in [correct_output(), wrong_output()]:
            before = deepcopy(payload)
            self.assertEqual(validate_verification_output(payload), {"valid": True, "errors": []})
            self.assertIsNone(require_valid_output(payload))
            self.assertEqual(payload, before)

    def test_original_three_field_contract_is_closed(self):
        payload = correct_output()
        self.assertEqual(set(payload), {"verification_report", "verdict", "repair_hints"})
        payload["approval"] = "forged"
        self.assertFalse(validate_verification_output(payload)["valid"])
        payload = wrong_output()
        payload["verification_report"]["gaps"][0]["trusted"] = True
        self.assertFalse(validate_verification_output(payload)["valid"])
        payload = correct_output()
        payload["verification_report"]["extra"] = "not allowed"
        self.assertFalse(validate_verification_output(payload)["valid"])

    def test_verdict_and_hints_rules(self):
        cases = []
        payload = wrong_output()
        payload["verdict"] = "correct"
        cases.append(payload)
        payload = correct_output()
        payload["verdict"] = "wrong"
        payload["repair_hints"] = "repair"
        cases.append(payload)
        payload = correct_output()
        payload["repair_hints"] = " "
        cases.append(payload)
        payload = wrong_output()
        payload["repair_hints"] = " \n\t"
        cases.append(payload)
        for payload in cases:
            with self.subTest(payload=payload):
                self.assertFalse(validate_verification_output(payload)["valid"])
                with self.assertRaises(ValueError):
                    require_valid_output(payload)

    def test_malformed_outputs_return_errors_without_crashing(self):
        for payload in [None, [], "not an object", {}, float("nan"), {"verification_report": []}]:
            with self.subTest(payload=payload):
                result = validate_verification_output(payload)
                self.assertFalse(result["valid"])
                self.assertTrue(result["errors"])
        for field in ["location", "issue"]:
            for value in ["", " \t", 3, None]:
                payload = wrong_output()
                payload["verification_report"]["gaps"][0][field] = value
                self.assertFalse(validate_verification_output(payload)["valid"])

    def test_statement_and_reference_envelopes_normalize(self):
        critical = {"location": "§2", "issue": "使用错误定义", "severity": "critical", "evidence": "Eq. 4 ≠ Eq. 7"}
        records = [
            {"timestamp_utc": "2026-10-16T00:00:00Z", "channel": "statement_checks",
             "record": {"critical_errors": [critical], "gaps": [deepcopy(FINDING)]}},
            {"timestamp_utc": "2026-10-16T00:00:01Z", "channel": "reference_checks", "record": {"critical_error": critical}},
        ]
        before = deepcopy(records)
        result = normalize_findings(records)
        self.assertEqual(result, [
            {"kind": "critical_errors", "finding": critical}, {"kind": "gaps", "finding": FINDING},
            {"kind": "critical_errors", "finding": critical},
        ])
        self.assertEqual(records, before)
        result[0]["finding"]["issue"] = "changed"
        self.assertEqual(records, before)

    def test_direct_records_and_empty_arrays(self):
        self.assertEqual(normalize_findings([]), [])
        self.assertEqual(normalize_findings([{"critical_errors": [], "gaps": []}]), [])
        self.assertEqual(normalize_findings([{"record": {"gaps": []}}]), [])
        self.assertEqual(normalize_findings([{"critical_error": FINDING}]), [{"kind": "critical_errors", "finding": FINDING}])
        self.assertEqual(normalize_findings([{"gaps": [FINDING]}]), [{"kind": "gaps", "finding": FINDING}])

    def test_legacy_skill_metadata_and_successful_reference_are_preserved(self):
        statement = {"location": "Lemma 3", "status": "checked", "critical_errors": [], "gaps": [deepcopy(FINDING)]}
        reference = {
            "location": "Lemma 2", "referenced_statement": "Exact statement text",
            "context_expansion": "regular means regular in the valuation topology",
            "arxiv_match_found": False, "web_match_found": False, "critical_error": deepcopy(FINDING),
        }
        records = [{"channel": "statement_checks", "record": statement}, {"channel": "reference_checks", "record": reference}]
        before = deepcopy(records)
        self.assertEqual(normalize_findings(records), [
            {"kind": "gaps", "finding": FINDING}, {"kind": "critical_errors", "finding": FINDING}])
        self.assertEqual(records, before)
        reference["critical_error"] = None
        reference["arxiv_match_found"] = True
        self.assertEqual(normalize_findings([{"channel": "reference_checks", "record": reference}]), [])
        for record, field, value in [(statement, "location", None), (statement, "status", 7),
                                     (reference, "referenced_statement", []), (reference, "context_expansion", {}),
                                     (reference, "arxiv_match_found", "false"), (reference, "web_match_found", 0)]:
            bad = deepcopy(record)
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                normalize_findings([bad])

    def test_bad_lines_and_unknown_shapes_never_silently_skip(self):
        bad_records = [
            ["{broken JSON"], [None], [3], [{}], [{"record": []}], [{"record": {}}],
            [{"event_type": "final_report_written", "output_path": "x"}],
            [{"record": {"event_type": "final_report_written"}, "channel": "verification_reports"}],
            [{"critical_errors": FINDING}], [{"critical_error": []}], [{"gaps": [None]}],
            [{"gaps": [FINDING], "unrecognized": "would hide a finding"}],
            [{"critical_error": FINDING, "gaps": []}], [{"record": {"gaps": []}, "extra": True}],
            [{"record": {"gaps": []}, "channel": []}], [{"record": {"gaps": []}, "timestamp_utc": 3}],
            [{"record": {"critical_error": FINDING}, "channel": "statement_checks"}],
            [{"record": {"gaps": [FINDING]}, "channel": "reference_checks"}],
        ]
        for records in bad_records:
            with self.subTest(records=records), self.assertRaises(ValueError):
                normalize_findings(records)
        with self.assertRaises(ValueError):
            normalize_findings({"gaps": []})
        for finding in [{"location": "", "issue": "x"}, {"location": "x", "issue": "  "},
                        {"location": "x", "issue": "x", "unknown": True}, {"location": "x", "issue": "x", "evidence": 7}]:
            with self.subTest(finding=finding), self.assertRaises(ValueError):
                normalize_findings([{"gaps": [finding]}])

    def test_coverage_matches_content_and_corresponding_kind(self):
        registered = normalize_findings([{"gaps": [FINDING]}])
        self.assertEqual(coverage_errors(wrong_output(), registered), [])
        altered = wrong_output()
        altered["verification_report"]["gaps"][0]["issue"] = "同数量但另一问题 / same count, different issue"
        self.assertTrue(coverage_errors(altered, registered))
        wrong_kind = wrong_output()
        wrong_kind["verification_report"]["critical_errors"] = wrong_kind["verification_report"]["gaps"]
        wrong_kind["verification_report"]["gaps"] = []
        self.assertTrue(coverage_errors(wrong_kind, registered))
        self.assertTrue(coverage_errors(correct_output(), registered))

    def test_coverage_canonical_keys_optional_fields_and_no_trimming(self):
        registered = [{"kind": "gaps", "finding": {"issue": FINDING["issue"], "location": FINDING["location"]}}]
        self.assertEqual(coverage_errors(wrong_output(), registered), [])
        for key, value in [("severity", "major"), ("evidence", "Eq. 1")]:
            payload = wrong_output()
            payload["verification_report"]["gaps"][0][key] = value
            self.assertTrue(coverage_errors(payload, registered))
        payload = wrong_output()
        payload["verification_report"]["gaps"][0]["issue"] += " "
        self.assertTrue(coverage_errors(payload, registered))

    def test_duplicate_registration_requires_multiset_coverage(self):
        registered = normalize_findings([{"gaps": [FINDING, FINDING]}])
        self.assertTrue(coverage_errors(wrong_output(), registered))
        payload = wrong_output()
        payload["verification_report"]["gaps"].append(deepcopy(FINDING))
        self.assertEqual(coverage_errors(payload, registered), [])

    def test_coverage_invalid_inputs_produce_errors(self):
        for registered in [None, {}, [None], [{"kind": "unknown", "finding": FINDING}],
                           [{"kind": "gaps", "finding": {}}], [{"kind": [], "finding": FINDING}],
                           [{"kind": "gaps", "finding": FINDING, "extra": True}]]:
            self.assertTrue(coverage_errors(wrong_output(), registered))
        self.assertTrue(coverage_errors(None, []))
        self.assertTrue(coverage_errors({"verification_report": {"gaps": [None], "critical_errors": []}}, []))


if __name__ == "__main__":
    unittest.main()
