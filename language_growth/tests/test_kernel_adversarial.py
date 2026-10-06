"""Coordinator-authored attacks on the proposal boundary, not learned-model tests."""

import copy
import json
from pathlib import Path
import unittest

from language_growth.kernel import check_program, validate_release


PACK = Path(__file__).resolve().parents[2] / "fixtures" / "language_growth"


class KernelAdversarialTests(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads((PACK / "catalog.json").read_text(encoding="utf-8"))
        self.case = json.loads((PACK / "cases/01_supported_read.json").read_text(encoding="utf-8"))

    def test_malformed_reference_types_fail_without_values_in_reason(self):
        for value in (None, False, 3, [], {}, ["private-secret"]):
            with self.subTest(value=value):
                proposal = copy.deepcopy(self.case["proposal"])
                proposal["nodes"][1]["proposition"] = value
                result = check_program(proposal, self.case["host"], self.catalog)
                self.assertEqual(result.status, "rejected")
                self.assertEqual(result.outputs, ())
                self.assertNotIn("private-secret", result.reason)

    def test_checked_meaning_is_not_aliased_to_mutable_inputs(self):
        proposal = copy.deepcopy(self.case["proposal"])
        catalog = copy.deepcopy(self.catalog)
        result = check_program(proposal, self.case["host"], catalog)
        before = result.to_dict()
        proposal["nodes"][0]["record_ref"] = "internal-discount-v1"
        catalog["records"][0]["fields"]["repairable"]["value"] = False
        self.assertEqual(result.to_dict(), before)
        with self.assertRaisesRegex(ValueError, "unreleasable result"):
            validate_release(result, self.case["host"], catalog)

    def test_missing_record_and_unregistered_field_are_distinct_diagnostics(self):
        proposal = copy.deepcopy(self.case["proposal"])
        proposal["nodes"][0]["record_ref"] = "missing-record"
        self.assertEqual(check_program(proposal, self.case["host"], self.catalog).reason, "unknown_reference")
        proposal = copy.deepcopy(self.case["proposal"])
        proposal["nodes"][0]["field"] = "unregistered-field"
        self.assertEqual(check_program(proposal, self.case["host"], self.catalog).reason, "unsupported_field")

    def test_repair_feedback_cannot_probe_hidden_record_existence(self):
        outcomes = []
        for ref in ("internal-discount-v1", "never-existent-record"):
            proposal = copy.deepcopy(self.case["proposal"])
            proposal["nodes"][0]["record_ref"] = ref
            result = check_program(proposal, self.case["host"], self.catalog)
            outcomes.append((result.status, result.reason, result.outputs))
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(outcomes[0], ("rejected", "unknown_reference", ()))

    def test_kernel_does_not_credit_irrelevant_supported_answers(self):
        case = json.loads((PACK / "cases/14_supported_but_wrong_question.json").read_text(encoding="utf-8"))
        result = check_program(case["proposal"], case["host"], self.catalog)
        self.assertEqual(result.status, "accepted")
        self.assertNotIn("correct", result.to_dict())
        self.assertNotIn("task_outcome", result.to_dict())
        self.assertFalse(case["expected"]["useful_complete_answer"])


if __name__ == "__main__":
    unittest.main()
