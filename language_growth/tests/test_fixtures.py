"""Observed fixture replay against the actual L0 kernel and renderer."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO

from fixtures.language_growth.validate import read_json
from language_growth.evaluate import decision_request_for_case, evaluate_case, evaluate_pack, main


PACK = Path(__file__).resolve().parents[2] / "fixtures" / "language_growth"
CATALOG = read_json(PACK / "catalog.json")


def case(number):
    return read_json(PACK / "cases" / next(path.name for path in (PACK / "cases").glob(f"{number:02d}_*.json")))


class FixtureEvaluationTests(unittest.TestCase):
    def test_cli_mismatch_and_release_failure_are_not_success(self):
        for mismatches, failures in ((["LG01"], []), ([], ["LG01"]), ([], [])):
            with self.subTest(mismatches=mismatches, failures=failures):
                summary = {"authored_boundary_expectation_mismatches": mismatches, "release_failures": failures}
                with patch("language_growth.evaluate.evaluate_pack", return_value=summary), redirect_stdout(StringIO()):
                    code = main(["--fixtures", str(PACK), "--out", "unused-mocked-output"])
                self.assertEqual(code, 1 if mismatches or failures else 0)

    def test_observed_boundary_and_repair_are_distinct_from_task_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "report"
            summary = evaluate_pack(PACK, destination)
            rows = [json.loads(line) for line in (destination / "case_results.jsonl").read_text(encoding="utf-8").splitlines()]
            by_id = {row["case_id"]: row for row in rows}
            self.assertEqual(len(rows), 24)
            self.assertEqual(summary["observed_kernel_status_counts"], {"accepted": 7, "rejected": 16, "not_run": 1})
            self.assertEqual(summary["authored_boundary_expectation_mismatches"], [])
            self.assertEqual(summary["release_failures"], [])
            self.assertIsNone(summary["learned_model_accuracy"])
            self.assertEqual(summary["decision_only_model_calls"], 0)
            self.assertEqual(by_id["LG15"]["observed"]["provider_precondition"], "fail")
            self.assertEqual(by_id["LG15"]["observed"]["kernel_status"], "not_run")
            self.assertEqual(by_id["LG15"]["attempts"][0]["status"], "provider_rejected")
            self.assertEqual(by_id["LG18"]["attempts"][0]["status"], "rejected")
            self.assertEqual(by_id["LG18"]["attempts"][1]["status"], "accepted")
            self.assertEqual(by_id["LG18"]["observed"]["scripted_repair_attempts"], 1)
            self.assertEqual(by_id["LG19"]["observed"]["repair_state"], "budget_exhausted")
            self.assertEqual(len(by_id["LG19"]["attempts"]), 1)
            self.assertEqual(by_id["LG14"]["observed"]["kernel_status"], "accepted")
            self.assertIsNone(by_id["LG14"]["observed"]["useful_complete_answer"])
            self.assertEqual(by_id["LG14"]["authored_expectation"]["task_outcome"], "task_wrong")
            self.assertEqual(by_id["LG17"]["observed"]["speech_kinds"], ["ask"])
            self.assertEqual(by_id["LG20"]["observed"]["kernel_reason_category"], "unsupported_operation")
            self.assertEqual(by_id["LG21"]["observed"]["kernel_reason_category"], "invalid_graph")
            self.assertNotIn("99 months", by_id["LG13"]["observed"]["rendered_text"])
            self.assertEqual(read_json(destination / "summary.json"), summary)

    def test_authored_task_label_cannot_change_observation(self):
        original = case(14)
        pristine_case = deepcopy(original)
        pristine_catalog = deepcopy(CATALOG)
        changed = deepcopy(original)
        changed["expected"]["task_outcome"] = "complete_correct_answer"
        changed["expected"]["useful_complete_answer"] = True
        before = evaluate_case(original, CATALOG)
        after = evaluate_case(changed, CATALOG)
        self.assertEqual(before["observed"], after["observed"])
        self.assertNotEqual(before["authored_expectation"], after["authored_expectation"])
        self.assertEqual(original, pristine_case)
        self.assertEqual(CATALOG, pristine_catalog)

    def test_decision_request_filters_private_records_and_gold(self):
        request = decision_request_for_case(case(1), CATALOG)
        serialized = json.dumps(request)
        self.assertEqual(request["request_id"], "LG01-D0")
        self.assertEqual(request["task_input"], case(1)["question"])
        self.assertNotIn("target_option_id", serialized)
        self.assertNotIn("useful_complete_answer", serialized)
        self.assertNotIn("proposal", request)
        self.assertNotIn("internal-discount-v1", serialized)
        self.assertNotIn("catalog-private-r1", serialized)
        self.assertTrue(all(item["audience"] == "public" for item in request["evidence"]))
        request["evidence"][0]["fields"].clear()
        self.assertTrue(CATALOG["records"][0]["fields"])
        with self.assertRaisesRegex(ValueError, "provider precondition"):
            decision_request_for_case(case(15), CATALOG)

    def test_existing_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "already exists"):
                evaluate_pack(PACK, Path(temporary))


if __name__ == "__main__":
    unittest.main()
