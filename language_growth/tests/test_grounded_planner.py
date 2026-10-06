"""Grounding intervention preserves domains and uses only visible premises."""

import copy
import json
import unittest

from language_growth.development_data import generate_development_tasks
from language_growth.free_running_dev import _initial_task, score_development_task
from language_growth.planner import plan_checked_answer
from language_growth.tests.test_planner import EXPECTED_CONFIG, REVISION, ScriptedProvider


class GroundedPlannerTests(unittest.TestCase):
    def run_task(self, presentation, selectors, task=None):
        task = task or generate_development_tasks(271828, 3, choice_format="production",
                                                 split="development_validation")[-1]
        provider = ScriptedProvider(selectors)
        training = task["training_only"]
        result = plan_checked_answer(_initial_task(task), training["host"], training["catalog"],
                                     provider, model="scripted", parent_revision=REVISION,
                                     expected_served_config=EXPECTED_CONFIG,
                                     candidate_presentation=presentation)
        return task, result, provider.requests

    def test_grounded_comparison_and_unchanged_domains(self):
        selectors = ["Compare two", "development-record-1 field warranty_months",
                     "development-record-2 field warranty_months", "less_than"]
        task, result, requests = self.run_task("grounded", selectors)
        self.assertTrue(score_development_task(task, result))
        _, reference, old_requests = self.run_task("reference", selectors)
        self.assertEqual(result.program, reference.program)
        self.assertEqual(result.provider_calls, 4)
        for request, old in zip(requests, old_requests):
            q, old_q = next(iter(request["questions"].values())), next(iter(old["questions"].values()))
            self.assertEqual(set(q["criteria"]), set(old_q["criteria"]))
        operand_options = next(iter(requests[1]["questions"].values()))["criteria"]
        self.assertIn('"entity": "Kestrel"', operand_options["v002"])
        self.assertIn('"value": 40', operand_options["v002"])
        question = next(iter(requests[-1]["questions"].values()))["instructions"]
        self.assertIn("Selected LEFT: Fact development-record-1 field warranty_months", question)
        self.assertIn("Selected RIGHT: Fact development-record-2 field warranty_months", question)
        self.assertIn('"value": 55', question)
        self.assertEqual(len(next(iter(requests[-1]["questions"].values()))["criteria"]), 3)

    def test_false_relation_is_still_offered_and_rejected(self):
        _, result, _ = self.run_task("grounded", ["Compare two", "development-record-1 field warranty_months",
                                                 "development-record-2 field warranty_months", "greater_than"])
        self.assertEqual(result.status, "kernel_rejected")
        self.assertIsNone(result.text)

    def test_no_private_record_or_gold_metadata_in_grounding(self):
        task = generate_development_tasks(271828, 3, choice_format="production",
                                          split="development_validation")[-1]
        private = copy.deepcopy(task["training_only"]["catalog"]["records"][0])
        private.update(record_ref="SECRET", entity_ref="SECRET-ENTITY", audience="private")
        task["training_only"]["catalog"]["records"].append(private)
        task["training_only"]["catalog"]["gold_label"] = "SECRET-GOLD"
        _, result, requests = self.run_task("grounded", ["Compare two", "development-record-1 field warranty_months",
                                                         "development-record-2 field warranty_months", "less_than"], task)
        self.assertEqual(result.status, "accepted")
        self.assertNotIn("SECRET", json.dumps(requests))
        for request in requests:
            self.assertEqual(json.loads(request["state"])["accepted_partial_program"], [])

    def test_invalid_presentation_fails_before_provider(self):
        _, result, requests = self.run_task("invalid", [])
        self.assertEqual(result.reason, "invalid_candidate_presentation")
        self.assertEqual(requests, [])


if __name__ == "__main__":
    unittest.main()
