"""CPU checks for synthetic development examples; no audit data is imported."""

import json
import unittest

from language_growth.development_data import generate_development_tasks, price_swap_counterfactual
from language_growth.kernel import check_program


class DevelopmentDataTests(unittest.TestCase):
    def test_seeded_tasks_are_reproducible_and_supported(self):
        tasks = generate_development_tasks(17, 30)
        self.assertEqual(tasks, generate_development_tasks(17, 30))
        self.assertNotEqual(tasks, generate_development_tasks(18, 30))
        self.assertEqual({task["training_only"]["kind"] for task in tasks},
                         {"read", "explain", "compare"})
        for task in tasks:
            training = task["training_only"]
            checked = check_program(training["program"], training["host"], training["catalog"])
            self.assertEqual(checked.status, "accepted")
            self.assertTrue(task["development_only"])
            self.assertEqual(len(task["model_inputs"]), len(training["targets"]))

    def test_model_requests_keep_current_targets_and_programs_out(self):
        for task in generate_development_tasks(23, 12):
            targets = task["training_only"]["targets"]
            for position, request in enumerate(task["model_inputs"]):
                state = json.loads(request["state"])
                self.assertEqual(state["accepted_partial_program"], [])
                self.assertEqual(len(state["proposed_slot_choices"]), position)
                self.assertNotIn("training_only", state)
                self.assertNotIn("target_option_id", state)
                self.assertNotIn("expected", state)
                self.assertNotIn("program", request["ssdm"])
                qid, question = next(iter(request["questions"].items()))
                self.assertEqual(question["type"], "choice")
                self.assertIn(targets[position]["option_id"], question["criteria"])
                if targets[position]["slot"] == "relation":
                    values = [r["fields"]["price"]["value"] for r in task["training_only"]["catalog"]["records"]]
                    options_text = json.dumps(question["criteria"])
                    for value in values:
                        self.assertNotIn(value, options_text)

    def test_price_swap_changes_relation_target_without_changing_options(self):
        task = next(task for task in generate_development_tasks(31, 30)
                    if task["training_only"]["kind"] == "compare"
                    and task["training_only"]["program"]["nodes"][2]["relation"] != "equal")
        swapped = price_swap_counterfactual(task)
        original_training = task["training_only"]
        swapped_training = swapped["training_only"]
        self.assertEqual(check_program(swapped_training["program"], swapped_training["host"],
                                       swapped_training["catalog"]).status, "accepted")
        self.assertNotEqual(original_training["program"]["nodes"][2]["relation"],
                            swapped_training["program"]["nodes"][2]["relation"])
        self.assertNotEqual(original_training["targets"][-1]["option_id"],
                            swapped_training["targets"][-1]["option_id"])
        self.assertEqual([r["questions"] for r in task["model_inputs"]],
                         [r["questions"] for r in swapped["model_inputs"]])
        self.assertNotEqual(task["model_inputs"][0]["state"], swapped["model_inputs"][0]["state"])
        self.assertNotEqual(original_training["host"]["context_ref"], swapped_training["host"]["context_ref"])
        self.assertNotEqual(task["model_inputs"][0]["ssdm"]["request_id"],
                            swapped["model_inputs"][0]["ssdm"]["request_id"])

    def test_context_identity_changes_with_evidence(self):
        tasks = generate_development_tasks(13, 12)
        refs = [task["training_only"]["host"]["context_ref"] for task in tasks]
        self.assertEqual(len(refs), len(set(refs)))
        for task in tasks:
            training = task["training_only"]
            self.assertEqual(set(training["catalog"]["snapshots"]), {training["host"]["context_ref"]})
            self.assertEqual(training["program"]["context_ref"], training["host"]["context_ref"])

    def test_equality_is_a_supported_target_and_price_swap_is_neutral(self):
        comparisons = [task for task in generate_development_tasks(31, 90)
                       if task["training_only"]["kind"] == "compare"]
        self.assertEqual({task["training_only"]["program"]["nodes"][2]["relation"] for task in comparisons},
                         {"greater_than", "less_than", "equal"})
        equal = next(task for task in comparisons
                     if task["training_only"]["program"]["nodes"][2]["relation"] == "equal")
        swapped = price_swap_counterfactual(equal)
        self.assertEqual(equal["training_only"]["targets"], swapped["training_only"]["targets"])
        self.assertEqual(equal["model_inputs"], swapped["model_inputs"])

    def test_option_ids_are_not_fixed_targets(self):
        tasks = generate_development_tasks(91, 60)
        operation_ids = {task["training_only"]["targets"][0]["option_id"] for task in tasks}
        relation_ids = {task["training_only"]["targets"][-1]["option_id"]
                        for task in tasks if task["training_only"]["kind"] == "compare"}
        self.assertGreater(len(operation_ids), 1)
        self.assertGreater(len(relation_ids), 1)

    def test_production_choice_cohort_matches_runtime_slot_shapes(self):
        tasks = generate_development_tasks(7, 3, choice_format="production")
        operation = next(iter(tasks[0]["model_inputs"][0]["questions"].values()))
        self.assertEqual(operation["criteria"], {
            "op001": "Assert one selected fact",
            "op002": "Explain a checked assertion of one selected fact",
            "op003": "Compare two selected typed values and assert the selected relation",
        })
        comparison = tasks[2]
        left = next(iter(comparison["model_inputs"][1]["questions"].values()))
        relation = next(iter(comparison["model_inputs"][3]["questions"].values()))
        self.assertEqual(left["criteria"]["v001"], "Fact development-record-1 field price")
        self.assertEqual(relation["criteria"], {
            "r001": "Relation greater_than", "r002": "Relation less_than", "r003": "Relation equal",
        })
        augmented = generate_development_tasks(7, 12, choice_format="augmented")
        production = generate_development_tasks(7, 12, choice_format="production")
        self.assertTrue(any(a["model_inputs"][0]["questions"] != p["model_inputs"][0]["questions"]
                            for a, p in zip(augmented, production)))

    def test_rejects_invalid_generation_requests(self):
        for seed, count in ((True, 1), (-1, 1), (0, 0), (0, 10001)):
            with self.subTest(seed=seed, count=count), self.assertRaises(ValueError):
                generate_development_tasks(seed, count)


if __name__ == "__main__":
    unittest.main()
