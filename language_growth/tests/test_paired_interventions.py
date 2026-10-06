"""Scripted external intervention checks using the actual CPU planner/kernel."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from language_growth.development_data import generate_development_tasks
from language_growth.free_running_dev import score_development_task
from language_growth.paired_interventions import run_paired_interventions
from language_growth.parent_bridge import legacy_request_fingerprint
from language_growth.tests.test_paired_comparison import CONFIG, REVISION, EvidenceProvider


def comparisons(**kwargs):
    return [task for task in generate_development_tasks(17, 90, **kwargs)
            if task["training_only"]["kind"] == "compare"]


def run(tasks, provider):
    return run_paired_interventions(tasks, provider, model="scripted",
                                    parent_revision=REVISION, expected_served_config=CONFIG)


class PairedInterventionTests(unittest.TestCase):
    def test_both_fields_formats_splits_and_repair_budgets(self):
        for split in ("development_train", "development_validation"):
            for fmt in ("production", "augmented"):
                for repair in (False, True):
                    tasks = comparisons(split=split, choice_format=fmt, allow_one_repair=repair)
                    selected = [next(task for task in tasks
                                     if task["training_only"]["program"]["nodes"][0]["field"] == field
                                     and task["training_only"]["program"]["nodes"][2]["relation"] != "equal")
                                for field in ("price", "warranty_months")]
                    with self.subTest(split=split, fmt=fmt, repair=repair):
                        provider = EvidenceProvider()
                        report = run(selected, provider)
                        self.assertEqual((report["task_total"], report["total"], report["pair_pass"]), (2, 4, 4))
                        self.assertEqual(len(provider.requests), 32)
                        self.assertEqual(report["condition"], "one_shot_no_repair")
                        self.assertEqual(report["dependency"]["dependency_commit"], "03db551")
                        for pair in report["pairs"]:
                            self.assertTrue(pair["input_check"]["state_changed"])
                            self.assertTrue(pair["input_check"]["task_text_unchanged"])
                            nuisance = pair["intervention"] == "nuisance_change"
                            self.assertEqual(pair["expected_relation_change"], not nuisance)
                            self.assertEqual(pair["observed_relation_change"], not nuisance)
                            self.assertEqual(pair["checked_output_invariant"], nuisance)
                            self.assertEqual(pair["surface_invariant"], nuisance)
                            self.assertTrue(pair["surface_matches_expected_change"])
                            for arm in (pair["base"], pair["counterfactual"]):
                                self.assertEqual((arm["provider_calls"], arm["forward_calls"],
                                                  arm["repair_requests"]), (4, 4, 0))
                                self.assertEqual(len(arm["calls"]), 4)
                                self.assertTrue(all(call["response"] and call["error"] is None
                                                    for call in arm["calls"]))
                                self.assertEqual(json.loads(arm["calls"][0]["request"]["state"]),
                                                 arm["initial_state"])
                        json.dumps(report, allow_nan=False)

    def test_equal_relation_controls_keep_identical_swap_state_and_output(self):
        tasks = comparisons(choice_format="production")
        equal = next(task for task in tasks
                     if task["training_only"]["program"]["nodes"][2]["relation"] == "equal")
        report = run([equal], EvidenceProvider())
        self.assertEqual((report["total"], report["pair_pass"]), (2, 2))
        swap, nuisance = report["pairs"]
        self.assertTrue(swap["equal_relation_control"])
        self.assertFalse(swap["input_check"]["state_changed"])
        self.assertEqual(swap["base"]["initial_state_sha256"],
                         swap["counterfactual"]["initial_state_sha256"])
        self.assertFalse(swap["expected_relation_change"])
        self.assertFalse(swap["observed_relation_change"])
        self.assertTrue(swap["surface_invariant"])
        self.assertTrue(nuisance["input_check"]["state_changed"])
        self.assertTrue(nuisance["surface_invariant"])

    def test_wrong_but_kernel_valid_answers_do_not_pass_invariance(self):
        class WrongField(EvidenceProvider):
            def decide(self, request):
                fake = deepcopy(request)
                state = json.loads(fake["state"])
                task = state["task_input"]
                state["task_input"] = (task.replace("warranty months", "price")
                                       if "warranty months" in task else task.replace("price", "warranty months"))
                fake["state"] = json.dumps(state)
                response = super().decide(fake)
                response["ssdm"]["diagnostics"]["request_fingerprint"] = legacy_request_fingerprint(request)
                return response

        report = run([comparisons()[0]], WrongField())
        self.assertEqual(report["pair_pass"], 0)
        for pair in report["pairs"]:
            self.assertTrue(pair["base"]["kernel_accepted"])
            self.assertTrue(pair["counterfactual"]["kernel_accepted"])
            self.assertFalse(pair["both_task_correct"])
        self.assertTrue(report["pairs"][0]["checked_output_invariant"])
        self.assertFalse(report["pairs"][1]["checked_output_invariant"])

    def test_errors_malformed_responses_and_invalid_pins_keep_all_pairs(self):
        class Malformed(EvidenceProvider):
            def decide(self, request):
                return {"bad": "response"}

        task = comparisons()[0]
        for provider in (EvidenceProvider("error_planner"), Malformed()):
            report = run([task], provider)
            self.assertEqual((report["total"], report["pair_pass"]), (2, 0))
            for pair in report["pairs"]:
                self.assertIsNone(pair["checked_output_invariant"])
                self.assertIsNone(pair["surface_invariant"])
                for arm in (pair["base"], pair["counterfactual"]):
                    self.assertEqual(arm["status"], "provider_error")
                    self.assertEqual(arm["provider_calls"], 1)
                    self.assertEqual(len(arm["calls"]), 1)
                    if isinstance(provider, Malformed):
                        self.assertEqual(arm["calls"][0]["response"], {"bad": "response"})
                    else:
                        self.assertEqual(arm["calls"][0]["error"],
                                         {"type": "RuntimeError", "message": "simulated planner failure"})
        provider = EvidenceProvider()
        report = run_paired_interventions([task], provider, model="scripted",
                                          parent_revision="unpinned", expected_served_config=CONFIG)
        self.assertEqual((report["total"], report["pair_pass"]), (2, 0))
        self.assertEqual(len(provider.requests), 4)
        self.assertTrue(all(arm["reason"] == "identity_mismatch"
                            for pair in report["pairs"] for arm in (pair["base"], pair["counterfactual"])))
        provider = EvidenceProvider()
        report = run_paired_interventions([task], provider, model="scripted",
                                          parent_revision="", expected_served_config=CONFIG)
        self.assertEqual((report["total"], report["pair_pass"]), (2, 0))
        self.assertEqual(provider.requests, [])

    def test_request_guard_records_blocked_program_leak_without_calling_provider(self):
        from language_growth.parent_bridge import decision_request

        def poison(**kwargs):
            request, metadata = decision_request(**kwargs)
            state = json.loads(request["state"])
            state["reference_program"] = {"label": "must not leave evaluator"}
            request["state"] = json.dumps(state)
            return request, metadata

        provider = EvidenceProvider()
        with patch("language_growth.planner.decision_request", side_effect=poison):
            report = run([comparisons()[0]], provider)
        self.assertEqual((report["total"], report["pair_pass"]), (2, 0))
        self.assertEqual(provider.requests, [])
        for pair in report["pairs"]:
            for arm in (pair["base"], pair["counterfactual"]):
                self.assertEqual((arm["request_attempts"], arm["provider_calls"]), (1, 0))
                self.assertEqual(arm["calls"][0]["error"]["message"],
                                 "provider_request_label_or_program_leakage")

    def test_counterfactual_preparation_failure_keeps_base_attempt(self):
        with patch("language_growth.paired_interventions._INTERVENTIONS", {
            "relevant_swap": lambda example: (_ for _ in ()).throw(ValueError("bad world"))}):
            provider = EvidenceProvider()
            report = run([comparisons()[0]], provider)
        self.assertEqual((report["total"], report["pair_pass"]), (1, 0))
        pair = report["pairs"][0]
        self.assertTrue(pair["base"]["task_correct"])
        self.assertEqual(pair["counterfactual"]["error"], {"type": "ValueError", "message": "bad world"})
        self.assertEqual(pair["counterfactual"]["calls"], [])
        self.assertEqual(len(provider.requests), 4)

    def test_scoring_happens_after_both_arms_and_gold_prefixes_are_unused(self):
        task = deepcopy(comparisons()[0])
        for request in task["model_inputs"][1:]:
            request["state"] = "forbidden gold prefix sentinel"
        task["training_only"]["targets"] = [{"label": "scorer-only sentinel"}]
        provider = EvidenceProvider()
        call_counts = []

        def score(example, result):
            call_counts.append(len(provider.requests))
            return score_development_task(example, result)

        with patch("language_growth.paired_interventions.score_development_task", side_effect=score):
            report = run([task], provider)
        self.assertEqual(call_counts, [8, 8, 16, 16])
        self.assertEqual(report["pair_pass"], 2)
        encoded = json.dumps(provider.requests)
        for forbidden in ("scorer-only sentinel", "forbidden gold prefix sentinel", "training_only",
                          '"program"', '"targets"', '"task_id"'):
            self.assertNotIn(forbidden, encoded)

    def test_poisoned_initial_state_is_rejected_without_forwarding_labels(self):
        task = deepcopy(comparisons()[0])
        state = json.loads(task["model_inputs"][0]["state"])
        state["target_option_id"] = "r001"
        task["model_inputs"][0]["state"] = json.dumps(state)
        provider = EvidenceProvider()
        report = run([task], provider)
        self.assertEqual((report["total"], report["pair_pass"]), (2, 0))
        # The clean regenerated counterfactual is independently attempted.
        self.assertEqual(len(provider.requests), 8)
        self.assertNotIn("target_option_id", json.dumps(provider.requests))
        for pair in report["pairs"]:
            self.assertEqual(pair["base"]["error"]["message"], "initial_state_mismatch")
            self.assertEqual(pair["base"]["provider_calls"], 0)

    def test_checked_program_with_tampered_surface_cannot_pass(self):
        from language_growth.planner import plan_checked_answer

        def tamper(*args, **kwargs):
            result = plan_checked_answer(*args, **kwargs)
            result.text = "unrelated output"
            return result

        with patch("language_growth.paired_interventions.plan_checked_answer", side_effect=tamper):
            report = run([comparisons()[0]], EvidenceProvider())
        self.assertEqual(report["pair_pass"], 0)
        for pair in report["pairs"]:
            self.assertTrue(pair["base"]["kernel_accepted"])
            self.assertFalse(pair["base"]["task_correct"])
            self.assertIsNone(pair["surface_invariant"])


if __name__ == "__main__":
    unittest.main()
