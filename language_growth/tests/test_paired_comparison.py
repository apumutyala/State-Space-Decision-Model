"""CPU scripted paired comparison checks; no model weights or audit data."""

from copy import deepcopy
from decimal import Decimal
import json
import unittest

from decide.schema import wire_answer

from language_growth.development_data import (comparison_swap_counterfactual,
                                               generate_development_tasks)
from language_growth.paired_comparison import (DIRECT_QID, direct_comparison_request,
                                               run_paired_comparisons)
from language_growth.parent_bridge import legacy_request_fingerprint


REVISION = "scripted-comparison:v1"
CONFIG = {"calibration": {"kind": "option_count_temperature", "id": "scripted-comparison",
                          "t0": 1.0, "t1": 0.0, "t_min": 0.05},
          "policy_masking": "none (served covers every supplied option)"}


def comparisons(seed=17, count=12):
    return [task for task in generate_development_tasks(seed, count, choice_format="production")
            if task["training_only"]["kind"] == "compare"]


class EvidenceProvider:
    """Selects from request text and permitted evidence only, never training_only."""

    def __init__(self, mode="correct"):
        self.mode = mode
        self.requests = []

    def decide(self, request):
        self.requests.append(deepcopy(request))
        qid, question = next(iter(request["questions"].items()))
        if self.mode == "error_direct" and qid == DIRECT_QID:
            raise RuntimeError("simulated direct failure")
        if self.mode == "error_planner" and qid.endswith("-operation"):
            raise RuntimeError("simulated planner failure")
        state = json.loads(request["state"])
        task = state["task_input"].lower()
        records = state["records"]
        ordered = sorted(records, key=lambda rec: task.index(rec["entity_ref"].lower()))
        field = "warranty_months" if "warranty months" in task else "price"
        left = Decimal(str(ordered[0]["fields"][field]["value"]))
        right = Decimal(str(ordered[1]["fields"][field]["value"]))
        relation = "equal" if left == right else ("greater_than" if left > right else "less_than")
        if self.mode == "wrong_direct" and qid == DIRECT_QID:
            relation = "greater_than" if relation == "equal" else (
                "less_than" if relation == "greater_than" else "greater_than")
        if qid == DIRECT_QID or qid.endswith("-relation"):
            description = f"Relation {relation}"
        elif qid.endswith("-operation"):
            description = "Compare two selected typed values and assert the selected relation"
        elif qid.endswith("-left"):
            description = f"Fact {ordered[0]['record_ref']} field {field}"
        elif qid.endswith("-right"):
            description = f"Fact {ordered[1]['record_ref']} field {field}"
        else:
            raise AssertionError(qid)
        options = question["criteria"]
        selected = next(oid for oid, text in options.items() if text == description)
        probabilities = {oid: 1.0 if oid == selected else 0.0 for oid in options}
        return {"answers": {qid: wire_answer(question, probabilities)},
                "ssdm": {"schema": "ssdm.decide.v0.1", "status": "ok",
                         "request_id": request["ssdm"]["request_id"],
                         "raw_probabilities": {qid: dict(probabilities)},
                         "served_probabilities": {qid: dict(probabilities)},
                         "served_config": {**deepcopy(CONFIG), "temperature_per_question": {qid: 1.0}},
                         "state_handle": None,
                         "diagnostics": {"revision_id": REVISION,
                                         "request_fingerprint": legacy_request_fingerprint(request),
                                         "request_fingerprint_algorithm": "ssdm.request_fingerprint.v0",
                                         "provenance": {"supplied": False, "applied": False},
                                         "state_handle_in": None, "state_reuse": False,
                                         "inference_calls": 1}}}


def run(tasks, provider, option_ids=None):
    return run_paired_comparisons(tasks, provider, model="scripted",
                                  parent_revision=REVISION, expected_served_config=CONFIG,
                                  option_ids=option_ids)


class PairedComparisonTests(unittest.TestCase):
    def test_correct_pair_has_same_initial_state_and_separate_work(self):
        task = comparisons()[0]
        provider = EvidenceProvider()
        report = run([task], provider)
        self.assertEqual((report["total"], report["direct_correct"], report["planner_correct"],
                          report["paired_correct"]), (1, 1, 1, 1))
        pair = report["pairs"][0]
        self.assertTrue(pair["same_initial_state"])
        self.assertEqual(pair["direct_initial_state_sha256"], pair["planner_initial_state_sha256"])
        self.assertEqual((pair["direct"]["provider_calls"], pair["direct"]["forward_calls"]), (1, 1))
        self.assertEqual((pair["planner"]["provider_calls"], pair["planner"]["forward_calls"]), (4, 4))
        self.assertEqual(len(pair["planner_calls"]), 4)
        self.assertIsNotNone(pair["direct"]["response"])
        self.assertEqual(pair["direct_request"], provider.requests[0])
        self.assertNotIn("training_only", json.dumps(pair["direct_request"]))
        self.assertEqual(set(pair["direct_request"]["questions"][DIRECT_QID]["criteria"].values()),
                         {"Relation greater_than", "Relation less_than", "Relation equal"})

    def test_opposite_direct_relation_fails_without_changing_planner(self):
        task = next(task for task in comparisons() if task["training_only"]["program"]["nodes"][2]["relation"] != "equal")
        report = run([task], EvidenceProvider("wrong_direct"))
        self.assertEqual((report["total"], report["direct_correct"], report["planner_correct"],
                          report["paired_correct"]), (1, 0, 1, 0))
        self.assertTrue(report["pairs"][0]["same_initial_state"])

    def test_counterfactual_values_change_targets_but_not_candidate_set(self):
        task = next(task for task in comparisons() if task["training_only"]["program"]["nodes"][2]["relation"] != "equal")
        swapped = comparison_swap_counterfactual(task)
        original_request = direct_comparison_request(task, model="scripted", parent_revision=REVISION)
        swapped_request = direct_comparison_request(swapped, model="scripted", parent_revision=REVISION)
        self.assertEqual(original_request["questions"], swapped_request["questions"])
        self.assertNotEqual(original_request["state"], swapped_request["state"])
        report = run([task, swapped], EvidenceProvider())
        self.assertEqual((report["total"], report["paired_correct"]), (2, 2))
        self.assertNotEqual(report["pairs"][0]["direct"]["target_id"],
                            report["pairs"][1]["direct"]["target_id"])
        self.assertTrue(all(pair["same_initial_state"] for pair in report["pairs"]))

    def test_equal_values_and_reversed_named_operands_remain_in_cohort(self):
        task = next(task for task in comparisons(count=30)
                    if task["training_only"]["program"]["nodes"][0]["record_ref"] == "development-record-2"
                    and task["training_only"]["program"]["nodes"][2]["relation"] == "equal")
        report = run([task], EvidenceProvider())
        self.assertEqual((report["direct_correct"], report["planner_correct"], report["paired_correct"]),
                         (1, 1, 1))

    def test_option_relabel_preserves_evidence_and_scoring(self):
        task = comparisons()[0]
        ids = {"greater_than": "z3", "less_than": "z1", "equal": "z2"}
        original = direct_comparison_request(task, model="scripted", parent_revision=REVISION)
        renamed = direct_comparison_request(task, model="scripted", parent_revision=REVISION, option_ids=ids)
        self.assertEqual(original["state"], renamed["state"])
        self.assertNotEqual(original["questions"], renamed["questions"])
        self.assertEqual(run([task], EvidenceProvider(), option_ids=ids)["paired_correct"], 1)
        with self.assertRaises(ValueError):
            direct_comparison_request(task, model="scripted", parent_revision=REVISION,
                                      option_ids={"greater_than": "x", "less_than": "x", "equal": "y"})

    def test_provider_errors_and_malformed_state_stay_in_denominators(self):
        task = comparisons()[0]
        direct_error = run([task], EvidenceProvider("error_direct"))
        pair = direct_error["pairs"][0]
        self.assertEqual((direct_error["total"], direct_error["direct_correct"], direct_error["planner_correct"]),
                         (1, 0, 1))
        self.assertEqual((pair["direct"]["status"], pair["direct"]["provider_calls"],
                          pair["direct"]["forward_calls"]), ("provider_error", 1, None))
        self.assertTrue(pair["same_initial_state"])
        planner_error = run([task], EvidenceProvider("error_planner"))
        self.assertEqual((planner_error["total"], planner_error["direct_correct"],
                          planner_error["planner_correct"]), (1, 1, 0))
        self.assertEqual(planner_error["pairs"][0]["planner"]["forward_calls"], None)
        poisoned = deepcopy(task)
        state = json.loads(poisoned["model_inputs"][0]["state"])
        state["target_option_id"] = "r001"
        poisoned["model_inputs"][0]["state"] = json.dumps(state)
        report = run([poisoned], EvidenceProvider())
        self.assertEqual(report["total"], 1)
        self.assertEqual((report["pairs"][0]["direct"]["status"],
                          report["pairs"][0]["direct"]["provider_calls"]), ("runner_error", 0))
        self.assertIsNone(report["pairs"][0]["same_initial_state"])

    def test_direct_rejects_extra_response_envelope_key(self):
        class ExtraEnvelopeProvider(EvidenceProvider):
            def decide(self, request):
                response = super().decide(request)
                if DIRECT_QID in request["questions"]:
                    response["ssdm"]["unexpected_key"] = "must fail closed"
                return response

        report = run([comparisons()[0]], ExtraEnvelopeProvider())
        pair = report["pairs"][0]
        self.assertEqual((report["total"], report["direct_correct"], report["planner_correct"],
                          report["paired_correct"]), (1, 0, 1, 0))
        self.assertEqual((pair["direct"]["status"], pair["direct"]["reason"],
                          pair["direct"]["provider_calls"], pair["direct"]["forward_calls"]),
                         ("provider_error", "invalid_response", 1, 1))
        self.assertEqual(pair["planner"]["provider_calls"], 4)
        self.assertTrue(pair["same_initial_state"])

    def test_direct_rejects_boolean_legacy_answer_numbers(self):
        class BooleanWireProvider(EvidenceProvider):
            def __init__(self, changed):
                super().__init__()
                self.changed = changed

            def decide(self, request):
                response = super().decide(request)
                if DIRECT_QID in request["questions"]:
                    answer = response["answers"][DIRECT_QID]
                    if self.changed == "confidence":
                        answer["confidence"] = True
                    else:
                        answer["probabilities"] = {key: bool(value)
                                                   for key, value in answer["probabilities"].items()}
                return response

        for changed in ("confidence", "probabilities"):
            with self.subTest(changed=changed):
                report = run([comparisons()[0]], BooleanWireProvider(changed))
                pair = report["pairs"][0]
                self.assertEqual((report["total"], report["direct_correct"],
                                  report["planner_correct"], report["paired_correct"]), (1, 0, 1, 0))
                self.assertEqual((pair["direct"]["status"], pair["direct"]["reason"],
                                  pair["direct"]["provider_calls"], pair["direct"]["forward_calls"]),
                                 ("provider_error", "invalid_probabilities", 1, 1))
                self.assertEqual(pair["planner"]["provider_calls"], 4)
                self.assertTrue(pair["same_initial_state"])

    def test_invalid_parent_pin_fails_both_arms_before_calls(self):
        provider = EvidenceProvider()
        bad_pin = deepcopy(CONFIG)
        del bad_pin["calibration"]["id"]
        report = run_paired_comparisons([comparisons()[0]], provider, model="scripted",
                                        parent_revision=REVISION, expected_served_config=bad_pin)
        pair = report["pairs"][0]
        self.assertEqual(provider.requests, [])
        self.assertEqual((report["total"], report["direct_correct"], report["planner_correct"],
                          report["paired_correct"]), (1, 0, 0, 0))
        self.assertEqual((pair["direct"]["status"], pair["direct"]["reason"],
                          pair["direct"]["provider_calls"], pair["direct"]["forward_calls"]),
                         ("precondition_failed", "invalid_served_config_pin", 0, 0))
        self.assertEqual((pair["planner"]["status"], pair["planner"]["reason"],
                          pair["planner"]["provider_calls"], pair["planner"]["forward_calls"]),
                         ("precondition_failed", "invalid_served_config_pin", 0, 0))
        self.assertIsNone(pair["same_initial_state"])

    def test_noncomparison_tasks_are_rejected_as_a_separate_cohort(self):
        read = generate_development_tasks(17, 1)[0]
        provider = EvidenceProvider()
        with self.assertRaises(ValueError):
            run([read], provider)
        self.assertEqual(provider.requests, [])


if __name__ == "__main__":
    unittest.main()
