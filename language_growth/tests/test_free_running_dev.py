"""CPU free-running development tests with a scripted nonlearned provider."""

import copy
import json
import unittest
from decimal import Decimal

from language_growth.development_data import generate_development_tasks
from language_growth.free_running_dev import run_development_tasks, score_development_task
from language_growth.planner import PlannerResult
from language_growth.parent_bridge import legacy_request_fingerprint

REVISION = "scripted-dev:v1"
CONFIG = {
    "calibration": {"kind": "option_count_temperature", "id": "scripted-dev",
                    "t0": 1.0, "t1": 0.0, "t_min": 0.05},
    "policy_masking": "none (served covers every supplied option)",
}


class ScriptedProvider:
    """Chooses from current request/state only; never receives gold targets."""

    def __init__(self, mode="correct"):
        self.mode = mode
        self.requests = []

    def decide(self, request):
        self.requests.append(request)
        if self.mode == "error":
            raise RuntimeError("synthetic provider failure")
        state = json.loads(request["state"])
        qid, question = next(iter(request["questions"].items()))
        options = question["criteria"]
        slot = qid.rsplit("-", 1)[-1]
        task = state["task_input"]
        records = state["records"]
        if slot == "operation":
            phrase = ("Explain a checked assertion" if task.startswith("Explain") else
                      "Assert one selected fact" if task.startswith("What") else
                      "Compare two selected typed values")
        elif slot == "fact":
            target = next(rec for rec in records if rec["entity_ref"] in task)
            field = "warranty_months" if "warranty months" in task else "price"
            if self.mode == "wrong_true":
                field = "price" if field == "warranty_months" else "warranty_months"
            phrase = f"Fact {target['record_ref']} field {field}"
        elif slot in ("left", "right"):
            index = (0 if slot == "left" else 1)
            if self.mode == "reverse":
                index = 1 - index
            phrase = f"Fact {records[index]['record_ref']} field price"
        else:
            previous = {choice["slot"]: choice["description"] for choice in state["proposed_slot_choices"]}
            left_ref = previous["left"].split(" ")[1]
            right_ref = previous["right"].split(" ")[1]
            by_ref = {rec["record_ref"]: rec for rec in records}
            left = Decimal(by_ref[left_ref]["fields"]["price"]["value"])
            right = Decimal(by_ref[right_ref]["fields"]["price"]["value"])
            relation = "equal" if left == right else ("greater_than" if left > right else "less_than")
            if self.mode == "false_relation":
                relation = "less_than" if relation == "greater_than" else "greater_than"
            phrase = f"Relation {relation}"
        matches = [oid for oid, description in options.items() if description.startswith(phrase)]
        assert len(matches) == 1, (phrase, options)
        selected = matches[0]
        probs = {oid: 1.0 if oid == selected else 0.0 for oid in options}
        return {
            "answers": {qid: {"type": "choice", "choice": selected,
                              "confidence": 1.0, "probabilities": dict(probs)}},
            "ssdm": {
                "schema": "ssdm.decide.v0.1",
                "status": "ok", "request_id": request["ssdm"]["request_id"],
                "raw_probabilities": {qid: dict(probs)},
                "served_probabilities": {qid: dict(probs)},
                "served_config": {**copy.deepcopy(CONFIG), "temperature_per_question": {qid: 1.0}},
                "state_handle": None,
                "diagnostics": {"revision_id": REVISION,
                                "request_fingerprint": legacy_request_fingerprint(request),
                                "request_fingerprint_algorithm": "ssdm.request_fingerprint.v0",
                                "provenance": {"supplied": False, "applied": False},
                                "inference_calls": 1},
            },
        }


def run(examples, provider, *, repair=False):
    return run_development_tasks(examples, provider, model="scripted",
                                 parent_revision=REVISION, expected_served_config=CONFIG,
                                 allow_repair=repair)


class FreeRunningDevelopmentTests(unittest.TestCase):
    def test_correct_tasks_run_from_initial_input_only(self):
        examples = generate_development_tasks(5, 3, choice_format="production")
        # Later generated requests carry gold histories. They are unavailable to
        # the runner and must never become planner input.
        for example in examples:
            for request in example["model_inputs"][1:]:
                request["state"] = "poisoned gold prefix"
        provider = ScriptedProvider()
        report = run(examples, provider)
        self.assertEqual((report["total"], report["kernel_accepted"], report["task_correct"]), (3, 3, 3))
        self.assertEqual(report["condition"], "one_shot")
        first_states = [json.loads(req["state"]) for req in provider.requests if
                        next(iter(req["questions"])).startswith("slot-001-")]
        self.assertEqual(len(first_states), 3)
        self.assertTrue(all(state["proposed_slot_choices"] == [] for state in first_states))

    def test_true_but_wrong_fact_is_task_failure(self):
        example = generate_development_tasks(7, 1, choice_format="production")[0]
        report = run([example], ScriptedProvider("wrong_true"))
        self.assertEqual((report["total"], report["kernel_accepted"], report["task_correct"]), (1, 1, 0))
        self.assertIsNotNone(report["observations"][0]["text"])

    def test_explanation_is_required(self):
        example = generate_development_tasks(7, 2, choice_format="production")[1]
        # The provider reads the correct fact but deliberately chooses an Assert
        # operation, which remains kernel-valid and text-producing.
        class NoExplanation(ScriptedProvider):
            def decide(self, request):
                response = super().decide(request)
                qid = next(iter(request["questions"]))
                if qid.endswith("operation"):
                    options = request["questions"][qid]["criteria"]
                    selected = next(oid for oid, desc in options.items() if desc == "Assert one selected fact")
                    probs = {oid: 1.0 if oid == selected else 0.0 for oid in options}
                    response["answers"][qid].update(choice=selected, probabilities=probs)
                    response["ssdm"]["raw_probabilities"][qid] = dict(probs)
                    response["ssdm"]["served_probabilities"][qid] = dict(probs)
                return response
        report = run([example], NoExplanation())
        self.assertEqual((report["kernel_accepted"], report["task_correct"]), (1, 0))

    def test_reversed_comparison_is_equivalent_for_equal_and_unequal_values(self):
        equal = generate_development_tasks(9, 3, choice_format="production")[2]
        self.assertEqual(equal["training_only"]["program"]["nodes"][2]["relation"], "equal")
        unequal = next(task for task in generate_development_tasks(11, 30, choice_format="production")
                       if task["training_only"]["kind"] == "compare"
                       and task["training_only"]["program"]["nodes"][2]["relation"] != "equal")
        report = run([equal, unequal], ScriptedProvider("reverse"))
        self.assertEqual((report["total"], report["kernel_accepted"], report["task_correct"]), (2, 2, 2))

    def test_false_program_and_errors_stay_in_denominator(self):
        comparison = generate_development_tasks(9, 3, choice_format="production")[2]
        false = run([comparison], ScriptedProvider("false_relation"))
        self.assertEqual((false["total"], false["kernel_accepted"], false["task_correct"]), (1, 0, 0))
        self.assertIsNone(false["observations"][0]["text"])
        error = run([comparison], ScriptedProvider("error"))
        self.assertEqual((error["total"], error["task_correct"]), (1, 0))
        self.assertEqual(error["observations"][0]["status"], "provider_error")
        malformed = copy.deepcopy(comparison)
        malformed["model_inputs"][0]["state"] = "not json"
        runner_error = run([malformed], ScriptedProvider())
        self.assertEqual((runner_error["total"], runner_error["task_correct"]), (1, 0))
        self.assertEqual(runner_error["observations"][0]["status"], "runner_error")

    def test_repair_condition_is_reported_separately(self):
        example = generate_development_tasks(7, 3, choice_format="production")[2]
        example["training_only"]["host"]["limits"].update(
            max_repair_attempts=1, max_repair_calls=4, max_proposal_calls=8)

        class RepairingProvider(ScriptedProvider):
            def decide(self, request):
                self.mode = "false_relation" if len(self.requests) < 4 else "correct"
                return super().decide(request)

        report = run([example], RepairingProvider(), repair=True)
        self.assertEqual(report["condition"], "one_repair")
        self.assertEqual((report["total"], report["task_correct"]), (1, 1))
        row = report["observations"][0]
        self.assertEqual((row["provider_calls"], row["repair_requests"]), (8, 4))
        self.assertEqual(row["effective_budget"], {
            "max_repair_attempts": 1, "max_repair_calls": 4, "max_proposal_calls": 8})
        one_shot = run([example], RepairingProvider())
        self.assertEqual(one_shot["task_correct"], 0)
        self.assertEqual(one_shot["observations"][0]["effective_budget"]["max_repair_attempts"], 0)

    def test_misconfigured_repair_arm_fails_before_any_provider_call(self):
        example = generate_development_tasks(7, 1, choice_format="production")[0]
        provider = ScriptedProvider()
        with self.assertRaisesRegex(ValueError, "one_repair requires"):
            run([example], provider, repair=True)
        self.assertEqual(provider.requests, [])

    def test_output_only_tampering_is_not_credited_from_valid_program(self):
        example = generate_development_tasks(7, 1, choice_format="production")[0]
        result = PlannerResult("accepted", text="An unrelated unsupported answer.",
                               program=example["training_only"]["program"])
        self.assertFalse(score_development_task(example, result))


if __name__ == "__main__":
    unittest.main()
