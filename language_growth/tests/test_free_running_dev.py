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
            phrase = ("Compare two selected typed values" if "compare" in task.lower() else
                      "Explain a checked assertion" if "checked" in task.lower() else
                      "Assert one selected fact")
        elif slot == "fact":
            target = next(rec for rec in records if rec["entity_ref"] in task)
            field = "warranty_months" if "warranty months" in task else "price"
            if self.mode == "wrong_true":
                field = "price" if field == "warranty_months" else "warranty_months"
            phrase = f"Fact {target['record_ref']} field {field}"
        elif slot in ("left", "right"):
            named_records = sorted(records, key=lambda record: task.index(record["entity_ref"]))
            index = (0 if slot == "left" else 1)
            if self.mode == "reverse" or self.mode == "wrong_" + slot:
                index = 1 - index
            field = "warranty_months" if "warranty months" in task else "price"
            phrase = f"Fact {named_records[index]['record_ref']} field {field}"
        else:
            previous = {choice["slot"]: choice["description"] for choice in state["proposed_slot_choices"]}
            left_ref = previous["left"].split(" ")[1]
            right_ref = previous["right"].split(" ")[1]
            by_ref = {rec["record_ref"]: rec for rec in records}
            field = previous["left"].split(" field ")[1]
            left = Decimal(str(by_ref[left_ref]["fields"][field]["value"]))
            right = Decimal(str(by_ref[right_ref]["fields"][field]["value"]))
            if self.mode in ("wrong_left", "wrong_right"):
                named = sorted(records, key=lambda record: task.index(record["entity_ref"]))
                left, right = (Decimal(str(record["fields"][field]["value"])) for record in named)
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
                                "state_handle_in": None, "state_reuse": False,
                                "inference_calls": 1},
            },
        }


def run(examples, provider, *, repair=False):
    return run_development_tasks(examples, provider, model="scripted",
                                 parent_revision=REVISION, expected_served_config=CONFIG,
                                 allow_repair=repair)


class FreeRunningDevelopmentTests(unittest.TestCase):
    def slots(self, report):
        return {row["slot"]: row for row in report["observations"][0]["slot_diagnostics"]["slots"]}

    def test_slot_diagnostics_attribute_each_comparison_error(self):
        example = next(task for task in generate_development_tasks(11, 30, choice_format="production")
                       if task["training_only"]["kind"] == "compare"
                       and task["training_only"]["program"]["nodes"][2]["relation"] != "equal")
        for mode, failed_slot in (("wrong_left", "left"), ("wrong_right", "right"),
                                  ("false_relation", "relation")):
            with self.subTest(mode=mode):
                report = run([example], ScriptedProvider(mode))
                slots = self.slots(report)
                self.assertEqual(set(slots), {"operation", "left", "right", "relation"})
                self.assertEqual([slot for slot, row in slots.items() if not row["correct"]], [failed_slot])
                self.assertTrue(all(not row["missing_choice"] and row["denominator"] == 1
                                    and row["status"] == "ok" for row in slots.values()))
                self.assertEqual(report["slot_diagnostics"]["total"], 4)
                self.assertEqual(report["slot_diagnostics"]["correct"], 3)
                self.assertEqual((report["kernel_accepted"], report["task_correct"]), (0, 0))

    def test_correct_read_explain_and_compare_slots(self):
        for fmt in ("production", "augmented"):
            with self.subTest(choice_format=fmt):
                examples = generate_development_tasks(5, 3, choice_format=fmt)
                report = run(examples, ScriptedProvider())
                self.assertEqual((report["kernel_accepted"], report["task_correct"]), (3, 3))
                diagnostics = report["slot_diagnostics"]
                self.assertEqual((diagnostics["total"], diagnostics["correct"], diagnostics["missing_choice"]),
                                 (8, 8, 0))
                self.assertEqual(diagnostics["by_slot"]["fact"], {"total": 2, "correct": 2, "missing_choice": 0})
                for row in report["observations"]:
                    for slot in row["slot_diagnostics"]["slots"]:
                        self.assertEqual(slot["selected_ids"], [slot["selected_id"]])
                        self.assertEqual(slot["selected_id"], slot["expected_id"])
                        self.assertEqual(slot["comparison_basis"],
                                         "option_description" if fmt == "augmented" else "option_id")

    def test_slot_failures_and_preprovider_errors_retain_every_expected_slot(self):
        example = generate_development_tasks(7, 3, choice_format="production")[2]

        class PartialFailure(ScriptedProvider):
            def decide(self, request):
                if len(self.requests) == 2:
                    self.mode = "error"
                return super().decide(request)

        report = run([example], PartialFailure())
        slots = self.slots(report)
        self.assertEqual((report["total"], report["kernel_accepted"], report["task_correct"]), (1, 0, 0))
        self.assertEqual(slots["right"]["status"], "provider_exception")
        self.assertEqual(slots["relation"]["status"], "not_reached")
        self.assertEqual(report["slot_diagnostics"]["total"], 4)
        self.assertEqual(report["slot_diagnostics"]["missing_choice"], 2)
        malformed = copy.deepcopy(example)
        malformed["model_inputs"][0]["state"] = "not json"
        for item in (example, malformed):
            provider = ScriptedProvider()
            preprovider = run_development_tasks([item], provider, model="scripted", parent_revision=REVISION,
                                                expected_served_config={})
            self.assertEqual(provider.requests, [])
            self.assertEqual(preprovider["slot_diagnostics"]["total"], 4)
            self.assertEqual(preprovider["slot_diagnostics"]["missing_choice"], 4)
            self.assertTrue(all(row["selected_ids"] == [None] and row["denominator"] == 1
                                for row in self.slots(preprovider).values()))

    def test_targets_are_read_only_after_all_batch_calls_and_never_leak(self):
        examples = generate_development_tasks(5, 3, choice_format="production")
        baseline_provider = ScriptedProvider()
        baseline = run(examples, baseline_provider)
        provider = ScriptedProvider()

        class GuardedTraining(dict):
            def get(self, key, default=None):
                if key == "targets":
                    self_calls = len(provider.requests)
                    if self_calls != 8:
                        raise AssertionError(f"targets read before final batch call: {self_calls}")
                return super().get(key, default)

        poisoned = copy.deepcopy(examples)
        for example in poisoned:
            example["training_only"]["targets"] = [dict(target, option_id="SCORER_ONLY_POISON")
                                                       for target in example["training_only"]["targets"]]
            example["training_only"] = GuardedTraining(example["training_only"])
            for request in example["model_inputs"][1:]:
                request["state"] = "UNREAD_GOLD_PREFIX"
        changed = run(poisoned, provider)
        self.assertEqual((changed["kernel_accepted"], changed["task_correct"]),
                         (baseline["kernel_accepted"], baseline["task_correct"]))
        self.assertEqual(changed["slot_diagnostics"]["correct"], 0)
        for original, actual in zip(baseline_provider.requests, provider.requests):
            self.assertEqual(original["state"], actual["state"])
            self.assertEqual(original["questions"], actual["questions"])
            serialized = json.dumps(actual)
            for forbidden in ("SCORER_ONLY_POISON", "UNREAD_GOLD_PREFIX", "training_only",
                              "slot_diagnostics", "target_id", "expected_id"):
                self.assertNotIn(forbidden, serialized)

    def test_missing_target_metadata_does_not_change_primary_scores(self):
        example = generate_development_tasks(5, 1, choice_format="production")[0]
        for targets in (None, [{"slot": "fact"}]):
            with self.subTest(targets=targets):
                example["training_only"]["targets"] = targets
                report = run([example], ScriptedProvider())
                self.assertEqual((report["total"], report["kernel_accepted"], report["task_correct"]), (1, 1, 1))
                self.assertEqual(report["slot_diagnostics"]["unavailable_tasks"], 1)
                self.assertEqual(report["observations"][0]["slot_diagnostics"]["status"], "unavailable")

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
        equal = next(task for task in generate_development_tasks(9, 90, choice_format="production")
                     if task["training_only"]["kind"] == "compare"
                     and task["training_only"]["program"]["nodes"][2]["relation"] == "equal")
        unequal = next(task for task in generate_development_tasks(11, 30, choice_format="production")
                       if task["training_only"]["kind"] == "compare"
                       and task["training_only"]["program"]["nodes"][2]["relation"] != "equal")
        report = run([equal, unequal], ScriptedProvider("reverse"))
        self.assertEqual((report["total"], report["kernel_accepted"], report["task_correct"]), (2, 2, 2))
        # Slot orientation is diagnostic; equivalent reversed meaning retains
        # the established task score even when left/right selections differ.
        self.assertEqual(report["slot_diagnostics"]["by_slot"]["left"]["correct"], 0)
        self.assertEqual(report["slot_diagnostics"]["by_slot"]["right"]["correct"], 0)

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
        slots = self.slots(report)
        self.assertEqual(row["slot_diagnostics"]["attempt_total"], 8)
        self.assertEqual(row["slot_diagnostics"]["total"], 4)
        self.assertTrue(all(slot["final_attempt"] == 2 and slot["correct"] for slot in slots.values()))
        relation = slots["relation"]
        self.assertEqual([attempt["correct"] for attempt in relation["attempts"]], [False, True])
        self.assertEqual([attempt["repair"] for attempt in relation["attempts"]], [False, True])
        self.assertEqual(relation["selected_ids"], [attempt["selected_id"] for attempt in relation["attempts"]])
        one_shot = run([example], RepairingProvider())
        self.assertEqual(one_shot["task_correct"], 0)
        self.assertEqual(one_shot["observations"][0]["effective_budget"]["max_repair_attempts"], 0)

    def test_interrupted_repair_cannot_reuse_first_attempt_slots(self):
        example = generate_development_tasks(7, 3, choice_format="production", allow_one_repair=True)[2]

        class InterruptedRepair(ScriptedProvider):
            def decide(self, request):
                self.mode = ("false_relation" if len(self.requests) < 4 else
                             "correct" if len(self.requests) < 6 else "error")
                return super().decide(request)

        report = run([example], InterruptedRepair(), repair=True)
        slots = self.slots(report)
        self.assertEqual(report["observations"][0]["status"], "provider_error")
        self.assertEqual(report["slot_diagnostics"]["total"], 4)
        self.assertEqual(report["slot_diagnostics"]["missing_choice"], 2)
        self.assertEqual(report["slot_diagnostics"]["correct"], 2)
        self.assertEqual(slots["right"]["selected_ids"], [slots["right"]["target_id"], None])
        self.assertFalse(slots["right"]["correct"])
        self.assertEqual(slots["right"]["status"], "provider_exception")
        self.assertEqual(slots["relation"]["final_attempt"], 2)
        self.assertEqual(slots["relation"]["status"], "not_reached")
        self.assertFalse(slots["relation"]["correct"])

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
