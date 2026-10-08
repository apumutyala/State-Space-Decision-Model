"""CPU scripted read probe; synthetic development inputs and no model weights."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from language_growth.development_data import generate_development_tasks
from language_growth.paired_read import DIRECT_QID, direct_read_request, run_paired_reads
from language_growth.tests.test_planner import EXPECTED_CONFIG, REVISION, ScriptedProvider


def reads(count=12):
    return [task for task in generate_development_tasks(17, count, choice_format="production")
            if task["training_only"]["kind"] == "read"]


class ReadProvider(ScriptedProvider):
    """Select from task text and visible facts only; response uses shared CPU fake."""

    def __init__(self, mode="correct"):
        super().__init__([])
        self.mode = mode

    def decide(self, request):
        qid = next(iter(request["questions"]))
        direct = qid == DIRECT_QID
        if ((self.mode == "error_direct" and direct)
                or (self.mode == "error_planner" and qid.endswith("-operation"))):
            raise RuntimeError("scripted failure")
        state = json.loads(request["state"])
        field = "warranty_months" if "warranty months" in state["task_input"].lower() else "price"
        record = next(rec for rec in state["records"] if rec["entity_ref"].lower() in state["task_input"].lower())
        if self.mode == "wrong_direct" and direct:
            record = next(rec for rec in state["records"] if rec != record)
        target = ("Assert one selected fact" if qid.endswith("-operation")
                  else f"Fact {record['record_ref']} field {field}")
        self.selectors.append(target)
        response = super().decide(request)
        if self.mode == "invalid_direct" and direct:
            response["ssdm"]["unexpected"] = "reject"
        if self.mode == "wrong_config" and direct:
            response["ssdm"]["served_config"]["calibration"]["id"] = "other-model-config"
        return response


def run(tasks, provider):
    return run_paired_reads(tasks, provider, model="scripted", parent_revision=REVISION,
                            expected_served_config=EXPECTED_CONFIG)


class PairedReadTests(unittest.TestCase):
    def test_same_initial_state_pins_and_all_raw_work(self):
        provider = ReadProvider()
        report = run(reads(), provider)
        self.assertEqual(report["paired_correct"], report["total"])
        for pair in report["pairs"]:
            self.assertTrue(pair["same_initial_state"])
            self.assertEqual((pair["direct"]["provider_calls"], pair["direct"]["forward_calls"]), (1, 1))
            self.assertEqual((pair["planner"]["provider_calls"], pair["planner"]["forward_calls"]), (2, 2))
            self.assertEqual(len(pair["direct_calls"]), 1)
            self.assertEqual(len(pair["planner_calls"]), 2)
            first = pair["planner_calls"][0]["request"]
            self.assertEqual(first["state"], pair["direct_request"]["state"])
            self.assertEqual(first["model"], pair["direct_request"]["model"])
            self.assertEqual(first["ssdm"]["parent_revision"], pair["direct_request"]["ssdm"]["parent_revision"])
            direct_config = pair["direct"]["trace"][0]["served_config"]
            planner_config = pair["planner_calls"][0]["response"]["ssdm"]["served_config"]
            for key in ("calibration", "policy_masking"):
                self.assertEqual(direct_config[key], planner_config[key])
            self.assertEqual(list(direct_config["temperature_per_question"].values()),
                             list(planner_config["temperature_per_question"].values()))
        self.assertEqual(report["qualification_requires"], ["strong_unrestricted_comparator", "full_parent_retention"])

    def test_all_permitted_registered_facts_without_gold_pruning(self):
        task = reads()[0]
        request = direct_read_request(task, model="scripted", parent_revision=REVISION)
        state = json.loads(request["state"])
        expected = {f"Fact {rec['record_ref']} field {field}" for rec in state["records"]
                    for field in rec["fields"] if field in state["registry"]["registered_fields"]}
        options = request["questions"][DIRECT_QID]["criteria"]
        self.assertEqual(set(options.values()), expected)
        self.assertEqual(list(options), ["f001", "f002", "f003", "f004"])
        poisoned = deepcopy(task)
        poisoned["training_only"]["program"] = {"gold_secret": "never exposed"}
        poisoned["training_only"]["targets"] = {"invented": "gold_secret"}
        poisoned["model_inputs"] = poisoned["model_inputs"][:1] + [{"state": "gold_secret"}]
        changed = direct_read_request(poisoned, model="scripted", parent_revision=REVISION)
        self.assertEqual(request["state"], changed["state"])
        self.assertEqual(request["questions"], changed["questions"])
        self.assertNotIn("gold_secret", json.dumps(changed))

    def test_target_lookup_only_after_call_and_wrong_supported_fact_not_corrected(self):
        import language_growth.paired_read as module
        provider = ReadProvider("wrong_direct")
        target = module._target_fact

        def after_call(example):
            self.assertEqual(len(provider.requests), 1)
            return target(example)

        with patch.object(module, "_target_fact", side_effect=after_call):
            report = run([reads()[0]], provider)
        pair = report["pairs"][0]
        self.assertEqual((report["direct_correct"], report["planner_correct"], report["paired_correct"]), (0, 1, 0))
        self.assertEqual(pair["direct"]["status"], "ok")
        self.assertNotEqual(pair["direct"]["selected_id"], pair["direct"]["target_id"])

    def test_invalid_reference_program_cannot_award_direct_or_planner_credit(self):
        for mutate, expected_reason in (
            (lambda program: program.update(schema="not-a-program"), "invalid_graph"),
            (lambda program: program["nodes"][0].update(unreviewed=True), "unsupported_field"),
        ):
            with self.subTest(expected_reason=expected_reason):
                task = deepcopy(reads()[0])
                mutate(task["training_only"]["program"])
                provider = ReadProvider()
                report = run([task], provider)
                pair = report["pairs"][0]
                self.assertEqual((report["total"], report["direct_correct"],
                                  report["planner_correct"], report["paired_correct"]), (1, 0, 0, 0))
                self.assertEqual((pair["direct"]["status"], pair["direct"]["reason"]),
                                 ("reference_rejected", expected_reason))
                self.assertEqual(pair["direct"]["provider_calls"], 1)
                self.assertEqual(len(pair["direct_calls"]), 1)
                self.assertTrue(pair["same_initial_state"])

    def test_failures_rejections_and_config_mismatch_keep_denominators_and_calls(self):
        for mode, reason, forward in (("error_direct", "provider_exception", None),
                                     ("invalid_direct", "invalid_response", 1),
                                     ("wrong_config", "served_config_mismatch", 1)):
            with self.subTest(mode=mode):
                report = run([reads()[0]], ReadProvider(mode))
                pair = report["pairs"][0]
                self.assertEqual((report["total"], report["direct_correct"], report["planner_correct"]), (1, 0, 1))
                self.assertEqual((pair["direct"]["reason"], pair["direct"]["provider_calls"],
                                  pair["direct"]["forward_calls"]), (reason, 1, forward))
                self.assertEqual(pair["direct_calls"][0]["exception"], mode == "error_direct")
                self.assertTrue(pair["same_initial_state"])
        report = run([reads()[0]], ReadProvider("error_planner"))
        pair = report["pairs"][0]
        self.assertEqual((report["total"], report["direct_correct"], report["planner_correct"]), (1, 1, 0))
        self.assertEqual((pair["planner"]["provider_calls"], pair["planner"]["forward_calls"]), (1, None))
        self.assertTrue(pair["planner_calls"][0]["exception"])

    def test_candidate_overflow_fails_without_pruning(self):
        task = reads()[0]
        task["training_only"]["host"]["limits"]["max_options_per_question"] = 3
        provider = ReadProvider()
        report = run([task], provider)
        pair = report["pairs"][0]
        self.assertEqual((report["total"], report["direct_correct"], report["planner_correct"]), (1, 0, 0))
        self.assertEqual((pair["direct"]["reason"], pair["direct"]["provider_calls"]), ("options_exceeded", 0))
        self.assertEqual(pair["planner"]["provider_calls"], 1)

    def test_invalid_parent_pin_and_malformed_initial_state_call_accounting(self):
        provider = ReadProvider()
        report = run_paired_reads([reads()[0]], provider, model="scripted", parent_revision=REVISION,
                                  expected_served_config={})
        pair = report["pairs"][0]
        self.assertEqual(provider.requests, [])
        self.assertEqual(pair["direct"]["reason"], "invalid_served_config_pin")
        self.assertEqual(pair["planner"]["reason"], "invalid_served_config_pin")
        poisoned = reads()[0]
        state = json.loads(poisoned["model_inputs"][0]["state"])
        state["target_option_id"] = "f001"
        poisoned["model_inputs"][0]["state"] = json.dumps(state)
        report = run([poisoned], provider)
        self.assertEqual(report["total"], 1)
        self.assertEqual(report["pairs"][0]["direct"]["provider_calls"], 0)
        self.assertEqual(report["pairs"][0]["direct"]["status"], "runner_error")

    def test_explain_compare_ask_and_sum_are_not_plain_read_cohorts(self):
        provider = ReadProvider()
        for kind in ("explain", "compare", "ask", "sum"):
            task = reads()[0]
            task["training_only"]["kind"] = kind
            with self.assertRaises(ValueError):
                run([task], provider)
        self.assertEqual(provider.requests, [])


if __name__ == "__main__":
    unittest.main()
