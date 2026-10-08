"""Tiny public development-only gate tests. No heldout or model weights loaded."""
from __future__ import annotations

from copy import deepcopy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from language_growth.gate_suites import LEVELS, build_suite, canonical, digest, export_suite
from language_growth.gate_planner import plan_gate_task
from language_growth.gates import (readiness, paired_ci, score_checked, score_comparator,
                                  normalized_answer, intervention_comparison, decision_regression, main)
from language_growth.gates import run_joint_adversarial
from language_growth.planner import PlannerResult
from language_growth.tests.test_planner import ScriptedProvider, EXPECTED_CONFIG, REVISION


class FakeParent(ScriptedProvider):
    def planner_kwargs(self):
        return {"model": "scripted", "parent_revision": REVISION, "expected_served_config": EXPECTED_CONFIG}


class GateTests(unittest.TestCase):
    def suite(self, level="L0"):
        return build_suite(level, "development_validation", worlds=2)

    def test_all_level_development_references_and_split_isolation(self):
        for level in LEVELS:
            tasks, refs = self.suite(level)
            train, _ = build_suite(level, "development_train", worlds=2)
            self.assertTrue(tasks)
            self.assertEqual(len(tasks), len(refs))
            self.assertTrue(set(t["task_id"] for t in tasks).isdisjoint(t["task_id"] for t in train))
            for task, ref in zip(tasks, refs):
                self.assertEqual(digest(task), ref["input_sha256"])
                self.assertNotIn("program", task)
                self.assertNotIn("meaning", task)
                self.assertNotIn("training_only", task)
            if level == "L4":
                self.assertEqual(len(set(t["world_id"] for t in tasks)), 1)

    def test_readiness_never_passes_missing_measurements(self):
        report = readiness("missing-checkpoint")
        self.assertFalse(report["qualified"])
        self.assertFalse(report["heldout_opened"])
        for result in report["levels"].values():
            self.assertFalse(result["passed"])
            self.assertTrue(all(not gate["passed"] for gate in result["gates"].values()))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--parent", "missing-checkpoint"]), 2)

    def test_model_run_requires_pins_before_loading(self):
        with contextlib.redirect_stderr(io.StringIO()), tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main(["--parent", "missing", "--run-development", "--out", str(Path(tmp)/"out")]), 2)

    def test_separate_export_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "suite"
            manifest = export_suite(out, "L1", "development_validation", worlds=2, seed=42)
            self.assertFalse(manifest["training_allowed"])
            self.assertFalse(manifest["promotion_eligible"])
            self.assertEqual(hashlib.sha256((out / "inputs.jsonl").read_bytes()).hexdigest(), manifest["inputs_sha256"])
            self.assertNotIn(b'"program"', (out / "inputs.jsonl").read_bytes())
            with self.assertRaises(ValueError):
                export_suite(out, "L1", "development_validation", worlds=2, seed=42)

    def test_actual_free_running_choices_for_each_landed_module(self):
        for level in LEVELS:
            tasks, refs = self.suite(level)
            for task, ref in zip(tasks, refs):
                kind = ref["kind"]
                reads = [node for node in ref["program"].get("nodes", []) if node["op"] == "ReadFact"]
                def description(node): return f"{node['record_ref']}) field {node['field']}"
                fact1 = description(reads[0]) if reads else None
                fact2 = description(reads[1]) if len(reads)>1 else fact1
                if kind == "read": selectors = ["ReadFact", fact1]
                elif kind == "compare": selectors = ["Compare", fact1, fact2, ref["program"]["nodes"][2]["relation"]]
                elif kind == "difference": selectors = ["ExactDifference", fact1, fact2]
                elif kind == "conjoin": selectors = ["Conjoin", fact1, fact2]
                elif kind == "hypothesis": selectors = ["Hypothesize", "Assumption alternate-warranty", fact2, "greater_than"]
                elif kind == "plan": selectors = ["Plan", "Workflow catalog-review", "Step inspect:", "Step draft:"]
                else: selectors = [kind, f"record {ref['program']['record_ref']}"]
                if level == "L3": selectors.append(task["production"])
                with self.subTest(level=level, kind=kind):
                    parent = FakeParent(selectors)
                    result = plan_gate_task(task, parent)
                    row = score_checked(task, ref, result)
                    self.assertTrue(row["correct"], (result.status, result.reason, result.trace))
                    self.assertEqual(result.provider_calls, len(selectors))
                    self.assertTrue(parent.requests)
                    for request in parent.requests:
                        self.assertNotIn("evaluator_only", request["state"])
                        self.assertNotIn("reference_program", request["state"])

    def test_wrong_task_and_altered_surface_fail(self):
        tasks, refs = self.suite()
        task, ref = tasks[0], refs[0]
        wanted = ref["program"]["nodes"][0]["record_ref"]
        wrong = next(r["record_ref"] for r in task["catalog"]["records"] if r["record_ref"] != wanted)
        result = plan_gate_task(task, FakeParent(["ReadFact", f"{wrong}) field warranty_months"]))
        self.assertEqual(result.status, "accepted")
        self.assertFalse(score_checked(task, ref, result)["correct"])
        result.text += " Unsupported claim."
        self.assertTrue(score_checked(task, ref, result)["unsupported_release"])

    def test_call_option_node_budgets_and_missing_registry_mapping_fail_closed(self):
        tasks, refs = self.suite()
        task, ref = deepcopy(tasks[0]), deepcopy(refs[0])
        wanted = ref["program"]["nodes"][0]["record_ref"]
        selectors = ["ReadFact", f"{wanted}) field warranty_months"]
        task["host"]["limits"]["max_proposal_calls"] = 1
        result = plan_gate_task(task, FakeParent(selectors))
        self.assertEqual((result.status,result.reason,result.provider_calls), ("exhausted","max_proposal_calls",1))
        task = deepcopy(tasks[0])
        task["host"]["limits"]["max_options_per_question"] = 1
        result = plan_gate_task(task, FakeParent(selectors))
        self.assertEqual(result.provider_calls, 0)
        self.assertNotEqual(result.status,"accepted")
        task = deepcopy(tasks[0])
        task["host"]["limits"]["max_program_nodes"] = 1
        result = plan_gate_task(task, FakeParent(selectors))
        self.assertEqual((result.status,result.reason), ("kernel_rejected","exceeded_limit"))
        task = deepcopy(tasks[0])
        del task["catalog"]["registries"][task["host"]["registry_ref"]]["field_predicates"]["warranty_months"]
        result = plan_gate_task(task, FakeParent(selectors))
        self.assertNotEqual(result.status,"accepted")
        self.assertIsNone(result.text)

    def test_all_false_relations_remain_offered(self):
        tasks, refs = self.suite()
        task, ref = tasks[1], refs[1]
        true_relation = ref["program"]["nodes"][2]["relation"]
        false_relation = "less_than" if true_relation != "less_than" else "greater_than"
        left,right=ref["program"]["nodes"][:2]
        parent = FakeParent(["Compare", f"{left['record_ref']}) field warranty_months", f"{right['record_ref']}) field warranty_months", false_relation])
        result = plan_gate_task(task, parent)
        self.assertEqual(result.status, "kernel_rejected")
        offered = next(iter(parent.requests[-1]["questions"].values()))["criteria"]
        self.assertEqual(set(offered.values()), {"equal", "less_than", "greater_than"})
        self.assertFalse(score_checked(task, ref, result)["correct"])

    def test_comparator_raw_answers_missing_errors_and_equivalences(self):
        tasks, refs = self.suite()
        rows = [{"task_id": t["task_id"], "input_sha256": digest(t), "status": "ok", "raw_text": json.dumps(r["meaning"])} for t, r in zip(tasks, refs)]
        scored = score_comparator(tasks, refs, rows)
        self.assertTrue(all(r["correct"] for r in scored))
        rows[0]["raw_text"] = '{"claims":[],"explanations":0,"correct":true}'
        rows.pop()
        scored = score_comparator(tasks, refs, rows)
        self.assertFalse(scored[0]["correct"])
        self.assertFalse(scored[-1]["correct"])
        answer = deepcopy(refs[1]["meaning"])
        claim = answer["claims"][0]
        claim["left"], claim["right"] = claim["right"], claim["left"]
        claim["relation"] = {"greater_than": "less_than", "less_than": "greater_than", "equal": "equal"}[claim["relation"]]
        claim["left"]["value"] = str(claim["left"]["value"]) + ".00"
        self.assertEqual(normalized_answer(answer), normalized_answer(refs[1]["meaning"]))
        claim["left"]["value"] = True
        self.assertNotEqual(normalized_answer(answer), normalized_answer(refs[1]["meaning"]))

    def test_world_clusters_and_duplicate_rejection(self):
        rows = [{"task_id": str(i), "world_id": str(i//2), "a": True, "b": False} for i in range(4)]
        ci = paired_ci(rows, "a", "b", draws=100)
        self.assertEqual((ci["point"], ci["lo"], ci["hi"]), (1,1,1))
        self.assertEqual(ci["n_worlds"], 2)
        self.assertFalse(ci["minimum_worlds_met"])
        with self.assertRaises(ValueError): paired_ci(rows + rows[:1], "a", "b", draws=100)

    def test_incomplete_references_never_truncate_denominator(self):
        tasks, refs = self.suite()
        with self.assertRaises(ValueError): score_comparator(tasks, refs[:1], [])
        with self.assertRaises(ValueError): run_joint_adversarial(tasks, refs[:1], None)
        with self.assertRaises(ValueError): intervention_comparison(tasks, refs[:1], {}, draws=100)

    def test_joint_adversarial_requires_specific_rejection_not_provider_failure(self):
        tasks, refs = self.suite()
        wanted = refs[0]["program"]["nodes"][0]["record_ref"]
        report = run_joint_adversarial(tasks[:1], refs[:1], FakeParent(["ReadFact", f"{wanted}) field warranty_months"]))
        self.assertTrue(report["criterion_observed"], report)
        failed = PlannerResult("provider_error", None, None, "backend_error")
        with patch("language_growth.gate_planner.plan_gate_task", return_value=failed):
            report = run_joint_adversarial(tasks[:1], refs[:1], None)
        self.assertFalse(report["criterion_observed"])
        self.assertTrue(all(not row["matched"] for row in report["observations"]))

    def test_interventions_cannot_pass_from_correct_booleans(self):
        tasks, refs = self.suite()
        result = PlannerResult("error", None, None, "missing")
        arms = {name: [result]*len(tasks) for name in ("semantic", "random", "output_only", "sham")}
        report = intervention_comparison(tasks, refs, arms, draws=100)
        self.assertFalse(report["passed"])
        self.assertFalse(report["criterion_observed"])
        arms.pop("random")
        with self.assertRaises(ValueError): intervention_comparison(tasks, refs, arms, draws=100)

    def test_authorized_dev_predictions_compute_outcomes_and_reject_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p, c, auth_path = root/"p.jsonl", root/"c.jsonl", root/"authorization.json"
            base = {"item_id": "one", "request_sha256": "a"*64, "checkpoint_sha256": "b"*64,
                    "probabilities": {"a": .8, "b": .2}, "correct": False}
            p.write_bytes(canonical(base)+b"\n")
            base["checkpoint_sha256"] = "c"*64
            c.write_bytes(canonical(base)+b"\n")
            auth = {"schema": "ssdm.language_authorized_dev_predictions.v1", "split": "DEV",
                    "purpose": "language_parent_decision_regression", "parent_checkpoint_sha256": "b"*64,
                    "candidate_checkpoint_sha256": "c"*64,
                    "parent_predictions_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "candidate_predictions_sha256": hashlib.sha256(c.read_bytes()).hexdigest(),
                    "items": [{"item_id": "one", "request_sha256": "a"*64, "cluster": "world", "options": ["a", "b"], "target": "a"}]}
            auth_path.write_bytes(canonical(auth))
            result = decision_regression(auth_path, hashlib.sha256(auth_path.read_bytes()).hexdigest(), p, c,
                                         candidate_sha256="c"*64, draws=100)
            self.assertEqual(result["accuracy_difference"]["point"], 0)
            self.assertFalse(result["passed"])
            auth["split"] = "TEST"
            auth_path.write_bytes(canonical(auth))
            with self.assertRaises(ValueError):
                decision_regression(auth_path, hashlib.sha256(auth_path.read_bytes()).hexdigest(), p, c,
                                    candidate_sha256="c"*64, draws=100)


if __name__ == "__main__":
    unittest.main()
