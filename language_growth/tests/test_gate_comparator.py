"""Comparator protocol checks without a model, torch, accelerator or service."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from language_growth.gate_comparator import (
    ComparatorError, canonical_input_sha256, comparator_prompt, gate_task_context,
    generate_observations, local_model_identity,
)
from language_growth.tests.test_conjoin_planner import world
from language_growth.gate_answer_format import answer_format


def task():
    host, catalog = world()
    return {"task_id": "L0-validation-1", "world_id": "world-1", "level": "L0",
            "split": "development_validation", "task_input": "What is alpha's price?",
            "host": host, "catalog": catalog,
            "answer_format": answer_format()}


class ComparatorProtocolTests(unittest.TestCase):
    def test_no_labels_ids_or_invisible_records_in_prompt(self):
        item = task()
        item["catalog"]["records"].append({"record_ref": "secret-record", "entity_ref": "hidden",
                                            "fields": {"password": "SECRET-VALUE"}})
        prompt = comparator_prompt(item)
        for hidden in ("secret-record", "SECRET-VALUE", item["task_id"], item["world_id"]):
            self.assertNotIn(hidden, prompt)
        payload = json.loads(prompt.split("\n", 1)[1])
        self.assertEqual(payload["task"], item["task_input"])
        self.assertEqual(len(payload["evidence"]["records"]), 3)
        item["reference_program"] = {}
        with self.assertRaises(ComparatorError):
            comparator_prompt(item)

    def test_raw_generation_failure_and_input_identity_preserved(self):
        first = task()
        second = deepcopy(first)
        second["task_id"] = "L0-validation-2"
        second["task_input"] = "What is beta's price?"
        calls = []

        def generate(prompt):
            calls.append(prompt)
            if len(calls) == 2:
                raise RuntimeError("sensitive payload")
            return {"raw_text": "not JSON: no post-processing", "output_tokens": 9}

        result = generate_observations([first, second], generate, identity={"model_id": "fake"})
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["observations"][0]["raw_text"], "not JSON: no post-processing")
        self.assertEqual(result["observations"][0]["input_sha256"], canonical_input_sha256(first))
        self.assertEqual(result["observations"][1]["status"], "error")
        self.assertNotIn("sensitive payload", json.dumps(result))

    def test_extensions_match_permitted_host_definitions_without_truth_filtering(self):
        item = task()
        typed = lambda value: {"value": value, "value_type": "decimal", "unit": "USD", "temporal_scope": "2026-10"}
        item["host"]["assumptions"] = {
            "sale": {"record_ref": "a-v1", "field": "price", "value": typed("0.01")},
            "secret": {"record_ref": "secret-v1", "field": "password", "value": typed("12345.67")}}
        item["host"]["workflows"] = {
            "public-action": {"audience": "public", "status": "current", "steps": {
                "step": {"action_ref": "describe", "after": [], "preconditions": [
                    {"record_ref": "a-v1", "field": "price", "expected": typed("999.99")}]}}},
            "private-action": {"audience": "private", "status": "current", "steps": {}}}
        context = gate_task_context(item)
        self.assertEqual(set(context["assumptions"]), {"sale"})
        self.assertEqual(set(context["workflows"]), {"public-action"})
        self.assertIn("999.99", json.dumps(context))  # false precondition stays offered
        self.assertNotIn("12345.67", comparator_prompt(item))
        item["host"]["assumptions"]["sale"]["evaluator_only"] = "GOLD_SECRET"
        with self.assertRaises(ComparatorError):
            comparator_prompt(item)

    def test_energy_scope_is_shared_without_duplicate_unedited_values(self):
        from language_growth.domains.energy import load_energy_module, energy_environment
        module = load_energy_module()
        item = task()
        item["host"], item["catalog"] = energy_environment(module)
        item["energy_module"] = module
        context = gate_task_context(item)
        self.assertEqual(context["energy_interpretation"]["uncertainty"], module["uncertainty"])
        self.assertNotIn("rows", context["energy_interpretation"])

    def test_duplicate_and_invalid_rows_reject_before_any_generation(self):
        calls = []
        with self.assertRaises(ComparatorError):
            generate_observations([task(), task()], calls.append, identity={})
        bad = task()
        bad["task_id"] = "bad"
        bad["gold"] = "answer"
        with self.assertRaises(ComparatorError):
            generate_observations([task(), bad], calls.append, identity={})
        self.assertEqual(calls, [])

    def test_nested_answer_format_cannot_carry_gold_or_mutate_protocol(self):
        item = task()
        item["answer_format"]["expected_program"] = "GOLD_SECRET"
        with self.assertRaises(ComparatorError):
            comparator_prompt(item)
        self.assertNotIn("expected_program", answer_format())

    def test_snapshot_identity_changes_with_weights_and_cannot_claim_strength(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ComparatorError):
                local_model_identity(root, model_id="example/open", revision="fixed")
            weights = root / "model.safetensors"
            weights.write_bytes(b"synthetic fixture only")
            before = local_model_identity(root, model_id="example/open", revision="fixed")
            weights.write_bytes(b"changed synthetic fixture")
            after = local_model_identity(root, model_id="example/open", revision="fixed")
            self.assertNotEqual(before["files_sha256"], after["files_sha256"])
            self.assertEqual(before["strong_comparator_status"], "requires_same_task_evidence")


if __name__ == "__main__":
    unittest.main()
