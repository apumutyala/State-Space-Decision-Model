import copy
from collections import Counter
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

from language_growth.development_data import development_split_spec
from language_growth.frozen_wording_eval import evaluate, MODEL_HASHES, MODEL_DIR
from language_growth.kernel import check_program
from language_growth.paraphrase_data import generate_paraphrase_tasks, paraphrase_manifest, TEMPLATES, NAMES, PRICE_CENTS_RANGE, WARRANTY_RANGE
from language_growth.planner import _candidate_sets, _as_options


class ParaphraseDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = generate_paraphrase_tasks()

    def test_deterministic_bounded_balance(self):
        self.assertEqual(self.tasks, generate_paraphrase_tasks())
        self.assertNotEqual(self.tasks, generate_paraphrase_tasks(161804))
        self.assertEqual(len({t["task_id"] for t in self.tasks}), 300)
        self.assertEqual(Counter(t["training_only"]["kind"] for t in self.tasks), dict(read=100, explain=100, compare=100))
        self.assertEqual(sorted(paraphrase_manifest(self.tasks)["template_counts"].values()), [33]*6+[34]*3)
        for seed in (True, -1, 1.5):
            with self.assertRaises(ValueError):
                generate_paraphrase_tasks(seed)

    def test_reference_programs_and_target_option_bindings(self):
        relations = set()
        for task in self.tasks:
            t = task["training_only"]
            self.assertEqual(check_program(t["program"], t["host"], t["catalog"]).status, "accepted")
            state = json.loads(task["model_inputs"][0]["state"])
            facts, operands, _ = _candidate_sets(t["host"], t["catalog"], state["task_input"])
            for target in t["targets"][1:]:
                slot = target["slot"]
                if slot == "relation":
                    relations.add(t["program"]["nodes"][2]["relation"])
                    options, binding = _as_options(["greater_than", "less_than", "equal"], lambda x: f"Relation {x}", "r")
                    self.assertEqual(binding[target["option_id"]], t["program"]["nodes"][2]["relation"])
                else:
                    items = facts if slot == "fact" else operands
                    options, binding = _as_options(items, lambda x: f"Fact {x['record_ref']} field {x['field']}", "f" if slot == "fact" else "v")
                    node = t["program"]["nodes"][1 if slot == "right" else 0]
                    self.assertEqual(binding[target["option_id"]]["record_ref"], node["record_ref"])
                    self.assertEqual(binding[target["option_id"]]["field"], node["field"])
            nodes = t["program"]["nodes"]
            names = {r["record_ref"]: r["entity_ref"] for r in t["catalog"]["records"]}
            if t["kind"] == "compare":
                self.assertLess(state["task_input"].index(names[nodes[0]["record_ref"]]), state["task_input"].index(names[nodes[1]["record_ref"]]))
        self.assertEqual(relations, {"greater_than", "less_than", "equal"})

    def test_labels_and_gold_prefixes_never_enter_model_inputs(self):
        for task in self.tasks:
            self.assertEqual(len(task["model_inputs"]), 1)
            state = json.loads(task["model_inputs"][0]["state"])
            self.assertEqual(state["proposed_slot_choices"], [])
            self.assertEqual(state["accepted_partial_program"], [])
            self.assertFalse(re.search(r"\b(read|explain|compare)\b", state["task_input"], re.I))
            self.assertFalse({"targets", "kind", "template_id", "training_only", "program", "gold"} & state.keys())
            altered = copy.deepcopy(task)
            altered["training_only"] = {"gold": "arbitrary"}
            self.assertEqual(task["model_inputs"], altered["model_inputs"])

    def test_domains_and_templates_are_disjoint_from_prior_development(self):
        for split in ("development_train", "development_validation"):
            prior = development_split_spec(split)
            self.assertFalse(set(NAMES) & set(prior["names"]))
            self.assertGreater(PRICE_CENTS_RANGE[0], prior["price_cents_range"][1])
            self.assertGreater(WARRANTY_RANGE[0], prior["warranty_range"][1])
            for kind, templates in TEMPLATES.items():
                self.assertEqual(len(set(templates)), 3)
                self.assertFalse(set(templates) & set(prior["templates"][kind]))

    def test_frozen_rollout_never_fits_and_retains_actual_scores(self):
        model = json.loads((MODEL_DIR/"model-17.json").read_text())
        before = copy.deepcopy(model)
        with patch("language_growth.transparent_cpu.fit", side_effect=AssertionError("no fitting")):
            raw = evaluate(self.tasks[:3], model, lambda: None)
        self.assertEqual(model, before)
        self.assertEqual(len(raw["observations"]), 3)
        self.assertTrue(raw["calls"])
        for call in raw["calls"]:
            self.assertTrue(call["logits"])
            self.assertIn("answers", call["response"])
        self.assertEqual(len(raw["slot_outcomes"]), 8)
        self.assertFalse(any(r["status"] == "runner_error" for r in raw["observations"]))


if __name__ == "__main__":
    unittest.main()
