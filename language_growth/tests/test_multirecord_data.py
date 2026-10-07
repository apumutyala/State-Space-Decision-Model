"""No-model CPU checks of the synthetic L1 data foundation."""

import hashlib
import json
import unittest
from collections import Counter, defaultdict
from decimal import Decimal

from language_growth.development_data import development_split_spec
from language_growth.kernel import check_program
from language_growth.multirecord_data import (
    generate_multirecord_tasks, multirecord_manifest, multirecord_rows,
    multirecord_split_spec,
)


class MultirecordDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = generate_multirecord_tasks(1707, 2)

    def test_deterministic_worlds_and_manifest(self):
        self.assertEqual(self.tasks, generate_multirecord_tasks(1707, 2))
        self.assertNotEqual(self.tasks, generate_multirecord_tasks(1708, 2))
        manifest = multirecord_manifest(self.tasks)
        self.assertEqual(manifest, multirecord_manifest(generate_multirecord_tasks(1707, 2)))
        self.assertEqual(manifest["sample_sha256"], hashlib.sha256(multirecord_rows(self.tasks)).hexdigest())
        self.assertEqual((manifest["world_count"], manifest["task_count"]), (2, 80))
        self.assertEqual(len({t["task_id"] for t in self.tasks}), 80)
        self.assertEqual(len({w["context_ref"] for w in manifest["worlds"]}), 2)
        for world in manifest["worlds"]:
            rows = [t for t in self.tasks if t["world_id"] == world["world_id"]]
            self.assertEqual(world["rows_sha256"], hashlib.sha256(multirecord_rows(rows)).hexdigest())
            self.assertEqual(world["task_ids"], [t["task_id"] for t in rows])
            self.assertEqual(world["catalog_sha256"], hashlib.sha256(json.dumps(
                rows[0]["evaluator_only"]["catalog"], sort_keys=True, separators=(",", ":"),
                ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest())
        self.assertIn("not_independently_validated", manifest["label_status"])

    def test_exact_four_records_and_checker_valid_v01_programs(self):
        refs = {f"l1-development-record-{i}" for i in range(1, 5)}
        for task in self.tasks:
            gold = task["evaluator_only"]
            self.assertEqual({r["record_ref"] for r in gold["catalog"]["records"]}, refs)
            self.assertEqual(len(gold["catalog"]["records"]), 4)
            self.assertEqual(len({r["entity_ref"] for r in gold["catalog"]["records"]}), 4)
            self.assertEqual(gold["program"]["schema"], "ssdm.semantic_program.v0.1")
            self.assertTrue({n["op"] for n in gold["program"]["nodes"]} <=
                            {"ReadFact", "Compare", "Assert", "ExplainDerivation"})
            self.assertEqual(check_program(gold["program"], gold["host"], gold["catalog"]).status, "accepted")
            self.assertEqual(set(gold["catalog"]["snapshots"]), {gold["host"]["context_ref"]})
            self.assertEqual(gold["program"]["context_ref"], gold["host"]["context_ref"])
            self.assertTrue(task["development_only"])

    def test_no_authored_targets_or_program_prefix_in_model_requests(self):
        state_keys = {"schema", "task_input", "audience", "context_ref", "policy_ref", "registry_ref",
                      "records", "registry", "query_constants", "unmet_obligations",
                      "accepted_partial_program", "proposed_slot_choices"}
        forbidden = {"evaluator_only", "training_only", "targets", "target_option_id", "program",
                     "label_status", "expected", "kind", "seed", "task_index", "world_index"}

        def all_keys(value):
            if isinstance(value, dict):
                return set(value).union(*(all_keys(v) for v in value.values()))
            if isinstance(value, list):
                return set().union(*(all_keys(v) for v in value))
            return set()

        for task in self.tasks:
            for request, target in zip(task["model_inputs"], task["evaluator_only"]["targets"]):
                state = json.loads(request["state"])
                self.assertEqual(set(state), state_keys)
                self.assertEqual(state["accepted_partial_program"], [])
                self.assertEqual(state["proposed_slot_choices"], [])
                self.assertFalse(all_keys(request) & forbidden)
                self.assertFalse(all_keys(state) & forbidden)
                self.assertEqual([r["fields"] for r in state["records"]],
                                 [r["fields"] for r in task["evaluator_only"]["catalog"]["records"]])
                question = next(iter(request["questions"].values()))
                self.assertEqual(question["type"], "choice")
                self.assertIn(target["option_id"], question["criteria"])
                self.assertEqual(len(state["records"]), 4)

    def test_all_candidate_choices_and_values_survive_choice_shuffle(self):
        production = generate_multirecord_tasks(1707, 1, choice_format="production")
        augmented = generate_multirecord_tasks(1707, 1, choice_format="augmented")
        expected_facts = {f"Fact l1-development-record-{i} field {f}"
                          for i in range(1, 5) for f in ("price", "warranty_months")}
        shifted = False
        for p, a in zip(production, augmented):
            self.assertEqual(p["task_id"], a["task_id"])
            self.assertEqual(p["evaluator_only"]["catalog"], a["evaluator_only"]["catalog"])
            for pr, ar, pt, at in zip(p["model_inputs"], a["model_inputs"],
                                      p["evaluator_only"]["targets"], a["evaluator_only"]["targets"]):
                self.assertEqual(pr["state"], ar["state"])
                pc = next(iter(pr["questions"].values()))["criteria"]
                ac = next(iter(ar["questions"].values()))["criteria"]
                self.assertEqual(set(pc.values()), set(ac.values()))
                self.assertEqual(pc[pt["option_id"]], ac[at["option_id"]])
                shifted |= pc != ac
                if pt["slot"] in ("fact", "left", "right"):
                    self.assertEqual(set(pc.values()), expected_facts)
                    self.assertEqual(len(pc), 8)
                elif pt["slot"] == "relation":
                    self.assertEqual(set(pc.values()), {f"Relation {r}" for r in
                                     ("greater_than", "less_than", "equal")})
                    self.assertEqual(len(pc), 3)
                else:
                    self.assertEqual(len(pc), 3)
        self.assertTrue(shifted)

    def test_complete_record_field_pair_order_equality_and_value_coverage(self):
        worlds = defaultdict(list)
        for task in self.tasks:
            worlds[task["world_id"]].append(task)
        for tasks in worlds.values():
            self.assertEqual(Counter(t["evaluator_only"]["kind"] for t in tasks),
                             {"read": 8, "explain": 8, "compare": 24})
            pairs = defaultdict(list)
            read_support = defaultdict(set)
            for task in tasks:
                gold = task["evaluator_only"]
                nodes = gold["program"]["nodes"]
                if gold["kind"] != "compare":
                    read_support[gold["kind"]].add((nodes[0]["record_ref"], nodes[0]["field"]))
                    continue
                pairs[gold["comparison_pair_id"]].append(task)
                self.assertEqual(nodes[0]["field"], nodes[1]["field"])
                self.assertNotEqual(nodes[0]["record_ref"], nodes[1]["record_ref"])
                records = {r["record_ref"]: r for r in gold["catalog"]["records"]}
                left = records[nodes[0]["record_ref"]]
                right = records[nodes[1]["record_ref"]]
                lf, rf = left["fields"][gold["field"]], right["fields"][gold["field"]]
                self.assertEqual((lf["unit"], lf["temporal_scope"], lf["quantity_kind"]),
                                 (rf["unit"], rf["temporal_scope"], rf["quantity_kind"]))
                a, b = Decimal(str(lf["value"])), Decimal(str(rf["value"]))
                expected = "equal" if a == b else ("greater_than" if a > b else "less_than")
                self.assertEqual(nodes[2]["relation"], expected)
                text = json.loads(task["model_inputs"][0]["state"])["task_input"]
                self.assertLess(text.index(left["entity_ref"]), text.index(right["entity_ref"]))
            expected_support = {(f"l1-development-record-{i}", f) for i in range(1, 5)
                                for f in ("price", "warranty_months")}
            self.assertEqual(read_support, {"read": expected_support, "explain": expected_support})
            self.assertEqual(len(pairs), 12)
            for pair in pairs.values():
                self.assertEqual(len(pair), 2)
                a, b = [t["evaluator_only"]["program"]["nodes"] for t in pair]
                self.assertEqual((a[0]["record_ref"], a[1]["record_ref"]),
                                 (b[1]["record_ref"], b[0]["record_ref"]))
                self.assertEqual(a[2]["relation"], {"equal": "equal", "greater_than": "less_than",
                                                   "less_than": "greater_than"}[b[2]["relation"]])
            for field in ("price", "warranty_months"):
                values = [r["fields"][field]["value"] for r in tasks[0]["evaluator_only"]["catalog"]["records"]]
                self.assertEqual(sorted(Counter(values).values()), [1, 1, 2])
                comparisons = [t for t in tasks if t["evaluator_only"]["kind"] == "compare"
                               and t["evaluator_only"]["field"] == field]
                self.assertEqual(Counter(t["evaluator_only"]["program"]["nodes"][2]["relation"]
                                         for t in comparisons), {"equal": 2, "less_than": 5, "greater_than": 5})
                self.assertEqual({t["evaluator_only"]["program"]["nodes"][0]["record_ref"]
                                  for t in comparisons}, {f"l1-development-record-{i}" for i in range(1, 5)})

    def test_splits_disjoint_from_each_other_and_all_l0_development_domains(self):
        specs = [development_split_spec(s) for s in ("development_train", "development_validation")]
        for split in ("l1_development_train", "l1_development_validation"):
            spec = multirecord_split_spec(split)
            for other in specs:
                self.assertFalse(set(spec["names"]) & set(other["names"]))
                for key in ("warranty_range", "price_cents_range"):
                    self.assertTrue(spec[key][1] < other[key][0] or spec[key][0] > other[key][1])
                for kind, template in spec["templates"].items():
                    self.assertNotIn(template, other["templates"][kind])
            specs.append(spec)
            for task in generate_multirecord_tasks(1707, 1, split=split):
                self.assertEqual(task["development_split"], split)
                self.assertTrue(task["evaluator_only"]["template_id"].startswith(split + ":"))
                for record in task["evaluator_only"]["catalog"]["records"]:
                    self.assertIn(record["entity_ref"], spec["names"])
                    for field, key in (("price", "price_cents_range"), ("warranty_months", "warranty_range")):
                        value = Decimal(str(record["fields"][field]["value"])) * (100 if field == "price" else 1)
                        self.assertTrue(spec[key][0] <= value <= spec[key][1])

    def test_invalid_generation_and_incomplete_or_mixed_manifest_rejected(self):
        for seed, worlds in ((True, 1), (-1, 1), (0, True), (0, 0), (0, 101)):
            with self.subTest(seed=seed, worlds=worlds), self.assertRaises(ValueError):
                generate_multirecord_tasks(seed, worlds)
        for kwargs in ({"split": "audit"}, {"split": "development_train"}, {"choice_format": "unknown"}):
            with self.assertRaises(ValueError):
                generate_multirecord_tasks(0, 1, **kwargs)
        for tasks in ([], self.tasks[:39], self.tasks + self.tasks[:1],
                      self.tasks + generate_multirecord_tasks(0, 1),
                      self.tasks + generate_multirecord_tasks(1707, 1, split="l1_development_validation")):
            with self.assertRaises(ValueError):
                multirecord_manifest(tasks)


if __name__ == "__main__":
    unittest.main()
