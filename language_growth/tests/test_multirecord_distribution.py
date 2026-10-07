"""Pinned multiworld coverage and separation controls; no model execution."""

from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import unittest

from language_growth.development_data import development_split_spec
from language_growth.multirecord_data import (
    MULTIWORLD_COUNT, MULTIWORLD_SEED, MULTIWORLD_TRAIN_SPLIT,
    MULTIWORLD_VALIDATION_SPLIT, generate_multirecord_tasks,
    multirecord_manifest, multirecord_rows, multirecord_split_spec,
)
from language_growth.multirecord_eval import (
    MULTIWORLD_FIXTURE_DIR, _canonical, _PINS, load_multirecord_evaluation,
)
from language_growth.training_records import L1_TRAINING_ALLOWLIST, export_training_records

SPLITS = (MULTIWORLD_TRAIN_SPLIT, MULTIWORLD_VALIDATION_SPLIT)
FIELDS = ("price", "warranty_months")
REFS = {f"l1-development-record-{i}" for i in range(1, 5)}


class MultirecordDistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohorts = {}
        for split in SPLITS:
            data = (MULTIWORLD_FIXTURE_DIR / f"{split}.jsonl").read_bytes()
            manifest_bytes = (MULTIWORLD_FIXTURE_DIR / f"{split}-manifest.json").read_bytes()
            cls.cohorts[split] = (
                [json.loads(line) for line in data.splitlines()], json.loads(manifest_bytes),
                hashlib.sha256(data).hexdigest(), hashlib.sha256(manifest_bytes).hexdigest(),
            )

    def test_exact_pinned_regeneration_and_unique_world_hashes(self):
        for split, (tasks, manifest, rows_hash, manifest_hash) in self.cohorts.items():
            pins = L1_TRAINING_ALLOWLIST[split] if split == MULTIWORLD_TRAIN_SPLIT else None
            expected = (pins["sample_sha256"], pins["manifest_sha256"]) if pins else _PINS[split]
            self.assertEqual((rows_hash, manifest_hash), expected)
            self.assertEqual(_canonical(manifest), _canonical(multirecord_manifest(tasks)))
            regenerated = generate_multirecord_tasks(MULTIWORLD_SEED, MULTIWORLD_COUNT, split=split)
            self.assertEqual(rows_hash, hashlib.sha256(multirecord_rows(regenerated)).hexdigest())
            self.assertEqual((manifest["world_count"], manifest["task_count"]), (32, 1280))
            for key in ("world_id", "world_index", "context_ref", "catalog_sha256", "rows_sha256"):
                self.assertEqual(len({w[key] for w in manifest["worlds"]}), 32)
            self.assertEqual(len({t["task_id"] for t in tasks}), 1280)

    def test_every_world_complete_and_balanced_with_paired_ordered_comparisons(self):
        for tasks, manifest, *_ in self.cohorts.values():
            worlds = defaultdict(list)
            for task in tasks:
                worlds[task["world_id"]].append(task)
            self.assertEqual(set(worlds), {w["world_id"] for w in manifest["worlds"]})
            for world in manifest["worlds"]:
                rows = worlds[world["world_id"]]
                self.assertEqual(len(rows), 40)
                self.assertEqual({t["evaluator_only"]["task_index"] for t in rows}, set(range(40)))
                self.assertEqual(world["task_ids"], [t["task_id"] for t in rows])
                self.assertEqual(world["rows_sha256"], hashlib.sha256(multirecord_rows(rows)).hexdigest())
                catalog = rows[0]["evaluator_only"]["catalog"]
                self.assertEqual(world["catalog_sha256"], hashlib.sha256(_canonical(catalog).encode("ascii")).hexdigest())
                self.assertEqual({r["record_ref"] for r in catalog["records"]}, REFS)
                self.assertTrue(all(t["evaluator_only"]["catalog"] == catalog for t in rows))
                self.assertEqual(Counter(t["evaluator_only"]["kind"] for t in rows),
                                 {"read": 8, "explain": 8, "compare": 24})
                self.assertEqual(Counter(t["evaluator_only"]["field"] for t in rows),
                                 {"price": 20, "warranty_months": 20})
                self.assertEqual(Counter(t["evaluator_only"]["choice_format"] for t in rows),
                                 {"production": 20, "augmented": 20})
                pairs = defaultdict(list)
                for field in FIELDS:
                    for kind in ("read", "explain"):
                        support = {(t["evaluator_only"]["program"]["nodes"][0]["record_ref"],
                                    t["evaluator_only"]["field"]) for t in rows
                                   if t["evaluator_only"]["kind"] == kind and t["evaluator_only"]["field"] == field}
                        self.assertEqual(support, {(ref, field) for ref in REFS})
                    comparisons = [t["evaluator_only"] for t in rows
                                   if t["evaluator_only"]["kind"] == "compare" and t["evaluator_only"]["field"] == field]
                    self.assertEqual(Counter(g["program"]["nodes"][2]["relation"] for g in comparisons),
                                     {"equal": 2, "less_than": 5, "greater_than": 5})
                    self.assertEqual({(g["program"]["nodes"][0]["record_ref"],
                                       g["program"]["nodes"][1]["record_ref"]) for g in comparisons},
                                     {(a, b) for a in REFS for b in REFS if a != b})
                    records = {r["record_ref"]: r for r in catalog["records"]}
                    for gold in comparisons:
                        nodes = gold["program"]["nodes"]
                        a, b = [Decimal(str(records[n["record_ref"]]["fields"][field]["value"])) for n in nodes[:2]]
                        expected = "equal" if a == b else ("less_than" if a < b else "greater_than")
                        self.assertEqual(nodes[2]["relation"], expected)
                        pairs[gold["comparison_pair_id"]].append(nodes)
                self.assertEqual(len(pairs), 12)
                for pair in pairs.values():
                    self.assertEqual(len(pair), 2)
                    self.assertEqual((pair[0][0]["record_ref"], pair[0][1]["record_ref"]),
                                     (pair[1][1]["record_ref"], pair[1][0]["record_ref"]))

    def test_bindings_change_across_worlds_for_every_name_and_record_field(self):
        for split, (tasks, *_rest) in self.cohorts.items():
            names, values = defaultdict(set), defaultdict(set)
            for task in tasks[::40]:
                for record in task["evaluator_only"]["catalog"]["records"]:
                    names[record["entity_ref"]].add(record["record_ref"])
                    for field in FIELDS:
                        values[(record["record_ref"], field)].add(record["fields"][field]["value"])
            self.assertEqual(set(names), set(multirecord_split_spec(split)["names"]))
            self.assertTrue(all(refs == REFS for refs in names.values()))
            self.assertEqual(len(values), 8)
            self.assertTrue(all(len(v) >= 16 for v in values.values()))

    def test_train_and_validation_domains_and_worlds_are_disjoint(self):
        train, validation = [self.cohorts[s] for s in SPLITS]
        for key in ("world_id", "context_ref", "catalog_sha256", "rows_sha256"):
            self.assertFalse({w[key] for w in train[1]["worlds"]} & {w[key] for w in validation[1]["worlds"]})
        self.assertFalse({t["task_id"] for t in train[0]} & {t["task_id"] for t in validation[0]})
        specs = [development_split_spec(s) for s in ("development_train", "development_validation")]
        for split, (tasks, *_rest) in self.cohorts.items():
            spec = multirecord_split_spec(split)
            for other in specs:
                self.assertFalse(set(spec["names"]) & set(other["names"]))
                for key in ("price_cents_range", "warranty_range"):
                    self.assertTrue(spec[key][1] < other[key][0] or other[key][1] < spec[key][0])
                for kind, template in spec["templates"].items():
                    self.assertNotIn(template, other["templates"][kind])
            for task in tasks[::40]:
                for record in task["evaluator_only"]["catalog"]["records"]:
                    self.assertIn(record["entity_ref"], spec["names"])
                    for field, key in (("price", "price_cents_range"), ("warranty_months", "warranty_range")):
                        value = Decimal(str(record["fields"][field]["value"])) * (100 if field == "price" else 1)
                        self.assertTrue(spec[key][0] <= value <= spec[key][1])
            specs.append(spec)

    def test_validation_never_exports_even_when_relabelled_as_train(self):
        self.assertEqual(set(L1_TRAINING_ALLOWLIST), {MULTIWORLD_TRAIN_SPLIT})
        tasks, manifest, *_ = self.cohorts[MULTIWORLD_VALIDATION_SPLIT]
        for split in (MULTIWORLD_VALIDATION_SPLIT, MULTIWORLD_TRAIN_SPLIT,
                      "development_train", "development_validation", "l1_development_train"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                export_training_records(tasks, split=split, l1_manifest=manifest)
        relabelled = json.loads(json.dumps(tasks))
        for task in relabelled:
            task["development_split"] = MULTIWORLD_TRAIN_SPLIT
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            export_training_records(relabelled, split=MULTIWORLD_TRAIN_SPLIT,
                                    l1_manifest=self.cohorts[MULTIWORLD_TRAIN_SPLIT][1])

    def test_only_new_validation_adapts_initial_input_with_l1_identity(self):
        with self.assertRaises(ValueError):
            load_multirecord_evaluation(MULTIWORLD_TRAIN_SPLIT)
        for split in (MULTIWORLD_VALIDATION_SPLIT,):
            adapted = load_multirecord_evaluation(split)
            self.assertEqual(len(adapted), 1280)
            for task in adapted:
                self.assertEqual(task["development_split"], split)
                self.assertEqual(len(task["model_inputs"]), 1)
                self.assertEqual(set(task["model_inputs"][0]), {"state"})
                state = json.loads(task["model_inputs"][0]["state"])
                self.assertEqual(state["proposed_slot_choices"], [])
                self.assertEqual(state["accepted_partial_program"], [])
                self.assertEqual(len(state["records"]), 4)

    def test_original_one_world_source_and_manifest_bytes_unchanged(self):
        fixture_dir = Path(__file__).resolve().parents[2] / "receipts/2026-10-07-L1-multirecord-data"
        for split in ("l1_development_train", "l1_development_validation"):
            for suffix, digest in zip((".jsonl", "-manifest.json"), _PINS[split]):
                self.assertEqual(hashlib.sha256((fixture_dir / (split + suffix)).read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
