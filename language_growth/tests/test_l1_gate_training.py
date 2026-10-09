"""Fit-only adapter tests for GatePlanner-supervised L1 semantic slots."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from language_growth.l1_gate_training import (
    EXPORT_SCHEMA,
    EXPECTED_FIT_WORLDS,
    EXPECTED_ROWS,
    EXPECTED_TASKS,
    export_l1_fit_training,
    generate_l1_fit_training_records,
)
from language_growth.gate_suites import digest
from language_growth.multirecord_calibration_split import split_multirecord_training
from language_growth.multirecord_data import TASKS_PER_WORLD


class L1GateTrainingTests(unittest.TestCase):
    def test_only_pinned_fit_worlds_replay_through_gateplanner(self):
        rows, manifest = generate_l1_fit_training_records(
            model="test-native-parent", parent_revision="pinned-test-revision")
        partitions, partition_manifest = split_multirecord_training()
        fit_worlds = frozenset(row["metadata"]["world_id"] for row in partitions["fit"])

        self.assertEqual(len(rows), EXPECTED_ROWS)
        self.assertEqual(manifest["tasks"], EXPECTED_TASKS)
        self.assertEqual(manifest["worlds"], EXPECTED_FIT_WORLDS)
        self.assertEqual(manifest["tasks_per_world"], TASKS_PER_WORLD)
        self.assertEqual(manifest["fit_world_ids"], sorted(fit_worlds))
        self.assertEqual(manifest["source_kind_counts"], {"compare": 576, "explain": 192, "read": 192})
        self.assertEqual(manifest["source_coverage"]["fields"], ["price", "warranty_months"])
        self.assertEqual(manifest["source_coverage"]["read_record_field_pairs_per_world"], 8)
        self.assertEqual(manifest["source_coverage"]["explain_record_field_pairs_per_world"], 8)
        self.assertEqual(manifest["source_coverage"]["ordered_comparison_field_triples_per_world"], 24)
        self.assertEqual(manifest["source_coverage"]["comparison_relations_per_field_in_fit"],
                         {"equal": 48, "greater_than": 120, "less_than": 120})
        self.assertEqual(manifest["slot_counts"], {
            "comparison-relation": 576,
            "fact": 384,
            "left-operand": 576,
            "right-operand": 576,
            "semantic-operation": 960,
        })
        target_counts = manifest["target_counts_by_slot"]
        self.assertEqual(target_counts["semantic-operation"], {
            "Compare": 576, "ExplainDerivation": 192, "ReadFact": 192})
        self.assertEqual(target_counts["comparison-relation"], {
            "equal": 96, "greater_than": 240, "less_than": 240})
        self.assertTrue(all(sum(target_counts[slot].values()) == count
                            for slot, count in manifest["slot_counts"].items()))
        self.assertEqual(manifest["target_counts_sha256"], digest(target_counts))
        self.assertFalse(manifest["validation_or_heldout_generated"])

        tasks = defaultdict(list)
        world_rows = defaultdict(list)
        for row in rows:
            meta = row["metadata"]
            self.assertEqual(meta["schema"], EXPORT_SCHEMA)
            self.assertIn(meta["source_world_id"], fit_worlds)
            self.assertEqual(meta["split"], "development_train")
            self.assertTrue(meta["diagnostic_teacher_forced"])
            self.assertFalse(meta["free_running"] or meta["promotion_eligible"])
            self.assertNotIn(meta["source_world_id"], repr(row["request"]))
            self.assertNotIn(meta["source_task_id"], repr(row["request"]))
            request_text = json.dumps(row["request"], sort_keys=True)
            for forbidden in ("source_world_id", "source_task_id", "reference_program", "training_only"):
                self.assertNotIn(forbidden, request_text)
            tasks[row["id"]].append(row)
            world_rows[meta["world_id"]].append(row)

        self.assertEqual(len(tasks), EXPECTED_TASKS)
        self.assertEqual(len(world_rows), EXPECTED_FIT_WORLDS)
        self.assertEqual({len(values) for values in world_rows.values()}, {128})
        self.assertTrue(all({row["group_id"] for row in values} == {world_id}
                            for world_id, values in world_rows.items()))
        self.assertEqual(len({row["metadata"]["source_task_id"] for row in rows}), EXPECTED_TASKS)

        # Source identity maps one-to-one into the adapter namespace; only the
        # fit cohort is emitted even though the pinned loader checks all splits.
        self.assertEqual(len(partitions["fit"]), EXPECTED_ROWS)
        self.assertEqual({w["world_id"] for w in partition_manifest["partitions"]["fit"]["worlds"]},
                         fit_worlds)
        self.assertEqual({len(w["task_ids"]) for w in partition_manifest["partitions"]["fit"]["worlds"]},
                         {TASKS_PER_WORLD})
        self.assertEqual({w["row_count"] for w in partition_manifest["partitions"]["fit"]["worlds"]},
                         {128})
        self.assertEqual(manifest["source_task_count"], EXPECTED_TASKS)

        slot_counts = Counter(row["metadata"]["slot"] for row in rows)
        self.assertEqual(dict(slot_counts), manifest["slot_counts"])
        self.assertGreater(manifest["slot_counts"]["semantic-operation"], 0)

    def test_deterministic_export_and_existing_output_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "first", Path(tmp) / "second"
            manifests = [export_l1_fit_training(
                path, model="test-native-parent", parent_revision="pinned-test-revision")
                for path in (first, second)]
            self.assertEqual(manifests[0], manifests[1])
            for name in ("training_records.jsonl", "training_records.jsonl.gz", "manifest.json"):
                a, b = [(path / name).read_bytes() for path in (first, second)]
                self.assertEqual(a, b)
            data = (first / "training_records.jsonl").read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), manifests[0]["training_records_sha256"])
            self.assertEqual(len(data.splitlines()), EXPECTED_ROWS)
            from train.run_cell import _cell_records, _development_split

            loaded = _cell_records(str(first / "training_records.jsonl.gz"), development=True)
            self.assertEqual(len(loaded), EXPECTED_ROWS)
            train_ids, eval_ids, split = _development_split(
                loaded, {"n_train_records": 18, "n_eval_records": 6, "seed": 1708})
            worlds_by_key = {key: loaded[key]["group_id"] for key in loaded}
            self.assertEqual(split["split_unit"], "world/task")
            self.assertTrue({worlds_by_key[key] for key in train_ids}.isdisjoint(
                worlds_by_key[key] for key in eval_ids))
            self.assertEqual(len(train_ids) + len(eval_ids), EXPECTED_ROWS)
            with self.assertRaisesRegex(ValueError, "already exists"):
                export_l1_fit_training(first)
            self.assertEqual(data, (first / "training_records.jsonl").read_bytes())


if __name__ == "__main__":
    unittest.main()
