"""No-model tests for the L1 split-before-trainer-consumer boundary."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest

from language_growth.l1_world_training import (
    CALIBRATION_RECORDS_SHA256,
    FIT_RECORDS_SHA256,
    PARTITION_MANIFEST_SHA256,
    load_l1_world_training_data,
)
from language_growth.multirecord_data import MULTIWORLD_TRAIN_SPLIT, MULTIWORLD_VALIDATION_SPLIT
from language_growth.multirecord_eval import MULTIWORLD_FIXTURE_DIR, _FIXTURE_DIR
from train.guard import SplitGuard


class TinyEncoder:
    """Minimal deterministic token IDs; the test never constructs or loads a model."""

    def __call__(self, text: str, trusted: bool) -> list[int]:
        return [1] if text else []

    def identity(self) -> dict:
        return {"kind": "test-only", "version": 1}


class L1WorldTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_l1_world_training_data()

    def test_exact_pins_counts_and_whole_world_disjointness(self):
        data = self.data
        self.assertEqual((len(data.fit_records), len(data.calibration_records), len(data.validation_tasks)),
                         (3072, 1024, 1280))
        self.assertEqual((len(data.fit_world_ids), len(data.calibration_world_ids), len(data.validation_world_ids)),
                         (24, 8, 32))
        self.assertFalse(data.fit_world_ids & data.calibration_world_ids)
        self.assertFalse(data.fit_world_ids & data.validation_world_ids)
        self.assertFalse(data.calibration_world_ids & data.validation_world_ids)
        self.assertEqual(FIT_RECORDS_SHA256,
                         data.partition_manifest["partitions"]["fit"]["sha256"])
        self.assertEqual(CALIBRATION_RECORDS_SHA256,
                         data.partition_manifest["partitions"]["calibration"]["sha256"])
        self.assertEqual(PARTITION_MANIFEST_SHA256,
                         hashlib.sha256((json.dumps(data.partition_manifest, sort_keys=True,
                             separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")).hexdigest())
        self.assertEqual(data.fit_world_ids | data.calibration_world_ids,
                         {w["world_id"] for w in data.partition_manifest["source_manifest"]["worlds"]})
        self.assertEqual(data.validation_world_ids,
                         {w["world_id"] for w in json.loads((MULTIWORLD_FIXTURE_DIR /
                              f"{MULTIWORLD_VALIDATION_SPLIT}-manifest.json").read_text(encoding="ascii"))["worlds"]})

    def test_records_keep_labels_and_world_lineage_outside_model_request(self):
        for record in self.data.fit_records + self.data.calibration_records:
            self.assertEqual(len(record["labels"]), 1)
            self.assertIn("world_id", record["metadata"])
            self.assertNotIn(record["metadata"]["world_id"], repr(record["request"]))
            self.assertNotIn("metadata", record["request"])
            self.assertNotIn("labels", record["request"])
        for task in self.data.validation_tasks:
            self.assertNotIn("targets", task["model_inputs"][0])
            self.assertIn("training_only", task)

    def test_fit_conversion_occurs_after_split_and_returns_world_sidecar(self):
        items, world_by_item_id = self.data.fit_items(
            TinyEncoder(), guard=SplitGuard.synthetic_only("pinned synthetic L1 development cohort"))
        self.assertEqual(len(items), 3072)
        self.assertEqual(len(world_by_item_id), len(items))
        self.assertEqual(set(world_by_item_id.values()), self.data.fit_world_ids)
        self.assertEqual(len(set(world_by_item_id)), len(items))
        for item in items:
            self.assertNotIn("world_id", item.row.meta)
            self.assertNotIn("labels", item.row.meta)
            self.assertIn(world_by_item_id[item.id], self.data.fit_world_ids)

    def test_reserved_calibration_rows_cannot_be_appended_before_fit_conversion(self):
        tampered = replace(
            self.data,
            fit_records=self.data.fit_records + self.data.calibration_records,
        )
        with self.assertRaisesRegex(ValueError, "fit records changed after the pinned world split"):
            tampered.fit_items(
                TinyEncoder(), guard=SplitGuard.synthetic_only("tampering regression"))

    def test_source_tampering_fails_before_any_items_are_built(self):
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td)
            for split in (MULTIWORLD_TRAIN_SPLIT, MULTIWORLD_VALIDATION_SPLIT):
                for suffix in (".jsonl", "-manifest.json"):
                    source = MULTIWORLD_FIXTURE_DIR / f"{split}{suffix}"
                    shutil.copyfile(source, dst / source.name)
            train_path = dst / f"{MULTIWORLD_TRAIN_SPLIT}.jsonl"
            train_path.write_bytes(train_path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_l1_world_training_data(fixture_dir=dst)

    def test_one_world_smoke_cohort_cannot_replace_the_multiworld_train_source(self):
        with tempfile.TemporaryDirectory() as td:
            dst = Path(td)
            for split in (MULTIWORLD_TRAIN_SPLIT, MULTIWORLD_VALIDATION_SPLIT):
                for suffix in (".jsonl", "-manifest.json"):
                    source = MULTIWORLD_FIXTURE_DIR / f"{split}{suffix}"
                    shutil.copyfile(source, dst / source.name)
            for suffix in (".jsonl", "-manifest.json"):
                smoke = _FIXTURE_DIR / f"l1_development_train{suffix}"
                shutil.copyfile(smoke, dst / f"{MULTIWORLD_TRAIN_SPLIT}{suffix}")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_l1_world_training_data(fixture_dir=dst)


if __name__ == "__main__":
    unittest.main()
