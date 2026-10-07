"""No-model controls for the fixed whole-world fit/calibration partition."""

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from language_growth.multirecord_calibration_split import (
    SPLIT_SEED, TRAINING_EXPORT_SHA256, _rows_bytes, calibration_world_ids,
    main, split_multirecord_training,
)
from language_growth.multirecord_data import MULTIWORLD_TRAIN_SPLIT, MULTIWORLD_VALIDATION_SPLIT
from language_growth.multirecord_eval import MULTIWORLD_FIXTURE_DIR, _canonical
from language_growth.multirecord_training import load_multirecord_training
from language_growth.training_records import L1_TRAINING_ALLOWLIST


class MultirecordCalibrationSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.source = load_multirecord_training()
        cls.partitions, cls.manifest = split_multirecord_training()
        cls.tasks = [json.loads(line) for line in
                     (MULTIWORLD_FIXTURE_DIR / f"{MULTIWORLD_TRAIN_SPLIT}.jsonl").read_bytes().splitlines()]

    def test_disjoint_complete_world_task_and_slot_membership_without_loss_or_duplication(self):
        fit, calibration = [self.partitions[p] for p in ("fit", "calibration")]
        worlds = [{r["metadata"]["world_id"] for r in rows} for rows in (fit, calibration)]
        tasks = [{r["id"] for r in rows} for rows in (fit, calibration)]
        self.assertEqual([len(w) for w in worlds], [24, 8])
        self.assertEqual([len(t) for t in tasks], [960, 320])
        self.assertEqual([len(r) for r in (fit, calibration)], [3072, 1024])
        self.assertFalse(worlds[0] & worlds[1])
        self.assertFalse(tasks[0] & tasks[1])
        self.assertEqual(worlds[0] | worlds[1], {w["world_id"] for w in self.source["worlds"]})
        expected = Counter(_canonical(r) for r in self.records)
        actual = Counter(_canonical(r) for r in fit + calibration)
        self.assertEqual(actual, expected)
        self.assertEqual(set(actual.values()), {1})
        for rows, membership in zip((fit, calibration), worlds):
            # Exact row equality includes all metadata, prior prefixes and labels;
            # the relative source order is unchanged within each output.
            self.assertEqual(rows, [r for r in self.records if r["metadata"]["world_id"] in membership])
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["metadata"]["world_id"]].append(row)
            for world, group in grouped.items():
                self.assertEqual((len(group), len({r["id"] for r in group})), (128, 40))
                slots = Counter(qid.rsplit("-", 1)[1] for r in group for qid in r["labels"])
                self.assertEqual(slots, {"operation": 40, "fact": 16, "left": 24, "right": 24, "relation": 24})
                task_rows = defaultdict(list)
                for row in group:
                    task_rows[row["id"]].append(row)
                for task in (t for t in self.tasks if t["world_id"] == world):
                    expected_qids = [next(iter(r["questions"])) for r in task["model_inputs"]]
                    self.assertEqual([next(iter(r["request"]["questions"]))
                                      for r in task_rows[task["task_id"]]], expected_qids)

    def test_identical_per_world_task_kind_field_relation_distributions(self):
        grouped = defaultdict(list)
        for task in self.tasks:
            grouped[task["world_id"]].append(task["evaluator_only"])
        baseline = None
        for world_id, tasks in grouped.items():
            kinds = Counter(t["kind"] for t in tasks)
            fields = Counter(t["field"] for t in tasks)
            formats = Counter(t["choice_format"] for t in tasks)
            joint = Counter((t["kind"], t["field"]) for t in tasks)
            relations = Counter((t["field"], t["program"]["nodes"][2]["relation"])
                                for t in tasks if t["kind"] == "compare")
            self.assertEqual(kinds, {"read": 8, "explain": 8, "compare": 24})
            self.assertEqual(fields, {"price": 20, "warranty_months": 20})
            self.assertEqual(formats, {"production": 20, "augmented": 20})
            for field in ("price", "warranty_months"):
                self.assertEqual({relation: relations[field, relation]
                                  for relation in ("equal", "less_than", "greater_than")},
                                 {"equal": 2, "less_than": 5, "greater_than": 5})
            signature = (kinds, fields, formats, joint, relations)
            if baseline is None:
                baseline = signature
            self.assertEqual(signature, baseline, world_id)

    def test_fixed_rule_determinism_and_input_order_independence(self):
        ids = [w["world_id"] for w in self.source["worlds"]]
        def independent_rank(w):
            return hashlib.sha256(json.dumps(["ssdm.l1_world_split.v1", 1708, w],
                                             separators=(",", ":")).encode("ascii")).hexdigest(), w
        expected = frozenset(sorted(ids, key=independent_rank)[:8])
        self.assertEqual(SPLIT_SEED, 1708)
        self.assertEqual(calibration_world_ids(ids), expected)
        self.assertEqual(calibration_world_ids(reversed(ids)), expected)
        self.assertEqual(calibration_world_ids(ids[7:] + ids[:7]), expected)
        self.assertEqual({r["metadata"]["world_id"] for r in self.partitions["calibration"]}, expected)
        # Repeated full loading/export/splitting is byte-deterministic too.
        partitions, manifest = split_multirecord_training()
        self.assertEqual(_canonical(manifest), _canonical(self.manifest))
        for name in partitions:
            self.assertEqual(_rows_bytes(partitions[name]), _rows_bytes(self.partitions[name]))

    def test_assignment_api_uses_only_ids_never_labels_or_row_content(self):
        changed = deepcopy(self.records)
        for row in changed:
            row["labels"] = {"arbitrary": "opposite"}
            row["request"] = {"arbitrary": "replaced task, evidence and prefix"}
        original_ids = {r["metadata"]["world_id"] for r in self.records}
        mutated_ids = {r["metadata"]["world_id"] for r in changed}
        self.assertEqual(calibration_world_ids(original_ids), calibration_world_ids(mutated_ids))
        # Altered exports cannot reach assignment even with genuine cohort metadata.
        with patch("language_growth.multirecord_calibration_split.load_multirecord_training",
                   return_value=(changed, self.source)):
            with patch("language_growth.multirecord_calibration_split.calibration_world_ids") as assignment:
                with self.assertRaisesRegex(ValueError, "export SHA-256 mismatch"):
                    split_multirecord_training()
                assignment.assert_not_called()

    def test_manifest_exact_counts_hashes_lineage_and_zero_scope(self):
        manifest = self.manifest
        self.assertEqual(manifest["source_training_export_sha256"], TRAINING_EXPORT_SHA256)
        self.assertEqual(hashlib.sha256(_rows_bytes(self.records)).hexdigest(), TRAINING_EXPORT_SHA256)
        self.assertEqual(manifest["source_cohort_hashes"], L1_TRAINING_ALLOWLIST[MULTIWORLD_TRAIN_SPLIT])
        self.assertEqual(manifest["source_manifest"], self.source)
        self.assertEqual((manifest["source_world_count"], manifest["source_task_count"], manifest["source_row_count"]),
                         (32, 1280, 4096))
        for name, rows in self.partitions.items():
            summary = manifest["partitions"][name]
            self.assertEqual(summary["row_count"], len(rows))
            self.assertEqual(summary["task_count"], len({r["id"] for r in rows}))
            self.assertEqual(summary["world_count"], len(summary["worlds"]))
            self.assertEqual(summary["sha256"], hashlib.sha256(_rows_bytes(rows)).hexdigest())
            self.assertEqual(summary["bytes"], len(_rows_bytes(rows)))
            self.assertTrue(all(w["row_count"] == 128 and len(w["task_ids"]) == 40 for w in summary["worlds"]))
        self.assertEqual(manifest["evaluation_only_cohort"]["split"], MULTIWORLD_VALIDATION_SPLIT)
        self.assertIs(manifest["evaluation_only_cohort"]["used_in_split"], False)
        self.assertEqual(manifest["cost_usd"], 0)
        for key in ("model_execution", "training_performed", "calibration_performed"):
            self.assertIs(manifest[key], False)

    def test_validation_and_one_world_smoke_fail_before_file_access(self):
        for split in (MULTIWORLD_VALIDATION_SPLIT, "l1_development_train", "l1_development_validation",
                      "development_train", "development_validation", "heldout"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                split_multirecord_training(split=split, fixture_dir=Path("does-not-exist"))
        for ids in ([], ["one-world"], ["duplicate"] * 32, list(range(32))):
            with self.assertRaises(ValueError):
                calibration_world_ids(ids)

    def test_substituted_validation_or_smoke_bytes_cannot_bypass_train_name(self):
        smoke = Path(__file__).resolve().parents[2] / "receipts/2026-10-07-L1-multirecord-data"
        for directory, source_split in ((MULTIWORLD_FIXTURE_DIR, MULTIWORLD_VALIDATION_SPLIT),
                                        (smoke, "l1_development_train")):
            with tempfile.TemporaryDirectory() as temporary:
                for suffix in (".jsonl", "-manifest.json"):
                    Path(temporary, MULTIWORLD_TRAIN_SPLIT + suffix).write_bytes(
                        (directory / (source_split + suffix)).read_bytes())
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    split_multirecord_training(fixture_dir=Path(temporary))

    def test_atomic_publication_failure_cleanup_and_existing_output_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "split"
            original_write = Path.write_bytes
            def fail_manifest(path, data):
                if path.name == "split-manifest.json":
                    self.assertTrue((path.parent / "fit.jsonl").exists())
                    self.assertTrue((path.parent / "calibration.jsonl").exists())
                    raise OSError("injected split manifest write failure")
                return original_write(path, data)
            with patch("sys.argv", ["multirecord_calibration_split", "--out", str(output)]):
                with patch("language_growth.multirecord_calibration_split.split_multirecord_training",
                           return_value=(self.partitions, self.manifest)):
                    with patch.object(Path, "write_bytes", fail_manifest):
                        with self.assertRaisesRegex(OSError, "injected split manifest write failure"):
                            main()
                    self.assertFalse(output.exists())
                    self.assertEqual(list(Path(directory).iterdir()), [])
                    main()
                saved = {p.name: p.read_bytes() for p in output.iterdir()}
                self.assertEqual(set(saved), {"fit.jsonl", "calibration.jsonl", "split-manifest.json"})
                self.assertEqual(json.loads(saved["split-manifest.json"]), self.manifest)
                for name in self.partitions:
                    self.assertEqual(saved[f"{name}.jsonl"], _rows_bytes(self.partitions[name]))
                with self.assertRaises(FileExistsError):
                    main()
                self.assertEqual(saved, {p.name: p.read_bytes() for p in output.iterdir()})
                self.assertEqual(len(list(Path(directory).iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
