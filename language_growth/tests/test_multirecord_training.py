"""No-model controls for pinned L1 teacher-forced data export."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from language_growth.development_data import generate_development_tasks
from language_growth.kernel import check_program
from language_growth.multirecord_eval import _canonical
from language_growth.multirecord_training import load_multirecord_training, main
from language_growth.parent_bridge import audience_state
from language_growth.training_records import L1_TRAINING_ALLOWLIST, export_training_records

FIXTURES = Path(__file__).resolve().parents[2] / "receipts/2026-10-07-L1-multirecord-data"
SPLIT = "l1_development_train"


def source():
    tasks = [json.loads(row) for row in (FIXTURES / f"{SPLIT}.jsonl").read_bytes().splitlines()]
    manifest = json.loads((FIXTURES / f"{SPLIT}-manifest.json").read_bytes())
    return tasks, manifest


class MultirecordTrainingTests(unittest.TestCase):
    def test_export_publication_failure_cleanup_success_and_retry_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "export"
            original_write = Path.write_bytes

            def fail_manifest(path, data):
                if path.name == "export-manifest.json":
                    self.assertTrue((path.parent / "training_records.jsonl").exists())
                    raise OSError("injected manifest write failure")
                return original_write(path, data)

            with patch("sys.argv", ["multirecord_training", "--out", str(output)]):
                with patch.object(Path, "write_bytes", fail_manifest):
                    with self.assertRaisesRegex(OSError, "injected manifest write failure"):
                        main()
                self.assertFalse(output.exists())
                self.assertEqual(list(Path(directory).iterdir()), [])
                main()
                saved = {p.name: p.read_bytes() for p in output.iterdir()}
                self.assertEqual(set(saved), {"training_records.jsonl", "export-manifest.json"})
                self.assertEqual(len(list(Path(directory).iterdir())), 1)
                manifest = json.loads(saved["export-manifest.json"])
                self.assertEqual(manifest["training_records_sha256"],
                                 hashlib.sha256(saved["training_records.jsonl"]).hexdigest())
                with self.assertRaises(FileExistsError):
                    main()
                self.assertEqual(saved, {p.name: p.read_bytes() for p in output.iterdir()})
                self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_every_slot_has_exact_prior_targets_and_current_label_is_separate(self):
        tasks, manifest = source()
        untouched = deepcopy(tasks)
        with patch("language_growth.multirecord_eval.check_program", wraps=check_program) as checker:
            records, actual_manifest = load_multirecord_training()
        self.assertEqual(checker.call_count, 40)
        self.assertEqual(actual_manifest, manifest)
        self.assertEqual((len(records), len({r["id"] for r in records})), (128, 40))
        cursor = 0
        for task in tasks:
            prior = []
            gold = task["evaluator_only"]
            for authored, target in zip(task["model_inputs"], gold["targets"]):
                row = records[cursor]
                cursor += 1
                state = json.loads(authored["state"])
                expected = json.loads(audience_state(state["task_input"], gold["host"], gold["catalog"]))
                expected["proposed_slot_choices"] = deepcopy(prior)
                request = deepcopy(authored)
                request["state"] = _canonical(expected)
                self.assertEqual(row["request"], request)
                qid, question = next(iter(request["questions"].items()))
                self.assertEqual(row["labels"], {qid: target["option_id"]})
                self.assertEqual(row["id"], task["task_id"])
                self.assertEqual(row["metadata"]["world_id"], task["world_id"])
                for forbidden in ("targets", "program", "labels", "evaluator_only", "training_only", "metadata"):
                    self.assertNotIn(f'"{forbidden}"', json.dumps(request))
                prior.append({**target, "description": question["criteria"][target["option_id"]]})
        self.assertEqual(tasks, untouched)

    def test_all_source_hashes_and_world_identities_survive_outside_requests(self):
        records, manifest = load_multirecord_training()
        worlds = {w["world_id"]: w for w in manifest["worlds"]}
        for row in records:
            meta = row["metadata"]
            world = worlds[meta["world_id"]]
            self.assertEqual(meta["task_id"], row["id"])
            self.assertIn(row["id"], world["task_ids"])
            for key, value in L1_TRAINING_ALLOWLIST[SPLIT].items():
                self.assertEqual(meta[key], value)
            self.assertEqual(meta["world_rows_sha256"], world["rows_sha256"])
            self.assertEqual(meta["catalog_sha256"], world["catalog_sha256"])
            self.assertEqual(meta["context_ref"], world["context_ref"])
        records[0]["metadata"]["world_id"] = "mutated"
        self.assertNotEqual(records[1]["metadata"]["world_id"], "mutated")

    def test_non_train_splits_fail_before_file_access(self):
        tasks, manifest = source()
        for split in ("l1_development_validation", "development_validation", "audit", "heldout"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                load_multirecord_training(split=split, fixture_dir=Path("does-not-exist"))
            if split != "development_validation":
                with self.assertRaises(ValueError):
                    export_training_records(tasks, split=split, l1_manifest=manifest)

    def test_source_bytes_and_manifest_are_both_pinned(self):
        for suffix in (".jsonl", "-manifest.json"):
            with tempfile.TemporaryDirectory() as directory:
                for src in FIXTURES.glob(f"{SPLIT}*"):
                    Path(directory, src.name).write_bytes(src.read_bytes())
                target = Path(directory, SPLIT + suffix)
                target.write_bytes(target.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    load_multirecord_training(fixture_dir=Path(directory))

    def test_direct_export_cannot_bypass_pins_or_reference_check(self):
        tasks, manifest = source()
        for mutation in ("label", "prefix", "identity", "program", "subset", "manifest"):
            changed, metadata = deepcopy(tasks), deepcopy(manifest)
            if mutation == "label":
                changed[0]["evaluator_only"]["targets"][0]["option_id"] = "op003"
            elif mutation == "prefix":
                changed[0]["model_inputs"][0]["state"] += " "
            elif mutation == "identity":
                changed[0]["world_id"] += "-wrong"
            elif mutation == "program":
                changed[0]["evaluator_only"]["program"]["nodes"] = []
            elif mutation == "subset":
                changed.pop()
            else:
                metadata["seed"] += 1
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                export_training_records(changed, split=SPLIT, l1_manifest=metadata)
        with self.assertRaises(ValueError):
            export_training_records(tasks, split=SPLIT)
        rejected = type("Rejected", (), {"status": "rejected"})()
        with patch("language_growth.multirecord_eval.check_program", return_value=rejected):
            with self.assertRaisesRegex(ValueError, "reference program mismatch"):
                export_training_records(tasks, split=SPLIT, l1_manifest=manifest)

    def test_l0_output_is_identical_to_exact_pr128_base(self):
        # Digest recomputed using the exporter from exact 4538e9d base, seed17.
        rows = export_training_records(generate_development_tasks(17, 24, choice_format="mixed"))
        digest = hashlib.sha256(_canonical(rows).encode("ascii")).hexdigest()
        self.assertEqual(digest, "93a221534d4ebbe7cea4b1c5dc9b8950a547d31ac4f68a7a0fea0fa8ac18e1c6")
        self.assertTrue(all(set(row) == {"id", "request", "labels"} for row in rows))


if __name__ == "__main__":
    unittest.main()
