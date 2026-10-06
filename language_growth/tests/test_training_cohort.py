"""Small, temporary CPU cohorts only; no evaluation data or model loads."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from language_growth.development_data import development_manifest, generate_development_tasks
from language_growth.training_cohort import main, verify_records, write_training_cohort
from language_growth.training_records import export_training_records


class TrainingCohortTests(unittest.TestCase):
    def test_deterministic_bytes_hashes_counts_and_real_d_adapter(self):
        with tempfile.TemporaryDirectory() as folder:
            left, right = Path(folder) / "left", Path(folder) / "right"
            a = write_training_cohort(left, seed=17, count=6)
            b = write_training_cohort(right, seed=17, count=6)
            for name in ("training_records.jsonl", "development_manifest.json"):
                self.assertEqual((left / name).read_bytes(), (right / name).read_bytes())
                self.assertNotIn(b"\r", (left / name).read_bytes())
                self.assertTrue((left / name).read_bytes().endswith(b"\n"))
            self.assertEqual(a["hashes"], b["hashes"])
            self.assertEqual(a["counts"], {"tasks": 6, "records": 16, "slots": 16,
                "by_kind_tasks": {"compare": 2, "explain": 2, "read": 2},
                "by_kind_slots": {"compare": 8, "explain": 4, "read": 4},
                "by_slot": {"operation": 6, "fact": 4, "left": 2, "right": 2, "relation": 2},
                "by_choice_format_tasks": {"production": 3, "augmented": 3}})
            self.assertEqual(a["verification"]["item_count"], 16)
            self.assertTrue(a["verification"]["labels_verified"])
            self.assertFalse(a["verification"]["model_loaded"])
            self.assertEqual(a["verification"]["encoder"]["kind"], "byte+4")
            self.assertEqual(a["hardware"]["device"], "cpu")
            self.assertEqual(a["cost"]["incremental_paid_usd"], 0)
            self.assertEqual(len(a["source"]["sha"]), 40)
            self.assertIn("language_growth/training_cohort.py", a["source"]["files_sha256"])
            self.assertEqual(json.loads((left / "receipt.json").read_bytes()), a)
            rows = (left / "training_records.jsonl").read_bytes().splitlines(keepends=True)
            self.assertEqual(len(rows), len(a["hashes"]["records"]))
            for line, entry in zip(rows, a["hashes"]["records"]):
                row = json.loads(line)
                self.assertEqual(entry["sha256"], hashlib.sha256(line).hexdigest())
                self.assertEqual(entry["item_id"], row["id"] + "/" + next(iter(row["labels"])))
            self.assertEqual(a["hashes"]["training_records_sha256"],
                             hashlib.sha256(b"".join(rows)).hexdigest())
            self.assertEqual(a["hashes"]["development_manifest_sha256"],
                             hashlib.sha256((left / "development_manifest.json").read_bytes()).hexdigest())
            tasks = generate_development_tasks(17, 6)
            self.assertEqual(a["hashes"]["tasks_sha256"], development_manifest(tasks)["sample_sha256"])
            self.assertEqual(len(a["hashes"]["tasks"]), 6)
            c = write_training_cohort(Path(folder) / "other-seed", seed=18, count=6)
            self.assertNotEqual(a["hashes"]["training_records_sha256"], c["hashes"]["training_records_sha256"])

    def test_nontrain_splits_refused_before_generation_or_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "refused"
            with patch("language_growth.training_cohort.generate_development_tasks") as generator:
                for split in ("development_validation", "audit", "Index", "held_out"):
                    with self.subTest(split=split), self.assertRaisesRegex(ValueError, "development_train"):
                        write_training_cohort(output, seed=17, count=3, split=split)
                    self.assertFalse(output.exists())
                with self.assertRaises(SystemExit) as error:
                    main(["--output", str(output), "--seed", "17", "--count", "3",
                          "--split", "development_validation"])
                self.assertEqual(error.exception.code, 2)
                generator.assert_not_called()

    def test_current_targets_and_programs_do_not_enter_requests(self):
        tasks = generate_development_tasks(17, 3)
        before = export_training_records(tasks)
        changed = deepcopy(tasks)
        for task in changed:
            # The last slot has no later teacher-forced consumer. Changing its
            # label must leave every request byte-identical.
            target = task["training_only"]["targets"][-1]
            options = next(iter(task["model_inputs"][-1]["questions"].values()))["criteria"]
            target["option_id"] = next(oid for oid in options if oid != target["option_id"])
            task["training_only"]["program"] = {"private_gold": "MUST_NOT_ENTER_REQUEST"}
        after = export_training_records(changed)
        self.assertEqual([r["request"] for r in before], [r["request"] for r in after])
        self.assertNotEqual([r["labels"] for r in before], [r["labels"] for r in after])
        verify_records(changed, after)
        for task in tasks:
            for index, request in enumerate(task["model_inputs"]):
                self.assertNotIn("MUST_NOT_ENTER_REQUEST", json.dumps(request))
                state = json.loads(request["state"])
                self.assertEqual(state["accepted_partial_program"], [])
                self.assertEqual(len(state["proposed_slot_choices"]), index)
                self.assertNotIn("labels", request)
                self.assertNotIn("targets", state)
                self.assertNotIn("program", state)

    def test_injected_supervision_is_rejected_at_all_request_boundaries(self):
        for location in ("request", "state", "ssdm", "question"):
            for key in ("labels", "targets", "program"):
                with self.subTest(location=location, key=key):
                    tasks = generate_development_tasks(17, 1)
                    request = tasks[0]["model_inputs"][0]
                    if location == "state":
                        state = json.loads(request["state"])
                        state[key] = "MUST_NOT_ENTER_REQUEST"
                        request["state"] = json.dumps(state, sort_keys=True, separators=(",", ":"))
                    else:
                        container = (request if location == "request" else request["ssdm"]
                                     if location == "ssdm" else next(iter(request["questions"].values())))
                        container[key] = "MUST_NOT_ENTER_REQUEST"
                    with self.assertRaises(ValueError):
                        verify_records(tasks, export_training_records(tasks))

    def test_nonempty_directory_and_file_refused_before_generation(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "cohort"
            output.mkdir()
            preserved = output / "unique.txt"
            preserved.write_bytes(b"preserve unique evidence")
            with patch("language_growth.training_cohort.generate_development_tasks") as generator:
                for target in (output, preserved):
                    with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                        write_training_cohort(target, seed=17, count=3)
                generator.assert_not_called()
            self.assertEqual(list(output.iterdir()), [preserved])
            self.assertEqual(preserved.read_bytes(), b"preserve unique evidence")

    def test_existing_empty_output_and_cli_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "cohort"
            output.mkdir()
            argv = ["--output", str(output), "--seed", "17", "--count", "3",
                    "--choice-format", "augmented"]
            self.assertEqual(main(argv), 0)
            receipt = json.loads((output / "receipt.json").read_bytes())
            self.assertEqual(receipt["command"][3:], argv)
            self.assertEqual(receipt["counts"]["by_choice_format_tasks"], {"production": 0, "augmented": 3})
            self.assertEqual(set(p.name for p in output.iterdir()),
                             {"training_records.jsonl", "development_manifest.json", "receipt.json"})
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                write_training_cohort(output, seed=17, count=3)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})


if __name__ == "__main__":
    unittest.main()
