"""CPU-light checks for fresh train-only K4 operation pair generation."""
import unittest
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from language_growth.gate_operation_pairs import development_operation_pairs, export_operation_pairs
from language_growth.gate_suites import digest
from language_growth.parent_bridge import decision_request


class GateOperationPairTests(unittest.TestCase):
    def test_train_validation_pairs_are_deterministic_and_request_matched(self):
        train = development_operation_pairs(split="development_train", worlds=32, seed=20261011)
        self.assertEqual(train, development_operation_pairs(split="development_train", worlds=32, seed=20261011))
        validation = development_operation_pairs(split="development_validation", worlds=8, seed=20261012)
        self.assertEqual(len(train), 32)
        self.assertEqual(len(validation), 8)
        self.assertEqual({pair["split"] for pair in train}, {"development_train"})
        self.assertEqual({pair["split"] for pair in validation}, {"development_validation"})
        self.assertFalse({pair["world_id"] for pair in train} & {pair["world_id"] for pair in validation})

        for pair in train + validation:
            source, effect = pair["source"], pair["effect"]
            self.assertEqual(source["host"], effect["host"])
            self.assertEqual(source["catalog"], effect["catalog"])
            self.assertNotEqual(source["task_input"], effect["task_input"])
            self.assertEqual(source["question_id"], "slot-001-semantic-operation")
            for field in ("question_id", "question", "options"):
                self.assertEqual(source[field], effect[field])
            self.assertEqual(list(source["options"].values()),
                             ["ReadFact", "Compare", "ExplainDerivation", "ExactDifference"])
            self.assertEqual((pair["source_target"], pair["effect_target"]), ("v002", "v004"))
            rebuilt, _ = decision_request(
                **source, model="gate-operation-fit", parent_revision="operation-fit:v1",
                request_id="independent-reconstruction", proposed_slot_choices=[])
            self.assertEqual(list(rebuilt["questions"]), [source["question_id"]])
            self.assertEqual(rebuilt["questions"][source["question_id"]]["instructions"], source["question"])
            self.assertEqual(rebuilt["questions"][source["question_id"]]["criteria"], source["options"])
        self.assertEqual(digest(train), "30aeb3f11aad31abcb913b01dc5e5fe77497094335f88ee62037b1e5af5c25f6")

    def test_rejects_non_development_or_unbounded_requests_before_generation(self):
        for kwargs in (
            {"split": "TEST", "worlds": 1, "seed": 1},
            {"split": "language_heldout", "worlds": 1, "seed": 1},
            {"split": "development_train", "worlds": 0, "seed": 1},
            {"split": "development_validation", "worlds": 4097, "seed": 1},
            {"split": "development_train", "worlds": True, "seed": 1},
            {"split": "development_train", "worlds": 1, "seed": False},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                development_operation_pairs(**kwargs)

    def test_export_writes_pinned_separate_discovery_and_validation_files(self):
        with TemporaryDirectory() as directory:
            out = Path(directory) / "pairs"
            manifest = export_operation_pairs(out, train_worlds=2, validation_worlds=2)
            train_bytes = (out / "discovery-pairs.json").read_bytes()
            validation_bytes = (out / "validation-pairs.json").read_bytes()
            self.assertEqual(manifest["train"]["pairs"], 2)
            self.assertEqual(manifest["validation"]["pairs"], 2)
            self.assertEqual(manifest["train"]["sha256"], hashlib.sha256(train_bytes).hexdigest())
            self.assertEqual(manifest["validation"]["sha256"], hashlib.sha256(validation_bytes).hexdigest())
            self.assertEqual(json.loads((out / "manifest.json").read_text(encoding="utf-8")), manifest)
            with self.assertRaises(ValueError):
                export_operation_pairs(out, train_worlds=2, validation_worlds=2)


if __name__ == "__main__":
    unittest.main()
