"""CPU-light checks for fresh development-only K4 operation pair generation."""
import unittest
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from language_growth.gate_operation_pairs import (
    _WORDING, _numeric_relation, development_operation_pairs, export_operation_pairs,
)
from language_growth.gate_suites import digest
from language_growth.parent_bridge import decision_request


class GateOperationPairTests(unittest.TestCase):
    def test_price_reference_relation_uses_numeric_not_lexical_order(self):
        self.assertEqual(_numeric_relation("1000", "999"), "greater_than")
        self.assertEqual(_numeric_relation("999", "1000"), "less_than")
        self.assertEqual(_numeric_relation("01000", "1000"), "equal")

    def test_train_validation_pairs_are_deterministic_and_request_matched(self):
        train = development_operation_pairs(split="development_train", worlds=32, seed=20261011)
        self.assertEqual(train, development_operation_pairs(split="development_train", worlds=32, seed=20261011))
        validation = development_operation_pairs(split="development_validation", worlds=8, seed=20261012)
        self.assertEqual(len(train), 32)
        self.assertEqual(len(validation), 8)
        self.assertEqual({pair["split"] for pair in train}, {"development_train"})
        self.assertEqual({pair["split"] for pair in validation}, {"development_validation"})
        self.assertFalse({pair["world_id"] for pair in train} & {pair["world_id"] for pair in validation})
        fields = ("warranty_months", "price")
        train_templates = {pair["pair_id"].rsplit("operation-prompt-", 1)[1] for pair in train}
        validation_templates = {pair["pair_id"].rsplit("operation-prompt-", 1)[1] for pair in validation}
        self.assertEqual(train_templates, {f"{field}-train-{index:02d}" for field in fields for index in range(1, 5)})
        self.assertEqual(validation_templates,
                         {f"{field}-validation-{index:02d}" for field in fields for index in range(1, 5)})
        self.assertEqual([sum(pair["pair_id"].endswith(f"train-{index:02d}") for pair in train)
                          for index in range(1, 5)], [len(train) // 4] * 4)
        self.assertEqual([sum(pair["pair_id"].endswith(f"validation-{index:02d}") for pair in validation)
                          for index in range(1, 5)], [len(validation) // 4] * 4)
        self.assertEqual([sum(f"-operation-prompt-{field}-" in pair["pair_id"] for pair in train)
                          for field in fields], [16, 16])
        self.assertEqual([sum(f"-operation-prompt-{field}-" in pair["pair_id"] for pair in validation)
                          for field in fields], [4, 4])
        self.assertFalse({pair["source"]["task_input"] for pair in train} &
                         {pair["source"]["task_input"] for pair in validation})
        self.assertTrue(all("minus" not in pair["effect"]["task_input"].lower()
                            for pair in validation))
        for operation in ("Compare", "ExactDifference"):
            train_bank = set(_WORDING["development_train"][operation])
            validation_bank = set(_WORDING["development_validation"][operation])
            self.assertEqual(len(train_bank), 4)
            self.assertEqual(len(validation_bank), 4)
            self.assertFalse(train_bank & validation_bank)
        self.assertTrue(any("compare" not in text.lower()
                            for text in _WORDING["development_train"]["Compare"]))
        self.assertTrue(all("minus" not in text.lower()
                            for text in _WORDING["development_validation"]["ExactDifference"]))

        for pair in train + validation:
            source, effect = pair["source"], pair["effect"]
            self.assertEqual(set(pair), {
                "pair_id", "world_id", "split", "concept", "source", "effect",
                "source_target", "effect_target"})
            self.assertEqual(set(source), {
                "task_input", "host", "catalog", "question_id", "question", "options"})
            field = pair["pair_id"].rsplit("operation-prompt-", 1)[1].split("-", 1)[0]
            self.assertIn(field, source["task_input"])
            self.assertIn(field, effect["task_input"])
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
        self.assertEqual(digest(train), "1c9e9c53276f72ad43f60e02871ed0d328ef2f89d00f6a2d42774b6738733740")
        self.assertEqual(digest(validation), "e02868c3a9addcf50316da5f43f511b008cdf6791a3fafa0ec1c4cc805eb5606")

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
            self.assertEqual(sum(manifest["template_pair_counts"]["train"].values()), 2)
            self.assertEqual(sum(manifest["template_pair_counts"]["validation"].values()), 2)
            self.assertEqual(manifest["field_pair_counts"]["train"],
                             {"price": 1, "warranty_months": 1})
            self.assertEqual(manifest["field_pair_counts"]["validation"],
                             {"price": 1, "warranty_months": 1})
            self.assertEqual(manifest["train"]["sha256"], hashlib.sha256(train_bytes).hexdigest())
            self.assertEqual(manifest["validation"]["sha256"], hashlib.sha256(validation_bytes).hexdigest())
            self.assertEqual(json.loads((out / "manifest.json").read_text(encoding="utf-8")), manifest)
            with self.assertRaises(ValueError):
                export_operation_pairs(out, train_worlds=2, validation_worlds=2)


if __name__ == "__main__":
    unittest.main()
