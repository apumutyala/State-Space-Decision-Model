"""CPU tests for the language-to-parent training seam; no model loads."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from language_growth.training_records import export_training_records
from language_growth.parent_bridge import decision_request


def example(split="development_train"):
    pack = Path(__file__).resolve().parents[2] / "fixtures/language_growth"
    host = json.loads((pack / "cases/01_supported_read.json").read_text(encoding="utf-8"))["host"]
    catalog = json.loads((pack / "catalog.json").read_text(encoding="utf-8"))
    request, _ = decision_request(task_input="What is the price?", host=host, catalog=catalog,
                                  model="synthetic", parent_revision="test:v1", request_id="test-1",
                                  question_id="slot-001-operation", question="Select an operation.",
                                  options={"a": "Read", "b": "Compare"})
    return {
        "development_only": True, "development_split": split, "task_id": "synthetic-example",
        "model_inputs": [request],
        "training_only": {"targets": [{"slot": "operation", "option_id": "b"}],
                          "host": host, "catalog": catalog,
                          "program": {"private_gold": "MUST_NOT_ENTER_REQUEST"}},
    }


class TrainingRecordsTests(unittest.TestCase):
    def test_gold_is_separate_and_request_is_copied_verbatim(self):
        task = example()
        row = export_training_records([task])[0]
        self.assertEqual(row["request"], task["model_inputs"][0])
        self.assertEqual(row["labels"], {"slot-001-operation": "b"})
        changed = deepcopy(task)
        changed["training_only"]["targets"][0]["option_id"] = "a"
        self.assertEqual(export_training_records([changed])[0]["request"], row["request"])
        original = task["model_inputs"][0]["state"]
        row["request"]["state"] = "mutated"
        self.assertEqual(task["model_inputs"][0]["state"], original)

    def test_current_label_and_missing_previous_target_prefix_are_rejected(self):
        task = example()
        state = json.loads(task["model_inputs"][0]["state"])
        state["target_option_id"] = "b"
        task["model_inputs"][0]["state"] = json.dumps(state, sort_keys=True, separators=(",", ":"))
        with self.assertRaisesRegex(ValueError, "state must match"):
            export_training_records([task])
        task = example()
        second = deepcopy(task["model_inputs"][0])
        second["questions"] = {"slot-002-fact": {"type": "choice", "instructions": "Pick fact",
                                                "criteria": {"x": "Fact"}}}
        task["model_inputs"].append(second)
        task["training_only"]["targets"].append({"slot": "fact", "option_id": "x"})
        with self.assertRaisesRegex(ValueError, "earlier target prefix"):
            export_training_records([task])

    def test_split_mixing_and_unnamed_splits_fail(self):
        with self.assertRaisesRegex(ValueError, "split mismatch"):
            export_training_records([example(), example("development_validation")])
        validation = export_training_records([example("development_validation")], split="development_validation")
        self.assertEqual(len(validation), 1)
        with self.assertRaises(ValueError):
            export_training_records([example()], split="audit")

    def test_alignment_and_option_validation(self):
        for key, value in (("slot", "fact"), ("option_id", "missing")):
            task = example()
            task["training_only"]["targets"][0][key] = value
            with self.assertRaises(ValueError):
                export_training_records([task])
        task = example()
        task["training_only"]["targets"].append({"slot": "fact", "option_id": "a"})
        with self.assertRaises(ValueError):
            export_training_records([task])

    def test_duplicates_and_empty_export_fail(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            export_training_records([example(), example()])
        with self.assertRaises(ValueError):
            export_training_records([])


if __name__ == "__main__":
    unittest.main()
