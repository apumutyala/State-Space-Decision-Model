"""CPU tests for the language-to-parent training seam; no model loads."""

from copy import deepcopy
import unittest

from language_growth.training_records import export_training_records


def example(split="development_train"):
    return {
        "development_only": True, "development_split": split, "task_id": "synthetic-example",
        "model_inputs": [{"model": "synthetic", "state": "permitted evidence",
                          "questions": {"slot-001-operation": {
                              "type": "choice", "instructions": "Select an operation.",
                              "criteria": {"a": "Read", "b": "Compare"}}}}],
        "training_only": {"targets": [{"slot": "operation", "option_id": "b"}],
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
        row["request"]["state"] = "mutated"
        self.assertEqual(task["model_inputs"][0]["state"], "permitted evidence")

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
