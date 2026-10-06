import json
import unittest

from language_growth.compare_curriculum_probe import (
    TRAIN_COMPARE, VALIDATION_COMPARE, comparison_text, make_row, model_text,
    replace_question,
)
from language_growth.development_data import development_split_spec, generate_development_tasks


class CompareCurriculumTests(unittest.TestCase):
    def test_treatment_preserves_target_candidates_and_only_changes_question(self):
        example = generate_development_tasks(20261006, 3)[2]
        original = make_row(example)
        changed = replace_question(original, comparison_text(example, TRAIN_COMPARE[2]))
        self.assertEqual(original["evaluation_only"], changed["evaluation_only"])
        self.assertEqual(model_text(original["request"])[1:], model_text(changed["request"])[1:])
        old_state = json.loads(original["request"]["state"])
        new_state = json.loads(changed["request"]["state"])
        self.assertNotEqual(old_state.pop("task_input"), new_state.pop("task_input"))
        self.assertEqual(old_state, new_state)

    def test_wording_and_name_domains_are_disjoint(self):
        train = development_split_spec("development_train")
        validation = development_split_spec("development_validation")
        self.assertFalse(set(train["names"]) & set(validation["names"]))
        self.assertNotIn(VALIDATION_COMPARE, TRAIN_COMPARE)
        self.assertNotIn(VALIDATION_COMPARE, train["templates"]["compare"])
        self.assertNotIn(VALIDATION_COMPARE, validation["templates"]["compare"])


if __name__ == "__main__":
    unittest.main()
