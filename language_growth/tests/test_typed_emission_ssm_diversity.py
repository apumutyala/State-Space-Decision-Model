import copy
import json
import unittest

from language_growth.development_data import generate_development_tasks
from language_growth.typed_emission_ssm_diversity import CORE_SHA, TRAIN_TEMPLATES, core, diversify


class DiversityTests(unittest.TestCase):
    def test_exact_model_and_only_question_changes(self):
        self.assertEqual(core.sha(core.__file__), CORE_SHA)
        rows = [core.row(t, t["task_id"], "original") for t in generate_development_tasks(9101, 18)]
        altered = diversify(rows)
        self.assertTrue(any(a["state"]["task_input"] != b["state"]["task_input"] for a, b in zip(rows, altered)))
        for old, new in zip(rows, altered):
            restored = copy.deepcopy(new)
            restored["state"]["task_input"] = old["state"]["task_input"]
            self.assertEqual(restored, old)

    def test_six_templates_and_original_assignment(self):
        rows = [core.row(t, t["task_id"], "original") for t in generate_development_tasks(9101, 18)]
        diverse = diversify(rows)
        self.assertTrue(all(len(templates) == 6 for templates in TRAIN_TEMPLATES.values()))
        for i, row in enumerate(diverse):
            target = core.gold_target(row["evaluation_only"]["program"])
            record = next(r for r in row["state"]["records"] if r["record_ref"] == target["left"])
            self.assertIn(record["entity_ref"], row["state"]["task_input"])
            self.assertEqual((i // 3) % 6, i // 3)


if __name__ == "__main__":
    unittest.main()
