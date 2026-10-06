import unittest

from language_growth.development_data import development_split_spec, generate_development_tasks
from language_growth.typed_emission_ssm import project, sha
from language_growth.typed_emission_ssm_transfer import (
    ALTERNATE_TEMPLATES, CORE_SHA, EVAL_SPLIT, TRAIN_SPLIT, core, paired_rows,
)


class TransferTests(unittest.TestCase):
    def test_exact_core_and_disjoint_domain(self):
        self.assertEqual(sha(core.__file__), CORE_SHA)
        train, val = map(development_split_spec, (TRAIN_SPLIT, EVAL_SPLIT))
        self.assertFalse(set(train["names"]) & set(val["names"]))
        for key in ("warranty_range", "price_cents_range"):
            self.assertLess(train[key][1], val[key][0])
        for kind in core.KINDS:
            self.assertFalse(set(train["templates"][kind]) & set(val["templates"][kind]))
        for kind, template in ALTERNATE_TEMPLATES.items():
            self.assertNotIn(template, train["templates"][kind])

    def test_pairs_have_validation_evidence_and_unchanged_candidates(self):
        tasks = generate_development_tasks(9102, 6, split=EVAL_SPLIT)
        rows = paired_rows(tasks)
        self.assertEqual(len(rows), 12)
        val_names = development_split_spec(EVAL_SPLIT)["names"]
        for original, partner in zip(rows[::2], rows[1::2]):
            self.assertEqual(original["source_split"], EVAL_SPLIT)
            self.assertEqual(partner["source_split"], EVAL_SPLIT)
            self.assertTrue(all(r["entity_ref"] in val_names for r in original["state"]["records"]))
            self.assertEqual([c["target"] for c in project(original["state"])],
                             [c["target"] for c in project(partner["state"])])
            if partner["member"] == "value_swap":
                self.assertEqual(original["state"]["task_input"], partner["state"]["task_input"])
            else:
                self.assertNotEqual(original["state"]["task_input"], partner["state"]["task_input"])
                self.assertEqual(original["evaluation_only"], partner["evaluation_only"])


if __name__ == "__main__":
    unittest.main()
