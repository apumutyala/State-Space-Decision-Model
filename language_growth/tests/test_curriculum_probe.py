import copy
import json
import unittest
from collections import Counter

from language_growth.curriculum_spec import CURRICULUM_SPLITS
from language_growth.development_data import development_split_spec, generate_development_tasks
from language_growth.kernel import check_program
from language_growth.training_records import export_training_records
from language_growth.curriculum_probe import paired_ci


class CurriculumTests(unittest.TestCase):
    def test_split_domains_and_balanced_templates(self):
        train = development_split_spec("development_curriculum")
        transfer = development_split_spec("development_transfer_v2")
        self.assertFalse(set(train["names"]) & set(transfer["names"]))
        for range_name in ("warranty_range", "price_cents_range"):
            self.assertLess(train[range_name][1], transfer[range_name][0])
        for kind in ("read", "explain", "compare"):
            self.assertEqual(len(train["templates"][kind]), 8)
            self.assertFalse(set(train["templates"][kind]) & set(transfer["templates"][kind]))
            self.assertFalse(set(development_split_spec("development_validation")["templates"][kind])
                             & set(transfer["templates"][kind]))
        tasks = generate_development_tasks(161803, 300, split="development_transfer_v2")
        self.assertEqual(Counter(t["training_only"]["kind"] for t in tasks),
                         Counter(read=100, explain=100, compare=100))
        for kind in ("read", "explain", "compare"):
            counts = Counter(t["training_only"]["template_id"] for t in tasks
                             if t["training_only"]["kind"] == kind)
            self.assertEqual(len(counts), 6)
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_curriculum_preserves_latent_worlds_and_keeps_labels_out(self):
        old = generate_development_tasks(271828, 48)
        new = generate_development_tasks(271828, 48, split="development_curriculum")
        for before, after in zip(old, new):
            for key in ("host", "catalog", "program", "targets"):
                self.assertEqual(before["training_only"][key], after["training_only"][key])
            evidence = after["training_only"]
            self.assertEqual(check_program(evidence["program"], evidence["host"], evidence["catalog"]).status,
                             "accepted")
        rows = export_training_records(new, split="development_curriculum")
        self.assertTrue(all(set(r) == {"id", "request", "labels"} for r in rows))
        for row in rows:
            self.assertNotIn("labels", row["request"])
            self.assertNotIn("training_only", row["request"])
        corrupted = copy.deepcopy(new)
        state = json.loads(corrupted[0]["model_inputs"][0]["state"])
        state["gold_label"] = "read"
        corrupted[0]["model_inputs"][0]["state"] = json.dumps(state)
        with self.assertRaises(ValueError):
            export_training_records(corrupted, split="development_curriculum")

    def test_paired_ci_retains_task_clusters(self):
        baseline = [{"task_id": f"t{i}", "kind": kind, "slot_correct": 0, "slot_total": total}
                    for i, (kind, total) in enumerate((("read", 2), ("explain", 2), ("compare", 4)))]
        candidate = [{**r, "slot_correct": r["slot_total"]} for r in baseline]
        result = paired_ci(baseline, candidate, "slot_correct", slots=True)
        self.assertEqual(result["difference"], 1.0)
        self.assertEqual(result["ci95"], [1.0, 1.0])
        candidate[0]["task_id"] = "different"
        with self.assertRaises(ValueError):
            paired_ci(baseline, candidate, "slot_correct", slots=True)


if __name__ == "__main__":
    unittest.main()
