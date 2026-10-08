"""Tests signed/tied L1 difference supervision on the actual GatePlanner path."""
import unittest

from language_growth.l1_difference_fit_supplement import generate_records


class L1DifferenceFitSupplementTests(unittest.TestCase):
    def test_balanced_train_only_worlds_replay_all_ordered_pairs(self):
        rows, coverage = generate_records()
        self.assertEqual((len(rows), coverage["task_ids"], len(coverage["world_ids"])), (1728, 576, 24))
        self.assertEqual(coverage["sign_counts"], {
            "price": {"positive": 120, "negative": 120, "tie": 48},
            "warranty_months": {"positive": 120, "negative": 120, "tie": 48},
        })
        self.assertEqual(coverage["slot_counts"], {
            "semantic-operation": 576, "left-operand": 576, "right-operand": 576,
        })
        self.assertFalse(any(row["metadata"]["free_running"] for row in rows))
        self.assertTrue(all(row["metadata"]["split"] == "development_train" for row in rows))
        self.assertTrue(all(row["metadata"]["promotion_eligible"] is False for row in rows))
        for row in rows:
            self.assertNotIn("labels", row["request"])
            self.assertNotIn("program", row["request"])


if __name__ == "__main__":
    unittest.main()
