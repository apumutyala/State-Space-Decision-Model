"""Tests exact record-world coverage for the combined fit-only plan."""
import unittest

from language_growth.l1_combined_fit_preflight import (
    CombinedPreflightError,
    _verify_world_partition,
)


class CombinedFitPreflightTests(unittest.TestCase):
    def test_accepts_exact_whole_world_partition(self):
        records = [{"group_id": "fit-a"}, {"group_id": "fit-a"}, {"group_id": "fit-b"}]
        _verify_world_partition(records, {"fit-a"}, {"fit-b"}, "test")

    def test_rejects_world_omitted_from_plan(self):
        records = [{"group_id": "fit-a"}, {"group_id": "fit-b"}]
        with self.assertRaisesRegex(CombinedPreflightError, "actual record worlds"):
            _verify_world_partition(records, {"fit-a"}, set(), "test")

    def test_rejects_plan_world_without_records(self):
        records = [{"group_id": "fit-a"}]
        with self.assertRaisesRegex(CombinedPreflightError, "actual record worlds"):
            _verify_world_partition(records, {"fit-a", "phantom"}, set(), "test")

    def test_rejects_world_split_across_partitions(self):
        records = [{"group_id": "fit-a"}]
        with self.assertRaisesRegex(CombinedPreflightError, "overlap"):
            _verify_world_partition(records, {"fit-a"}, {"fit-a"}, "test")

    def test_rejects_missing_world_identity(self):
        with self.assertRaisesRegex(CombinedPreflightError, "missing a world identity"):
            _verify_world_partition([{}], set(), set(), "test")


if __name__ == "__main__":
    unittest.main()
