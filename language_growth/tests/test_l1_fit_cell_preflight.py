"""Tests the fit-world-only L1 training preflight without importing Torch."""
from pathlib import Path
import tempfile
import unittest

from language_growth.l1_fit_cell_preflight import PreflightError, build_plan, write_new_plan


def tiny_shape():
    """Build the pinned cohort's dimensions, without its task text or any other split."""
    records, census = [], []

    def add(world_no, task_no, slot):
        world = f"fit-world-{world_no:02d}"
        row_id = f"fit-task-{world_no:02d}-{task_no:02d}"
        n_options = 3 if slot == "semantic-operation" else 8
        request = {"questions": {"q": {"criteria": {f"o{i}": f"choice {i}" for i in range(n_options)}}}}
        records.append({"id": row_id, "group_id": world, "request": request,
                        "labels": {"q": "o0"},
                        "metadata": {"slot": slot, "world_id": world, "split": "development_train",
                                     "source_cohort": "l1_multiworld_development_train",
                                     "diagnostic_teacher_forced": True, "free_running": False,
                                     "promotion_eligible": False}})
        census_slot = f"slot-001-{slot}"
        census.append({"record_line": len(records), "record_id": row_id, "slot": census_slot,
                       "prompt_tokens": 1400, "options": n_options,
                       "longest_prompt_plus_continuation_tokens": 1404,
                       "max_continuation_tokens": 4})

    for world_no in range(24):
        for task_no in range(16):
            add(world_no, task_no, "semantic-operation")
            add(world_no, task_no, "fact")
        for task_no in range(16, 40):
            add(world_no, task_no, "semantic-operation")
            add(world_no, task_no, "left-operand")
            add(world_no, task_no, "right-operand")
            add(world_no, task_no, "comparison-relation")
    return records, census


class L1FitCellPreflightTests(unittest.TestCase):
    def test_uses_complete_disjoint_fit_worlds_and_preserves_no_authorization(self):
        records, census = tiny_shape()
        plan = build_plan(records, census)
        source = plan["source"]
        self.assertEqual((source["records"], source["worlds"]), (3072, 24))
        self.assertEqual((source["training_rows"], source["diagnostic_rows"]), (2304, 768))
        self.assertEqual(len(source["training_worlds"]), 18)
        self.assertEqual(len(source["diagnostic_worlds"]), 6)
        self.assertFalse(set(source["training_worlds"]) & set(source["diagnostic_worlds"]))
        self.assertEqual(set(source["training_worlds"]) | set(source["diagnostic_worlds"]),
                         {r["group_id"] for r in records})
        self.assertEqual(plan["sequence"]["max_tokens"], 2048)
        self.assertEqual(plan["sequence"]["max_options"], 8)
        self.assertFalse(plan["training_authorized"])
        self.assertIsNone(plan["parent"]["stage1_report_sha256"])

    def test_rejects_token_identity_mismatch_and_oversize(self):
        records, census = tiny_shape()
        census[0]["record_id"] = "other-fit-row"
        with self.assertRaisesRegex(PreflightError, "identity mismatch"):
            build_plan(records, census)
        records, census = tiny_shape()
        census[0]["longest_prompt_plus_continuation_tokens"] = 2049
        with self.assertRaisesRegex(PreflightError, "exceeds"):
            build_plan(records, census)
        records, census = tiny_shape()
        census[0]["max_continuation_tokens"] = -1
        with self.assertRaisesRegex(PreflightError, "exceeds"):
            build_plan(records, census)
        records, census = tiny_shape()
        census[0]["longest_prompt_plus_continuation_tokens"] = 1399
        with self.assertRaisesRegex(PreflightError, "exceeds"):
            build_plan(records, census)

    def test_rejects_duplicate_world_slot_rows(self):
        records, census = tiny_shape()
        records[1]["metadata"]["slot"] = records[0]["metadata"]["slot"]
        with self.assertRaisesRegex(PreflightError, "invalid identity/provenance"):
            build_plan(records, census)

    def test_rejects_multi_question_row(self):
        records, census = tiny_shape()
        records[0]["request"]["questions"]["extra"] = records[0]["request"]["questions"]["q"]
        with self.assertRaisesRegex(PreflightError, "exactly one question"):
            build_plan(records, census)

    def test_output_is_exclusive_and_preserves_existing_content(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan.json"
            output.write_text("keep me", encoding="utf-8")
            with self.assertRaisesRegex(PreflightError, "refusing to overwrite"):
                write_new_plan(output, {"new": True})
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me")

            dangling = Path(directory) / "dangling.json"
            try:
                dangling.symlink_to(Path(directory) / "missing-target.json")
            except OSError:
                self.skipTest("symlink creation is unavailable on this host")
            with self.assertRaisesRegex(PreflightError, "refusing to overwrite"):
                write_new_plan(dangling, {"new": True})


if __name__ == "__main__":
    unittest.main()
