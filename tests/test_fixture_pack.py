"""Packaging checks that do not import or execute the language runtime."""

from pathlib import Path
import shutil
import tempfile
import unittest

from fixtures.language_growth.validate import read_json, validate_pack


PACK = Path(__file__).resolve().parents[1] / "fixtures" / "language_growth"


class FixturePackTests(unittest.TestCase):
    def test_packaging_and_stable_decision_ids(self):
        cases, records = validate_pack(PACK)
        self.assertEqual((cases, records), (21, 9))
        manifest = read_json(PACK / "manifest.json")
        ids = [entry["case_id"] for entry in manifest["cases"]]
        self.assertEqual(ids, [f"LG{index:02d}" for index in range(1, 22)])
        for entry in manifest["cases"]:
            case = read_json(PACK / entry["path"])
            decision = case["decision_only"]
            self.assertEqual(decision["question_id"], f"{case['case_id']}-D0")
            self.assertEqual(decision["question"], case["question"])
            self.assertNotIn("target_option_id", case["host"])

    def test_unknown_operation_and_arbitrary_text_are_independent(self):
        unknown = read_json(PACK / "cases" / "20_unknown_operation_and_text.json")
        text = read_json(PACK / "cases" / "21_arbitrary_text_assertion.json")
        self.assertEqual(unknown["proposal"]["nodes"], [{"id": "n1", "op": "FreeText"}])
        self.assertEqual(text["proposal"]["nodes"][1]["op"], "Assert")
        self.assertEqual(text["proposal"]["nodes"][1]["proposition"], "Atlas is repairable because all Atlas products are perfect.")
        self.assertEqual(unknown["expected"]["kernel_reason_category"], "unsupported assertion")
        self.assertEqual(text["expected"]["kernel_reason_category"], "invalid graph")

    def test_unlisted_case_fails_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            fake_root = Path(temporary) / "repo"
            clone = fake_root / "fixtures" / "language_growth"
            shutil.copytree(PACK, clone)
            for relative in ("interfaces/DECISION_LANGUAGE_V0.md", "plans/codex/LANGUAGE_GROWTH_PLAN.md"):
                link = fake_root / relative
                link.parent.mkdir(parents=True, exist_ok=True)
                link.write_text("fixture link target", encoding="utf-8")
            (clone / "cases" / "unlisted.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unlisted or missing"):
                validate_pack(clone)


if __name__ == "__main__":
    unittest.main()
