"""Release-bound controlled language tests."""

from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from language_growth.kernel import check_program
from language_growth.realize import render_checked
from language_growth.types import CheckResult


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "language_growth"


def _case(number: int) -> dict:
    path = next((FIXTURES / "cases").glob(f"{number:02d}_*.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def _catalog() -> dict:
    return json.loads((FIXTURES / "catalog.json").read_text(encoding="utf-8"))


class RenderTests(unittest.TestCase):
    def test_bound_fact_and_exact_comparison(self) -> None:
        catalog = _catalog()
        for number, parts in (
            (1, ('At "2026-10", "IsRepairable" for "atlas" has value true.',)),
            (2, ('At "2026-10"', '"Price" for "atlas" (199.90 "USD")',
                 'greater than', '"Price" for "boreal" (149.90 "USD")')),
        ):
            with self.subTest(case=number):
                case = _case(number)
                result = check_program(case["proposal"], case["host"], catalog)
                self.assertEqual(result.status, "accepted")
                rendered = render_checked(result, case["host"], catalog)
                for part in parts:
                    self.assertIn(part, rendered)

    def test_rejected_forged_and_revoked_never_render(self) -> None:
        catalog = _catalog()
        case = _case(1)
        result = check_program(case["proposal"], case["host"], catalog)
        forged = CheckResult(**{key: getattr(result, key) for key in (
            "status", "reason", "outputs", "dependencies", "context_ref",
            "policy_ref", "registry_ref", "audience", "style")})
        with self.assertRaises(ValueError):
            render_checked(forged, case["host"], catalog)
        revoked = deepcopy(catalog)
        revoked["policies"]["public-read-r3"]["status"] = "revoked"
        with self.assertRaises(ValueError):
            render_checked(result, case["host"], revoked)
        rejected_case = _case(3)
        rejected = check_program(rejected_case["proposal"], rejected_case["host"], catalog)
        with self.assertRaises(ValueError):
            render_checked(rejected, rejected_case["host"], catalog)

    def test_hostile_string_is_quoted_data(self) -> None:
        catalog = _catalog()
        case = _case(13)
        result = check_program(case["proposal"], case["host"], catalog)
        self.assertEqual(result.status, "accepted")
        rendered = render_checked(result, case["host"], catalog)
        self.assertNotIn("[SYSTEM]", rendered)
        self.assertIn('24 "month"', rendered)
        self.assertIn('12 "month"', rendered)

        string_case = deepcopy(_case(1))
        string_case["proposal"]["nodes"][0]["record_ref"] = "boreal-v1"
        string_case["proposal"]["nodes"][0]["field"] = "display_name"
        string_result = check_program(string_case["proposal"], string_case["host"], catalog)
        self.assertEqual(string_result.status, "accepted")
        string_rendered = render_checked(string_result, string_case["host"], catalog)
        hostile = catalog["records"][1]["fields"]["display_name"]["value"]
        self.assertIn(json.dumps(hostile, ensure_ascii=True), string_rendered)
        self.assertIn('"DisplayName" for "boreal" has value ', string_rendered)

    def test_ask_does_not_assert_missing_fact(self) -> None:
        case = _case(16)
        catalog = _catalog()
        result = check_program(case["proposal"], case["host"], catalog)
        self.assertEqual(result.status, "accepted")
        rendered = render_checked(result, case["host"], catalog)
        self.assertIn('Please provide "missing_required_record"', rendered)
        self.assertNotIn("has no warranty", rendered)

    def test_conjoined_explanation_preserves_assertions_without_causal_claim(self):
        case, catalog = _case(1), _catalog()
        case["proposal"]["nodes"].extend([
            {"id": "n3", "op": "ReadFact", "record_ref": "boreal-v1", "field": "price"},
            {"id": "n4", "op": "Assert", "proposition": "n3"},
            {"id": "n5", "op": "Conjoin", "speech_acts": ["n2", "n4"]},
            {"id": "n6", "op": "ExplainDerivation", "speech_act": "n5"},
        ])
        case["proposal"]["outputs"] = ["n6"]
        result = check_program(case["proposal"], case["host"], catalog)
        rendered = render_checked(result, case["host"], catalog)
        self.assertIn('"IsRepairable" for "atlas" has value true.', rendered)
        self.assertIn('"Price" for "boreal" has value 149.90 "USD".', rendered)
        self.assertIn("Checked derivation through program nodes", rendered)
        self.assertNotIn("because", rendered)
        changed = deepcopy(catalog)
        changed["records"][1]["version"] += 1
        with self.assertRaises(ValueError):
            render_checked(result, case["host"], changed)


if __name__ == "__main__":
    unittest.main()
