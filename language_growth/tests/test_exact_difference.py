"""Authored CPU contract examples; no model or qualification data."""

from __future__ import annotations

import copy
import json
import unittest
from decimal import Decimal, Inexact, Rounded, localcontext
from pathlib import Path

from language_growth.kernel import check_program
from language_growth.realize import render_checked
from language_growth.types import ExactDifference


FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/language_growth/exact_difference.json"


class ExactDifferenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.catalog = self.fixture["catalog"]
        self.host = self.fixture["host"]
        self.program = copy.deepcopy(self.fixture["cases"][0]["proposal"])

    def check(self):
        return check_program(self.program, self.host, self.catalog)

    def constants(self, left, right, left_type="decimal", right_type="decimal"):
        for name, value, value_type in (("left", left, left_type), ("right", right, right_type)):
            self.host["query_constants"][name] = {
                "value": value, "value_type": value_type, "unit": "USD",
                "temporal_scope": "2026-10", "quantity_kind": "price",
            }
        self.program["nodes"] = [
            {"id": "n1", "op": "ExactDifference", "left": {"constant_ref": "left"}, "right": {"constant_ref": "right"}},
            {"id": "n2", "op": "Assert", "proposition": "n1"},
        ]
        self.program["outputs"] = ["n2"]

    def test_minimal_positive_and_negative_fixtures(self):
        for case in self.fixture["cases"]:
            with self.subTest(case=case["case_id"]):
                result = check_program(case["proposal"], self.host, self.catalog)
                self.assertEqual((result.status, result.reason), (case["expected_status"], case["expected_reason"]))
                if result.status == "accepted":
                    self.assertIsInstance(result.outputs[0].content, ExactDifference)
                    self.assertEqual(str(result.outputs[0].content.value), case["expected_value"])
                    self.assertEqual(result.dependencies, ("atlas-v1", "boreal-v1"))
                    self.assertEqual(render_checked(result, self.host, self.catalog), case["expected_render"])
                else:
                    self.assertEqual(result.outputs, ())
                    with self.assertRaises(ValueError):
                        render_checked(result, self.host, self.catalog)

    def test_ambient_decimal_precision_and_traps_cannot_round(self):
        self.constants("123456789012345678901234567890.01", "0.02")
        with localcontext() as context:
            context.prec = 2
            context.traps[Inexact] = True
            context.traps[Rounded] = True
            result = self.check()
            self.assertEqual(result.status, "accepted")
            self.assertEqual(str(result.outputs[0].content.value), "123456789012345678901234567889.99")
            self.assertIn("123456789012345678901234567889.99", render_checked(result, self.host, self.catalog))

    def test_integer_mixed_fact_constant_and_zero_values(self):
        self.constants(3, 8, "integer", "integer")
        self.assertEqual(self.check().outputs[0].content.value, -5)
        self.assertIs(type(self.check().outputs[0].content.value), int)
        self.constants(200, "199.90", "integer", "decimal")
        self.assertEqual(str(self.check().outputs[0].content.value), "0.10")
        self.constants("0.00", "0.00")
        self.assertEqual(str(self.check().outputs[0].content.value), "0.00")
        self.program = copy.deepcopy(self.fixture["cases"][0]["proposal"])
        self.program["nodes"][2]["right"] = {"constant_ref": "right"}
        self.assertEqual(str(self.check().outputs[0].content.value), "199.90")

    def test_scope_kind_and_nonnumeric_operands_fail(self):
        self.constants("3.00", "2.00")
        for key, invalid, reason in (
            ("temporal_scope", "2026-09", "scope_mismatch"),
            ("quantity_kind", "shipping_cost", "type_unit_mismatch"),
            ("unit", "EUR", "type_unit_mismatch"),
        ):
            with self.subTest(key=key):
                original = self.host["query_constants"]["right"][key]
                self.host["query_constants"]["right"][key] = invalid
                self.assertEqual(self.check().reason, reason)
                self.host["query_constants"]["right"][key] = original
        for value, value_type in ((True, "boolean"), ("2.00", "string")):
            with self.subTest(value_type=value_type):
                self.constants("3.00", value, right_type=value_type)
                self.assertEqual(self.check().reason, "type_unit_mismatch")

    def test_resource_caps_cover_inputs_alignment_and_result(self):
        for left, right, left_type, right_type in (
            (10 ** 256, 0, "integer", "integer"),
            ("9" * 257, "0", "decimal", "decimal"),
            ("0." + "0" * 64 + "1", "0", "decimal", "decimal"),
            (10 ** 255, "0.1", "integer", "decimal"),
            (10 ** 256 - 1, -1, "integer", "integer"),
        ):
            with self.subTest(left_type=left_type, right_type=right_type, left=str(left)[:30]):
                self.constants(left, right, left_type, right_type)
                self.assertEqual(self.check().reason, "exceeded_limit")
        self.constants("0." + "0" * 63 + "1", "0")
        self.assertEqual(self.check().outputs[0].content.value, Decimal("1e-64"))
        self.constants(10 ** 256 - 1, 0, "integer", "integer")
        self.assertEqual(self.check().status, "accepted")

    def test_opt_in_no_division_inline_values_or_arithmetic_chaining(self):
        self.constants("3.00", "0.00")
        registry = self.catalog["registries"][self.host["registry_ref"]]
        registry["allowed_ops"].remove("ExactDifference")
        self.assertEqual(self.check().reason, "unsupported_operation")
        registry["allowed_ops"].extend(["ExactDifference", "ExactDivide"])
        self.program["nodes"][0]["op"] = "ExactDivide"
        self.assertEqual(self.check().reason, "unsupported_operation")
        self.program["nodes"][0]["op"] = "ExactDifference"
        self.program["nodes"][0]["conversion"] = "USD_to_EUR"
        self.assertEqual(self.check().reason, "unsupported_field")
        del self.program["nodes"][0]["conversion"]
        self.program["nodes"][0]["right"] = {"value": 0}
        self.assertEqual(self.check().reason, "unsupported_field")
        self.program["nodes"][0]["right"] = {"constant_ref": "right"}
        self.program["nodes"].insert(1, {"id": "n3", "op": "ExactDifference", "left": {"node": "n1", "project": "object"}, "right": {"constant_ref": "right"}})
        self.assertEqual(self.check().reason, "invalid_graph")

    def test_live_evidence_constant_and_registry_changes_block_release(self):
        result = self.check()
        self.catalog["records"][0]["fields"]["price"]["value"] = "200.00"
        with self.assertRaises(ValueError):
            render_checked(result, self.host, self.catalog)
        self.setUp()
        self.constants("3.00", "2.00")
        result = self.check()
        self.host["query_constants"]["right"]["value"] = "1.00"
        with self.assertRaises(ValueError):
            render_checked(result, self.host, self.catalog)
        result = self.check()
        self.catalog["registries"][self.host["registry_ref"]]["allowed_ops"].remove("ExactDifference")
        with self.assertRaises(ValueError):
            render_checked(result, self.host, self.catalog)

    def test_difference_conjunction_explanation_and_scope_binding(self):
        self.program["nodes"].extend([
            {"id": "n5", "op": "Assert", "proposition": "n1"},
            {"id": "n6", "op": "Conjoin", "speech_acts": ["n4", "n5"]},
            {"id": "n7", "op": "ExplainDerivation", "speech_act": "n6"},
        ])
        self.program["outputs"] = ["n7"]
        result = self.check()
        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.dependencies, ("atlas-v1", "boreal-v1"))
        self.assertIn('is 50.00 "USD".', render_checked(result, self.host, self.catalog))
        self.assertEqual(result.to_dict()["outputs"][0]["content"]["content"][0]["content"]["quantity_kind"], "price")
        self.host["query_constants"]["other-scope"] = {
            "value": 1, "value_type": "integer", "unit": "USD",
            "temporal_scope": "2026-09", "quantity_kind": "price",
        }
        self.program["nodes"].insert(4, {
            "id": "n8", "op": "ExactDifference",
            "left": {"constant_ref": "other-scope"}, "right": {"constant_ref": "other-scope"},
        })
        self.program["nodes"][5]["proposition"] = "n8"
        self.assertEqual(self.check().reason, "scope_mismatch")


if __name__ == "__main__":
    unittest.main()
