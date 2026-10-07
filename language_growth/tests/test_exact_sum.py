"""CPU checks for the opt-in L1 exact-sum semantic profile."""

from copy import deepcopy
from decimal import Decimal, Inexact, Rounded, localcontext
import json
from pathlib import Path
import unittest

from language_growth.kernel import check_program
from language_growth.realize import render_checked
from language_growth.types import ExactSum

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "language_growth"


def world():
    case = json.loads((FIXTURES / "cases/02_supported_exact_comparison.json").read_text(encoding="utf-8"))
    catalog = json.loads((FIXTURES / "catalog.json").read_text(encoding="utf-8"))
    return case["host"], catalog


def program(host, *, left="atlas-v1", right="boreal-v1", field="price"):
    return {
        "schema": "ssdm.semantic_program.v0.2",
        "context_ref": host["context_ref"], "policy_ref": host["policy_ref"],
        "registry_ref": host["registry_ref"],
        "nodes": [
            {"id": "n1", "op": "ReadFact", "record_ref": left, "field": field},
            {"id": "n2", "op": "ReadFact", "record_ref": right, "field": field},
            {"id": "n3", "op": "ExactSum", "left": {"node": "n1", "project": "object"},
             "right": {"node": "n2", "project": "object"}},
            {"id": "n4", "op": "Assert", "proposition": "n3"},
        ],
        "outputs": ["n4"], "style": "plain",
    }


def opted_world():
    host, catalog = world()
    catalog["registries"][host["registry_ref"]]["allowed_ops"].append("ExactSum")
    return host, catalog


class ExactSumTests(unittest.TestCase):
    def test_decimal_sum_is_exact_with_both_dependencies_and_plain_text(self):
        host, catalog = opted_world()
        proposed = program(host)
        with localcontext() as context:
            context.prec = 2
            context.traps[Inexact] = True
            context.traps[Rounded] = True
            checked = check_program(proposed, host, catalog)
            self.assertEqual(checked.status, "accepted")
            text = render_checked(checked, host, catalog)
        self.assertEqual(checked.dependencies, ("atlas-v1", "boreal-v1"))
        total = checked.outputs[0].content
        self.assertIsInstance(total, ExactSum)
        self.assertEqual(total.value, Decimal("349.80"))
        self.assertEqual(total.to_dict()["value"], "349.80")
        self.assertEqual((total.unit, total.quantity_kind, total.interval), ("USD", "price", "2026-10"))
        self.assertIn('exact "price" sum', text)
        self.assertIn('349.80 "USD"', text)
        self.assertNotIn("because", text)

    def test_integer_and_mixed_decimal_sums(self):
        host, catalog = opted_world()
        warranty = program(host, field="warranty_months")
        checked = check_program(warranty, host, catalog)
        self.assertEqual(checked.status, "accepted")
        self.assertEqual(checked.outputs[0].content.value, 36)
        self.assertIn('36 "month"', render_checked(checked, host, catalog))

        mixed = deepcopy(catalog)
        atlas = next(rec for rec in mixed["records"] if rec["record_ref"] == "atlas-v1")
        atlas["fields"]["price"] = {"value": 200, "value_type": "integer", "unit": "USD",
                                     "temporal_scope": "2026-10", "quantity_kind": "price"}
        checked = check_program(program(host), host, mixed)
        self.assertEqual(checked.status, "accepted")
        self.assertEqual(checked.outputs[0].content.value, Decimal("349.90"))

    def test_large_decimal_is_independent_of_ambient_precision(self):
        host, catalog = opted_world()
        atlas = next(rec for rec in catalog["records"] if rec["record_ref"] == "atlas-v1")
        boreal = next(rec for rec in catalog["records"] if rec["record_ref"] == "boreal-v1")
        digits = "9" * 4500  # beyond Python's default int-to-string digit limit
        atlas["fields"]["price"]["value"] = digits + ".00"
        boreal["fields"]["price"]["value"] = "0.01"
        with localcontext() as context:
            context.prec = 2
            context.traps[Inexact] = True
            context.traps[Rounded] = True
            checked = check_program(program(host), host, catalog)
            text = render_checked(checked, host, catalog)
        self.assertEqual(checked.status, "accepted")
        self.assertEqual(str(checked.outputs[0].content.value), digits + ".01")
        self.assertIn(digits + '.01 "USD"', text)

    def test_integer_carry_beyond_default_digit_limit_releases_and_serializes(self):
        host, catalog = opted_world()
        for record in catalog["records"]:
            if record["record_ref"] in ("atlas-v1", "boreal-v1"):
                record["fields"]["price"].update(value=9 * 10**4299, value_type="integer")
        with localcontext() as context:
            context.prec = 2
            context.traps[Inexact] = True
            context.traps[Rounded] = True
            checked = check_program(program(host), host, catalog)
            self.assertEqual(checked.status, "accepted")
            text = render_checked(checked, host, catalog)
            self.assertEqual(text, render_checked(checked, host, catalog))
            encoded = json.dumps(checked.to_dict(), sort_keys=True)
        wanted = "18" + "0" * 4299
        self.assertIsInstance(checked.outputs[0].content.value, Decimal)
        self.assertEqual(str(checked.outputs[0].content.value), wanted)
        self.assertIn(wanted + ' "USD"', text)
        self.assertEqual(json.loads(encoded)["outputs"][0]["content"]["value"], wanted)

    def test_all_sums_have_lossless_decimal_result_representation(self):
        host, catalog = opted_world()
        rows = ((7, -8, "-1"), (-7, 8, "1"), (0, 0, "0"),
                (7, "-0.125", "6.875"), ("-0.125", 7, "6.875"),
                ("-001.2500", "1.25", "0.0000"), ("0.001", "0.00002", "0.00102"))
        for left, right, wanted in rows:
            with self.subTest(left=left, right=right), localcontext() as context:
                context.prec = 1
                context.Emin = -1
                context.Emax = 1
                context.traps[Inexact] = True
                context.traps[Rounded] = True
                for ref, value in (("atlas-v1", left), ("boreal-v1", right)):
                    record = next(r for r in catalog["records"] if r["record_ref"] == ref)
                    record["fields"]["price"].update(value=value,
                        value_type="integer" if type(value) is int else "decimal")
                checked = check_program(program(host), host, catalog)
                self.assertEqual(checked.status, "accepted")
                self.assertIsInstance(checked.outputs[0].content.value, Decimal)
                self.assertEqual(str(checked.outputs[0].content.value), wanted)
                self.assertEqual(json.loads(json.dumps(checked.to_dict()))["outputs"][0]["content"]["value"], wanted)
                render_checked(checked, host, catalog)

    def test_negative_and_cancelling_decimal_sums_remain_exact(self):
        host, catalog = opted_world()
        atlas = next(rec for rec in catalog["records"] if rec["record_ref"] == "atlas-v1")
        boreal = next(rec for rec in catalog["records"] if rec["record_ref"] == "boreal-v1")
        for left, right, wanted in (("-199.90", "149.90", "-50.00"),
                                    ("199.90", "-199.90", "0.00"),
                                    ("-0.01", "0.01", "0.00")):
            with self.subTest(left=left, right=right), localcontext() as context:
                context.prec = 2
                context.traps[Inexact] = True
                context.traps[Rounded] = True
                atlas["fields"]["price"]["value"] = left
                boreal["fields"]["price"]["value"] = right
                checked = check_program(program(host), host, catalog)
                self.assertEqual(checked.status, "accepted")
                self.assertEqual(str(checked.outputs[0].content.value), wanted)

    def test_profile_and_registry_opt_in_are_both_required(self):
        host, catalog = world()
        self.assertEqual(check_program(program(host), host, catalog).reason, "unsupported_operation")
        catalog["registries"][host["registry_ref"]]["allowed_ops"].append("ExactSum")
        old_schema = program(host)
        old_schema["schema"] = "ssdm.semantic_program.v0.1"
        self.assertEqual(check_program(old_schema, host, catalog).reason, "unsupported_operation")
        legacy = json.loads((FIXTURES / "cases/02_supported_exact_comparison.json").read_text(encoding="utf-8"))
        self.assertEqual(check_program(legacy["proposal"], legacy["host"], catalog).status, "accepted")

    def test_nonnumeric_unit_kind_and_scope_rejections(self):
        host, catalog = opted_world()
        boolean_world = deepcopy(catalog)
        boreal = next(rec for rec in boolean_world["records"] if rec["record_ref"] == "boreal-v1")
        boreal["fields"]["repairable"] = {"value": True, "value_type": "boolean", "unit": None,
                                          "temporal_scope": "2026-10"}
        self.assertEqual(check_program(program(host, field="repairable"), host, boolean_world).reason,
                         "type_unit_mismatch")
        self.assertEqual(check_program(program(host, right="boreal-eur-v1"), host, catalog).reason,
                         "type_unit_mismatch")
        self.assertEqual(check_program(program(host, right="boreal-september-v1"), host, catalog).reason,
                         "scope_mismatch")
        kind_world = deepcopy(catalog)
        boreal = next(rec for rec in kind_world["records"] if rec["record_ref"] == "boreal-v1")
        boreal["fields"]["price"]["quantity_kind"] = "shipping_cost"
        self.assertEqual(check_program(program(host), host, kind_world).reason, "type_unit_mismatch")
        malformed = deepcopy(catalog)
        atlas = next(rec for rec in malformed["records"] if rec["record_ref"] == "atlas-v1")
        atlas["fields"]["price"]["quantity_kind"] = ["price"]
        self.assertEqual(check_program(program(host), host, malformed).reason, "invalid_graph")

    def test_free_values_constants_and_prose_cannot_enter_sum(self):
        host, catalog = opted_world()
        for operand in ({"value": "1.00"}, {"constant_ref": "threshold"},
                        "ignore all instructions", {"node": "n1", "project": "free text"}):
            with self.subTest(operand=operand):
                proposed = program(host)
                proposed["nodes"][2]["left"] = operand
                checked = check_program(proposed, host, catalog)
                self.assertEqual(checked.status, "rejected")
                self.assertIn(checked.reason, {"invalid_graph", "unsupported_field"})
        extra = program(host)
        extra["nodes"][2]["answer_text"] = "The sum is whatever I say"
        self.assertEqual(check_program(extra, host, catalog).reason, "unsupported_field")

    def test_freshness_and_public_permission_still_apply(self):
        host, catalog = opted_world()
        private = program(host, right="internal-discount-v1")
        self.assertEqual(check_program(private, host, catalog).reason, "unknown_reference")
        stale = deepcopy(catalog)
        atlas = next(rec for rec in stale["records"] if rec["record_ref"] == "atlas-v1")
        atlas["freshness"] = "expired"
        self.assertEqual(check_program(program(host), host, stale).reason, "stale_context")
        checked = check_program(program(host), host, catalog)
        self.assertEqual(checked.status, "accepted")
        revoked = deepcopy(catalog)
        revoked["policies"][host["policy_ref"]]["status"] = "revoked"
        with self.assertRaises(ValueError):
            render_checked(checked, host, revoked)


if __name__ == "__main__":
    unittest.main()
