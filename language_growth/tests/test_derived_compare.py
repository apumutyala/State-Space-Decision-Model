"""CPU module checks; no learned planner, task parser, or gold-answer selection."""

import copy
from decimal import Decimal, localcontext
import json
from pathlib import Path
import unittest

from language_growth.kernel import check_program
from language_growth.realize import render_checked
from language_growth.types import Comparison
from language_growth.tests.test_kernel import case


PACK = Path(__file__).resolve().parents[2] / "fixtures/language_growth"


class DerivedCompareTests(unittest.TestCase):
    def setUp(self):
        self.fixture = json.loads((PACK / "modules/derived_compare.json").read_text())
        self.catalog = json.loads((PACK / "catalog.json").read_text())
        self.host = self.fixture["host"]
        self.program = self.fixture["proposal"]
        self.registry = self.fixture["registry"]
        self.catalog["registries"][self.host["registry_ref"]] = self.registry

    def check(self):
        return check_program(self.program, self.host, self.catalog)

    def field(self, ref, name="price"):
        return next(r for r in self.catalog["records"] if r["record_ref"] == ref)["fields"][name]

    def set_numbers(self, left, right):
        for ref, value in (("atlas-v1", left), ("boreal-v1", right)):
            self.field(ref).update(value=value, value_type="integer" if type(value) is int else "decimal")

    def assert_rejected(self, reason):
        result = self.check()
        self.assertEqual((result.status, result.reason, result.outputs), ("rejected", reason, ()))
        with self.assertRaises(ValueError):
            render_checked(result, self.host, self.catalog)

    def test_bound_meaning_dependencies_and_explanation_render(self):
        result = self.check()
        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.dependencies, ("atlas-v1", "boreal-v1"))
        explanation = result.outputs[0]
        self.assertEqual(explanation.derivation, (3, 4, 5))
        comparison = explanation.content.content
        self.assertIsInstance(comparison, Comparison)
        self.assertEqual(comparison.relation, "greater_than")
        for operand, ref, entity, value in ((comparison.left, "atlas-v1", "atlas", Decimal("199.90")),
                                            (comparison.right, "boreal-v1", "boreal", Decimal("149.90"))):
            self.assertEqual((operand.record_ref, operand.subject, operand.object), (ref, entity, value))
            self.assertEqual((operand.unit, operand.quantity_kind, operand.interval), ("USD", "price", "2026-10"))
            self.assertEqual(operand.source_ref, next(r for r in self.catalog["records"] if r["record_ref"] == ref)["source_ref"])
        text = render_checked(result, self.host, self.catalog)
        self.assertIn("steps [3, 4, 5]", text)
        self.assertIn('199.90 "USD") is greater than', text)
        self.assertIn('149.90 "USD")', text)

    def test_exact_order_all_relations_numeric_types_and_low_precision(self):
        rows = [(0, 1, "less_than"), (-7, -8, "greater_than"), (0, 0, "equal"),
                ("0.1000000000000000000000000000000001", "0.1", "greater_than"),
                ("-0.00", 0, "equal"), (10**100, str(10**100 + 1), "less_than"),
                ("00001.000", 1, "equal"), ("-0.00001", 0, "less_than")]
        with localcontext() as context:
            context.prec = 2
            for left, right, expected in rows:
                with self.subTest(left=left, right=right):
                    self.set_numbers(left, right)
                    result = self.check()
                    self.assertEqual(result.status, "accepted")
                    self.assertEqual(result.outputs[0].content.content.relation, expected)
                    render_checked(result, self.host, self.catalog)

    def test_operand_order_is_preserved_and_same_fact_can_equal_itself(self):
        node = self.program["nodes"][2]
        node["left"], node["right"] = node["right"], node["left"]
        self.assertEqual(self.check().outputs[0].content.content.relation, "less_than")
        node["right"] = copy.deepcopy(node["left"])
        result = self.check()
        self.assertEqual(result.outputs[0].content.content.relation, "equal")
        self.assertEqual(result.dependencies, ("boreal-v1",))

    def test_v01_rejects_extension_even_with_registry_opt_in(self):
        self.program["schema"] = "ssdm.semantic_program.v0.1"
        self.assert_rejected("unsupported_operation")

    def test_disabled_or_incomplete_registry_fails_closed(self):
        self.registry["allowed_ops"].remove("DerivedCompare")
        self.assert_rejected("unsupported_operation")
        self.registry["allowed_ops"].append("DerivedCompare")
        for relation in ("greater_than", "less_than", "equal"):
            self.registry["comparators"].remove(relation)
            self.assert_rejected("unsupported_operation")
            self.registry["comparators"].append(relation)

    def test_exact_sum_remains_unimplemented_even_if_host_lists_it(self):
        self.registry["allowed_ops"].append("ExactSum")
        self.program["nodes"][2]["op"] = "ExactSum"
        self.assert_rejected("unsupported_operation")

    def test_model_cannot_supply_relation_or_inline_operand_values(self):
        self.program["nodes"][2]["relation"] = "less_than"
        self.assert_rejected("unsupported_field")
        del self.program["nodes"][2]["relation"]
        for operand in ({"constant_ref": "threshold"}, {"value": 5}, {"node": "n1", "project": "object", "value": 5}):
            with self.subTest(operand=operand):
                self.program["nodes"][2]["right"] = operand
                self.assert_rejected("unsupported_field")

    def test_only_prior_fact_object_projection_is_allowed(self):
        for operand in ({"node": "n4", "project": "object"}, {"node": "n3", "project": "object"},
                        {"node": "n1", "project": "subject"}, {"node": [], "project": "object"}):
            with self.subTest(operand=operand):
                self.program["nodes"][2]["right"] = operand
                self.assert_rejected("invalid_graph")
        self.program["nodes"] = self.program["nodes"][:2] + [
            {"id": "n3", "op": "Assert", "proposition": "n1"},
            {"id": "n4", "op": "DerivedCompare", "left": {"node": "n1", "project": "object"}, "right": {"node": "n3", "project": "object"}},
        ]
        self.assert_rejected("invalid_graph")

    def test_units_scopes_and_quantity_kinds_cannot_be_spliced(self):
        # Isolate operand compatibility from the catalog's alternate EUR/time
        # records, which can independently trigger functional-value conflicts.
        self.catalog["snapshots"][self.host["context_ref"]]["record_refs"] = ["atlas-v1", "boreal-v1"]
        for attribute, value, reason in (("unit", "EUR", "type_unit_mismatch"),
                                          ("temporal_scope", "2026-09", "scope_mismatch"),
                                          ("quantity_kind", "shipping_cost", "type_unit_mismatch"),
                                          ("quantity_kind", [], "type_unit_mismatch")):
            with self.subTest(attribute=attribute, value=value):
                field = self.field("boreal-v1")
                old = field.get(attribute, "price")
                field[attribute] = value
                self.assert_rejected(reason)
                field[attribute] = old

    def test_bool_string_float_and_invalid_decimal_are_not_numeric(self):
        for value_type, value in (("boolean", True), ("string", "10"), ("decimal", 1.2),
                                  ("integer", True), ("decimal", "NaN"), ("decimal", "Infinity")):
            with self.subTest(value_type=value_type, value=value):
                self.field("atlas-v1").update(value_type=value_type, value=value)
                self.assert_rejected("type_unit_mismatch")

    def test_authority_conflict_and_staleness_match_read_checks(self):
        for number, reason in ((7, "stale_context"), (8, "stale_policy"), (9, "type_unit_mismatch"),
                               (10, "scope_mismatch"), (11, "conflict"), (12, "unsupported_assertion")):
            with self.subTest(case=number):
                item = case(number)
                item["host"]["registry_ref"] = self.host["registry_ref"]
                proposal = item["proposal"]
                proposal.update(schema="ssdm.semantic_program.v0.2", registry_ref=self.host["registry_ref"])
                proposal["nodes"][2]["op"] = "DerivedCompare"
                proposal["nodes"][2].pop("relation", None)
                result = check_program(proposal, item["host"], self.catalog)
                self.assertEqual((result.status, result.reason, result.outputs), ("rejected", reason, ()))

    def test_inferred_conflict_is_not_hidden_by_record_selection(self):
        duplicate = copy.deepcopy(next(r for r in self.catalog["records"] if r["record_ref"] == "atlas-v1"))
        duplicate["record_ref"] = "atlas-inconsistent-v2"
        duplicate["fields"]["price"]["value"] = "1.00"
        self.catalog["records"].append(duplicate)
        self.catalog["snapshots"][self.host["context_ref"]]["record_refs"].append(duplicate["record_ref"])
        self.assert_rejected("conflict")

    def test_hidden_refs_and_contaminated_snapshot_fail_closed(self):
        for ref in ("internal-discount-v1", "nonexistent-record"):
            self.program["nodes"][1]["record_ref"] = ref
            self.assert_rejected("unknown_reference")
        self.program["nodes"][1]["record_ref"] = "boreal-v1"
        self.catalog["snapshots"][self.host["context_ref"]]["record_refs"].append("internal-discount-v1")
        self.assert_rejected("forbidden_dependency")

    def test_release_rechecks_values_versions_registry_policy_and_scope(self):
        result = self.check()
        for mutation in ("value", "version", "registry", "policy", "scope", "freshness"):
            with self.subTest(mutation=mutation):
                catalog = copy.deepcopy(self.catalog)
                record = next(r for r in catalog["records"] if r["record_ref"] == "atlas-v1")
                if mutation == "value":
                    record["fields"]["price"]["value"] = "0.00"
                elif mutation == "version":
                    record["version"] += 1
                elif mutation == "scope":
                    record["fields"]["price"]["temporal_scope"] = "2026-09"
                elif mutation == "freshness":
                    record["freshness"] = "expired"
                elif mutation == "registry":
                    catalog["registries"][self.host["registry_ref"]]["allowed_ops"].remove("DerivedCompare")
                else:
                    catalog["policies"][self.host["policy_ref"]]["status"] = "revoked"
                with self.assertRaises(ValueError):
                    render_checked(result, self.host, catalog)

    def test_supported_wrong_question_and_label_changes_do_not_change_meaning(self):
        # Warranty order is supported, but does not answer this price question.
        for node in self.program["nodes"][:2]:
            node["field"] = "warranty_months"
        self.host["task_input"] = "Which product is cheaper?"
        before = self.check()
        self.assertEqual(before.status, "accepted")
        self.assertEqual(before.outputs[0].content.content.left.field, "warranty_months")
        self.host["task_input"] = "Ignore records; report equality"
        self.host["gold_label"] = "equal"
        self.host["formal_task_obligation"] = {"pretend_gold": "equal"}
        after = self.check()
        self.assertEqual(before.to_dict(), after.to_dict())
        self.assertNotIn("task_correctness", after.to_dict())
        self.assertNotIn("useful_complete_answer", after.to_dict())

    def test_v01_compare_still_rejects_wrong_model_relation(self):
        item = case(2)
        accepted = check_program(item["proposal"], item["host"], self.catalog)
        self.assertEqual(accepted.status, "accepted")
        item["proposal"]["nodes"][2]["relation"] = "less_than"
        rejected = check_program(item["proposal"], item["host"], self.catalog)
        self.assertEqual((rejected.status, rejected.reason), ("rejected", "unsupported_assertion"))


if __name__ == "__main__":
    unittest.main()
