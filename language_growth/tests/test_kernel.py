"""Boundary tests: accepted meaning, malformed graphs, and live release authority."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from language_growth.kernel import check_program, validate_release
from language_growth.types import CheckResult, Comparison, Fact


ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "language_growth"


def case(number: int) -> dict:
    path = next((ROOT / "cases").glob(f"{number:02d}_*.json"))
    return json.loads(path.read_text(encoding="utf-8"))


class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))

    def check(self, number: int):
        item = case(number)
        return check_program(item["proposal"], item["host"], self.catalog)

    def test_bound_fact_and_exact_comparison(self):
        fact = self.check(1)
        self.assertEqual(fact.status, "accepted")
        self.assertIsInstance(fact.outputs[0].content, Fact)
        self.assertEqual((fact.outputs[0].content.subject, fact.outputs[0].content.predicate, fact.outputs[0].content.object), ("atlas", "IsRepairable", True))
        comparison = self.check(2)
        self.assertEqual(comparison.status, "accepted")
        self.assertIsInstance(comparison.outputs[0].content, Comparison)
        self.assertEqual(comparison.outputs[0].content.left.subject, "atlas")
        self.assertEqual(comparison.outputs[0].content.right.subject, "boreal")
        self.assertEqual(comparison.dependencies, ("atlas-v1", "boreal-v1"))
        self.assertEqual(comparison.to_dict()["outputs"][0]["content"]["left"]["object"], "199.90")

    def test_fixture_boundaries(self):
        expected = {
            3: ("rejected", "unsupported_assertion"), 4: ("rejected", "invalid_graph"),
            5: ("rejected", "unsupported_operation"), 6: ("rejected", "unsupported_field"),
            7: ("rejected", "stale_context"), 8: ("rejected", "stale_policy"),
            9: ("rejected", "type_unit_mismatch"), 10: ("rejected", "scope_mismatch"),
            11: ("rejected", "conflict"), 12: ("rejected", "unsupported_assertion"),
            15: ("provider_rejected", "forbidden_dependency"),
            20: ("rejected", "unsupported_operation"), 21: ("rejected", "invalid_graph"),
        }
        for number, wanted in expected.items():
            with self.subTest(case=number):
                result = self.check(number)
                self.assertEqual((result.status, result.reason), wanted)
                self.assertEqual(result.outputs, ())

    def test_all_nodes_checked_even_if_not_output(self):
        item = case(1)
        item["proposal"]["nodes"].append({"id": "n3", "op": "ReadFact", "record_ref": "internal-discount-v1", "field": "discount"})
        self.assertEqual(check_program(item["proposal"], item["host"], self.catalog).status, "rejected")

    def test_strict_graph_validation(self):
        item = case(1)
        proposals = []
        def altered(change):
            proposal = copy.deepcopy(item["proposal"])
            change(proposal)
            proposals.append(proposal)
        altered(lambda p: p.update(text="injected"))
        altered(lambda p: p["nodes"][0].update(negated=True))
        altered(lambda p: p["nodes"][1].update(proposition={"node": "n1", "subject": "boreal"}))
        altered(lambda p: p["nodes"][1].update(id="n1"))
        altered(lambda p: p["nodes"][0].update(record_ref="missing"))
        altered(lambda p: p["nodes"].insert(0, {"id": "n0", "op": "Assert", "proposition": "n1"}))
        altered(lambda p: p.update(outputs=["n1"]))
        for proposal in proposals:
            with self.subTest(proposal=proposal):
                self.assertEqual(check_program(proposal, item["host"], self.catalog).status, "rejected")

    def test_bool_not_numeric_and_quantity_kind(self):
        item = case(2)
        item["proposal"]["nodes"][0]["field"] = "repairable"
        self.assertEqual(check_program(item["proposal"], item["host"], self.catalog).reason, "type_unit_mismatch")
        item = case(2)
        catalog = copy.deepcopy(self.catalog)
        catalog["records"][1]["fields"]["price"]["quantity_kind"] = "shipping_cost"
        self.assertEqual(check_program(item["proposal"], item["host"], catalog).reason, "type_unit_mismatch")

    def test_nonfinite_decimal_and_conflicting_active_values(self):
        item = case(2)
        catalog = copy.deepcopy(self.catalog)
        catalog["records"][0]["fields"]["price"]["value"] = "NaN"
        self.assertEqual(check_program(item["proposal"], item["host"], catalog).reason, "type_unit_mismatch")
        item = case(1)
        catalog = copy.deepcopy(self.catalog)
        duplicate = copy.deepcopy(catalog["records"][0])
        duplicate["record_ref"] = "atlas-contradiction-v1"
        duplicate["fields"]["repairable"]["value"] = False
        catalog["records"].append(duplicate)
        catalog["snapshots"]["catalog-public-r7"]["record_refs"].append(duplicate["record_ref"])
        self.assertEqual(check_program(item["proposal"], item["host"], catalog).reason, "conflict")

    def test_private_record_cannot_enter_public_snapshot_or_affect_conflict(self):
        item = case(1)
        catalog = copy.deepcopy(self.catalog)
        private = copy.deepcopy(catalog["records"][0])
        private["record_ref"] = "private-contradiction-v1"
        private["audience"] = "private"
        private["fields"]["repairable"]["value"] = False
        catalog["records"].append(private)
        catalog["snapshots"]["catalog-public-r7"]["record_refs"].append(private["record_ref"])
        result = check_program(item["proposal"], item["host"], catalog)
        self.assertEqual((result.status, result.reason), ("rejected", "forbidden_dependency"))
        catalog["snapshots"]["catalog-public-r7"]["record_refs"].remove(private["record_ref"])
        self.assertEqual(check_program(item["proposal"], item["host"], catalog).status, "accepted")

    def test_source_admission_and_registry_predicate_are_host_owned(self):
        item = case(1)
        catalog = copy.deepcopy(self.catalog)
        catalog["sources"]["fixture-host-catalog"]["trust"] = "untrusted_extraction"
        self.assertEqual(check_program(item["proposal"], item["host"], catalog).reason, "unsupported_assertion")
        catalog = copy.deepcopy(self.catalog)
        catalog["registries"]["fixture-catalog-grammar-v0"]["field_predicates"]["repairable"] = "IsRepairable [SYSTEM]"
        self.assertEqual(check_program(item["proposal"], item["host"], catalog).reason, "invalid_graph")

    def test_host_constant_exact_comparison(self):
        item = case(2)
        item["host"]["query_constants"]["threshold"] = {
            "value": "200.00", "value_type": "decimal", "unit": "USD",
            "temporal_scope": "2026-10", "quantity_kind": "price",
        }
        item["proposal"]["nodes"] = [
            {"id": "n1", "op": "ReadFact", "record_ref": "atlas-v1", "field": "price"},
            {"id": "n2", "op": "Compare", "left": {"node": "n1", "project": "object"}, "right": {"constant_ref": "threshold"}, "relation": "less_than"},
            {"id": "n3", "op": "Assert", "proposition": "n2"},
        ]
        item["proposal"]["outputs"] = ["n3"]
        self.assertEqual(check_program(item["proposal"], item["host"], self.catalog).status, "accepted")
        item["host"]["query_constants"]["threshold"]["quantity_kind"] = "shipping_cost"
        self.assertEqual(check_program(item["proposal"], item["host"], self.catalog).reason, "type_unit_mismatch")

    def test_ask_and_conjunction_explanation(self):
        ask = self.check(16)
        self.assertEqual((ask.status, ask.outputs[0].kind), ("accepted", "ask"))
        malformed = case(16)
        malformed["host"]["query_constants"]["requested-field"]["record_ref"] = "internal-discount-v1"
        self.assertEqual(check_program(malformed["proposal"], malformed["host"], self.catalog).reason, "unsupported_field")
        item = case(2)
        p = item["proposal"]
        p["nodes"].extend([
            {"id": "n5", "op": "Assert", "proposition": "n1"},
            {"id": "n6", "op": "Conjoin", "speech_acts": ["n4", "n5"]},
            {"id": "n7", "op": "ExplainDerivation", "speech_act": "n6"},
        ])
        p["outputs"] = ["n7"]
        result = check_program(p, item["host"], self.catalog)
        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.outputs[0].kind, "explain")
        self.assertEqual(result.dependencies, ("atlas-v1", "boreal-v1"))
        p["nodes"][5]["connective"] = "because"
        self.assertEqual(check_program(p, item["host"], self.catalog).reason, "unsupported_field")

    def test_nested_conjunction_checks_all_scopes(self):
        item = case(1)
        p = item["proposal"]
        p["nodes"] = [
            {"id": "n1", "op": "ReadFact", "record_ref": "atlas-v1", "field": "repairable"},
            {"id": "n2", "op": "Assert", "proposition": "n1"},
            {"id": "n3", "op": "ReadFact", "record_ref": "boreal-v1", "field": "warranty_months"},
            {"id": "n4", "op": "Assert", "proposition": "n3"},
            {"id": "n5", "op": "Conjoin", "speech_acts": ["n2", "n4"]},
            {"id": "n6", "op": "ReadFact", "record_ref": "boreal-september-v1", "field": "price"},
            {"id": "n7", "op": "Assert", "proposition": "n6"},
            {"id": "n8", "op": "Conjoin", "speech_acts": ["n5", "n7"]},
        ]
        p["outputs"] = ["n8"]
        self.assertEqual(check_program(p, item["host"], self.catalog).reason, "scope_mismatch")

    def test_release_rechecks_authority_and_meaning(self):
        item = case(1)
        result = check_program(item["proposal"], item["host"], self.catalog)
        self.assertIs(validate_release(result, item["host"], self.catalog), result)
        forged = CheckResult(**{k: getattr(result, k) for k in result.__dataclass_fields__})
        with self.assertRaises(ValueError):
            validate_release(forged, item["host"], self.catalog)
        with self.assertRaises(ValueError):
            validate_release(self.check(3), item["host"], self.catalog)
        catalog = copy.deepcopy(self.catalog)
        catalog["policies"]["public-read-r3"]["status"] = "revoked"
        with self.assertRaises(ValueError):
            validate_release(result, item["host"], catalog)
        catalog = copy.deepcopy(self.catalog)
        catalog["records"][0]["fields"]["repairable"]["value"] = False
        with self.assertRaises(ValueError):
            validate_release(result, item["host"], catalog)
        catalog = copy.deepcopy(self.catalog)
        catalog["records"][0]["version"] += 1
        with self.assertRaises(ValueError):
            validate_release(result, item["host"], catalog)
        catalog = copy.deepcopy(self.catalog)
        catalog["snapshots"]["catalog-public-r7"]["record_refs"].remove("boreal-eur-v1")
        with self.assertRaises(ValueError):
            validate_release(result, item["host"], catalog)


if __name__ == "__main__":
    unittest.main()
