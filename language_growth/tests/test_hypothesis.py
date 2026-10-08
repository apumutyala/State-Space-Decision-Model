"""Illustrative host-declared conditional reasoning; no learned evaluation."""

import copy
import json
import unittest
from pathlib import Path

from language_growth.kernel import check_program
from language_growth.realize import render_checked
from language_growth.types import AssumptionValue, ConditionalContent


FIXTURES = Path(__file__).resolve().parents[2] / "fixtures/language_growth/hypothesis"


def fixture(name="registered_assumption"):
    item = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return item["host"], item["catalog"], item["proposal"]


class HypothesisTests(unittest.TestCase):
    def test_registered_assumption_stays_conditional(self):
        host, catalog, proposal = fixture()
        before = copy.deepcopy(catalog)
        result = check_program(proposal, host, catalog)
        self.assertEqual(result.status, "accepted")
        act = result.outputs[0]
        self.assertEqual(act.kind, "hypothesize")
        self.assertIsInstance(act.content, ConditionalContent)
        self.assertIsInstance(act.content.conclusion.left, AssumptionValue)
        text = render_checked(result, host, catalog)
        self.assertEqual(text, 'If at "2026-10", hypothetical "Price" for "atlas" had value 99.90 "USD", then at "2026-10", the value of hypothetical "Price" for "atlas" (99.90 "USD") is less than the value of recorded baseline "Price" for "boreal" (149.90 "USD").')
        self.assertEqual(catalog, before)
        self.assertEqual(act.content.assumption.premise.to_dict()["object"], "199.90")
        self.assertEqual(act.to_dict()["content"]["assumption"]["epistemic_class"], "hypothesis")

    def test_same_slot_counterfactual_labels_hypothetical_and_recorded_baseline(self):
        host, catalog, proposal = fixture()
        proposal["nodes"][0]["record_ref"] = "atlas-v1"
        result = check_program(proposal, host, catalog)
        self.assertEqual(result.status, "accepted")
        text = render_checked(result, host, catalog)
        self.assertEqual(text, 'If at "2026-10", hypothetical "Price" for "atlas" had value 99.90 "USD", then at "2026-10", the value of hypothetical "Price" for "atlas" (99.90 "USD") is less than the value of recorded baseline "Price" for "atlas" (199.90 "USD").')
        proposal["nodes"][-1]["op"] = "Assert"
        self.assertEqual(check_program(proposal, host, catalog).reason, "unsupported_assertion")

    def test_inline_model_fact_refuses(self):
        host, catalog, proposal = fixture("inline_model_fact")
        result = check_program(proposal, host, catalog)
        self.assertEqual((result.status, result.reason), ("rejected", "unsupported_field"))
        self.assertEqual(result.outputs, ())

    def test_assumption_cannot_escape_as_assertion(self):
        host, catalog, proposal = fixture()
        proposal["nodes"][-1]["op"] = "Assert"
        result = check_program(proposal, host, catalog)
        self.assertEqual((result.status, result.reason), ("rejected", "unsupported_assertion"))

    def test_assumption_cannot_escape_through_exact_difference(self):
        host, catalog, proposal = fixture()
        catalog["registries"][host["registry_ref"]]["allowed_ops"].append("ExactDifference")
        proposal["nodes"][-2]["op"] = "ExactDifference"
        proposal["nodes"][-2].pop("relation")
        proposal["nodes"][-1]["op"] = "Assert"
        result = check_program(proposal, host, catalog)
        self.assertEqual((result.status, result.reason), ("rejected", "unsupported_assertion"))
        self.assertEqual(result.outputs, ())

    def test_false_conditional_conclusion_refuses(self):
        host, catalog, proposal = fixture()
        proposal["nodes"][-2]["relation"] = "greater_than"
        self.assertEqual(check_program(proposal, host, catalog).reason, "unsupported_assertion")

    def test_unregistered_or_hidden_assumption_refuses(self):
        host, catalog, proposal = fixture()
        host["assumptions"].clear()
        self.assertEqual(check_program(proposal, host, catalog).reason, "unknown_reference")
        host, catalog, proposal = fixture()
        host["assumptions"]["atlas-sale"]["record_ref"] = "hidden-record"
        self.assertEqual(check_program(proposal, host, catalog).reason, "unknown_reference")

    def test_assumption_requires_opt_in_and_preserves_types(self):
        for operation in ("Hypothesize", "ReadFact"):
            host, catalog, proposal = fixture()
            catalog["registries"][host["registry_ref"]]["allowed_ops"].remove(operation)
            # Remove the explicit read: the assumption's implicit premise read
            # must independently require ReadFact permission.
            proposal["nodes"] = proposal["nodes"][1:]
            proposal["nodes"][0]["right"] = {"constant_ref": "threshold"}
            host["query_constants"]["threshold"] = {
                "value": "149.90", "value_type": "decimal", "unit": "USD",
                "temporal_scope": "2026-10", "quantity_kind": "price"}
            self.assertEqual(check_program(proposal, host, catalog).reason, "unsupported_operation")
        for key, value, reason in (("unit", "EUR", "type_unit_mismatch"),
                                   ("temporal_scope", "2026-09", "scope_mismatch"),
                                   ("quantity_kind", "shipping_cost", "type_unit_mismatch")):
            host, catalog, proposal = fixture()
            host["assumptions"]["atlas-sale"]["value"][key] = value
            self.assertEqual(check_program(proposal, host, catalog).reason, reason)

    def test_live_assumption_change_or_revocation_denies_release(self):
        for change in (lambda h: h["assumptions"]["atlas-sale"]["value"].update(value="89.90"),
                       lambda h: h["assumptions"].clear()):
            host, catalog, proposal = fixture()
            result = check_program(proposal, host, catalog)
            self.assertEqual(result.status, "accepted")
            change(host)
            with self.assertRaises(ValueError):
                render_checked(result, host, catalog)

    def test_factual_assertion_unchanged_with_assumption_registry(self):
        host, catalog, proposal = fixture()
        proposal["nodes"] = [{"id": "n1", "op": "ReadFact", "record_ref": "atlas-v1", "field": "price"},
                             {"id": "n2", "op": "Assert", "proposition": "n1"}]
        proposal["outputs"] = ["n2"]
        result = check_program(proposal, host, catalog)
        self.assertEqual(result.status, "accepted")
        self.assertEqual(render_checked(result, host, catalog), 'At "2026-10", "Price" for "atlas" has value 199.90 "USD".')


if __name__ == "__main__":
    unittest.main()
