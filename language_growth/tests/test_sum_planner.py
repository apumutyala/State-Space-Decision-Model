"""Scripted parent selection of the opt-in sum operation; no learned claim."""

from copy import deepcopy
import json
import unittest

from language_growth.planner import plan_checked_answer
from language_growth.tests.test_planner import ScriptedProvider, case, catalog, REVISION, EXPECTED_CONFIG


SUM = "Sum two selected numeric facts"


def run(host, evidence, provider):
    return plan_checked_answer("What is the sum of Atlas and Boreal prices?", host, evidence, provider,
                               model="scripted", parent_revision=REVISION,
                               expected_served_config=EXPECTED_CONFIG)


class SumPlannerTests(unittest.TestCase):
    def setUp(self):
        self.host = case(2)["host"]
        self.catalog = catalog()
        self.catalog["registries"][self.host["registry_ref"]]["allowed_ops"].append("ExactSum")

    def test_parent_selects_operation_and_both_facts(self):
        provider = ScriptedProvider([SUM, "Fact atlas-v1 field price", "Fact boreal-v1 field price"])
        result = run(self.host, self.catalog, provider)
        self.assertEqual(result.status, "accepted")
        self.assertEqual((result.provider_calls, result.forward_calls), (3, 3))
        self.assertEqual(result.program["schema"], "ssdm.semantic_program.v0.2")
        self.assertEqual(result.program["nodes"][2]["op"], "ExactSum")
        self.assertIn('349.80 "USD"', result.text)
        self.assertNotIn("349.80", json.dumps(provider.requests))
        for request in provider.requests[1:]:
            options = next(iter(request["questions"].values()))["criteria"]
            self.assertTrue(all(description.startswith("Fact ") for description in options.values()))

    def test_no_registry_opt_in_means_no_sum_choice(self):
        self.catalog["registries"][self.host["registry_ref"]]["allowed_ops"].remove("ExactSum")
        provider = ScriptedProvider(["Assert one selected fact", "Fact atlas-v1 field price"])
        result = run(self.host, self.catalog, provider)
        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.program["schema"], "ssdm.semantic_program.v0.1")
        first = next(iter(provider.requests[0]["questions"].values()))["criteria"]
        self.assertFalse(any(SUM in text for text in first.values()))

    def test_old_operation_still_uses_v01_when_sum_is_available(self):
        provider = ScriptedProvider(["Assert one selected fact", "Fact atlas-v1 field price"])
        result = run(self.host, self.catalog, provider)
        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.program["schema"], "ssdm.semantic_program.v0.1")

    def test_incompatible_units_and_nonnumeric_values_never_render(self):
        for mode in ("units", "boolean"):
            evidence = deepcopy(self.catalog)
            fact = next(row for row in evidence["records"] if row["record_ref"] == "boreal-v1")["fields"]["price"]
            if mode == "units":
                right = "Fact boreal-eur-v1 field price"
            else:
                fact.update(value=True, value_type="boolean")
                right = "Fact boreal-v1 field price"
            provider = ScriptedProvider([SUM, "Fact atlas-v1 field price", right])
            result = run(self.host, evidence, provider)
            self.assertEqual((result.status, result.reason), ("kernel_rejected", "type_unit_mismatch"))
            self.assertIsNone(result.text)

    def test_revocation_during_selection_prevents_output(self):
        def revoke(_):
            if len(provider.requests) == 3:
                self.catalog["policies"][self.host["policy_ref"]]["status"] = "revoked"

        provider = ScriptedProvider([SUM, "Fact atlas-v1 field price", "Fact boreal-v1 field price"],
                                    contaminate=revoke)
        result = run(self.host, self.catalog, provider)
        self.assertNotEqual(result.status, "accepted")
        self.assertIsNone(result.text)


if __name__ == "__main__":
    unittest.main()
