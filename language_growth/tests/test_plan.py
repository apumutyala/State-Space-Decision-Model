"""Focused non-executing registered-plan boundary; no learned evaluation."""

import copy
import json
import unittest
from pathlib import Path

from language_growth.kernel import check_program
from language_growth.realize import render_checked
from language_growth.types import PlanContent


def fixture():
    item = json.loads((Path(__file__).resolve().parents[2] /
                       "fixtures/language_growth/plan/registered_workflow.json").read_text())
    return item["host"], item["catalog"], item["proposal"]


class PlanTests(unittest.TestCase):
    def test_registered_plan_is_conditional_and_has_no_effects(self):
        host, catalog, proposal = fixture()
        before = copy.deepcopy((host, catalog, proposal))
        result = check_program(proposal, host, catalog)
        self.assertEqual(result.status, "accepted")
        self.assertIsInstance(result.outputs[0].content, PlanContent)
        content = result.to_dict()["outputs"][0]["content"]
        self.assertEqual((content["epistemic_class"], content["execution"]), ("plan", "none"))
        self.assertEqual([step["status"] for step in content["steps"]], ["proposed", "proposed"])
        self.assertEqual(result.dependencies, ("atlas-v1",))
        self.assertEqual(render_checked(result, host, catalog),
            'Proposed plan for workflow "repair-request"; no actions executed.\n'
            'Proposed step 1 ("inspect"): if at "2026-10", recorded "IsRepairable" for "atlas" has value true, propose action "inspect-atlas-repair-options"; execution requires separate authorization and precondition recheck.\n'
            'Proposed step 2 ("draft-request") after proposed steps ["inspect"]: if at "2026-10", recorded "WarrantyMonths" for "atlas" has value 24 "month", propose action "draft-atlas-repair-request"; execution requires separate authorization and precondition recheck.')
        self.assertEqual((host, catalog, proposal), before)

    def test_model_cannot_supply_action_authority_effect_or_completion(self):
        for key, value in (("action_ref", "send-request"), ("authority", "granted"),
                           ("execute", True), ("complete", True), ("preconditions", [])):
            with self.subTest(key=key):
                host, catalog, proposal = fixture()
                proposal["nodes"][0][key] = value
                self.assertEqual(check_program(proposal, host, catalog).reason, "unsupported_field")

    def test_only_registered_ordered_steps_are_accepted(self):
        for refs, reason in ((["invented"], "unknown_reference"),
                             (["draft-request", "inspect"], "invalid_graph"),
                             (["draft-request"], "invalid_graph"),
                             (["inspect", "inspect"], "invalid_graph"),
                             (["inspect"] * 33, "exceeded_limit")):
            with self.subTest(refs=refs):
                host, catalog, proposal = fixture()
                proposal["nodes"][0]["step_refs"] = refs
                self.assertEqual(check_program(proposal, host, catalog).reason, reason)

    def test_premises_use_existing_authority_and_exact_typed_preconditions(self):
        cases = (
            (lambda h, c: c["registries"][h["registry_ref"]]["allowed_ops"].remove("ReadFact"), "unsupported_operation"),
            (lambda h, c: c["registries"][h["registry_ref"]]["allowed_ops"].remove("Plan"), "unsupported_operation"),
            (lambda h, c: c["snapshots"][h["context_ref"]]["record_refs"].clear(), "unknown_reference"),
            (lambda h, c: c["records"][0].update(epistemic_class="hypothesis"), "unsupported_assertion"),
            (lambda h, c: c["records"][0].update(freshness="stale"), "stale_context"),
            (lambda h, c: c["records"][0].update(conflicts=["atlas-v1"]), "conflict"),
            (lambda h, c: c["records"][0]["fields"]["repairable"].update(value=False), "unsupported_assertion"),
            (lambda h, c: h["workflows"]["repair-request"].update(audience="private"), "forbidden_dependency"),
            (lambda h, c: h["workflows"].clear(), "unknown_reference"),
        )
        for change, reason in cases:
            with self.subTest(reason=reason):
                host, catalog, proposal = fixture()
                change(host, catalog)
                self.assertEqual(check_program(proposal, host, catalog).reason, reason)
        for key, value, reason in (("unit", "day", "type_unit_mismatch"),
                                   ("temporal_scope", "2026-09", "scope_mismatch"),
                                   ("quantity_kind", "price", "type_unit_mismatch"),
                                   ("value_type", "decimal", "type_unit_mismatch")):
            host, catalog, proposal = fixture()
            expected = host["workflows"]["repair-request"]["steps"]["draft-request"]["preconditions"][0]["expected"]
            expected[key] = value
            if key == "value_type":
                expected["value"] = "24"
            self.assertEqual(check_program(proposal, host, catalog).reason, reason)

    def test_plan_cannot_be_promoted_into_a_fact(self):
        host, catalog, proposal = fixture()
        proposal["nodes"].append({"id": "n2", "op": "Assert", "proposition": "n1"})
        proposal["outputs"] = ["n2"]
        self.assertEqual(check_program(proposal, host, catalog).reason, "invalid_graph")
        proposal["nodes"][-1] = {"id": "n2", "op": "Plan", "workflow_ref": "repair-request", "step_refs": ["inspect"]}
        proposal["nodes"].append({"id": "n3", "op": "Conjoin", "speech_acts": ["n1", "n2"]})
        proposal["outputs"] = ["n3"]
        self.assertEqual(check_program(proposal, host, catalog).reason, "invalid_graph")
        proposal["nodes"] = proposal["nodes"][:1] + [{"id": "n2", "op": "ExplainDerivation", "speech_act": "n1"}]
        proposal["outputs"] = ["n2"]
        result = check_program(proposal, host, catalog)
        self.assertEqual(result.status, "accepted")
        self.assertIn("Proposed step 1", render_checked(result, host, catalog))

    def test_release_rechecks_workflow_registration_and_premises(self):
        for change in (lambda h, c: h["workflows"].clear(),
                       lambda h, c: h["workflows"]["repair-request"]["steps"]["inspect"].update(action_ref="send"),
                       lambda h, c: h["workflows"]["repair-request"].update(status="revoked"),
                       lambda h, c: c["records"][0]["fields"]["repairable"].update(value=False)):
            host, catalog, proposal = fixture()
            result = check_program(proposal, host, catalog)
            self.assertEqual(result.status, "accepted")
            change(host, catalog)
            with self.assertRaises(ValueError):
                render_checked(result, host, catalog)


if __name__ == "__main__":
    unittest.main()
