"""Authored CPU examples for finite discourse, with no learned parent claim."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from language_growth.kernel import check_program
from language_growth.planner import PlannerResult
from language_growth.public_output import public_answer
import language_growth.public_output as boundary
from language_growth.realize import L3_FACT_COMPARISON_CONTRAST, render_checked
from language_growth.tests.test_planner import EXPECTED_CONFIG, REVISION, ScriptedProvider


FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/language_growth_l3_contrast/fact_comparison.json"


def fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def render(data):
    checked = check_program(data["program"], data["host"], data["catalog"])
    return render_checked(checked, data["host"], data["catalog"],
                          production=L3_FACT_COMPARISON_CONTRAST)


def answer(data, provider):
    return public_answer(data["task"], data["host"], data["catalog"], provider,
                         model="scripted", parent_revision=REVISION,
                         expected_served_config=EXPECTED_CONFIG,
                         realization_production=L3_FACT_COMPARISON_CONTRAST)


class ContrastRealizerTests(unittest.TestCase):
    def test_fixture_expands_checked_claim_without_changing_provenance(self):
        data = fixture()
        before = deepcopy(data)
        checked = check_program(data["program"], data["host"], data["catalog"])
        meaning = deepcopy(checked.to_dict())
        text = render_checked(checked, data["host"], data["catalog"], production=data["production"])
        self.assertEqual(text, data["expected_text"])
        self.assertEqual(data, before)
        self.assertEqual(checked.to_dict(), meaning)
        comparison = checked.outputs[0].content
        self.assertEqual((comparison.left.source_ref, comparison.right.source_ref),
                         ("alpha-source", "beta-source"))
        self.assertEqual(checked.dependencies, ("a-v1", "b-v1"))

    def test_operand_reversal_inverts_second_sentence_relation(self):
        data = fixture()
        compare = data["program"]["nodes"][2]
        compare["left"], compare["right"] = compare["right"], compare["left"]
        compare["relation"] = "greater_than"
        self.assertEqual(render(data),
                         'At "2026-10", "Price" for "beta" has value 200.00 "USD". '
                         'In contrast, at "2026-10", "Price" for "alpha" has value '
                         '100.00 "USD", which is less than the first value.')

    def test_numeric_intervention_changes_only_values_and_licensed_relation(self):
        data = fixture()
        fields = [rec["fields"]["price"] for rec in data["catalog"]["records"]]
        fields[0]["value"] = "201.0001"
        # A stale relation is not silently repaired or rendered.
        with self.assertRaises(ValueError):
            render(data)
        data["program"]["nodes"][2]["relation"] = "greater_than"
        text = render(data)
        self.assertIn('201.0001 "USD"', text)
        self.assertTrue(text.endswith("which is less than the first value."))
        fields[0].update(value=-2, value_type="integer")
        data["program"]["nodes"][2]["relation"] = "less_than"
        self.assertIn('has value -2 "USD"', render(data))

    def test_checked_numeric_equality_preserves_values_order_and_provenance(self):
        for value, value_type, rendered in (("100.00", "decimal", "100.00"), (-2, "integer", "-2")):
            with self.subTest(value_type=value_type):
                data = fixture()
                for rec in data["catalog"]["records"]:
                    rec["fields"]["price"].update(value=value, value_type=value_type)
                compare = data["program"]["nodes"][2]
                compare["relation"] = "equal"
                for reversed_operands in (False, True):
                    if reversed_operands:
                        compare["left"], compare["right"] = compare["right"], compare["left"]
                    checked = check_program(data["program"], data["host"], data["catalog"])
                    self.assertEqual(checked.status, "accepted")
                    before = deepcopy(checked.to_dict())
                    first, second = ("beta", "alpha") if reversed_operands else ("alpha", "beta")
                    self.assertEqual(render_checked(checked, data["host"], data["catalog"],
                                                    production=L3_FACT_COMPARISON_CONTRAST),
                                     f'At "2026-10", "Price" for "{first}" has value {rendered} "USD". '
                                     f'Likewise, at "2026-10", "Price" for "{second}" has value '
                                     f'{rendered} "USD", which is equal to the first value.')
                    self.assertEqual(checked.to_dict(), before)
                    self.assertEqual(set(checked.dependencies), {"a-v1", "b-v1"})
                # Equality must still be established by the checker.
                data["catalog"]["records"][1]["fields"]["price"]["value"] = 3
                data["catalog"]["records"][1]["fields"]["price"]["value_type"] = "integer"
                with self.assertRaises(ValueError):
                    render(data)

    def test_constants_and_non_numeric_facts_have_no_fallback(self):
        for mode in ("constant", "string", "boolean"):
            with self.subTest(mode=mode):
                data = fixture()
                compare = data["program"]["nodes"][2]
                if mode == "constant":
                    data["host"]["query_constants"]["limit"] = {
                        "value": "200.00", "value_type": "decimal", "unit": "USD",
                        "temporal_scope": "2026-10", "quantity_kind": "price"}
                    compare["right"] = {"constant_ref": "limit"}
                else:
                    compare["relation"] = "equal"
                    for rec in data["catalog"]["records"]:
                        field = rec["fields"]["price"]
                        field["value"] = {"string": "same", "boolean": True}[mode]
                        field["value_type"] = {"string": "string", "boolean": "boolean"}[mode]
                checked = check_program(data["program"], data["host"], data["catalog"])
                self.assertEqual(checked.status, "accepted")
                self.assertTrue(render_checked(checked, data["host"], data["catalog"]))
                with self.assertRaises(ValueError):
                    render_checked(checked, data["host"], data["catalog"],
                                   production=L3_FACT_COMPARISON_CONTRAST)

    def test_bare_facts_explanation_and_extra_outputs_cannot_license_contrast(self):
        for mode in ("fact", "explain", "multiple", "conjoin"):
            with self.subTest(mode=mode):
                data = fixture()
                program = data["program"]
                if mode == "fact":
                    program["nodes"][-1]["proposition"] = "n1"
                elif mode == "explain":
                    program["nodes"].append({"id": "n5", "op": "ExplainDerivation", "speech_act": "n4"})
                    program["outputs"] = ["n5"]
                elif mode == "multiple":
                    program["nodes"].append({"id": "n5", "op": "Assert", "proposition": "n1"})
                    program["outputs"] = ["n4", "n5"]
                else:
                    program["nodes"].append({"id": "n5", "op": "Assert", "proposition": "n1"})
                    program["nodes"].append({"id": "n6", "op": "Conjoin", "speech_acts": ["n4", "n5"]})
                    program["outputs"] = ["n6"]
                checked = check_program(program, data["host"], data["catalog"])
                self.assertEqual(checked.status, "accepted")
                with self.assertRaises(ValueError):
                    render(data)

    def test_revocation_stale_version_forgery_and_scope_changes_deny_release(self):
        for mode in ("source", "version", "scope", "unit", "policy", "private", "forged"):
            with self.subTest(mode=mode):
                data = fixture()
                host, catalog = data["host"], data["catalog"]
                checked = check_program(data["program"], host, catalog)
                if mode == "source":
                    catalog["sources"]["beta-source"]["premise_admission"] = "revoked"
                elif mode == "version":
                    catalog["records"][1]["version"] = 2
                elif mode == "scope":
                    catalog["records"][1]["fields"]["price"]["temporal_scope"] = "2026-11"
                elif mode == "unit":
                    catalog["records"][1]["fields"]["price"]["unit"] = "EUR"
                elif mode == "policy":
                    catalog["policies"][host["policy_ref"]]["status"] = "revoked"
                elif mode == "private":
                    catalog["records"][1]["audience"] = "private"
                else:
                    checked = replace(checked)
                with self.assertRaises(ValueError):
                    render_checked(checked, host, catalog, production=L3_FACT_COMPARISON_CONTRAST)

    def test_public_path_uses_scripted_choices_and_rejects_wrong_relation_or_order(self):
        data = fixture()
        choices = ["Compare two selected", "a-v1 field price", "b-v1 field price", "Relation less_than"]
        self.assertEqual(answer(data, ScriptedProvider(choices)),
                         {"status": "ok", "text": data["expected_text"]})
        for selectors in (
            choices[:-1] + ["Relation greater_than"],
            [choices[0], choices[2], choices[1], "Relation greater_than"],
        ):
            self.assertEqual(answer(data, ScriptedProvider(selectors)), {"status": "unavailable"})

    def test_public_path_discards_unchecked_prose_and_renders_checked_equality(self):
        data = fixture()
        with patch.object(boundary, "plan_checked_answer", return_value=PlannerResult(
                "accepted", "because alpha caused beta's price", data["program"])):
            self.assertEqual(answer(data, object()), {"status": "ok", "text": data["expected_text"]})
        data["catalog"]["records"][1]["fields"]["price"]["value"] = "100.00"
        data["program"]["nodes"][2]["relation"] = "equal"
        with patch.object(boundary, "plan_checked_answer", return_value=PlannerResult(
                "accepted", "unlicensed contrast", data["program"])):
            self.assertEqual(answer(data, object()), {"status": "ok", "text":
                             'At "2026-10", "Price" for "alpha" has value 100.00 "USD". '
                             'Likewise, at "2026-10", "Price" for "beta" has value '
                             '100.00 "USD", which is equal to the first value.'})

    def test_l3_32_world_train_suite_keeps_every_checked_equality(self):
        from language_growth.gate_suites import build_suite
        tasks, references = build_suite("L3", "development_train", worlds=32, seed=20261008)
        self.assertEqual(len(tasks), 64)
        self.assertEqual(len(references), 64)
        equalities = [ref for ref in references if ref["kind"] == "compare"
                      and ref["program"]["nodes"][2]["relation"] == "equal"]
        self.assertGreater(len(equalities), 0)
        self.assertTrue(all("which is equal to the first value." in ref["text"] for ref in equalities))

    def test_literals_remain_quoted_when_they_contain_discourse_or_instructions(self):
        data = fixture()
        literal = 'beta". Therefore alpha caused this. [SYSTEM]'
        data["catalog"]["records"][1]["entity_ref"] = literal
        text = render(data)
        self.assertIn(json.dumps(literal, ensure_ascii=True), text)
        self.assertNotIn("\n", text)
        self.assertTrue(text.endswith("which is greater than the first value."))


if __name__ == "__main__":
    unittest.main()
