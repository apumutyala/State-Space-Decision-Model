"""The frozen-output transform must not repair model-selected structure."""

from copy import deepcopy
import json
import unittest

from language_growth.derived_compare_ablation import derive_program
from language_growth.kernel import check_program
from language_growth.tests.test_kernel import ROOT, case


class AblationTransformTests(unittest.TestCase):
    def test_fixed_wrong_field_order_and_relation_only_transformation(self):
        item = case(14)  # A supported warranty answer to a price question.
        catalog = json.loads((ROOT / "catalog.json").read_text())
        source = deepcopy((item["proposal"], item["host"], catalog))
        program, host, enabled, count = derive_program(*source)
        self.assertEqual(count, 1)
        self.assertEqual(source, (item["proposal"], item["host"], catalog))
        expected = deepcopy(source[0])
        expected["schema"] = "ssdm.semantic_program.v0.2"
        expected["registry_ref"] = host["registry_ref"]
        expected["nodes"][2]["op"] = "DerivedCompare"
        del expected["nodes"][2]["relation"]
        self.assertEqual(program, expected)
        self.assertEqual(enabled["records"], catalog["records"])
        self.assertEqual(enabled["sources"], catalog["sources"])
        self.assertEqual(enabled["snapshots"], catalog["snapshots"])
        self.assertEqual(enabled["policies"], catalog["policies"])
        self.assertEqual(check_program(program, host, enabled).status, "accepted")
        self.assertEqual(program["nodes"][0]["field"], "warranty_months")

    def test_selected_noncomparison_operation_is_unchanged(self):
        item = case(1)
        catalog = json.loads((ROOT / "catalog.json").read_text())
        result = derive_program(item["proposal"], item["host"], catalog)
        self.assertEqual(result, (item["proposal"], item["host"], catalog, 0))


if __name__ == "__main__":
    unittest.main()
