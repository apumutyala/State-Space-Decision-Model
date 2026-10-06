"""Decision request isolation regressions for the fixture evaluator."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from fixtures.language_growth.validate import read_json
from language_growth.evaluate import decision_request_for_case


PACK = Path(__file__).resolve().parents[2] / "fixtures" / "language_growth"


class DecisionRequestIsolationTests(unittest.TestCase):
    def setUp(self):
        self.catalog = read_json(PACK / "catalog.json")
        self.case = read_json(PACK / "cases" / "01_supported_read.json")

    def test_forbidden_snapshot_entry_rejects_before_serialization(self):
        catalog = deepcopy(self.catalog)
        catalog["snapshots"]["catalog-public-r7"]["record_refs"].append("internal-discount-v1")
        atlas = next(record for record in catalog["records"] if record["record_ref"] == "atlas-v1")
        atlas["conflicts"].append("internal-discount-v1")
        with self.assertRaisesRegex(ValueError, "^provider precondition failed$"):
            decision_request_for_case(self.case, catalog)

    def test_private_catalog_changes_do_not_change_public_request(self):
        baseline = decision_request_for_case(self.case, self.catalog)
        changed = deepcopy(self.catalog)
        private = next(record for record in changed["records"] if record["record_ref"] == "internal-discount-v1")
        private["fields"]["discount"]["value"] = "0.99"
        private["source_ref"] = "secret-source"
        atlas = next(record for record in changed["records"] if record["record_ref"] == "atlas-v1")
        atlas["conflicts"] = ["internal-discount-v1"]
        request = decision_request_for_case(self.case, changed)
        self.assertEqual(request, baseline)
        text = json.dumps(request)
        self.assertNotIn("internal-discount-v1", text)
        self.assertNotIn("secret-source", text)
        self.assertNotIn("target_option_id", text)
        self.assertEqual(list(request["questions"]["LG01-D0"]["criteria"]),
                         [option["id"] for option in self.case["decision_only"]["options"]])

    def test_host_obligation_cannot_name_hidden_record(self):
        case = deepcopy(self.case)
        case["host"]["unmet_obligations"] = [{
            "obligation_id": "hidden",
            "category": "missing_required_record",
            "allowed_refs": ["internal-discount-v1"],
        }]
        with self.assertRaisesRegex(ValueError, "^provider precondition failed$"):
            decision_request_for_case(case, self.catalog)

    def test_caller_literal_does_not_become_a_private_membership_oracle(self):
        case = deepcopy(self.case)
        case["host"]["query_constants"]["caller-literal"] = {
            "value": "internal-discount-v1", "value_type": "string",
            "unit": None, "temporal_scope": "2026-10", "quantity_kind": "identifier",
        }
        baseline = decision_request_for_case(case, self.catalog)
        changed = deepcopy(self.catalog)
        private = next(record for record in changed["records"] if record["record_ref"] == "internal-discount-v1")
        private["record_ref"] = "another-private-id"
        for snapshot in changed["snapshots"].values():
            snapshot["record_refs"] = ["another-private-id" if ref == "internal-discount-v1" else ref for ref in snapshot["record_refs"]]
        self.assertEqual(decision_request_for_case(case, changed), baseline)

    def test_query_constant_cannot_name_hidden_record(self):
        case = deepcopy(self.case)
        case["host"]["query_constants"]["leaky-constant"] = {"record_ref": "internal-discount-v1"}
        with self.assertRaisesRegex(ValueError, "^provider precondition failed$"):
            decision_request_for_case(case, self.catalog)

    def test_arbitrary_query_and_obligation_metadata_is_rejected(self):
        case = deepcopy(self.case)
        case["host"]["query_constants"]["unregistered-shape"] = {"record_ref": "atlas-v1"}
        with self.assertRaisesRegex(ValueError, "^provider precondition failed$"):
            decision_request_for_case(case, self.catalog)
        case = deepcopy(self.case)
        case["host"]["unmet_obligations"] = [{
            "obligation_id": "x",
            "category": "missing_required_record",
            "allowed_refs": ["atlas-v1"],
            "private_hint": "internal-discount-v1",
        }]
        with self.assertRaisesRegex(ValueError, "^provider precondition failed$"):
            decision_request_for_case(case, self.catalog)


if __name__ == "__main__":
    unittest.main()
