"""Direct decision comparison preserves task/evidence and excludes labels."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from language_growth.decision_view import direct_decision_request
from language_growth.parent_bridge import BridgeError, decision_request

PACK = Path(__file__).resolve().parents[2] / "fixtures/language_growth"


class DecisionViewTests(unittest.TestCase):
    def setUp(self):
        self.case = json.loads((PACK / "cases/01_supported_read.json").read_text(encoding="utf-8"))
        self.catalog = json.loads((PACK / "catalog.json").read_text(encoding="utf-8"))

    def build(self, case=None, **kwargs):
        return direct_decision_request(case or self.case, self.catalog,
                                       model="synthetic", parent_revision="test:v1", **kwargs)

    def test_gold_and_authored_program_do_not_change_request(self):
        changed = deepcopy(self.case)
        changed["expected"] = {"secret": "DO_NOT_SERIALIZE"}
        changed["proposal"] = {"secret": "DO_NOT_SERIALIZE"}
        changed["decision_only"]["target_option_id"] = "DO_NOT_SERIALIZE"
        self.assertEqual(self.build(), self.build(changed))
        self.assertNotIn("DO_NOT_SERIALIZE", json.dumps(self.build(changed)))

    def test_direct_and_initial_planner_request_have_identical_state(self):
        direct = self.build()
        planner, digest = decision_request(
            task_input=self.case["question"], host=self.case["host"], catalog=self.catalog,
            model="synthetic", parent_revision="test:v1", request_id="planner",
            question_id="slot-operation", question="Select the operation.",
            options={"op001": "Assert one fact", "op002": "Compare two facts"},
        )
        self.assertEqual(direct["request"]["state"], planner["state"])
        self.assertEqual(direct["state_sha256"], digest)

    def test_relabel_changes_ids_not_descriptions_or_evidence(self):
        original, renamed = self.build(), self.build(relabel=True)
        self.assertEqual(original["state_sha256"], renamed["state_sha256"])
        qid = self.case["decision_only"]["question_id"]
        before = original["request"]["questions"][qid]["criteria"]
        after = renamed["request"]["questions"][qid]["criteria"]
        for old, new in self.case["decision_only"]["relabel_control"]["id_map"].items():
            self.assertEqual(before[old], after[new])
        self.assertNotEqual(original["request_fingerprint"], renamed["request_fingerprint"])

    def test_mismatched_evidence_and_duplicate_options_are_rejected(self):
        changed = deepcopy(self.case)
        changed["decision_only"]["evidence_context_ref"] = "another-context"
        with self.assertRaises(BridgeError):
            self.build(changed)
        changed = deepcopy(self.case)
        changed["decision_only"]["options"].append(changed["decision_only"]["options"][0])
        with self.assertRaises(BridgeError):
            self.build(changed)


if __name__ == "__main__":
    unittest.main()
