"""No-model counterfactual mutation, target, separation and byte-integrity audit."""

import copy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import unittest

from language_growth.kernel import check_program
from language_growth.semantic_axis_fixtures import (
    FIELDS, LEVELS, OUTPUT_DIR, SPLITS, TASKS, canonical, generate_split,
    manifest, rows, sha,
)


def differences(left, right, path=()):
    if type(left) is not type(right):
        return [path]
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return [path]
        return [p for key in left for p in differences(left[key], right[key], path + (key,))]
    if isinstance(left, list):
        if len(left) != len(right):
            return [path]
        return [p for i in range(len(left)) for p in differences(left[i], right[i], path + (i,))]
    return [] if left == right else [path]


def all_keys(value):
    if isinstance(value, dict):
        return set(value).union(*(all_keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(all_keys(v) for v in value))
    return set()


class SemanticAxisFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohorts = {}
        for split in SPLITS:
            request_bytes = (OUTPUT_DIR / f"{split}-requests.jsonl").read_bytes()
            scorer_bytes = (OUTPUT_DIR / f"{split}-scorer.jsonl").read_bytes()
            cls.cohorts[split] = ([json.loads(line) for line in request_bytes.splitlines()],
                                  [json.loads(line) for line in scorer_bytes.splitlines()])

    def test_regeneration_counts_and_manifest_bytes(self):
        for split, (requests, groups) in self.cohorts.items():
            generated_requests, generated_groups = generate_split(split)
            self.assertEqual(rows(requests), rows(generated_requests))
            self.assertEqual(rows(groups), rows(generated_groups))
            expected_manifest = canonical(manifest(split, requests, groups)) + b"\n"
            self.assertEqual(expected_manifest, (OUTPUT_DIR / f"{split}-manifest.json").read_bytes())
            self.assertEqual((len(requests), len(groups)), (1152, 64))
            self.assertEqual(len({r["ssdm"]["request_id"] for r in requests}), len(requests))
            self.assertEqual(len({g["world_id"] for g in groups}), 32)
            self.assertEqual(len({g["axis_group_id"] for g in groups}), 64)
            self.assertLess(sum(p.stat().st_size for p in OUTPUT_DIR.glob(f"{split}*")), 20_000_000)

    def test_one_typed_value_change_all_wording_options_and_structure_fixed(self):
        for requests, groups in self.cohorts.values():
            for group in groups:
                axis = group["axis"]
                self.assertIn(axis["field"], FIELDS)
                variants = group["variants"]
                self.assertEqual([v["level"] for v in variants], list(LEVELS))
                self.assertEqual(len({v["host"]["context_ref"] for v in variants}), 3)
                record_index = next(i for i, record in enumerate(variants[0]["catalog"]["records"])
                                    if record["record_ref"] == axis["record_ref"])
                expected_path = ("records", record_index, "fields", axis["field"], "value")
                for variant in variants:
                    catalog = variant["catalog"]
                    unbound = copy.deepcopy(catalog)
                    snapshot = unbound["snapshots"].pop(variant["host"]["context_ref"])
                    unbound["snapshots"]["content"] = snapshot
                    self.assertEqual(variant["host"]["context_ref"], f"synthetic-scalar:{sha(unbound)}")
                    self.assertEqual([t["kind"] for t in variant["tasks"]], list(TASKS))
                for previous, current in zip(variants, variants[1:]):
                    before, after = copy.deepcopy(previous["catalog"]), copy.deepcopy(current["catalog"])
                    before.pop("snapshots"); after.pop("snapshots")
                    self.assertEqual(differences(before, after), [expected_path])
                    for left_task, right_task in zip(previous["tasks"], current["tasks"]):
                        left = copy.deepcopy(requests[left_task["request_index"]])
                        right = copy.deepcopy(requests[right_task["request_index"]])
                        # Question IDs, instructions, option IDs/order and descriptions are byte-exact.
                        self.assertEqual(canonical(left["questions"]), canonical(right["questions"]))
                        self.assertEqual(list(left["questions"]["answer"]["criteria"]),
                                         list(right["questions"]["answer"]["criteria"]))
                        self.assertNotEqual(left["ssdm"]["request_id"], right["ssdm"]["request_id"])
                        for request in (left, right):
                            request["state"] = json.loads(request["state"])
                            request["state"].pop("context_ref")
                            request["ssdm"].pop("context_ref")
                            request["ssdm"].pop("request_id")
                        self.assertEqual(differences(left, right), [("state",) + expected_path])

    def test_independent_axes_share_base_world_and_cover_records(self):
        for _, groups in self.cohorts.values():
            for price, warranty in zip(groups[::2], groups[1::2]):
                self.assertEqual(price["world_id"], warranty["world_id"])
                self.assertEqual((price["axis"]["field"], warranty["axis"]["field"]), FIELDS)
                self.assertEqual(price["axis"]["record_ref"], warranty["axis"]["record_ref"])
                self.assertEqual(price["axis"]["comparator_record_ref"], warranty["axis"]["comparator_record_ref"])
                masked = []
                for group in (price, warranty):
                    catalog = copy.deepcopy(group["variants"][0]["catalog"])
                    catalog.pop("snapshots")
                    for record in catalog["records"]:
                        if record["record_ref"] == group["axis"]["record_ref"]:
                            for field in FIELDS:
                                record["fields"][field]["value"] = "masked"
                    masked.append(catalog)
                self.assertEqual(masked[0], masked[1])
            self.assertEqual({g["axis"]["record_ref"] for g in groups}, {f"record-{i}" for i in range(1, 5)})

    def test_kernel_valid_programs_targets_and_reverse_relations(self):
        for requests, groups in self.cohorts.values():
            indices = []
            for group in groups:
                scalar = []
                for variant in group["variants"]:
                    axis = group["axis"]
                    field = axis["field"]
                    records = {r["record_ref"]: r for r in variant["catalog"]["records"]}
                    a = Decimal(str(records[axis["record_ref"]]["fields"][field]["value"]))
                    b = Decimal(str(records[axis["comparator_record_ref"]]["fields"][field]["value"]))
                    self.assertEqual("equal" if a == b else "low" if a < b else "high", variant["level"])
                    scalar.append(a)
                    for task in variant["tasks"]:
                        indices.append(task["request_index"])
                        checked = check_program(task["program"], variant["host"], variant["catalog"])
                        self.assertEqual(checked.status, "accepted")
                        self.assertEqual(checked.outputs[0].to_dict(), task["checked_output"])
                        output = checked.outputs[0]
                        if task["kind"] == "explanation":
                            self.assertEqual(output.kind, "explain")
                            output = output.content
                        self.assertEqual(output.kind, "assert")
                        request = requests[task["request_index"]]
                        answer = request["questions"]["answer"]["criteria"][task["target_option_id"]]
                        if "compare" in task["kind"]:
                            content = output.content
                            left, right = content.left.object, content.right.object
                            relation = "equal" if left == right else "less_than" if left < right else "greater_than"
                            self.assertEqual((task["answer_value"], answer, content.relation), (relation,) * 3)
                            wrong = copy.deepcopy(task["program"])
                            wrong["nodes"][2]["relation"] = "greater_than" if relation != "greater_than" else "less_than"
                            self.assertEqual(check_program(wrong, variant["host"], variant["catalog"]).reason,
                                             "unsupported_assertion")
                        else:
                            fact = output.content
                            self.assertEqual(Decimal(str(fact.object)), Decimal(str(task["answer_value"])))
                            fact_text = f"{fact.subject}: {fact.field} = {task['answer_value']} {fact.unit} in 2026-10"
                            expected = (f"Read {fact.record_ref}.{fact.field} from accepted synthetic-premises, assert {fact_text}, then explain that derivation."
                                        if task["kind"] == "explanation" else fact_text)
                            self.assertEqual(answer, expected)
                    forward, reverse = variant["tasks"][1:3]
                    inverse = {"less_than": "greater_than", "greater_than": "less_than", "equal": "equal"}
                    self.assertEqual(reverse["answer_value"], inverse[forward["answer_value"]])
                self.assertTrue(scalar[0] < scalar[1] < scalar[2])
            self.assertEqual(sorted(indices), list(range(len(requests))))

    def test_responsiveness_and_unrelated_field_invariance(self):
        for _, groups in self.cohorts.values():
            for group in groups:
                for task_index in range(4):
                    tasks = [v["tasks"][task_index] for v in group["variants"]]
                    self.assertEqual(len({t["target_option_id"] for t in tasks}), 3)
                    self.assertEqual(len({canonical(t["checked_output"]) for t in tasks}), 3)
                for task_index in (4, 5):
                    tasks = [v["tasks"][task_index] for v in group["variants"]]
                    self.assertEqual(len({t["target_option_id"] for t in tasks}), 1)
                    self.assertEqual(len({canonical(t["checked_output"]) for t in tasks}), 1)

    def test_model_requests_exclude_scorer_metadata_and_authored_prefixes(self):
        forbidden = {"axis", "axis_group_id", "world_id", "split", "level", "target_option_id", "targets",
                     "answer_value", "evaluator_only", "program", "checked_output", "label_status", "seed"}
        for requests, _ in self.cohorts.values():
            for request in requests:
                self.assertEqual(set(request), {"model", "state", "questions", "ssdm"})
                self.assertFalse(all_keys(request) & forbidden)
                state = json.loads(request["state"])
                self.assertFalse(all_keys(state) & forbidden)
                self.assertEqual(state["accepted_partial_program"], [])
                self.assertEqual(state["proposed_slot_choices"], [])
                self.assertEqual(len(state["records"]), 4)

    def test_split_disjointness_and_receipt_hashes(self):
        def identities(groups):
            return ({g["world_id"] for g in groups},
                    {g["axis_group_id"] for g in groups},
                    {r["entity_ref"] for g in groups for r in g["variants"][0]["catalog"]["records"]},
                    {sha(v["catalog"]) for g in groups for v in g["variants"]})
        left, right = [identities(self.cohorts[s][1]) for s in SPLITS]
        for a, b in zip(left, right):
            self.assertFalse(a & b)
        for field in FIELDS:
            values = [{Decimal(str(r["fields"][field]["value"])) for g in self.cohorts[s][1]
                       for v in g["variants"] for r in v["catalog"]["records"]} for s in SPLITS]
            self.assertFalse(values[0] & values[1])
        receipt = json.loads((OUTPUT_DIR / "verification.json").read_bytes())
        root = Path(__file__).resolve().parents[2]
        for entry in receipt["artifacts"]:
            content = (root / entry["path"]).read_bytes()
            self.assertEqual(len(content), entry["bytes"])
            self.assertEqual(hashlib.sha256(content).hexdigest(), entry["sha256"])


if __name__ == "__main__":
    unittest.main()
