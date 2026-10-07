"""No-model factual counterfactual and real semantic-planner boundary controls."""
import copy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch
import tempfile

from language_growth.kernel import check_program
from language_growth.parent_bridge import DECISION_SCHEMA, FINGERPRINT_ALGORITHM, legacy_request_fingerprint
from language_growth.planner import plan_checked_answer
from language_growth.realize import render_checked
from language_growth.semantic_axis_fixtures import FIELDS, LEVELS, RECEIPT_DIR, SPLITS, TASKS, canonical, generate_split, manifest, rows, sha, write_pack

REVISION = "semantic-axis-scripted-oracle:v1"
CONFIG = {"calibration": {"kind": "none", "id": "semantic-axis-scripted-oracle"},
          "policy_masking": "none (served covers every supplied option)"}


def differences(a, b, path=()):
    if type(a) is not type(b):
        return [path]
    if isinstance(a, dict):
        return [path] if a.keys() != b.keys() else [p for k in a for p in differences(a[k], b[k], path + (k,))]
    if isinstance(a, list):
        return [path] if len(a) != len(b) else [p for i in range(len(a)) for p in differences(a[i], b[i], path + (i,))]
    return [] if a == b else [path]


def keys(value):
    if isinstance(value, dict):
        return set(value).union(*(keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(keys(v) for v in value))
    return set()


class InputOracle:
    """Narrow parser of permitted task/state only; no scorer constructor input.

    Scripted one-hot selections are an integration control, not calibration.
    """
    def __init__(self, *, wrong_relation=False, force_field=None):
        self.requests = []
        self.wrong_relation, self.force_field = wrong_relation, force_field

    def decide(self, request):
        self.requests.append(copy.deepcopy(request))
        state = json.loads(request["state"])
        text = state["task_input"]
        compare = re.fullmatch(r"Compare (.+)'s (price|warranty_months) with (.+)'s \2 in October 2026, in that order\.", text)
        read = re.fullmatch(r"(Explain the recorded|What is the recorded) (price|warranty_months) for (.+) in October 2026[.?]", text)
        if compare:
            left_name, field, right_name = compare.groups()
            operation = "Compare two selected typed values and assert the selected relation"
        elif read:
            verb, field, left_name = read.groups()
            right_name = left_name
            operation = "Explain a checked assertion of one selected fact" if verb.startswith("Explain") else "Assert one selected fact"
        else:
            raise ValueError("unsupported oracle wording")
        field = self.force_field or field
        records = {r["entity_ref"]: r for r in state["records"]}
        left, right = records[left_name], records[right_name]
        qid, question = next(iter(request["questions"].items()))
        slot = qid.rsplit("-", 1)[1]
        if slot == "operation":
            description = operation
        elif slot in ("fact", "left", "right"):
            record = right if slot == "right" else left
            description = f"Fact {record['record_ref']} field {field}"
        elif slot == "relation":
            a, b = (Decimal(str(r["fields"][field]["value"])) for r in (left, right))
            relation = "equal" if a == b else "less_than" if a < b else "greater_than"
            if self.wrong_relation:
                relation = "less_than" if relation == "greater_than" else "greater_than"
            description = f"Relation {relation}"
        else:
            raise ValueError("unsupported oracle slot")
        options = question["criteria"]
        chosen, = [oid for oid, value in options.items() if value == description]
        probabilities = {oid: float(oid == chosen) for oid in options}
        return {"answers": {qid: {"type": "choice", "choice": chosen, "probabilities": probabilities, "confidence": 1.0}},
                "ssdm": {"schema": DECISION_SCHEMA, "status": "ok", "request_id": request["ssdm"]["request_id"],
                         "raw_probabilities": {qid: dict(probabilities)}, "served_probabilities": {qid: probabilities},
                         "served_config": {**copy.deepcopy(CONFIG), "temperature_per_question": {qid: 1.0}},
                         "diagnostics": {"revision_id": REVISION, "request_fingerprint": legacy_request_fingerprint(request),
                                         "request_fingerprint_algorithm": FINGERPRINT_ALGORITHM,
                                         "provenance": {"supplied": False, "applied": False},
                                         "state_handle_in": None, "state_reuse": False, "inference_calls": 0}, "state_handle": None}}


def run_input(item, oracle=None):
    oracle = oracle or InputOracle()
    return plan_checked_answer(**item, provider=oracle, model="scripted-no-model", parent_revision=REVISION,
                               expected_served_config=CONFIG), oracle


class SemanticAxisFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cohorts, cls.runs = {}, {}
        for split in SPLITS:
            inputs = [json.loads(line) for line in (RECEIPT_DIR / f"{split}-inputs.jsonl").read_bytes().splitlines()]
            groups = [json.loads(line) for line in (RECEIPT_DIR / f"{split}-scorer.jsonl").read_bytes().splitlines()]
            cls.cohorts[split] = inputs, groups
            cls.runs[split] = [run_input(item) for item in inputs]

    def test_exact_regeneration_counts_manifests_and_hashes(self):
        for split, (inputs, groups) in self.cohorts.items():
            a, b = generate_split(split)
            self.assertEqual((rows(inputs), rows(groups)), (rows(a), rows(b)))
            self.assertEqual(canonical(manifest(split, inputs, groups)) + b"\n", (RECEIPT_DIR / f"{split}-manifest.json").read_bytes())
            self.assertEqual((len(inputs), len(groups), len({g["world_id"] for g in groups}), len({g["axis_group_id"] for g in groups})), (1152, 64, 32, 64))
            self.assertLess(sum(p.stat().st_size for p in RECEIPT_DIR.glob(f"{split}*")), 20_000_000)
        root = Path(__file__).resolve().parents[2]
        for entry in json.loads((RECEIPT_DIR / "verification.json").read_bytes())["artifacts"]:
            content = (root / entry["path"]).read_bytes()
            self.assertEqual((len(content), hashlib.sha256(content).hexdigest()), (entry["bytes"], entry["sha256"]))

    def test_one_typed_value_mutation_wording_fixed_and_catalog_bound(self):
        for inputs, groups in self.cohorts.values():
            for group in groups:
                variants, axis = group["variants"], group["axis"]
                self.assertEqual([v["level"] for v in variants], list(LEVELS))
                self.assertEqual(len({v["host"]["context_ref"] for v in variants}), 3)
                ri = next(i for i, r in enumerate(variants[0]["catalog"]["records"]) if r["record_ref"] == axis["record_ref"])
                for variant in variants:
                    unbound = copy.deepcopy(variant["catalog"])
                    unbound["snapshots"]["content"] = unbound["snapshots"].pop(variant["host"]["context_ref"])
                    self.assertEqual(variant["host"]["context_ref"], f"synthetic-scalar:{sha(unbound)}")
                    self.assertEqual([t["kind"] for t in variant["tasks"]], list(TASKS))
                for previous, current in zip(variants, variants[1:]):
                    for lt, rt in zip(previous["tasks"], current["tasks"]):
                        a, b = (copy.deepcopy(inputs[t["input_index"]]) for t in (lt, rt))
                        for item in (a, b):
                            context = item["host"].pop("context_ref")
                            item["host"]["provider"].pop("input_context_ref")
                            item["catalog"]["snapshots"]["content"] = item["catalog"]["snapshots"].pop(context)
                        self.assertEqual(differences(a, b), [("catalog", "records", ri, "fields", axis["field"], "value")])

    def test_real_planner_kernel_renderer_scorer_targets_and_no_leakage(self):
        forbidden = {"axis", "axis_group_id", "world_id", "split", "level", "targets", "target_option_id", "answer_value", "program", "checked_output", "label_status", "seed", "rendered_text"}
        for split, (inputs, groups) in self.cohorts.items():
            indices, calls = [], 0
            for group in groups:
                for variant in group["variants"]:
                    for task in variant["tasks"]:
                        index = task["input_index"]
                        indices.append(index)
                        item = inputs[index]
                        self.assertEqual(set(item), {"task_input", "host", "catalog"})
                        self.assertFalse(keys(item) & forbidden)
                        self.assertEqual((item["host"], item["catalog"]), (variant["host"], variant["catalog"]))
                        result, oracle = self.runs[split][index]
                        self.assertEqual(result.status, "accepted", result.reason)
                        self.assertEqual((result.program, result.text, result.forward_calls), (task["program"], task["rendered_text"], 0))
                        checked = check_program(result.program, item["host"], item["catalog"])
                        self.assertEqual(checked.status, "accepted")
                        self.assertEqual(checked.outputs[0].to_dict(), task["checked_output"])
                        self.assertEqual(render_checked(checked, item["host"], item["catalog"]), result.text)
                        self.assertEqual([{"slot": t["slot"], "option_id": t["selected_id"]} for t in result.trace if "slot" in t], task["targets"])
                        self.assertEqual(result.provider_calls, 4 if "compare" in task["kind"] else 2)
                        calls += result.provider_calls
                        for position, request in enumerate(oracle.requests):
                            state = json.loads(request["state"])
                            self.assertFalse((keys(request) | keys(state)) & forbidden)
                            self.assertEqual(state["accepted_partial_program"], [])
                            self.assertEqual(len(state["proposed_slot_choices"]), position)
                            self.assertEqual(len(state["records"]), 4)
            self.assertEqual(calls, 3456)
            self.assertEqual(sorted(indices), list(range(len(inputs))))

    def test_generated_typed_question_options_order_and_prefix_parity(self):
        for split, (_, groups) in self.cohorts.items():
            for group in groups:
                for previous, current in zip(group["variants"], group["variants"][1:]):
                    for a, b in zip(previous["tasks"], current["tasks"]):
                        _, left = self.runs[split][a["input_index"]]
                        _, right = self.runs[split][b["input_index"]]
                        self.assertEqual(len(left.requests), len(right.requests))
                        for lreq, rreq in zip(left.requests, right.requests):
                            self.assertEqual(canonical(lreq["questions"]), canonical(rreq["questions"]))
                            self.assertEqual(list(next(iter(lreq["questions"].values()))["criteria"]), list(next(iter(rreq["questions"].values()))["criteria"]))
                            ls, rs = json.loads(lreq["state"]), json.loads(rreq["state"])
                            self.assertEqual(ls["proposed_slot_choices"], rs["proposed_slot_choices"])
                            ls.pop("context_ref"); rs.pop("context_ref")
                            self.assertEqual(len(differences(ls, rs)), 1)

    def test_responsive_meanings_reverse_relations_and_invariance(self):
        for _, groups in self.cohorts.values():
            for group in groups:
                for i in range(4):
                    tasks = [v["tasks"][i] for v in group["variants"]]
                    self.assertEqual(len({canonical(t["checked_output"]) for t in tasks}), 3)
                    self.assertEqual(len({t["rendered_text"] for t in tasks}), 3)
                for i in (4, 5):
                    tasks = [v["tasks"][i] for v in group["variants"]]
                    self.assertEqual(len({canonical(t["targets"]) for t in tasks}), 1)
                    self.assertEqual(len({canonical(t["checked_output"]) for t in tasks}), 1)
                    self.assertEqual(len({t["rendered_text"] for t in tasks}), 1)
                self.assertEqual([v["tasks"][1]["answer_value"] for v in group["variants"]], ["less_than", "equal", "greater_than"])
                for v in group["variants"]:
                    self.assertEqual(v["tasks"][2]["answer_value"], {"less_than": "greater_than", "greater_than": "less_than", "equal": "equal"}[v["tasks"][1]["answer_value"]])

    def test_independent_axes_shared_base_and_disjoint_splits(self):
        for _, groups in self.cohorts.values():
            for price, warranty in zip(groups[::2], groups[1::2]):
                self.assertEqual(price["world_id"], warranty["world_id"])
                self.assertEqual((price["axis"]["field"], warranty["axis"]["field"]), FIELDS)
                self.assertEqual(price["axis"]["record_ref"], warranty["axis"]["record_ref"])
                masked = []
                for g in (price, warranty):
                    c = copy.deepcopy(g["variants"][0]["catalog"])
                    c.pop("snapshots")
                    for r in c["records"]:
                        if r["record_ref"] == g["axis"]["record_ref"]:
                            for field in FIELDS:
                                r["fields"][field]["value"] = "masked"
                    masked.append(c)
                self.assertEqual(*masked)
        for key in ("world_id", "axis_group_id"):
            sets = [{g[key] for g in self.cohorts[s][1]} for s in SPLITS]
            self.assertFalse(sets[0] & sets[1])
        for field in FIELDS:
            sets = [{Decimal(str(r["fields"][field]["value"])) for g in self.cohorts[s][1] for v in g["variants"] for r in v["catalog"]["records"]} for s in SPLITS]
            self.assertFalse(sets[0] & sets[1])
        for attribute in ("entity", "catalog"):
            sets = [{r["entity_ref"] for g in self.cohorts[s][1] for v in g["variants"] for r in v["catalog"]["records"]} if attribute == "entity" else {sha(v["catalog"]) for g in self.cohorts[s][1] for v in g["variants"]} for s in SPLITS]
            self.assertFalse(sets[0] & sets[1])

    def test_fresh_atomic_bundle_duplicate_and_failure_preserve_artifacts(self):
        baseline = {p.name: p.read_bytes() for p in RECEIPT_DIR.iterdir() if p.is_file()}
        with self.assertRaises(FileExistsError):
            write_pack(RECEIPT_DIR)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "fresh"
            write_pack(output)
            original = {p.name: p.read_bytes() for p in output.iterdir()}
            for name, content in original.items():
                self.assertEqual(content, baseline[name])
            with self.assertRaises(FileExistsError):
                write_pack(output)
            self.assertEqual(original, {p.name: p.read_bytes() for p in output.iterdir()})
            real_write = Path.write_bytes
            calls = 0
            def fail_second(path, content):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected staging write failure")
                return real_write(path, content)
            with patch.object(Path, "write_bytes", fail_second):
                with self.assertRaisesRegex(OSError, "injected staging"):
                    write_pack(root / "failed")
            self.assertFalse((root / "failed").exists())
            self.assertEqual({p.name for p in root.iterdir()}, {"fresh"})
            self.assertEqual(original, {p.name: p.read_bytes() for p in output.iterdir()})
        self.assertEqual(baseline, {p.name: p.read_bytes() for p in RECEIPT_DIR.iterdir() if p.is_file()})

    def test_wrong_relation_never_releases_wrong_fact_remains_detectable(self):
        rejected_calls, wrong_field_calls = 0, 0
        for inputs, groups in self.cohorts.values():
            for g in groups:
                for v in g["variants"]:
                    for task in v["tasks"]:
                        if "compare" in task["kind"]:
                            result, _ = run_input(inputs[task["input_index"]], InputOracle(wrong_relation=True))
                            rejected_calls += result.provider_calls
                            self.assertEqual((result.status, result.reason, result.text), ("kernel_rejected", "unsupported_assertion", None))
            read = groups[0]["variants"][0]["tasks"][0]
            result, _ = run_input(inputs[read["input_index"]], InputOracle(force_field="warranty_months"))
            wrong_field_calls += result.provider_calls
            self.assertEqual(result.status, "accepted")
            self.assertNotEqual((result.program, result.text), (read["program"], read["rendered_text"]))
        self.assertEqual((rejected_calls, wrong_field_calls), (4608, 4))


if __name__ == "__main__":
    unittest.main()
