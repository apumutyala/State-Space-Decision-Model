"""CPU infrastructure controls only; no model or learned L1 result."""

import copy
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from language_growth.free_running_dev import run_development_tasks
from language_growth.multirecord_eval import _adapt_task, load_multirecord_evaluation
from language_growth.parent_bridge import legacy_request_fingerprint
from language_growth.training_records import export_training_records
from language_growth.tests.test_free_running_dev import CONFIG, REVISION

FIXTURES = Path(__file__).resolve().parents[2] / "receipts/2026-10-07-L1-multirecord-data"
SPLITS = ("l1_development_train", "l1_development_validation")


class EvidenceOracle:
    """Scripted control deriving each choice solely from current public input."""

    def __init__(self, mode="correct"):
        self.mode, self.requests = mode, []

    def decide(self, request):
        self.requests.append(copy.deepcopy(request))
        if self.mode == "error":
            raise RuntimeError("scripted provider failure")
        state = json.loads(request["state"])
        qid, question = next(iter(request["questions"].items()))
        slot, task = qid.rsplit("-", 1)[-1], state["task_input"]
        records = state["records"]
        field = "warranty_months" if "warranty months" in task else "price"
        if slot == "operation":
            kind = "compare" if ("compare" in task.lower() or "relation" in task) else (
                "explain" if "explanation" in task or "explain" in task.lower() else "read")
            phrase = {"compare": "Compare two selected typed values",
                      "explain": "Explain a checked assertion", "read": "Assert one selected fact"}[kind]
        elif slot in ("fact", "left", "right"):
            named = sorted((r for r in records if r["entity_ref"] in task),
                           key=lambda r: task.index(r["entity_ref"]))
            index = int(slot == "right")
            if self.mode == "reverse" and slot in ("left", "right"):
                index = 1 - index
            if self.mode == "wrong_true":
                field = "price" if field == "warranty_months" else "warranty_months"
            phrase = f"Fact {named[index]['record_ref']} field {field}"
        else:
            prior = {c["slot"]: c["description"] for c in state["proposed_slot_choices"]}
            by_ref = {r["record_ref"]: r for r in records}
            values = [Decimal(str(by_ref[prior[s].split()[1]]["fields"][field]["value"]))
                      for s in ("left", "right")]
            relation = "equal" if values[0] == values[1] else (
                "greater_than" if values[0] > values[1] else "less_than")
            if self.mode == "false_relation":
                relation = "less_than" if relation == "greater_than" else "greater_than"
            phrase = f"Relation {relation}"
        options = question["criteria"]
        selected = next(oid for oid, desc in options.items() if desc.startswith(phrase))
        probs = {oid: float(oid == selected) for oid in options}
        return {
            "answers": {qid: {"type": "choice", "choice": selected,
                              "confidence": 1.0, "probabilities": dict(probs)}},
            "ssdm": {"schema": "ssdm.decide.v0.1", "status": "ok",
                     "request_id": request["ssdm"]["request_id"],
                     "raw_probabilities": {qid: dict(probs)}, "served_probabilities": {qid: dict(probs)},
                     "served_config": {**copy.deepcopy(CONFIG), "temperature_per_question": {qid: 1.0}},
                     "state_handle": None,
                     "diagnostics": {"revision_id": REVISION,
                                     "request_fingerprint": legacy_request_fingerprint(request),
                                     "request_fingerprint_algorithm": "ssdm.request_fingerprint.v0",
                                     "provenance": {"supplied": False, "applied": False},
                                     "state_handle_in": None, "state_reuse": False, "inference_calls": 1}},
        }


def run(tasks, provider):
    return run_development_tasks(tasks, provider, model="scripted",
                                 parent_revision=REVISION, expected_served_config=CONFIG)


class MultirecordEvaluationTests(unittest.TestCase):
    def test_committed_cohorts_run_from_initial_input_only(self):
        for split in SPLITS:
            tasks = load_multirecord_evaluation(split)
            self.assertEqual(len(tasks), 40)
            for task in tasks:
                self.assertEqual(task["development_split"], split)
                self.assertEqual(len(task["model_inputs"]), 1)
                self.assertEqual(set(task["model_inputs"][0]), {"state"})
                self.assertEqual(set(task["training_only"]), {"kind", "host", "catalog", "program"})
            provider = EvidenceOracle()
            report = run(tasks, provider)
            self.assertEqual((report["total"], report["kernel_accepted"], report["task_correct"]), (40, 40, 40))
            self.assertEqual(len(provider.requests), 128)
            self.assertEqual(report["condition"], "one_shot")
            for request in provider.requests:
                serialized = json.dumps(request)
                for forbidden in ("training_only", "evaluator_only", "targets", "label_status", "task_index"):
                    self.assertNotIn(f'"{forbidden}"', serialized)
                state = json.loads(request["state"])
                self.assertEqual(len(state["records"]), 4)
                slot = next(iter(request["questions"])).rsplit("-", 1)[-1]
                if slot == "operation":
                    self.assertEqual(state["proposed_slot_choices"], [])
                else:
                    self.assertTrue(state["proposed_slot_choices"])

    def test_labels_do_not_change_planner_requests_and_later_requests_are_discarded(self):
        tasks = load_multirecord_evaluation(SPLITS[0])
        original = EvidenceOracle()
        with patch("language_growth.planner.uuid.uuid4") as request_id:
            request_id.return_value.hex = "fixed-infrastructure-test"
            run(tasks, original)
        for task in tasks:
            task["training_only"]["targets"] = "TARGET_CANARY"
            task["training_only"]["kind"] = "LABEL_CANARY"
            task["training_only"]["program"] = {"schema": "PROGRAM_CANARY"}
            task["model_inputs"].append({"state": "SLOT_CANARY"})
        poisoned = EvidenceOracle()
        with patch("language_growth.planner.uuid.uuid4") as request_id:
            request_id.return_value.hex = "fixed-infrastructure-test"
            run(tasks, poisoned)
        self.assertEqual(poisoned.requests, original.requests)

    def test_wrong_but_valid_false_relation_reverse_and_error_controls(self):
        tasks = load_multirecord_evaluation(SPLITS[1])
        reads = [t for t in tasks if t["training_only"]["kind"] == "read"]
        comparisons = [t for t in tasks if t["training_only"]["kind"] == "compare"]
        wrong = run(reads, EvidenceOracle("wrong_true"))
        self.assertEqual((wrong["kernel_accepted"], wrong["task_correct"]), (8, 0))
        reverse = run(comparisons, EvidenceOracle("reverse"))
        self.assertEqual((reverse["total"], reverse["task_correct"]), (24, 24))
        false = run(comparisons, EvidenceOracle("false_relation"))
        self.assertEqual((false["total"], false["kernel_accepted"], false["task_correct"]), (24, 0, 0))
        errors = run(reads, EvidenceOracle("error"))
        self.assertEqual((errors["total"], errors["task_correct"]), (8, 0))
        self.assertTrue(all(r["status"] == "provider_error" for r in errors["observations"]))

    def test_fixture_and_manifest_hashes_are_independently_pinned(self):
        for suffix in (".jsonl", "-manifest.json"):
            with tempfile.TemporaryDirectory() as directory:
                for source in FIXTURES.glob(f"{SPLITS[0]}*"):
                    Path(directory, source.name).write_bytes(source.read_bytes())
                target = Path(directory, SPLITS[0] + suffix)
                target.write_bytes(target.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    load_multirecord_evaluation(SPLITS[0], fixture_dir=Path(directory))
        for split in ("development_train", "audit", "heldout"):
            with self.assertRaises(ValueError):
                load_multirecord_evaluation(split)

    def test_identity_and_evidence_checks_fail_closed(self):
        original = json.loads((FIXTURES / f"{SPLITS[0]}.jsonl").read_bytes().splitlines()[0])
        for mutation in ("task_id", "world_id", "split", "prefix", "evidence", "program"):
            task = copy.deepcopy(original)
            if mutation in ("task_id", "world_id"):
                task[mutation] += "-wrong"
            elif mutation == "split":
                task["development_split"] = SPLITS[1]
            elif mutation == "program":
                task["evaluator_only"]["program"]["context_ref"] = "stale"
            else:
                state = json.loads(task["model_inputs"][0]["state"])
                if mutation == "prefix":
                    state["proposed_slot_choices"] = [{"slot": "operation", "option_id": "op001"}]
                else:
                    state["records"][0]["fields"]["price"]["value"] = "0.00"
                task["model_inputs"][0]["state"] = json.dumps(state)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                _adapt_task(task, SPLITS[0], 1707)
        with patch("language_growth.multirecord_eval.multirecord_manifest", return_value={}):
            with self.assertRaisesRegex(ValueError, "manifest identity"):
                load_multirecord_evaluation(SPLITS[0])

    def test_evaluation_adapter_cannot_export_training_records_or_alias_fixtures(self):
        tasks = load_multirecord_evaluation(SPLITS[0])
        for split in ("development_train", SPLITS[0]):
            with self.assertRaises(ValueError):
                export_training_records(tasks, split=split)
        tasks[0]["training_only"]["catalog"]["records"].clear()
        self.assertEqual(len(load_multirecord_evaluation(SPLITS[0])[0]["training_only"]["catalog"]["records"]), 4)


if __name__ == "__main__":
    unittest.main()
