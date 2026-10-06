"""Synthetic clarification tasks exercise real planner choices and task scoring (CPU)."""

from __future__ import annotations

import copy
import json
import unittest
from collections import Counter
from decimal import Decimal

from language_growth.development_data import development_manifest, generate_development_tasks
from language_growth.free_running_dev import run_development_tasks
from language_growth.parent_bridge import decision_request, legacy_request_fingerprint

REVISION = "scripted-clarification:v1"
CONFIG = {
    "calibration": {"kind": "option_count_temperature", "id": "scripted-clarification",
                    "t0": 1.0, "t1": 0.0, "t_min": 0.05},
    "policy_masking": "none (served covers every supplied option)",
}


def _requested(state):
    task = state["task_input"]
    record = next(rec for rec in state["records"] if rec["entity_ref"] in task)
    field = "warranty_months" if "warranty months" in task else "price"
    return record, field


class SemanticProvider:
    """Chooses from the current task and permitted state; never reads training_only."""

    def __init__(self, *, wrong_ask=False, ask_even_when_answerable=False):
        self.wrong_ask = wrong_ask
        self.ask_even_when_answerable = ask_even_when_answerable
        self.requests = []

    def decide(self, request):
        self.requests.append(copy.deepcopy(request))
        state = json.loads(request["state"])
        qid, question = next(iter(request["questions"].items()))
        options = question["criteria"]
        slot = qid.rsplit("-", 1)[-1]
        task = state["task_input"]
        record, field = _requested(state)
        requested_missing = field not in record["fields"]
        if slot == "operation":
            if self.ask_even_when_answerable and state["unmet_obligations"]:
                phrase = "Ask for a registered missing input"
            elif requested_missing:
                phrase = "Ask for a registered missing input"
            elif "compare" in task.lower():
                phrase = "Compare two selected typed values"
            elif "derivation" in task.lower() or "checked" in task.lower():
                phrase = "Explain a checked assertion"
            else:
                phrase = "Assert one selected fact"
        elif slot == "fact":
            phrase = f"Fact {record['record_ref']} field {field}"
        elif slot in ("left", "right"):
            named = sorted(state["records"], key=lambda rec: task.index(rec["entity_ref"]))
            chosen = named[0 if slot == "left" else 1]
            phrase = f"Fact {chosen['record_ref']} field {field}"
        elif slot == "relation":
            previous = {choice["slot"]: choice["description"] for choice in state["proposed_slot_choices"]}
            refs = [previous[side].split(" ")[1] for side in ("left", "right")]
            by_ref = {rec["record_ref"]: rec for rec in state["records"]}
            values = [Decimal(str(by_ref[ref]["fields"][field]["value"])) for ref in refs]
            relation = "equal" if values[0] == values[1] else (
                "greater_than" if values[0] > values[1] else "less_than")
            phrase = f"Relation {relation}"
        elif slot == "obligation":
            obligations = state["unmet_obligations"]
            if self.ask_even_when_answerable and not requested_missing:
                chosen = obligations[0]
            elif self.wrong_ask:
                chosen = next(ob for ob in obligations if not any(
                    state["query_constants"][ref] == {"entity_ref": record["entity_ref"], "field": field}
                    for ref in ob["allowed_refs"]))
            else:
                chosen = next(ob for ob in obligations if any(
                    state["query_constants"][ref] == {"entity_ref": record["entity_ref"], "field": field}
                    for ref in ob["allowed_refs"]))
            phrase = f"Obligation {chosen['obligation_id']} category {chosen['category']}"
        elif slot == "ask_reference":
            previous = {choice["slot"]: choice["description"] for choice in state["proposed_slot_choices"]}
            obligation_id = previous["obligation"].split(" ")[1]
            obligation = next(ob for ob in state["unmet_obligations"] if ob["obligation_id"] == obligation_id)
            phrase = f"Requested reference {obligation['allowed_refs'][0]}"
        else:
            raise AssertionError(slot)
        matches = [oid for oid, description in options.items() if description.startswith(phrase)]
        if len(matches) != 1:
            raise AssertionError((slot, phrase, options))
        selected = matches[0]
        probabilities = {oid: float(oid == selected) for oid in options}
        return {
            "answers": {qid: {"type": "choice", "choice": selected, "confidence": 1.0,
                              "probabilities": dict(probabilities)}},
            "ssdm": {"schema": "ssdm.decide.v0.1", "status": "ok",
                     "request_id": request["ssdm"]["request_id"],
                     "raw_probabilities": {qid: dict(probabilities)},
                     "served_probabilities": {qid: dict(probabilities)},
                     "served_config": {**copy.deepcopy(CONFIG), "temperature_per_question": {qid: 1.0}},
                     "state_handle": None,
                     "diagnostics": {"revision_id": REVISION,
                                     "request_fingerprint": legacy_request_fingerprint(request),
                                     "request_fingerprint_algorithm": "ssdm.request_fingerprint.v0",
                                     "provenance": {"supplied": False, "applied": False},
                                     "state_handle_in": None, "state_reuse": False,
                                     "inference_calls": 1}},
        }


def run(tasks, provider):
    return run_development_tasks(tasks, provider, model="scripted", parent_revision=REVISION,
                                 expected_served_config=CONFIG)


class ClarificationTests(unittest.TestCase):
    def test_missing_layouts_overlap_and_question_controls_same_catalog(self):
        layouts = {"ask_missing": set(), "read_with_unrelated_obligation": set()}
        for seed in range(8):
            for task in generate_development_tasks(seed, 25, clarification_mode=True):
                kind = task["training_only"]["kind"]
                if kind in layouts:
                    records = task["training_only"]["catalog"]["records"]
                    layouts[kind].add(tuple(tuple(sorted(record["fields"])) for record in records))
        self.assertEqual(layouts["ask_missing"], layouts["read_with_unrelated_obligation"])
        # All six two-missing-field layouts occur in both arms, including an empty record.
        self.assertEqual(len(layouts["ask_missing"]), 6)
        self.assertTrue(any(not fields for layout in layouts["ask_missing"] for fields in layout))

        task = generate_development_tasks(17, 4, clarification_mode=True)[3]
        host, catalog = task["training_only"]["host"], task["training_only"]["catalog"]
        question = next(iter(task["model_inputs"][0]["questions"].values()))
        records = catalog["records"]
        missing = next((record, field) for record in records
                       for field in ("price", "warranty_months") if field not in record["fields"])
        present = next((record, field) for record in records for field in record["fields"])
        requests = []
        for number, (record, field) in enumerate((missing, present), 1):
            request, _ = decision_request(
                task_input=f"What is {record['entity_ref']}'s {field.replace('_', ' ')} in October 2026?"
                           " Ask for the needed input if it is absent.",
                host=host, catalog=catalog, model="development-only", parent_revision=REVISION,
                request_id=f"paired-question-{number}", question_id="slot-001-operation",
                question="Select the semantic operation for the user's task.", options=question["criteria"])
            requests.append(request)
        states = [json.loads(request["state"]) for request in requests]
        self.assertEqual(states[0]["records"], states[1]["records"])
        self.assertEqual(states[0]["unmet_obligations"], states[1]["unmet_obligations"])
        self.assertEqual(states[0]["query_constants"], states[1]["query_constants"])
        provider = SemanticProvider()
        selected = []
        for request in requests:
            qid, item = next(iter(request["questions"].items()))
            choice = provider.decide(request)["answers"][qid]["choice"]
            selected.append(item["criteria"][choice])
        self.assertEqual(selected, ["Ask for a registered missing input", "Assert one selected fact"])

    def test_opt_in_generation_is_balanced_and_split_explicit(self):
        tasks = generate_development_tasks(17, 25, choice_format="production", clarification_mode=True)
        self.assertEqual(Counter(t["training_only"]["kind"] for t in tasks),
                         {kind: 5 for kind in ("read", "explain", "compare", "ask_missing",
                                               "read_with_unrelated_obligation")})
        self.assertEqual(tasks, generate_development_tasks(17, 25, choice_format="production",
                                                           clarification_mode=True))
        self.assertEqual({t["training_only"]["kind"] for t in generate_development_tasks(17, 9)},
                         {"read", "explain", "compare"})
        legacy = development_manifest(generate_development_tasks(17, 15, choice_format="mixed"))
        self.assertEqual(legacy["sample_sha256"],
                         "6c1e6b24c56e10ad29cca3c20d2f2e5f9ed0b557e6533c6ca64d9e27c66dd74f")
        self.assertNotIn("clarification_mode", legacy)
        manifest = development_manifest(tasks)
        self.assertTrue(manifest["clarification_mode"])
        self.assertEqual(manifest["kind_counts"]["ask_missing"], 5)
        validation = generate_development_tasks(19, 25, split="development_validation",
                                                clarification_mode=True)
        self.assertFalse(set(manifest["template_ids"]["ask_missing"]) &
                         set(development_manifest(validation)["template_ids"]["ask_missing"]))
        self.assertFalse({rec["entity_ref"] for task in tasks for rec in task["training_only"]["catalog"]["records"]}
                         & {rec["entity_ref"] for task in validation
                            for rec in task["training_only"]["catalog"]["records"]})
        for task in tasks:
            kind = task["training_only"]["kind"]
            state = json.loads(task["model_inputs"][0]["state"])
            self.assertEqual(state["proposed_slot_choices"], [])
            self.assertNotIn("targets", state)
            self.assertNotIn("target_option_id", state)
            if kind in ("ask_missing", "read_with_unrelated_obligation"):
                self.assertIn("Ask for a registered missing input",
                              next(iter(task["model_inputs"][0]["questions"].values()))["criteria"].values())
                self.assertTrue(state["unmet_obligations"])
                self.assertEqual(len(state["unmet_obligations"]), 2)
                record, field = _requested(state)
                self.assertEqual(field not in record["fields"], kind == "ask_missing")

    def test_actual_planner_choices_solve_ask_and_answerable_obligation_tasks(self):
        tasks = generate_development_tasks(29, 10, choice_format="production", clarification_mode=True)
        provider = SemanticProvider()
        report = run(tasks, provider)
        self.assertEqual((report["total"], report["kernel_accepted"], report["task_correct"]), (10, 10, 10))
        by_id = {task["task_id"]: task for task in tasks}
        for row in report["observations"]:
            kind = by_id[row["task_id"]]["training_only"]["kind"]
            self.assertGreater(row["provider_calls"], 0)
            if kind == "ask_missing":
                self.assertEqual(row["program"]["nodes"][0]["op"], "Ask")
                self.assertEqual(row["provider_calls"], 3)
                task = by_id[row["task_id"]]
                task_wording = json.loads(task["model_inputs"][0]["state"])["task_input"]
                for index, slot in ((1, "obligation"), (2, "ask_reference")):
                    generated = next(iter(task["model_inputs"][index]["questions"].values()))["criteria"]
                    live = [request for request in provider.requests
                            if json.loads(request["state"])["task_input"] == task_wording
                            and request["ssdm"]["context_ref"] == task["training_only"]["host"]["context_ref"]
                            and next(iter(request["questions"])).endswith("-" + slot)]
                    self.assertEqual(len(live), 1)
                    self.assertEqual(next(iter(live[0]["questions"].values()))["criteria"], generated)
            if kind == "read_with_unrelated_obligation":
                self.assertEqual(row["program"]["nodes"][0]["op"], "ReadFact")
                self.assertEqual(row["provider_calls"], 2)
                task = by_id[row["task_id"]]
                generated = next(iter(task["model_inputs"][1]["questions"].values()))["criteria"]
                task_wording = json.loads(task["model_inputs"][0]["state"])["task_input"]
                live = [request for request in provider.requests
                        if json.loads(request["state"])["task_input"] == task_wording
                        and request["ssdm"]["context_ref"] == task["training_only"]["host"]["context_ref"]
                        and next(iter(request["questions"])).endswith("-fact")]
                self.assertEqual(len(live), 1)
                self.assertEqual(next(iter(live[0]["questions"].values()))["criteria"], generated)
        self.assertTrue(all("training_only" not in request for request in provider.requests))
        mixed = run(generate_development_tasks(29, 10, choice_format="mixed", clarification_mode=True),
                    SemanticProvider())
        self.assertEqual((mixed["total"], mixed["kernel_accepted"], mixed["task_correct"]), (10, 10, 10))

    def test_wrong_question_ask_is_kernel_valid_but_task_wrong(self):
        ask = generate_development_tasks(31, 5, clarification_mode=True)[3]
        report = run([ask], SemanticProvider(wrong_ask=True))
        row = report["observations"][0]
        self.assertEqual((row["status"], row["kernel_accepted"], row["task_correct"]),
                         ("accepted", True, False))
        self.assertEqual(row["program"]["nodes"][0]["op"], "Ask")
        self.assertNotEqual(row["program"]["nodes"][0]["obligation_ref"],
                            ask["training_only"]["program"]["nodes"][0]["obligation_ref"])

    def test_unrelated_obligation_does_not_make_ask_correct(self):
        answerable = generate_development_tasks(31, 5, clarification_mode=True)[4]
        report = run([answerable], SemanticProvider(ask_even_when_answerable=True))
        row = report["observations"][0]
        self.assertEqual((row["status"], row["kernel_accepted"], row["task_correct"]),
                         ("accepted", True, False))
        self.assertEqual(row["program"]["nodes"][0]["op"], "Ask")


if __name__ == "__main__":
    unittest.main()
