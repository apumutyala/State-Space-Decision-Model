"""Contract tests for the discovery-only scaffold; the parent is a deterministic fake, not a model result."""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal
import json
import unittest
from unittest.mock import patch

from decide.adapter import Decider
from decide.backends.base import Plan, ScoreResult
from decide.calibration import OptionCountTemperature
from language_growth.causal_relation_screen import (
    MIN_RUNTIME_AVAILABLE_BYTES, MIN_START_AVAILABLE_BYTES, PINNED_DISCOVERY_INPUT_SHA256,
    RelationScaffoldProvider, run_relation_screen, select_screen_inputs,
)
from language_growth.parent_bridge import decision_request
from language_growth.planner import plan_checked_answer
from language_growth.semantic_axis_fixtures import generate_split, rows


def _parse(task):
    import re
    match = re.fullmatch(
        r"Compare (?P<left>.+?)'s (?P<field>[A-Za-z_][A-Za-z0-9_]*) with "
        r"(?P<right>.+?)'s (?P=field) in October 2026, in that order\.", task)
    if match is None:
        raise AssertionError(f"test fake parent got unsupported wording: {task}")
    return match.group("left"), match.group("right"), match.group("field")


class FakeRelationBackend:
    name = "test-only-deterministic-parent"

    def __init__(self, *, wrong=False, fail=False):
        self.wrong, self.fail = wrong, fail
        self.requests = []

    def revision(self):
        return {"model_id": "fake-parent", "revision_id": "fake-parent-v1",
                "backend": "rwkv7_layout_bound_causal_experiment", "edit_id": None,
                "edit_sign": None, "max_context_tokens": 2048, "layout_version": "m2-layout-v2",
                "runtime_identity_sha256": "a" * 64}

    def plan(self, request):
        qid = next(iter(request["questions"]))
        return Plan(inference_calls=1, max_call_input_tokens=4, total_input_tokens=4,
                    per_question={qid: {"prompt_tokens": 3, "options": 3}})

    def score(self, request, check_deadline):
        check_deadline()
        self.requests.append(deepcopy(request))
        if self.fail:
            raise RuntimeError("test backend failure")
        state = json.loads(request["state"])
        left, right, field = _parse(state["task_input"])
        by_name = {record["entity_ref"]: record for record in state["records"]}
        lv = Decimal(str(by_name[left]["fields"][field]["value"]))
        rv = Decimal(str(by_name[right]["fields"][field]["value"]))
        desired = "equal" if lv == rv else "less_than" if lv < rv else "greater_than"
        if self.wrong:
            desired = {"equal": "less_than", "less_than": "greater_than", "greater_than": "equal"}[desired]
        qid, question = next(iter(request["questions"].items()))
        scores = {option_id: (1.0 if desc == f"Relation {desired}" else -1.0)
                  for option_id, desc in question["criteria"].items()}
        return ScoreResult(logits={qid: scores}, inference_calls=1, total_input_tokens=4,
                           extras={"test_fake_parent": True})


class CausalRelationScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs, _scorer_groups = generate_split("semantic_axis_discovery")
        cls.input_hash = __import__("hashlib").sha256(rows(cls.inputs)).hexdigest()
        if cls.input_hash != PINNED_DISCOVERY_INPUT_SHA256:
            raise AssertionError("generated discovery inputs differ from the committed manifest pin")

    def _decider(self, *, wrong=False, fail=False):
        backend = FakeRelationBackend(wrong=wrong, fail=fail)
        decider = Decider(backend, calibration=OptionCountTemperature())
        return decider, backend

    def test_selects_exact_discovery_offsets_and_rejects_bad_hash_or_scorer_fields(self):
        selected, digest = select_screen_inputs(self.inputs, expected_input_sha256=PINNED_DISCOVERY_INPUT_SHA256)
        self.assertEqual([offset for offset, _ in selected], [1, 2, 5, 7, 8, 11, 13, 14, 17])
        self.assertEqual(digest, self.input_hash)
        with self.assertRaisesRegex(ValueError, "pinned manifest"):
            select_screen_inputs(self.inputs, expected_input_sha256="0" * 64)
        altered = deepcopy(self.inputs)
        altered[0]["target_option_id"] = "leaked"
        changed_hash = __import__("hashlib").sha256(rows(altered)).hexdigest()
        with self.assertRaisesRegex(ValueError, "only task_input, host and catalog"):
            select_screen_inputs(altered, expected_input_sha256=changed_hash)

    def test_run_keeps_scaffold_fixed_and_saves_only_parent_relation_responses(self):
        decider, backend = self._decider()
        revision = decider.revision()["revision_id"]
        calibration_identity = decider.calibration.identity()
        with patch("language_growth.causal_relation_screen._available_memory_bytes",
                   return_value=16 * 1024**3):
            output = run_relation_screen(self.inputs, lambda: decider,
                                         expected_input_sha256=PINNED_DISCOVERY_INPUT_SHA256,
                                         expected_parent_revision=revision,
                                         expected_calibration_identity=calibration_identity)
        self.assertEqual(output["status"], "complete")
        self.assertEqual(output["selected_offsets"], [1, 2, 5, 7, 8, 11, 13, 14, 17])
        self.assertEqual(len(output["results"]), 9)
        self.assertEqual(len(backend.requests), 9)
        self.assertTrue(all(item["status"] == "accepted" for item in output["results"]))
        self.assertTrue(all(item["provider_calls"] == 4 and item["model_forward_calls"] == 1
                            for item in output["results"]))
        self.assertTrue(all(len(item["scripted_scaffold"]) == 3
                            and [event["slot"] for event in item["scripted_scaffold"]] ==
                            ["operation", "left", "right"] for item in output["results"]))
        parent_events = [item["parent_relation_calls"][0] for item in output["results"]]
        self.assertTrue(all(event["status"] == "returned"
                            and event["request"] == backend_request
                            for event, backend_request in zip(parent_events, backend.requests)))
        self.assertTrue(all(set(event["response"]["ssdm"]["raw_probabilities"]) ==
                            {next(iter(event["request"]["questions"]))} for event in parent_events))
        # Same comparison task wording and entity binding across low/equal/high; only scalar evidence changes.
        selection_fields = ("slot", "option_id", "description")
        first = [{key: event[key] for key in selection_fields}
                 for event in output["results"][0]["scripted_scaffold"]]
        middle = [{key: event[key] for key in selection_fields}
                  for event in output["results"][3]["scripted_scaffold"]]
        high = [{key: event[key] for key in selection_fields}
                for event in output["results"][6]["scripted_scaffold"]]
        self.assertEqual(first, middle)
        self.assertEqual(middle, high)
        self.assertNotIn("scorer", json.dumps(output).lower())

    def test_wrong_parent_relation_is_preserved_and_checker_rejects_without_fallback(self):
        decider, backend = self._decider(wrong=True)
        item = self.inputs[1]
        scaffold = RelationScaffoldProvider(decider)
        revision = decider.revision()
        with patch("language_growth.causal_relation_screen._available_memory_bytes",
                   return_value=16 * 1024**3):
            result = plan_checked_answer(
                item["task_input"], item["host"], item["catalog"], scaffold,
                model=revision["model_id"], parent_revision=revision["revision_id"],
                expected_served_config={"calibration": decider.calibration.identity(),
                                        "policy_masking": "none (served covers every supplied option)"},
            )
        self.assertEqual(result.status, "kernel_rejected")
        self.assertIsNone(result.text)
        self.assertEqual(len(backend.requests), 1)
        parent = scaffold.parent_relation_events[0]["response"]
        qid = next(iter(parent["answers"]))
        self.assertEqual(parent["answers"][qid]["choice"], max(parent["answers"][qid]["probabilities"],
                                                                key=parent["answers"][qid]["probabilities"].get))
        self.assertEqual(scaffold.parent_relation_events[0]["request"], backend.requests[0])

    def test_parent_failure_stays_failure_and_duplicate_entity_names_fail_before_parent(self):
        decider, backend = self._decider(fail=True)
        item = self.inputs[1]
        scaffold = RelationScaffoldProvider(decider)
        revision = decider.revision()
        with patch("language_growth.causal_relation_screen._available_memory_bytes",
                   return_value=16 * 1024**3):
            result = plan_checked_answer(
                item["task_input"], item["host"], item["catalog"], scaffold,
                model=revision["model_id"], parent_revision=revision["revision_id"],
                expected_served_config={"calibration": decider.calibration.identity(),
                                        "policy_masking": "none (served covers every supplied option)"},
            )
        self.assertIsNone(result.text)
        self.assertEqual(result.status, "provider_rejected")
        self.assertEqual(scaffold.parent_relation_events[0]["response"]["ssdm"]["error"]["code"], "backend_error")
        self.assertEqual(len(backend.requests), 1)

        duplicate = deepcopy(item)
        records = duplicate["catalog"]["records"]
        for record in records:
            record["entity_ref"] = "same-name"
        duplicate["task_input"] = "Compare same-name's price with same-name's price in October 2026, in that order."
        decider2, backend2 = self._decider()
        scaffold2 = RelationScaffoldProvider(decider2)
        rev2 = decider2.revision()
        rejected = plan_checked_answer(
            duplicate["task_input"], duplicate["host"], duplicate["catalog"], scaffold2,
            model=rev2["model_id"], parent_revision=rev2["revision_id"],
            expected_served_config={"calibration": decider2.calibration.identity(),
                                    "policy_masking": "none (served covers every supplied option)"},
        )
        self.assertIsNone(rejected.text)
        self.assertEqual(rejected.status, "provider_error")
        self.assertEqual(backend2.requests, [])

    def test_resource_guards_run_before_parent_factory_and_before_relation_forward(self):
        factory_called = []
        pin_decider, _ = self._decider()
        pins = {"expected_parent_revision": pin_decider.revision()["revision_id"],
                "expected_calibration_identity": pin_decider.calibration.identity()}
        with patch("language_growth.causal_relation_screen._available_memory_bytes",
                   return_value=MIN_START_AVAILABLE_BYTES - 1):
            refused = run_relation_screen(self.inputs, lambda: factory_called.append(True),
                                          expected_input_sha256=PINNED_DISCOVERY_INPUT_SHA256, **pins)
        self.assertEqual(refused["status"], "incomplete")
        self.assertEqual(refused["stop_reason"], "start_memory_guard")
        self.assertEqual(factory_called, [])

        decider, backend = self._decider()
        memory = iter([MIN_START_AVAILABLE_BYTES, MIN_RUNTIME_AVAILABLE_BYTES - 1])
        with patch("language_growth.causal_relation_screen._available_memory_bytes",
                   side_effect=lambda: next(memory)):
            stopped = run_relation_screen(self.inputs, lambda: decider,
                                          expected_input_sha256=PINNED_DISCOVERY_INPUT_SHA256,
                                          expected_parent_revision=decider.revision()["revision_id"],
                                          expected_calibration_identity=decider.calibration.identity())
        self.assertEqual(stopped["status"], "incomplete")
        self.assertEqual(stopped["stop_reason"], "runtime_memory_guard")
        self.assertEqual(len(stopped["results"]), 1)
        self.assertEqual(stopped["results"][0]["parent_relation_calls"][0]["status"], "runtime_memory_guard")
        self.assertEqual(backend.requests, [])

    def test_memory_telemetry_failure_and_wrong_backend_never_start_relation(self):
        factory_called = []
        pin_decider, _ = self._decider()
        pins = {"expected_parent_revision": pin_decider.revision()["revision_id"],
                "expected_calibration_identity": pin_decider.calibration.identity()}
        with patch("language_growth.causal_relation_screen._available_memory_bytes",
                   side_effect=OSError("telemetry unavailable")):
            failed_start = run_relation_screen(self.inputs, lambda: factory_called.append(True),
                                               expected_input_sha256=PINNED_DISCOVERY_INPUT_SHA256, **pins)
        self.assertEqual(failed_start["status"], "incomplete")
        self.assertEqual(failed_start["stop_reason"], "start_memory_read_failed")
        self.assertEqual(factory_called, [])

        decider, backend = self._decider()
        with patch("language_growth.causal_relation_screen._available_memory_bytes",
                   side_effect=[MIN_START_AVAILABLE_BYTES, OSError("telemetry unavailable")]):
            failed_runtime = run_relation_screen(self.inputs, lambda: decider,
                                                 expected_input_sha256=PINNED_DISCOVERY_INPUT_SHA256,
                                                 expected_parent_revision=decider.revision()["revision_id"],
                                                 expected_calibration_identity=decider.calibration.identity())
        self.assertEqual(failed_runtime["status"], "incomplete")
        self.assertEqual(failed_runtime["stop_reason"], "runtime_memory_read_failed")
        self.assertEqual(backend.requests, [])

        wrong_backend = FakeRelationBackend()
        wrong_backend.revision = lambda: {"model_id": "fake", "revision_id": "wrong", "backend": "oracle"}
        wrong_decider = Decider(wrong_backend, calibration=OptionCountTemperature())
        with patch("language_growth.causal_relation_screen._available_memory_bytes",
                   return_value=16 * 1024**3):
            refused_parent = run_relation_screen(self.inputs, lambda: wrong_decider,
                                                 expected_input_sha256=PINNED_DISCOVERY_INPUT_SHA256,
                                                 expected_parent_revision=wrong_decider.revision()["revision_id"],
                                                 expected_calibration_identity=wrong_decider.calibration.identity())
        self.assertEqual(refused_parent["status"], "incomplete")
        self.assertEqual(refused_parent["stop_reason"], "backend_not_unchanged_layout_bound_parent")
        self.assertEqual(wrong_backend.requests, [])

    def test_parent_revision_and_calibration_pins_fail_before_relation(self):
        decider, backend = self._decider()
        with patch("language_growth.causal_relation_screen._available_memory_bytes",
                   return_value=16 * 1024**3):
            wrong_revision = run_relation_screen(
                self.inputs, lambda: decider, expected_input_sha256=PINNED_DISCOVERY_INPUT_SHA256,
                expected_parent_revision="unexpected-parent-v0",
                expected_calibration_identity=decider.calibration.identity())
            wrong_calibration = run_relation_screen(
                self.inputs, lambda: decider, expected_input_sha256=PINNED_DISCOVERY_INPUT_SHA256,
                expected_parent_revision=decider.revision()["revision_id"],
                expected_calibration_identity={"kind": "other-calibration", "id": "wrong"})
        self.assertEqual(wrong_revision["stop_reason"], "parent_revision_mismatch")
        self.assertEqual(wrong_calibration["stop_reason"], "calibration_identity_mismatch")
        self.assertEqual(backend.requests, [])

    def test_context_change_between_scaffold_and_relation_refuses_before_parent(self):
        item = self.inputs[1]
        parent, backend = self._decider()
        scaffold = RelationScaffoldProvider(parent)
        revision = parent.revision()

        def make_request(host, catalog, qid, prompt, options, prior):
            return decision_request(task_input=item["task_input"], host=host, catalog=catalog,
                                    model=revision["model_id"], parent_revision=revision["revision_id"],
                                    request_id=f"test-{qid}", question_id=qid, question=prompt,
                                    options=options, proposed_slot_choices=prior)[0]

        op = {"op003": "Compare two selected typed values and assert the selected relation"}
        response = scaffold.decide(make_request(item["host"], item["catalog"], "slot-001-operation",
                                                "Select the semantic operation for the user's task.", op, []))
        op_choice = response["answers"]["slot-001-operation"]["choice"]
        prior = [{"slot": "operation", "option_id": op_choice, "description": op[op_choice]}]
        state = json.loads(decision_request(
            task_input=item["task_input"], host=item["host"], catalog=item["catalog"],
            model=revision["model_id"], parent_revision=revision["revision_id"],
            request_id="inspect-state", question_id="slot-002-left", question="Select the left typed operand.",
            options={"v001": "unused"}, proposed_slot_choices=prior)[0]["state"])
        left_entity, right_entity, field = _parse(item["task_input"])
        by_name = {record["entity_ref"]: record for record in state["records"]}
        facts = [{"record_ref": record["record_ref"], "field": name}
                 for record in state["records"] for name in sorted(record["fields"])
                 if name in state["registry"]["registered_fields"]]
        operand_options = {f"v{i:03d}": f"Fact {fact['record_ref']} field {fact['field']}"
                           for i, fact in enumerate(facts, 1)}
        left_desc = f"Fact {by_name[left_entity]['record_ref']} field {field}"
        left_id = next(k for k, v in operand_options.items() if v == left_desc)
        response = scaffold.decide(make_request(item["host"], item["catalog"], "slot-002-left",
                                                "Select the left typed operand.", operand_options, prior))
        self.assertEqual(response["answers"]["slot-002-left"]["choice"], left_id)
        prior = [*prior, {"slot": "left", "option_id": left_id, "description": left_desc}]
        right_desc = f"Fact {by_name[right_entity]['record_ref']} field {field}"
        right_id = next(k for k, v in operand_options.items() if v == right_desc)
        scaffold.decide(make_request(item["host"], item["catalog"], "slot-003-right",
                                     "Select the right typed operand.", operand_options, prior))
        prior = [*prior, {"slot": "right", "option_id": right_id, "description": right_desc}]

        changed_host, changed_catalog = deepcopy(item["host"]), deepcopy(item["catalog"])
        old_context = changed_host["context_ref"]
        new_context = old_context + ":changed"
        changed_catalog["snapshots"][new_context] = deepcopy(changed_catalog["snapshots"][old_context])
        changed_host["context_ref"] = new_context
        changed_host["provider"]["input_context_ref"] = new_context
        relation_options = {"r001": "Relation less_than", "r002": "Relation equal",
                            "r003": "Relation greater_than"}
        with self.assertRaisesRegex(ValueError, "identity changed"):
            scaffold.decide(make_request(changed_host, changed_catalog, "slot-004-relation",
                                         "Select the comparison relation.", relation_options, prior))
        self.assertEqual(backend.requests, [])


if __name__ == "__main__":
    unittest.main()
