from __future__ import annotations

import json

from language_growth.development_data import _SPLITS
from language_growth.fact_binding_screen import generate_binding_screen
from language_growth.kernel import check_program
from language_growth.planner import _as_options, _candidate_sets, _fact_description


def test_screen_is_frozen_balanced_paired_and_checker_valid():
    rows = generate_binding_screen()
    assert len(rows) == 32
    assert len({r["pair_id"] for r in rows}) == 16
    assert sum(r["training_only"]["kind"] == "read" for r in rows) == 16
    assert sum(r["training_only"]["kind"] == "explain" for r in rows) == 16
    old_tokens = {word.casefold() for split in _SPLITS.values()
                  for name in split["names"] for word in name.split()}
    screen_tokens = {word.casefold() for row in rows for rec in row["training_only"]["catalog"]["records"]
                     for word in rec["entity_ref"].split()}
    assert old_tokens.isdisjoint(screen_tokens)
    for index in range(0, len(rows), 2):
        left, right = rows[index:index + 2]
        a, b = left["training_only"], right["training_only"]
        assert left["pair_id"] == right["pair_id"]
        assert (a["kind"], a["field"], a["target_record_ref"]) == (
            b["kind"], b["field"], b["target_record_ref"])
        assert a["catalog"]["records"] != b["catalog"]["records"]
        assert a["catalog"]["snapshots"][a["host"]["context_ref"]]["record_refs"] == list(reversed(
            b["catalog"]["snapshots"][b["host"]["context_ref"]]["record_refs"]))
        assert check_program(a["program"], a["host"], a["catalog"]).status == "accepted"
        assert check_program(b["program"], b["host"], b["catalog"]).status == "accepted"
        for task in (left, right):
            request = task["model_inputs"][0]
            assert set(request) == {"model", "state", "questions", "ssdm"}
            assert set(request["questions"]) == {"slot-001-operation"}
            state = json.loads(request["state"])
            assert "training_only" not in json.dumps(request)
            assert "target_record_ref" not in json.dumps(request)
            assert state["proposed_slot_choices"] == []
            assert state["accepted_partial_program"] == []


def test_visible_name_candidate_changes_text_only_for_same_ordered_options():
    task = generate_binding_screen()[0]
    train = task["training_only"]
    task_input = json.loads(task["model_inputs"][0]["state"])["task_input"]
    old_facts, old_operands, old_obligations = _candidate_sets(train["host"], train["catalog"], task_input)
    new_facts, new_operands, new_obligations = _candidate_sets(
        train["host"], train["catalog"], task_input, candidate_style="entity_name")
    old_options, old_bindings = _as_options(old_facts, _fact_description, "f")
    new_options, new_bindings = _as_options(new_facts, _fact_description, "f")
    assert list(old_options) == list(new_options)
    assert old_obligations == new_obligations == []
    assert [(x["record_ref"], x["field"]) for x in old_operands if x["kind"] == "fact"] == [
        (x["record_ref"], x["field"]) for x in new_operands if x["kind"] == "fact"]
    assert [(old_bindings[k]["record_ref"], old_bindings[k]["field"]) for k in old_options] == [
        (new_bindings[k]["record_ref"], new_bindings[k]["field"]) for k in new_options]
    assert all(new_bindings[k]["entity_ref"] in new_options[k] for k in new_options)
    assert all(new_bindings[k]["record_ref"] in new_options[k] for k in new_options)
