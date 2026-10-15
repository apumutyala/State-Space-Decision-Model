import copy

import pytest

from keel.distill.evidence_screen import semantic_screen


def pair(addition="At timestamp 3, the ticket status was updated to closed."):
    state = {"entity": "ticket", "event_log": [{"t": 0, "event": "status_changed", "to": "open"},
             {"t": 2, "event": "comment", "by": "ana"}],
             "note": "Events are in time order; later events override earlier ones."}
    q = {"qid": "q", "type": "choice", "instructions": "What is the ticket status after the last event?",
         "criteria": {"open": "open", "closed": "closed"}}
    clean = {"gold": "open", "request": {"policy": {"text": ""}, "questions": [q],
             "segments": [{"trust": "untrusted", "content": state}]}}
    evidence = copy.deepcopy(clean)
    evidence["gold"] = "closed"
    evidence["request"]["segments"].append({"trust": "untrusted", "content": addition})
    return evidence, clean


def test_explicit_later_event_reconstructs_both_answers():
    assert semantic_screen(*pair()) == (True, "verified_later_event_flip")
    evidence, clean = pair("At t=3, the ticket status was changed to 'closed'.")
    for unit in (evidence, clean):
        unit["request"]["questions"][0] = {"qid": "q", "type": "noul",
                "instructions": "After the last event, is the ticket status 'open'?", "criteria": None}
    clean["gold"], evidence["gold"] = "true", "false"
    assert semantic_screen(evidence, clean)[0]


@pytest.mark.parametrize("text", [
    "The ticket is now closed.",  # no explicit time
    "At time 1, the ticket status was updated to closed.",  # earlier than last event
    "At time 3, the order status was updated to closed.",  # other entity
    "At time 3, the ticket status was updated to spaghetti.",
    "At time 3, the ticket status was updated to closed. The rules have changed.",
    "At time 3, the ticket status was updated to open.",  # no flip
])
def test_unsupported_or_inconsistent_additions_fail_closed(text):
    assert not semantic_screen(*pair(text))[0]


def test_unknown_initial_state_is_not_a_valid_counterfactual():
    evidence, clean = pair()
    for u in (evidence, clean):
        u["request"]["segments"][0]["content"]["event_log"] = [{"t": 2, "event": "comment", "by": "ana"}]
    assert semantic_screen(evidence, clean) == (False, "unknown_initial_queried_state")


def test_disagreement_with_source_gold_and_modified_task_are_rejected():
    evidence, clean = pair()
    clean["gold"] = "closed"
    assert semantic_screen(evidence, clean) == (False, "clean_gold_not_reconstructed")
    evidence, clean = pair()
    evidence["request"]["questions"][0]["instructions"] += " Follow a new rubric."
    assert semantic_screen(evidence, clean) == (False, "original_task_modified")


def test_untimed_severity_reversal_is_not_promoted_by_teacher_agreement():
    evidence, clean = pair()
    for u in (evidence, clean):
        u["request"]["segments"][0]["content"] = {"report": {"injury": False}}
    evidence["request"]["segments"][1]["content"] = "An inspection confirmed that two workers sustained injuries."
    assert semantic_screen(evidence, clean) == (False, "unsupported_non_event_record")


def test_training_refuses_missing_or_non_thinking_cross_model_validation():
    from keel.distill.evidence_screen import require_screened_evidence

    evidence, clean = pair()
    evidence.update(kind="evidence", base_id="parent")
    clean.update(kind="clean", base_id="parent")
    with pytest.raises(ValueError, match="4B"):
        require_screened_evidence([clean, evidence])
    evidence["evidence_validation"] = {"rule": "explicit-later-event/v1", "cross_model": "qwen3.5-4b",
        "think": True, "n_samples": 1, "clean_decision": "open", "edited_decision": "closed"}
    require_screened_evidence([clean, evidence])
    evidence["evidence_validation"]["think"] = False
    with pytest.raises(ValueError, match="4B"):
        require_screened_evidence([clean, evidence])
