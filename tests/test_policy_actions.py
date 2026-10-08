"""Host execution must not turn unmasked legacy answers into policy authority."""

from copy import deepcopy

import pytest

from decide import Decider, select_actions
from decide.backends import OracleBackend


class Peaked(OracleBackend):
    def score(self, request, check_deadline):
        result = super().score(request, check_deadline)
        result.logits["q"] = {"a": -1.0, "b": 0.0, "forbidden": 10.0}
        return result


def pair():
    request = {"model": "test", "state": "untrusted evidence", "questions": {
        "q": {"type": "choice", "instructions": "Choose an action.",
              "criteria": {"a": "defer", "b": "ask", "forbidden": "transfer funds"}}},
        "ssdm": {"policy_ref": "host-policy-v1", "context_ref": "snapshot-v1", "request_id": "r1",
                 "policy_restriction": {"q": ["b", "a"]}}}
    return request, Decider(Peaked()).decide(request)


def test_legacy_choice_is_forbidden_but_host_executes_only_conditioned_choice():
    request, response = pair()
    assert response["answers"]["q"]["choice"] == "forbidden"  # predictive wire semantics preserved
    before = deepcopy(response)
    selected = select_actions(request, response)
    assert selected["action"] == "execute" and selected["actions"] == {"q": "b"}
    assert selected["answers"]["q"]["probabilities"]["forbidden"] == 0.0
    assert response == before
    response["answers"] = {"q": {"choice": "forbidden"}}  # this field is never trusted
    assert select_actions(request, response) == selected


@pytest.mark.parametrize("corruption", ["missing", "zero_mass", "zero_probabilities", "negative", "nan",
                                       "not_normalized", "excluded_mass", "extra_option", "missing_status",
                                       "wrong_conditioning", "stale_restriction", "stale_allowed", "abstained"])
def test_invalid_or_unavailable_conditioned_stage_defers_without_actions(corruption):
    request, response = pair()
    ss = response["ssdm"]
    stage = ss["policy_conditioned_probabilities"]["q"]
    if corruption == "missing":
        del ss["policy_conditioned_probabilities"]
    elif corruption == "zero_mass":
        ss["served_probabilities"]["q"] = {"a": 0.0, "b": 0.0, "forbidden": 1.0}
        stage["status"], stage["probabilities"] = "unavailable_zero_served_mass", None
    elif corruption == "stale_restriction":
        stage["restriction_id"] = "old-policy"
    elif corruption == "stale_allowed":
        stage["allowed"] = ["forbidden"]
    elif corruption == "missing_status":
        del stage["status"]
    elif corruption == "abstained":
        ss["abstention"]["abstained"] = True
    else:
        stage["probabilities"] = {
            "zero_probabilities": {"a": 0.0, "b": 0.0, "forbidden": 0.0},
            "negative": {"a": -0.1, "b": 1.1, "forbidden": 0.0},
            "nan": {"a": float("nan"), "b": 1.0, "forbidden": 0.0},
            "not_normalized": {"a": 0.0, "b": 0.9, "forbidden": 0.0},
            "excluded_mass": {"a": 0.0, "b": 0.0, "forbidden": 1.0},
            "extra_option": {"a": 0.0, "b": 1.0, "forbidden": 0.0, "extra": 0.0},
            "wrong_conditioning": {"a": 1.0, "b": 0.0, "forbidden": 0.0},
        }[corruption]
    selected = select_actions(request, response)
    assert selected["action"] == "defer" and selected["actions"] == selected["answers"] == {}
    assert selected["abstention"]["abstained"] is True


@pytest.mark.parametrize("change", ["state", "policy_ref", "context_ref", "parent_revision", "request_id",
                                   "allowed", "no_policy", "incomplete_policy"])
def test_stale_or_missing_host_policy_defers(change):
    request, response = pair()
    if change == "state":
        request["state"] = "changed evidence"
    elif change == "allowed":
        request["ssdm"]["policy_restriction"]["q"] = ["a"]
    elif change == "no_policy":
        del request["ssdm"]["policy_restriction"]
    elif change == "incomplete_policy":
        request["questions"]["q2"] = deepcopy(request["questions"]["q"])
    else:
        request["ssdm"][change] = "changed"
    assert select_actions(request, response)["action"] == "defer"


def test_one_unavailable_question_defers_the_whole_request():
    request, _ = pair()
    request["questions"]["q2"] = deepcopy(request["questions"]["q"])
    request["ssdm"]["policy_restriction"]["q2"] = ["a", "b"]
    response = Decider(OracleBackend()).decide(request)
    del response["ssdm"]["policy_conditioned_probabilities"]["q2"]
    assert select_actions(request, response)["actions"] == {}
