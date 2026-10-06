"""Regression: max_inference_calls is enforced at run time, not only against the plan.

Track L finding (session log): at 42b8bb2 a backend whose plan under-reports its
inference calls could exceed the cap, because the callback only checked the deadline.
"""

from conftest import base_request

from decide import Decider
from decide.backends import OracleBackend
from decide.backends.base import Plan


def _env(req, **limits):
    req = dict(req)
    req["ssdm"] = {"schema": "ssdm.decide.v0.1", "limits": limits}
    return req


class Underreports(OracleBackend):
    """Plan claims 1 call; score() calls the callback once per question (more than 1)."""

    def plan(self, request):
        p = super().plan(request)
        return Plan(inference_calls=1, max_call_input_tokens=p.max_call_input_tokens,
                    total_input_tokens=p.max_call_input_tokens)


class SilentExtraCalls(OracleBackend):
    """Plan claims 1 call, never uses the callback beyond once, reports more calls."""

    def plan(self, request):
        p = super().plan(request)
        return Plan(inference_calls=1, max_call_input_tokens=p.max_call_input_tokens,
                    total_input_tokens=p.max_call_input_tokens)

    def score(self, request, check_deadline):
        r = super().score(request, lambda: None)
        check_deadline()
        r.inference_calls = len(request["questions"]) + 5
        return r


def _status(resp):
    s = resp["ssdm"]
    return s["status"], (s["error"] or {}).get("code")


def test_base_request_has_multiple_questions():
    assert len(base_request()["questions"]) > 1


def test_underreported_plan_is_stopped_by_callback():
    resp = Decider(Underreports()).decide(_env(base_request(), max_inference_calls=1))
    assert _status(resp) == ("exhausted", "inference_calls_exceeded")
    assert resp["answers"] == {} and resp["ssdm"]["served_probabilities"] == {}
    assert resp["ssdm"]["diagnostics"]["inference_calls"] == 1  # stopped before call 2 started


def test_reported_calls_over_cap_rejected():
    resp = Decider(SilentExtraCalls()).decide(_env(base_request(), max_inference_calls=1))
    assert _status(resp) == ("exhausted", "inference_calls_exceeded")
    assert resp["answers"] == {}


def test_within_budget_still_ok():
    n = len(base_request()["questions"])
    resp = Decider(Underreports()).decide(_env(base_request(), max_inference_calls=n))
    assert _status(resp)[0] == "ok"
