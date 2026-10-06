"""Regression tests for Codex's PR #6 review blockers: error boundary, message sanitisation,
work accounting."""

import json

import pytest
from conftest import base_request

from decide import Decider
from decide.backends import OracleBackend, ScoreResult
from decide.backends.base import BackendUnsupported

SENTINEL = "PRIVATE-SENTINEL-7f3a9c"


def run(backend):
    return Decider(backend).decide(base_request())


def status(resp):
    s = resp["ssdm"]
    return s["status"], (s["error"] or {}).get("code")


def assert_no_sentinel(resp):
    assert SENTINEL not in json.dumps(resp)


# (1) error boundary covers the success path ------------------------------------------------
class EmptyResult(OracleBackend):
    def score(self, request, check_deadline):
        return ScoreResult({}, 1, 1)  # Codex's reproducer: uncaught KeyError before the fix


class MissingOption(OracleBackend):
    def score(self, request, check_deadline):
        r = super().score(request, check_deadline)
        del r.logits["status"]["draft"]
        return r


class NotADict(OracleBackend):
    def score(self, request, check_deadline):
        r = super().score(request, check_deadline)
        r.logits["status"] = [1.0, 2.0, 3.0]
        return r


class StringLogit(OracleBackend):
    def score(self, request, check_deadline):
        r = super().score(request, check_deadline)
        r.logits["status"]["draft"] = "high"
        return r


class BadTokens(OracleBackend):
    def score(self, request, check_deadline):
        r = super().score(request, check_deadline)
        r.total_input_tokens = None
        return r


class NotAResult(OracleBackend):
    def score(self, request, check_deadline):
        return None


@pytest.mark.parametrize("backend", [EmptyResult(), MissingOption(), NotADict(), BadTokens(), NotAResult()])
def test_malformed_backend_result_is_explicit_error_not_exception(backend):
    resp = run(backend)  # must not raise
    assert status(resp) == ("error", "backend_error")
    assert resp["answers"] == {} and resp["error"]["code"] == "backend_error"


def test_non_numeric_logit_is_non_finite_logits():
    assert status(run(StringLogit())) == ("error", "non_finite_logits")


def test_exception_while_building_success_response_is_caught():
    class BadCalibration:
        def temperature(self, k):
            raise ZeroDivisionError(SENTINEL)

        def identity(self):
            return {"id": "x"}

    resp = Decider(OracleBackend(), calibration=BadCalibration()).decide(base_request())
    assert status(resp) == ("error", "backend_error")  # failure after score() was entered
    assert_no_sentinel(resp)


# (2) no raw exception / backend text in responses ---------------------------------------------
class Leaky(OracleBackend):
    def score(self, request, check_deadline):
        check_deadline()
        raise RuntimeError(f"db password {SENTINEL}")


class LeakyPlan(OracleBackend):
    def plan(self, request):
        raise RuntimeError(SENTINEL)


class LeakyUnsupported(OracleBackend):
    def plan(self, request):
        raise BackendUnsupported("model_cannot_do_this", f"secret {SENTINEL}")


class LeakyCode(OracleBackend):
    def plan(self, request):
        raise BackendUnsupported(f"Code With {SENTINEL}!", "x")


@pytest.mark.parametrize("backend,expected", [
    (Leaky(), ("error", "backend_error")),
    (LeakyPlan(), ("error", "internal_error")),
    (LeakyUnsupported(), ("unsupported", "model_cannot_do_this")),
    (LeakyCode(), ("unsupported", "backend_unsupported")),
])
def test_exception_text_never_returned(backend, expected):
    resp = run(backend)
    assert status(resp) == expected
    assert_no_sentinel(resp)
    assert resp["error"]["message"] == resp["ssdm"]["error"]["message"]
    assert "withheld" in resp["error"]["message"]


# (3) every attempted call is counted; unknown work is null, not zero ---------------------------
class FailsAfterTwoCalls(OracleBackend):
    def score(self, request, check_deadline):
        check_deadline()
        check_deadline()
        raise RuntimeError("boom")


class FailsBeforeAnyCallReport(OracleBackend):
    def score(self, request, check_deadline):
        raise RuntimeError("boom")


def calls(resp):
    d = resp["ssdm"]["diagnostics"]
    return d["inference_calls"], d["inference_calls_accounting"]


def test_failed_score_counts_started_calls():
    assert calls(run(FailsAfterTwoCalls()))[0] == 2


def test_failed_score_without_any_report_is_unknown_not_zero():
    n, note = calls(run(FailsBeforeAnyCallReport()))
    assert n is None and "unknown" in note


def test_malformed_result_counts_reported_calls():
    assert calls(run(EmptyResult()))[0] == 1


def test_rejection_before_score_counts_zero():
    req = base_request()
    req["ssdm"] = {"schema": "ssdm.decide.v0.1", "limits": {"max_questions": 1}}
    resp = Decider(OracleBackend()).decide(req)
    assert status(resp) == ("exhausted", "questions_exceeded")
    assert calls(resp) == (0, "no inference call attempted")


def test_deadline_counts_calls_started_before_expiry():
    req = base_request()
    req["ssdm"] = {"schema": "ssdm.decide.v0.1", "limits": {"deadline_ms": 5}}
    resp = Decider(OracleBackend(sleep_per_call_s=0.004)).decide(req)
    assert status(resp) == ("exhausted", "deadline_exceeded")
    n, _ = calls(resp)
    assert n is not None and 1 <= n <= 3


def test_ok_response_counts_match_backend():
    resp = run(OracleBackend())
    assert status(resp) == ("ok", None)
    assert resp["ssdm"]["diagnostics"]["inference_calls"] == 3  # one per question
