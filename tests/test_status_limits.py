"""Explicit status codes and limits: no silent truncation, pruning or fallback."""

import pytest
from conftest import base_request

from decide import Decider
from decide.backends import OracleBackend


def run(req, **kw):
    backend = kw.pop("backend", None) or OracleBackend()
    return Decider(backend, **kw).decide(req)


def status(resp):
    s = resp["ssdm"]
    return s["status"], (s["error"] or {}).get("code")


def with_env(req, **env):
    req = dict(req)
    req["ssdm"] = {"schema": "ssdm.decide.v0", **env}
    return req


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda r: r.pop("model"), ("error", "invalid_request")),
        (lambda r: r.pop("state"), ("error", "invalid_request")),
        (lambda r: r.__setitem__("questions", {}), ("error", "invalid_request")),
        (lambda r: r["questions"]["status"].__setitem__("type", "multi"), ("error", "unknown_question_type")),
        (lambda r: r["questions"]["status"].__setitem__("criteria", {}), ("error", "empty_criteria")),
        (lambda r: r["questions"]["urgency"].__setitem__("criteria", []), ("error", "empty_criteria")),
        (lambda r: r["questions"]["urgency"].__setitem__("criteria", ["only"]), ("error", "invalid_request")),
        (lambda r: r["questions"]["large"].__setitem__("criteria", {"maybe": "x"}), ("error", "invalid_request")),
        (lambda r: r.__setitem__("images", ["img"]), ("unsupported", "media_unsupported")),
        (
            lambda r: r["questions"]["status"].__setitem__("criteria", {f"o{i}": "d" for i in range(256)}),
            ("unsupported", "too_many_options"),
        ),
    ],
)
def test_invalid_and_unsupported_requests(mutate, expected):
    req = base_request()
    mutate(req)
    resp = run(req)
    assert status(resp) == expected
    assert resp["answers"] == {}


def test_255_options_is_allowed_not_pruned():
    req = base_request()
    req["questions"]["status"]["criteria"] = {f"o{i:03d}": "d" for i in range(255)}
    resp = run(req)
    assert status(resp) == ("ok", None)
    assert len(resp["answers"]["status"]["probabilities"]) == 255
    assert len(resp["ssdm"]["served_probabilities"]["status"]) == 255


def test_non_object_request():
    assert status(run(["not", "a", "dict"])) == ("error", "invalid_request")


def test_max_choices_exceeded():
    resp = run(with_env(base_request(), limits={"max_choices": 2}))
    assert status(resp) == ("exhausted", "choices_exceeded")


def test_max_input_tokens_exceeded_no_truncation():
    req = base_request()
    req["state"] = "word " * 5000
    resp = run(with_env(req, limits={"max_input_tokens": 512}))
    assert status(resp) == ("exhausted", "input_tokens_exceeded")
    assert "no truncation" in resp["ssdm"]["error"]["message"]


def test_backend_context_limit_caps_request_limit():
    req = base_request()
    req["state"] = "word " * 5000
    resp = Decider(OracleBackend(), backend_max_input_tokens=100).decide(with_env(req, limits={"max_input_tokens": 10**6}))
    assert status(resp) == ("exhausted", "input_tokens_exceeded")


def test_max_inference_calls_exceeded():
    resp = run(with_env(base_request(), limits={"max_inference_calls": 1}))
    assert status(resp) == ("exhausted", "inference_calls_exceeded")


def test_deadline_exceeded_returns_no_partial_answers():
    resp = run(with_env(base_request(), limits={"deadline_ms": 1}), backend=OracleBackend(sleep_per_call_s=0.01))
    assert status(resp) == ("exhausted", "deadline_exceeded")
    assert resp["answers"] == {} and resp["ssdm"]["served_probabilities"] == {}


@pytest.mark.parametrize("bad", [0, -1, 1.5, "10", True])
def test_bad_limit_values(bad):
    assert status(run(with_env(base_request(), limits={"max_choices": bad}))) == ("error", "invalid_request")


def test_unknown_limit_and_envelope_keys_rejected():
    assert status(run(with_env(base_request(), limits={"max_tokens": 5}))) == ("error", "invalid_request")
    assert status(run(with_env(base_request(), surprise=1))) == ("error", "invalid_request")


def test_envelope_schema_version_unsupported():
    req = base_request()
    req["ssdm"] = {"schema": "ssdm.decide.v9"}
    assert status(run(req)) == ("unsupported", "envelope_version_unsupported")


def test_bad_audience():
    assert status(run(with_env(base_request(), audience="everyone"))) == ("error", "invalid_request")


def test_parent_revision_pinning_no_fallback():
    backend = OracleBackend()
    good = backend.revision()["revision_id"]
    assert status(run(with_env(base_request(), parent_revision=good), backend=backend)) == ("ok", None)
    resp = run(with_env(base_request(), parent_revision="qwen3.5:other@abc"), backend=backend)
    assert status(resp) == ("unsupported", "parent_revision_mismatch")


def test_backend_unsupported_and_crash_are_explicit():
    resp = run(base_request(), backend=OracleBackend(fail_with=("model_cannot_do_this", "nope")))
    assert status(resp) == ("unsupported", "model_cannot_do_this")

    class Boom(OracleBackend):
        def score(self, request, check_deadline):
            raise RuntimeError("CUDA out of memory")

    resp = run(base_request(), backend=Boom())
    assert status(resp) == ("error", "backend_error")
    assert "CUDA out of memory" in resp["ssdm"]["error"]["message"]


def test_backend_missing_option_is_error_not_pruned():
    class Drops(OracleBackend):
        def score(self, request, check_deadline):
            r = super().score(request, check_deadline)
            r.logits["status"].pop("draft")
            return r

    assert status(run(base_request(), backend=Drops())) == ("error", "backend_error")


def test_non_finite_logits_rejected():
    class Nan(OracleBackend):
        def score(self, request, check_deadline):
            r = super().score(request, check_deadline)
            r.logits["status"]["draft"] = float("nan")
            return r

    assert status(run(base_request(), backend=Nan())) == ("error", "non_finite_logits")


def test_policy_restriction_requires_flag():
    req = with_env(base_request(), policy_restriction={"status": ["paid", "overdue"]})
    assert status(run(req)) == ("unsupported", "policy_restriction_requires_v0_1_deltas")


def test_policy_conditioned_is_separate_and_never_substituted():
    plain = run(base_request(), enable_v0_1_deltas=True)
    req = with_env(base_request(), policy_restriction={"status": ["paid", "draft"]})
    resp = run(req, enable_v0_1_deltas=True)
    s = resp["ssdm"]
    assert status(resp) == ("ok", None)
    # served/raw/wire answers unchanged by the restriction (benchmark path)
    assert s["served_probabilities"] == plain["ssdm"]["served_probabilities"]
    assert resp["answers"] == plain["answers"]
    pcp = s["policy_conditioned_probabilities"]["status"]
    served = s["served_probabilities"]["status"]
    assert pcp["allowed"] == ["draft", "paid"]
    assert pcp["excluded_served_mass"] == pytest.approx(served["overdue"])
    assert sum(pcp["probabilities"].values()) == pytest.approx(1.0)
    assert "policy_conditioned_probabilities" not in plain["ssdm"]


@pytest.mark.parametrize(
    "restriction",
    [{"nope": ["paid"]}, {"status": []}, {"status": ["refunded"]}, {"status": "paid"}],
)
def test_policy_restriction_validation(restriction):
    req = with_env(base_request(), policy_restriction=restriction)
    assert status(run(req, enable_v0_1_deltas=True)) == ("error", "invalid_request")
