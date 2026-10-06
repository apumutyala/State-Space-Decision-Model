"""Interface conformance: DECISION_LANGUAGE_V0 response fields + SystemOne wire semantics."""

import math

import pytest
from conftest import base_request

from decide import Decider, OptionCountTemperature, decide, set_default_backend
from decide.backends import OracleBackend
from decide.schema import canonical_options

STATUSES = {"ok", "unsupported", "exhausted", "error"}
REQUIRED_SSDM = {
    "schema",
    "request_id",
    "status",
    "error",
    "abstention",
    "raw_probabilities",
    "served_probabilities",
    "served_config",
    "diagnostics",
    "proposed_evidence_refs",
    "state_handle",
}


def check_response(req, resp):
    s = resp["ssdm"]
    expected_keys = {"model", "answers", "usage", "ssdm"} | ({"error"} if s["status"] != "ok" else set())
    assert set(resp) == expected_keys
    assert REQUIRED_SSDM <= set(s)
    assert s["status"] in STATUSES
    assert s["abstention"] == {"abstained": False, "reason": None, "mechanism": "not_implemented_v0"}
    assert s["state_handle"] is None
    assert "semantic_records" not in s  # never emitted before its gate
    d = s["diagnostics"]
    assert d["versions"]["adapter"] and d["versions"]["interface"]
    assert "latency_ms" in d and "inference_calls" in d and "revision_id" in d
    if s["status"] != "ok":
        assert resp["answers"] == {}
        assert s["error"]["code"] and s["error"]["message"]
        assert resp["error"] == {"status": s["status"], **s["error"]}  # legacy-visible failure signal
        return
    assert s["error"] is None
    assert set(resp["answers"]) == set(req["questions"])
    for qid, q in req["questions"].items():
        served = s["served_probabilities"][qid]
        raw = s["raw_probabilities"][qid]
        ids = [o for o, _ in canonical_options(q)]
        assert list(served) == ids and list(raw) == ids  # every supplied option, no pruning
        assert math.isclose(sum(served.values()), 1.0, abs_tol=1e-9)
        assert math.isclose(sum(raw.values()), 1.0, abs_tol=1e-9)
        a = resp["answers"][qid]
        assert a["type"] == q["type"]
        if q["type"] == "noul":
            assert set(a) == {"type", "noul"}
            assert a["noul"] == round(served["true"], 4)
        elif q["type"] == "choice":
            assert set(a) == {"type", "choice", "confidence", "probabilities"}
            assert a["choice"] == max(sorted(served), key=served.__getitem__)
            assert a["confidence"] == round(served[a["choice"]], 4)  # selected p
            assert list(a["probabilities"]) == [str(k) for k in q["criteria"]]
        else:
            assert set(a) == {"type", "score", "confidence", "legend", "probabilities"}
            expected = sum(i * served[str(i)] for i in range(len(q["criteria"])))
            assert a["score"] == round(expected, 4)  # raw expected ordinal index
            assert a["confidence"] == round(max(served.values()), 4)
            assert a["legend"] == {str(i): c for i, c in enumerate(q["criteria"])}
        ent = d["entropy_nats"][qid]
        assert 0 <= ent <= math.log(len(ids)) + 1e-9


def test_ok_response_conforms():
    req = base_request()
    resp = Decider(OracleBackend(labels={"status": "overdue"})).decide(req)
    assert resp["ssdm"]["status"] == "ok"
    check_response(req, resp)
    assert resp["answers"]["status"]["choice"] == "overdue"
    assert resp["model"] == "test"
    assert resp["usage"]["output_tokens"] == 0


def test_module_level_decide_and_state_handle_is_discarded():
    set_default_backend(OracleBackend())
    req = base_request()
    a = decide(req)
    b = decide(req, state_handle="opaque-handle-from-elsewhere")
    assert b["ssdm"]["diagnostics"]["state_handle_in"] == "discarded:stateless"
    assert b["ssdm"]["state_handle"] is None
    assert a["ssdm"]["served_probabilities"] == b["ssdm"]["served_probabilities"]


def test_identity_calibration_raw_equals_served():
    resp = Decider(OracleBackend()).decide(base_request())
    s = resp["ssdm"]
    assert s["raw_probabilities"] == s["served_probabilities"]
    assert s["served_config"]["calibration"]["status"] == "uncalibrated_identity"


def test_option_count_temperature_changes_served_not_raw_or_argmax():
    req = base_request()
    base = Decider(OracleBackend()).decide(req)
    cal = OptionCountTemperature(t0=1.0, t1=0.5)
    hot = Decider(OracleBackend(), calibration=cal).decide(req)
    assert hot["ssdm"]["raw_probabilities"] == base["ssdm"]["raw_probabilities"]
    for qid in req["questions"]:
        k = len(base["ssdm"]["raw_probabilities"][qid])
        assert hot["ssdm"]["served_config"]["temperature_per_question"][qid] == pytest.approx(cal.temperature(k))
        sb, sh = base["ssdm"]["served_probabilities"][qid], hot["ssdm"]["served_probabilities"][qid]
        assert max(sb, key=sb.get) == max(sh, key=sh.get)
        if cal.temperature(k) > 1:
            assert max(sh.values()) <= max(sb.values()) + 1e-12
    check_response(req, hot)


def test_revision_and_versions_recorded():
    resp = Decider(OracleBackend()).decide(base_request())
    d = resp["ssdm"]["diagnostics"]
    assert d["revision_id"].startswith("oracle:")
    assert d["model_revision"]["revision_id"] == d["revision_id"]
    assert d["versions"]["deltas"] == "disabled"
    assert len(d["request_fingerprint"]) == 64


def test_envelope_fields_echoed():
    req = base_request()
    req["ssdm"] = {
        "schema": "ssdm.decide.v0.1",
        "request_id": "r-1",
        "context_ref": "catalog-revision-7",
        "policy_ref": "public-read-policy-3",
        "registry_ref": "catalog-grammar-1",
        "audience": "public",
        "limits": {"max_choices": 8},
    }
    resp = Decider(OracleBackend()).decide(req)
    s = resp["ssdm"]
    assert s["request_id"] == "r-1"
    assert s["diagnostics"]["context_ref"] == "catalog-revision-7"
    assert s["diagnostics"]["limits_applied"]["max_questions"] == 8
    assert s["diagnostics"]["limits_applied"]["_source"]["max_questions"] == "request:max_choices_alias"
    assert s["diagnostics"]["request_fingerprint_algorithm"] == "ssdm.request_fingerprint.v0.1"
    assert s["schema"] == "ssdm.decide.v0.1"


def test_legacy_request_without_envelope_gets_pure_wire_answers():
    req = base_request()
    resp = Decider(OracleBackend()).decide(req)
    # legacy SystemOne clients read only model/answers/usage
    assert resp["answers"]["large"]["type"] == "noul"
    assert resp["ssdm"]["request_id"] is None


def test_attribution_only_with_flag_and_labelled():
    req = base_request()
    off = Decider(OracleBackend(emit_attribution=True)).decide(req)
    assert "attribution" not in off["ssdm"]
    on = Decider(OracleBackend(emit_attribution=True), enable_v0_1_deltas=True).decide(req)
    assert on["ssdm"]["attribution"]["label"] == "accounting attribution, not causal effect"
    assert on["answers"] == off["answers"]


def test_non_ok_response_conforms_and_has_legacy_error():
    req = base_request()
    req["questions"]["status"]["type"] = "multi"
    resp = Decider(OracleBackend()).decide(req)
    check_response(req, resp)
    assert resp["error"]["status"] == "error"
