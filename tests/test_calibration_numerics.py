"""Synthetic CPU regressions for exact feature-temperature identity and numeric refusal."""

import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from decide import Decider  # noqa: E402
from decide.calibration import (CalibrationNotServable, FeatureTemperature as Serving,
                                OptionCountTemperature, content_set_sha256,
                                disjointness_proof_sha256, softmax)  # noqa: E402
from train.calibration import FeatureTemperature as Training, features  # noqa: E402


CALIB_CONTENT = ("a" * 64, "b" * 64)
EVAL_CONTENT = ("c" * 64,)
FITTED_ON = content_set_sha256(CALIB_CONTENT)
DISJOINTNESS = (("eval", content_set_sha256(EVAL_CONTENT)),)
PROOF = disjointness_proof_sha256(FITTED_ON, dict(DISJOINTNESS))
PROOF_ARGS = dict(t_min=0.07, fitted_on=FITTED_ON, disjointness=DISJOINTNESS,
                  calib_content=CALIB_CONTENT, proof_sha256=PROOF)


class StubBackend:
    def __init__(self, evidence_tokens=10):
        self.evidence_tokens = evidence_tokens

    def revision(self):
        return {"revision_id": "synthetic-stub", "backend": "stub"}

    def plan(self, request):
        return SimpleNamespace(inference_calls=1, max_call_input_tokens=2, total_input_tokens=2)

    def score(self, request, check_deadline):
        check_deadline()
        return SimpleNamespace(logits={"q": {"a": 2.0, "b": 0.0}},
                               features={"q": {"evidence_tokens": self.evidence_tokens}},
                               inference_calls=1, total_input_tokens=2, extras={}, attribution=None)


def serve(calibration, evidence_tokens=10):
    decider = Decider(StubBackend(evidence_tokens), calibration=calibration)
    request = {"model": "synthetic", "state": "x",
               "questions": {"q": {"type": "choice", "instructions": "pick",
                                   "criteria": {"a": "A", "b": "B"}}},
               "ssdm": {"schema": "ssdm.decide.v0.1", "request_id": "synthetic-request",
                        "parent_revision": decider.revision()["revision_id"]}}
    return decider.decide(request)


def test_full_precision_weights_change_identity_and_served_probabilities():
    a = Serving(w=(0.8, 0.45, -0.2), **PROOF_ARGS)
    b = Serving(w=(0.8, 0.450000004, -0.2), **PROOF_ARGS)
    for calibration in (a, b):
        calibration.assert_servable()
        calibration.verify_disjoint("eval", EVAL_CONTENT)
    assert a.identity()["w"] != b.identity()["w"]
    assert a.identity()["id"] != b.identity()["id"]
    ta = a.temperature_for(7, {"evidence_tokens": 10**9})
    tb = b.temperature_for(7, {"evidence_tokens": 10**9})
    assert ta != tb
    assert softmax([2.0, 0.0, -1.0], ta) != softmax([2.0, 0.0, -1.0], tb)


def test_nonfinite_temperature_refuses_serving_and_training_apply_fit():
    huge = Serving(w=(0.8, 1e308, 0.0), **PROOF_ARGS)
    huge.assert_servable()  # the content proof is valid; derived temperature is not
    response = serve(huge)
    assert response["ssdm"]["status"] == "error"
    assert response["ssdm"]["error"]["code"] == "calibration_temperature_invalid"
    assert response["answers"] == {} and response["ssdm"]["served_probabilities"] == {}
    assert "1e+308" not in response["ssdm"]["error"]["message"]
    with pytest.raises(FloatingPointError):
        huge.temperature_for(2, {"evidence_tokens": 10})

    training = Training(t_min=0.07, w=[0.8, 1e308, 0.0])
    phi = features([10], [2])
    lp = torch.log_softmax(torch.tensor([[2.0, 0.0]], dtype=torch.float64), dim=-1)
    with pytest.raises(FloatingPointError):
        training.apply(lp, phi)
    bad_phi = torch.tensor([[1.0, float("inf"), math.log(2)]], dtype=torch.float64)
    with pytest.raises(FloatingPointError):
        Training.fit(lp, torch.tensor([0]), bad_phi, ["one"], steps=1)
    with pytest.raises(ValueError):
        softmax([2.0, 0.0], float("inf"))
    plain = serve(OptionCountTemperature(t0=float("inf")))
    assert plain["ssdm"]["error"]["code"] == "calibration_temperature_invalid"
    nonpositive = serve(OptionCountTemperature(t0=-1.0, t_min=-0.2))
    assert nonpositive["ssdm"]["error"]["code"] == "calibration_temperature_invalid"


def test_custom_minimum_defaults_to_actual_identity_in_both_paths():
    serving = Serving(t_min=0.07)
    training = Training(t_min=0.07)
    assert serving.is_identity and serving.identity()["status"] == "uncalibrated_identity"
    assert serving.temperature_for(7, {"evidence_tokens": 10**9}) == pytest.approx(1.0)
    assert training.temperature(features([0, 10**6], [2, 7])).tolist() == pytest.approx([1.0, 1.0])
    assert serving.identity()["id"] == training.identity()["id"]
    response = serve(serving, 0)
    assert response["ssdm"]["status"] == "ok"
    assert response["ssdm"]["served_probabilities"] == response["ssdm"]["raw_probabilities"]

    wrong_default = Serving(t_min=0.07, w=(math.log(math.expm1(0.95)), 0.0, 0.0))
    assert not wrong_default.is_identity and wrong_default.identity()["status"] == "unfitted_manual"
    with pytest.raises(CalibrationNotServable):
        wrong_default.assert_servable()
    for cls in (Serving, Training):
        with pytest.raises(ValueError):
            cls(t_min=1.1)
    option_count = OptionCountTemperature(t_min=1.1)
    assert not option_count.is_identity
    assert option_count.identity()["status"] == "unfitted_manual"


def test_explicit_fitted_weights_can_use_a_larger_positive_minimum():
    serving = Serving(w=(0.2, 0.0, 0.0), t_min=1.1, fitted_on=FITTED_ON,
                      disjointness=DISJOINTNESS, calib_content=CALIB_CONTENT, proof_sha256=PROOF)
    training = Training(w=[0.2, 0.0, 0.0], t_min=1.1)
    serving.assert_servable()
    assert serving.temperature_for(2, {"evidence_tokens": 0}) == pytest.approx(
        training.temperature(features([0], [2])).item())
