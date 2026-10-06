"""Malformed inputs fail loudly; they never come back as a perfect calibration or a silent pairing.
(Codex review of PR #7.) Synthetic only; no suite needed."""

import math

import numpy as np
import pytest

from harness import bootstrap, calibration, mde
from harness.calibration import CalibrationInputError
from harness.score import Bench

ESTIMATORS = [
    lambda c, y: calibration.binned_ece(c, y),
    lambda c, y: calibration.debiased_ece(c, y),
    lambda c, y: calibration.smooth_ece(c, y),
]


@pytest.mark.parametrize("est", ESTIMATORS)
@pytest.mark.parametrize(
    "conf,correct",
    [
        ([], []),  # empty: unavailable, not 0
        ([-0.2, 0.5], [1, 0]),  # below 0
        ([1.2, 0.5], [1, 0]),  # above 1
        ([math.nan, 0.5], [1, 0]),  # NaN
        ([math.inf, 0.5], [1, 0]),  # inf
        ([0.7, 0.5], [0.5, 1]),  # non-binary correctness
        ([0.7, 0.5], [2, 1]),
        ([0.7, 0.5], [1]),  # length mismatch
        ([[0.7, 0.5]], [[1, 0]]),  # not 1-D
    ],
)
def test_estimators_reject_malformed_input(est, conf, correct):
    with pytest.raises(CalibrationInputError):
        est(conf, correct)


@pytest.mark.parametrize("w", [[1, -1], [0, 0], [math.nan, 1], [1], [math.inf, 1]])
def test_binned_ece_rejects_bad_weights(w):
    with pytest.raises(CalibrationInputError):
        calibration.binned_ece([0.7, 0.5], [1, 0], w)


def test_debiased_needs_two_fields():
    with pytest.raises(CalibrationInputError):
        calibration.debiased_ece([0.7], [1])


def test_empty_report_blocks_are_unavailable_not_zero():
    rep = calibration.report([])
    assert rep["overall"] == {"n": 0, "available": False}
    assert rep["board_protocol"] == {"n": 0, "available": False}


def _row(q):
    return {"_evaluation": {"run_id": "r1", "catalog_id": 25, "track": "t", "proxy_tokens": 10}, "questions": {"q1": q}, "expected": {"q1": "A"}}


CHOICE = {"type": "choice", "criteria": {"A": "a", "B": "b"}}


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "choice", "choice": "A", "probabilities": {"A": math.nan, "B": 0.5}},
        {"type": "choice", "choice": "A", "probabilities": {"A": 1.4, "B": -0.4}},
        {"type": "choice", "choice": "A", "probabilities": {"A": 0.9}},  # missing option
        {"type": "choice", "choice": "C", "probabilities": {"A": 0.5, "B": 0.5}},  # choice not offered
        {"type": "choice", "choice": "A", "probabilities": {"A": 0.3, "B": 0.3}},  # does not sum to 1
    ],
)
def test_fields_rejects_malformed_results(answer):
    results = {"r1": {"status": "ok", "response": {"answers": {"q1": answer}}}}
    with pytest.raises(CalibrationInputError):
        calibration.fields([_row(CHOICE)], results)


def test_fields_rejects_bad_noul_and_accepts_good():
    q = {"type": "noul", "criteria": {}}
    bad = {"r1": {"status": "ok", "response": {"answers": {"q1": {"type": "noul", "noul": 1.5}}}}}
    with pytest.raises(CalibrationInputError):
        calibration.fields([_row(q)], bad)
    good = {"r1": {"status": "ok", "response": {"answers": {"q1": {"type": "choice", "choice": "A", "probabilities": {"A": 0.8, "B": 0.2}}}}}}
    assert calibration.fields([_row(CHOICE)], good)[0][3:] == (0.8, 1.0)


def _bench(n, clusters=("a", "b")):
    S = np.ones((len(clusters), 2))
    return Bench(n, list(clusters), S, lambda T: (T[:, 0] / T[:, 1], T[:, 0] / T[:, 1]), "test")


def test_pairing_requires_identical_benchmark_sets():
    a = {1: _bench(1), 3: _bench(3)}
    b = {1: _bench(1), 3: _bench(3), 4: _bench(4)}  # extra key used to be ignored silently
    with pytest.raises(ValueError, match="identical benchmark sets"):
        bootstrap.replicate_skills([a, b], B=10)
    with pytest.raises(ValueError, match="identical benchmark sets"):
        bootstrap.replicate_skills([b, a], B=10)
    with pytest.raises(ValueError, match="cluster order"):
        bootstrap.replicate_skills([{1: _bench(1)}, {1: _bench(1, ("b", "a"))}], B=10)
    with pytest.raises(ValueError):
        bootstrap.replicate_skills([{}, {}], B=10)


@pytest.mark.parametrize("bad", [0, -1.0, math.nan, math.inf, None, "1", True])
def test_mde_helpers_validate_inputs(bad):
    for call in (
        lambda: mde.mde(bad),
        lambda: mde.rescale_se(bad, 10, 20),
        lambda: mde.rescale_se(1.0, bad, 20),
        lambda: mde.rescale_se(1.0, 10, bad),
        lambda: mde.required_n(bad, 10, 1.0),
        lambda: mde.required_n(1.0, bad, 1.0),
        lambda: mde.required_n(1.0, 10, bad),
        lambda: mde.table(bad, 10),
        lambda: mde.table(1.0, bad),
        lambda: mde.paired_accuracy_se(0.2, 0.1, bad),
    ):
        with pytest.raises((ValueError, TypeError)):
            call()


@pytest.mark.parametrize("alpha,power", [(0, 0.8), (1, 0.8), (0.05, 0), (0.05, 1.2)])
def test_mde_rejects_bad_alpha_power(alpha, power):
    with pytest.raises(ValueError):
        mde.mde(1.0, alpha, power)


def test_mde_accepts_numpy_scalars():
    assert mde.mde(np.float64(1.0)) == pytest.approx(mde.mde(1.0))
    assert mde.table(np.float64(1.0), np.int64(100))[2]["clusters"] == 100


def test_mde_report_defines_zero_se():
    r = mde.report(0.0, 100)
    assert r["available"] is False and r["se"] == 0.0 and "no MDE" in r["note"]
    ok = mde.report(np.float64(0.5), 100)
    assert ok["available"] is True and ok["table"][2]["mde"] == pytest.approx(mde.mde(0.5))
    for bad in (-0.1, math.nan, math.inf):
        with pytest.raises(ValueError):
            mde.report(bad, 100)
