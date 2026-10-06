"""Bootstrap, calibration and MDE behave as specified (synthetic data; DEV where noted)."""

import numpy as np
import pytest

from harness import bootstrap, calibration, mde, predictors, score


def test_binned_ece_known_value():
    conf = [0.95] * 100 + [0.55] * 100
    cor = [1] * 80 + [0] * 20 + [1] * 55 + [0] * 45
    ece, rel = calibration.binned_ece(conf, cor)
    assert ece == pytest.approx(0.5 * 0.15 + 0.5 * 0.0)
    assert len(rel) == 2


def test_debiased_and_smooth_ece_near_zero_when_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.3, 1.0, 20000)
    y = (rng.uniform(size=p.size) < p).astype(float)
    assert calibration.debiased_ece(p, y)["debiased_l2"] < 0.02
    plain, _ = calibration.binned_ece(p, y)
    assert calibration.smooth_ece(p, y)["smece"] < 0.03
    assert plain < 0.03


def test_ece_detects_overconfidence():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.5, 1.0, 20000)
    y = (rng.uniform(size=p.size) < p - 0.1).astype(float)
    assert calibration.debiased_ece(p, y)["debiased_l2"] == pytest.approx(0.1, abs=0.02)
    assert calibration.smooth_ece(p, y)["smece"] == pytest.approx(0.1, abs=0.02)


def test_debiased_beats_plain_on_small_bins():
    rng = np.random.default_rng(2)
    plain, deb = [], []
    for _ in range(200):
        p = rng.uniform(0.5, 1.0, 150)
        y = (rng.uniform(size=150) < p).astype(float)
        plain.append(calibration.binned_ece(p, y, bins=15)[0])
        deb.append(calibration.debiased_ece(p, y)["debiased_l2"])
    assert np.mean(deb) < np.mean(plain)


def test_mde_formulas():
    assert mde.mde(1.0) == pytest.approx(1.959964 + 0.841621, abs=1e-5)
    assert mde.rescale_se(2.0, 100, 400) == pytest.approx(1.0)
    n = mde.required_n(1.0, 100, mde.mde(1.0))
    assert n == pytest.approx(100)


def test_identical_systems_have_zero_width_difference(dev_rows):
    res = predictors.predict(dev_rows, "noisy", seed=5, acc=0.6)
    b = score.decompose(dev_rows, res)
    out = bootstrap.compare(b, b, B=200, seed=0)
    d = out["diff"]["index"]
    assert d["point"] == d["lo"] == d["hi"] == 0.0


def test_paired_bootstrap_detects_known_shift_and_ci_brackets_point(dev_rows):
    base = score.decompose(dev_rows, predictors.predict(dev_rows, "noisy", seed=6, acc=0.55))
    cand = score.decompose(dev_rows, predictors.predict(dev_rows, "noisy", seed=7, acc=0.60))
    out = bootstrap.compare(cand, base, B=400, seed=1)
    d = out["diff"]["index"]
    assert d["lo"] < d["point"] < d["hi"]
    assert d["excludes_zero"] and d["point"] > 0
    # paired SE must be smaller than the unpaired combination of the two marginal SEs
    unpaired = (out["candidate"]["index"]["se"] ** 2 + out["baseline"]["index"]["se"] ** 2) ** 0.5
    assert d["se"] < unpaired * 1.05


def test_point_column_equals_official(dev_rows):
    res = predictors.predict(dev_rows, "noisy", seed=8, acc=0.5, status_drop=0.02)
    b = score.decompose(dev_rows, res)
    rep = bootstrap.replicate_skills([b], B=50, seed=0)[0]
    off = score.official(dev_rows, res)
    for n, (raw, skill) in rep.items():
        assert skill[0] == pytest.approx(off["values"][n]["skill"], abs=1e-4)


def test_smooth_ece_handles_boundary_mass():
    # one-hot confidences (all 1.0) at 60% accuracy: every estimator must give 0.4
    conf = np.ones(5000)
    y = np.r_[np.ones(3000), np.zeros(2000)]
    assert calibration.binned_ece(conf, y)[0] == pytest.approx(0.4)
    assert calibration.smooth_ece(conf, y)["smece"] == pytest.approx(0.4, abs=0.01)


def test_case_clusters_are_coarser_and_keep_the_point_estimate(dev_rows):
    res = predictors.predict(dev_rows, "noisy", seed=9, acc=0.6)
    g = score.decompose(dev_rows, res, clusters="group")
    c = score.decompose(dev_rows, res, clusters="case")
    assert score.point(g)["index"] == pytest.approx(score.point(c)["index"], abs=1e-9)
    assert sum(len(b.clusters) for b in c.values()) < sum(len(b.clusters) for b in g.values())
    # POP909 and cfcolor: never finer than the kit's song / user clustering
    for n, key in ((22, "song_id"), (23, "user_id")):
        if n in c:
            assert len(c[n].clusters) <= len({r["metadata"][key] for r in dev_rows if r["_evaluation"]["catalog_id"] == n})
