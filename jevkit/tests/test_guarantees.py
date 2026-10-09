import math
import random

import pytest

from decide import Decider
from decide.backends import OracleBackend
from jevkit import guarantees as G
from jevkit import systemone as S1

from ._fixtures import choice, noul, score
from .test_perturb import content_model, first_listed_model


# ------------------------------------------------------------------------------------------- (a) invariance
def test_invariance_exact_for_content_model_all_perms_and_relabels():
    rep = G.invariance_report([choice("a", k=4), choice("b", k=6), noul("n"), score("s")], content_model)
    assert rep["n_checked"] == 3 and rep["invariant_on_all_checked"] and rep["flips"] == 0
    row = next(r for r in rep["rows"] if r["id"] == "a")
    assert row["variants"] == math.factorial(4) - 1 + 3  # every other order of 4 options + 3 relabelings
    assert next(r for r in rep["rows"] if r["id"] == "s")["applicable"] is False


def test_invariance_detects_position_bias():
    r = G.invariance_check(choice("a", k=3), first_listed_model)
    assert not r["exact"] and r["flips"] > 0 and r["max_abs_dp"] == pytest.approx(0.55)


def test_decider_canonical_order_is_reorder_invariant():
    ad = S1.SystemOneAdapter(Decider(OracleBackend()))
    r = G.invariance_check(choice("a", k=4), lambda it: ad.answer_item(it)["answer"], relabels=0, tol=1e-4)
    assert r["exact"] and r["flips"] == 0


# ------------------------------------------------------------------------------------------- (b) conformal + LTT
def test_qhat_rank_and_infinite_case():
    s = [0.1 * i for i in range(10)]
    assert G.conformal_qhat(s, 0.2) == pytest.approx(s[math.ceil(11 * 0.8) - 1])
    assert G.conformal_qhat(s[:3], 0.1) == math.inf
    assert G.prediction_set({"a": 0.7, "b": 0.2, "c": 0.1}, 0.8) == ["a", "b"]


def _synthetic(n, seed, k=4, temp=1.0):
    """Calibrated-by-construction decisions: the label is drawn from the stated distribution."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        logits = [rng.gauss(0, 2.0) for _ in range(k)]
        z = sum(math.exp(x / temp) for x in logits)
        probs = {f"o{j}": math.exp(x / temp) / z for j, x in enumerate(logits)}
        r, acc, lab = rng.random(), 0.0, None
        for o, p in probs.items():
            acc += p
            if r <= acc:
                lab = o
                break
        out.append(G.Scored(f"x{i}", "choice" if i % 2 else "noul", probs, lab or "o0"))
    return out


def test_split_conformal_coverage_holds():
    cal, test = _synthetic(2000, 1), _synthetic(4000, 2)
    rep = G.conformal_report(cal, test, alpha=0.1, epsilon=0.1, delta=0.05)
    assert rep["sets"]["all"]["coverage"] >= 0.88
    assert set(rep["sets"]["per_type"]) == {"choice", "noul"}


def _true_acted_risk(fit, lam, pool):
    """Calibrated by construction, so E[error | act] = mean over acted of (1 - p of the acted label): no label noise."""
    acted = [(r, y) for r in pool for ok, y in [G._acts(r, fit, lam)] if ok]
    return sum(1 - r.probs[y] for r, y in acted) / len(acted) if acted else 0.0


def test_ltt_bounds_acted_error():
    """P(true acted-on risk > epsilon) <= delta: count violations over independent calibration draws."""
    pool = _synthetic(20000, 99)
    eps, delta, trials, bad = 0.15, 0.1, 10, 0
    for t in range(trials):
        cal = _synthetic(2000, 1000 + t)
        fit = G.fit_conformal(cal, 0.1)
        lt = G.ltt_select(cal, fit, epsilon=eps, delta=delta)
        assert lt["lambda"] is not None and lt["accepted_error_ucb"] <= eps + 1e-9
        bad += _true_acted_risk(fit, lt["lambda"], pool) > eps
        sel = G.selective_eval(pool[:2000], fit, lt["lambda"])
        assert 0 < sel["act_rate"] < 1
    assert bad <= 3  # expected <= delta * trials = 1 (seeds fixed, so the test is deterministic)
    fs = G.ltt_select(_synthetic(2000, 7), G.fit_conformal(_synthetic(2000, 7), 0.1), eps, delta, method="fixed_sequence")
    assert fs["method"] == "fixed_sequence" and fs["test_level"] == delta


def test_binomial_helpers():
    assert G.binom_cdf(2, 5, 0.3) == pytest.approx(sum(math.comb(5, i) * 0.3**i * 0.7 ** (5 - i) for i in range(3)))
    u = G.clopper_pearson_upper(1, 50, 0.05)
    assert G.binom_cdf(1, 50, u) == pytest.approx(0.05, abs=1e-6) and 0.02 < u < 0.15
    assert G.clopper_pearson_upper(0, 0, 0.05) == 1.0


def test_scored_from_answers_drops_invalid_and_label_only():
    its = [choice("a", k=2), choice("b", k=2), noul("n")]
    ans = {"a": {"type": "choice", "probabilities": {"o0": 0.9, "o1": 0.1}}, "b": {"type": "choice", "choice": "o0"},
           "n": {"type": "noul", "noul": 0.3}}
    sc = G.scored_from_answers(its, ans)
    assert [s.id for s in sc] == ["a", "n"] and sc[1].label == "yes"


# ------------------------------------------------------------------------------------------- (c) injection-robust
def test_degenerate_intervals_give_the_ordinary_set():
    logits = {"a": 2.0, "b": 0.5, "c": -1.0}
    iv = {k: (v, v) for k, v in logits.items()}
    for q in (0.2, 0.6, 0.9):
        assert G.robust_set(iv, q) == G.prediction_set(G.softmax(logits), q)


def test_robust_set_contains_clean_set_for_any_point_in_box():
    rng = random.Random(0)
    for _ in range(200):
        b = {k: rng.gauss(0, 2) for k in "abcd"}
        w = {k: abs(rng.gauss(0, 1)) for k in "abcd"}
        iv = {k: (b[k] - w[k], b[k] + w[k]) for k in "abcd"}
        q = rng.random()
        rob = set(G.robust_set(iv, q))
        clean = {k: rng.uniform(*iv[k]) for k in "abcd"}  # any untrusted text = any point in the box
        assert set(G.prediction_set(G.softmax(clean), q)) <= rob


def test_independence():
    assert G.independent_label({"a": (3, 4), "b": (0, 2.9)}) == "a"
    assert G.independent_label({"a": (1, 4), "b": (0, 2)}) is None
    with pytest.raises(ValueError):
        G.robust_set({"a": (1, 0), "b": (0, 1)}, 0.5)


def test_injection_report_coverage_and_box_checks():
    cal = _synthetic(1000, 5, k=2)
    cal = [G.Scored(c.id, "choice", {"a": c.probs["o0"], "b": c.probs["o1"]}, "a" if c.label == "o0" else "b")
           for c in cal]
    rng = random.Random(1)
    test = []
    for i in range(300):
        b = {"a": rng.gauss(0, 2), "b": rng.gauss(0, 2)}
        clean = {k: v + rng.uniform(-0.3, 0.3) for k, v in b.items()}
        pc = G.softmax(clean)
        lab = "a" if rng.random() < pc["a"] else "b"
        test.append({"id": f"t{i}", "type": "choice", "label": lab,
                     "intervals": {k: [v - 0.3, v + 0.3] for k, v in b.items()}, "clean_logits": clean,
                     "attacked_logits": [{k: v + rng.uniform(-0.3, 0.3) for k, v in b.items()}]})
    rep = G.injection_robust_report(cal, test, alpha=0.1)
    assert rep["coverage"] >= 0.85 and rep["clean_set_not_subset_of_robust"] == 0
    assert rep["clean_logits_outside_box"] == 0 and rep["attacked_logits_outside_box"] == 0
    assert rep["act_set_attack_success"] == 0 and 0 <= rep["independence_rate"] <= 1
    test[0]["attacked_logits"] = [{"a": 99.0, "b": 0.0}]  # a wrong trust label: the guarantee is void, and says so
    rep = G.injection_robust_report(cal, test, alpha=0.1)
    assert rep["attacked_logits_outside_box"] == 1 and rep["act_set_attack_success"] is None
