"""Injection-robust conformal sets (docs/guarantees/INJECTION_ROBUST_CONFORMAL.md), checked on synthetic G16 boxes.

Each decision has trusted-only logit intervals [lo_k, hi_k] (G16: b_k -/+ c||a_k||_1) and a clean logit inside the
box; the gold label is drawn from softmax(clean logits), so the clean model is calibrated. At test time an adversary
controls the untrusted text, i.e. may put the observed logits ANYWHERE in the box, and picks the point that excludes
the gold label from the set if one exists (it knows the gold label and q_hat: the strongest adversary).

Checks, averaged over many seeds (the guarantee is marginal over calibration and test draws):
* matched robust set  C_lo = {y : s_lo(y) <= Q(s_lo(cal))}           coverage >= 1 - alpha, and <= 1 - alpha + 1/(n+1)
* jevkit robust set   C_rob = {y : s_lo(y) <= Q(clean scores(cal))}   coverage >= 1 - alpha (dominance)
* pessimistic matched C_hi = {y : s_hi(y) <= Q(s_hi(cal))}            coverage >= 1 - alpha
* set inclusion, every point: C_lo subset C_rob subset C_lo|hi-cal (the theorem's ordering)
* naive set calibrated on clean point scores, evaluated on attacked point scores: FAILS under the same adversary
* mismatched pessimistic test score with clean calibration (s_hi vs clean): FAILS (wrong bound)
All sets except the naive one are functions of the intervals only; the adversary's choice cannot change them.
"""
from __future__ import annotations

import itertools
import math
import random

from jevkit.guarantees import (Scored, conformal_qhat, fit_conformal, injection_robust_report, prediction_set,
                               robust_min_score, robust_set, softmax)

ALPHA = 0.1
K = 4
N_CAL = 150
N_TEST = 100
SEEDS = 100


def s_lo(iv: dict, y: str) -> float:
    """Smallest nonconformity 1 - p_y over the box (y at hi, others at lo). Same as jevkit robust_min_score."""
    return robust_min_score(iv, y)


def s_hi(iv: dict, y: str) -> float:
    """Largest nonconformity over the box (y at lo, others at hi)."""
    z = {k: (v[0] if k == y else v[1]) for k, v in iv.items()}
    return 1.0 - softmax(z)[y]


def draw(rng: random.Random) -> dict:
    labels = [f"opt_{i}" for i in range(K)]
    b = {k: rng.gauss(0.0, 1.5) for k in labels}
    w = {k: rng.uniform(0.0, 1.5) for k in labels}  # c * ||a_k||_1
    iv = {k: (b[k] - w[k], b[k] + w[k]) for k in labels}
    clean = {k: rng.uniform(iv[k][0], iv[k][1]) for k in labels}  # benign untrusted text: some point in the box
    p = softmax(clean)
    u, acc, y = rng.random(), 0.0, labels[-1]
    for k in labels:
        acc += p[k]
        if u <= acc:
            y = k
            break
    return {"iv": iv, "clean": clean, "y": y}


CORNERS = list(itertools.product((0, 1), repeat=K))


def adversary(iv: dict, y: str, set_fn) -> dict:
    """Any point in the box; 1 - p_y is monotone in each coordinate, so the corners contain the worst case for
    excluding y. Returns observed logits that exclude y from set_fn(iv, logits) if any corner does, else the
    y-at-lo / others-at-hi corner."""
    keys = list(iv)
    worst = {k: (iv[k][0] if k == y else iv[k][1]) for k in keys}
    if y not in set_fn(iv, worst):
        return worst
    for c in CORNERS:
        z = {k: iv[k][bit] for k, bit in zip(keys, c)}
        if y not in set_fn(iv, z):
            return z
    return worst


def run_seed(seed: int) -> dict:
    rng = random.Random(f"inj|{seed}")
    cal = [draw(rng) for _ in range(N_CAL)]
    test = [draw(rng) for _ in range(N_TEST)]
    q_clean = conformal_qhat([1.0 - softmax(r["clean"])[r["y"]] for r in cal], ALPHA)
    q_lo = conformal_qhat([s_lo(r["iv"], r["y"]) for r in cal], ALPHA)
    q_hi = conformal_qhat([s_hi(r["iv"], r["y"]) for r in cal], ALPHA)
    # jevkit's own fit on the clean scores must give the same q_hat
    fit = fit_conformal([Scored(f"c{i}", "choice", softmax(r["clean"]), r["y"]) for i, r in enumerate(cal)],
                        ALPHA, mondrian=False)
    assert fit["qhat"]["all"] == q_clean

    sets = {
        "matched_lo": lambda iv, z: {y for y in iv if s_lo(iv, y) <= q_lo},
        "jevkit_clean_cal": lambda iv, z: set(robust_set(iv, q_clean)),
        "lo_test_hi_cal": lambda iv, z: {y for y in iv if s_lo(iv, y) <= q_hi},
        "matched_hi": lambda iv, z: {y for y in iv if s_hi(iv, y) <= q_hi},
        "WRONG_hi_test_clean_cal": lambda iv, z: {y for y in iv if s_hi(iv, y) <= q_clean},
        "naive_point": lambda iv, z: set(prediction_set(softmax(z), q_clean)),
    }
    cov = {k: 0 for k in sets}
    size = {k: 0 for k in sets}
    clean_cov = clean_size = 0
    for r in test:
        iv, y = r["iv"], r["y"]
        clean_set = prediction_set(softmax(r["clean"]), q_clean)
        clean_cov += y in clean_set
        clean_size += len(clean_set)
        got = {}
        z_rand = {k: rng.uniform(a, b) for k, (a, b) in iv.items()}
        for name, fn in sets.items():
            if name == "naive_point":
                z = adversary(iv, y, fn)
                assert all(iv[k][0] <= v <= iv[k][1] for k, v in z.items())
                s = fn(iv, z)
            else:  # interval-only sets: the adversary's point is irrelevant, bit for bit
                s = fn(iv, None)
                assert s == fn(iv, r["clean"]) == fn(iv, z_rand) == fn(iv, adversary(iv, y, sets["naive_point"]))
            got[name] = s
            cov[name] += y in s
            size[name] += len(s)
        assert got["matched_lo"] <= got["jevkit_clean_cal"] <= got["lo_test_hi_cal"]
        assert set(prediction_set(softmax(r["clean"]), q_clean)) <= got["jevkit_clean_cal"]
    out = {f"cov_{k}": v / N_TEST for k, v in cov.items()}
    out.update({f"size_{k}": v / N_TEST for k, v in size.items()})
    out["cov_clean_no_attack"] = clean_cov / N_TEST
    out["size_clean_no_attack"] = clean_size / N_TEST
    return out


def _mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows)


_ROWS: list[dict] | None = None


def rows() -> list[dict]:
    global _ROWS
    if _ROWS is None:
        _ROWS = [run_seed(s) for s in range(SEEDS)]
    return _ROWS


# per-seed coverage sd is about 0.04 here, so the mean over 100 seeds has SE about 0.004; slack = 3 SE
SLACK = 0.012
TARGET = 1 - ALPHA


def test_robust_sets_cover_under_worst_case_injection():
    r = rows()
    for name in ("matched_lo", "jevkit_clean_cal", "lo_test_hi_cal", "matched_hi"):
        assert _mean(r, f"cov_{name}") >= TARGET - SLACK, (name, _mean(r, f"cov_{name}"))
    # matched calibration is exact split conformal on a continuous score: no over-coverage beyond 1/(n+1)
    assert _mean(r, "cov_matched_lo") <= TARGET + 1 / (N_CAL + 1) + SLACK
    assert _mean(r, "cov_matched_hi") <= TARGET + 1 / (N_CAL + 1) + SLACK


def test_clean_conformal_is_valid_without_attack():
    assert _mean(rows(), "cov_clean_no_attack") >= TARGET - SLACK


def test_naive_point_set_fails_under_the_same_adversary():
    c = _mean(rows(), "cov_naive_point")
    assert c < TARGET - 0.10, c


def test_wrong_bound_fails():
    # pessimistic test score against clean calibration: s_hi >= clean score, so the threshold is too small
    c = _mean(rows(), "cov_WRONG_hi_test_clean_cal")
    assert c < TARGET - 0.05, c


def test_matched_calibration_is_tightest_valid_choice_with_lo_test_score():
    r = rows()
    a, b, c = (_mean(r, f"size_{k}") for k in ("matched_lo", "jevkit_clean_cal", "lo_test_hi_cal"))
    assert a <= b <= c
    assert a < b  # strictly smaller on average here: the jevkit (clean-calibrated) set over-covers


def test_jevkit_report_agrees_and_flags_out_of_box_attacks():
    rng = random.Random("report")
    cal = [draw(rng) for _ in range(200)]
    test = [draw(rng) for _ in range(50)]
    cal_s = [Scored(f"c{i}", "choice", softmax(r["clean"]), r["y"]) for i, r in enumerate(cal)]
    rows_in = [{"id": f"t{i}", "type": "choice", "label": r["y"], "intervals": {k: list(v) for k, v in r["iv"].items()},
                "clean_logits": r["clean"],
                "attacked_logits": [{k: (v[0] if k == r["y"] else v[1]) for k, v in r["iv"].items()}]}
               for i, r in enumerate(test)]
    rep = injection_robust_report(cal_s, rows_in, alpha=ALPHA, mondrian=False)
    assert rep["clean_logits_outside_box"] == 0 and rep["attacked_logits_outside_box"] == 0
    assert rep["clean_set_not_subset_of_robust"] == 0 and rep["act_set_attack_success"] == 0
    # a logit outside its box means wrong trust labels: the report must refuse to claim attack success 0
    rows_in[0]["attacked_logits"] = [{k: v[1] + 5.0 for k, v in rows_in[0]["intervals"].items()}]
    rep2 = injection_robust_report(cal_s, rows_in, alpha=ALPHA, mondrian=False)
    assert rep2["attacked_logits_outside_box"] == 1 and rep2["act_set_attack_success"] is None


def test_qhat_rank_matches_theorem():
    # the theorem's rank in exact rational arithmetic (a float ceil here would share any rounding bug with the code)
    from fractions import Fraction
    for n in list(range(1, 200)) + [999, 1000, 8599, 8600, 17233, 20000]:
        for a in (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3):
            q = (n + 1) * (1 - Fraction(str(a)))
            k = -((-q.numerator) // q.denominator)
            xs = [i / n for i in range(n)]
            want = math.inf if k > n else xs[k - 1]
            assert conformal_qhat(xs, a) == want, (n, a)


def test_robust_set_with_injected_calibration_text_still_covers():
    """A6 is not needed when calibration logits stay in their boxes (A5): any in-box calibration scores are >= s_lo,
    so q_hat(injected clean scores) >= q_lo and C_rob contains the matched set C_lo (Theorem A)."""
    covs = []
    for seed in range(40):
        rng = random.Random(f"injcal|{seed}")
        cal = [draw(rng) for _ in range(N_CAL)]
        test = [draw(rng) for _ in range(N_TEST)]
        q_lo = conformal_qhat([s_lo(r["iv"], r["y"]) for r in cal], ALPHA)
        # strongest calibration attack: every calibration row's logits at the corner that minimises its gold score
        q_inj = conformal_qhat([1.0 - softmax({k: (v[1] if k == r["y"] else v[0]) for k, v in r["iv"].items()})[r["y"]]
                                for r in cal], ALPHA)
        q_rand = conformal_qhat([1.0 - softmax({k: rng.uniform(*v) for k, v in r["iv"].items()})[r["y"]]
                                 for r in cal], ALPHA)
        assert q_inj == q_lo and q_lo <= q_rand
        covs.append(sum(r["y"] in robust_set(r["iv"], q_inj) for r in test) / N_TEST)
    assert sum(covs) / len(covs) >= TARGET - 0.02, sum(covs) / len(covs)


def test_print_summary(capsys):
    r = rows()
    keys = sorted(r[0])
    with capsys.disabled():
        print("\n[injection-conformal] means over %d seeds (alpha=%.2f, n_cal=%d, n_test=%d, K=%d)"
              % (SEEDS, ALPHA, N_CAL, N_TEST, K))
        for k in keys:
            print(f"  {k:32s} {_mean(r, k):.4f}")


# ------------------------------------------------------------------ jevkit matched mode on ssdm.g16sym.export.v1 records
def _record(r: dict, did: str, score: dict | None = None, gold: bool = True) -> dict:
    sc = score or r["clean"]
    rec = {"schema": "ssdm.g16sym.export.v1", "decision_id": did, "vetoed": [],
           "options": [{"id": k, "lo": v[0], "hi": v[1], "score": sc[k]} for k, v in r["iv"].items()]}
    if gold:
        rec["gold"] = r["y"]
    return rec


def test_matched_mode_covers_over_seeds_and_ignores_untrusted_scores():
    covs, sizes = [], []
    for seed in range(60):
        rng = random.Random(f"matched|{seed}")
        cal = [draw(rng) for _ in range(300)]
        test = [draw(rng) for _ in range(N_TEST)]
        rep = injection_robust_report([_record(r, f"c{i}") for i, r in enumerate(cal)],
                                      [_record(r, f"t{i}") for i, r in enumerate(test)], alpha=ALPHA, mode="matched",
                                      epsilon=0.10, delta=0.10)
        # the adversary moves every observed score to the worst in-box corner: nothing in the report may change
        att = [_record(r, f"t{i}", score={k: (v[0] if k == r["y"] else v[1]) for k, v in r["iv"].items()})
               for i, r in enumerate(test)]
        rep2 = injection_robust_report([_record(r, f"c{i}") for i, r in enumerate(cal)], att, alpha=ALPHA,
                                       mode="matched", epsilon=0.10, delta=0.10)
        assert rep == rep2
        # sets are exactly {y : s_lo(y) <= q_hat}, q_hat from the FIT half's matched scores
        assert rep["mode"] == "matched" and rep["n_with_gold"] == N_TEST
        covs.append(rep["coverage"])
        sizes.append(rep["mean_set_size"])
        if rep["ltt"]["lambda"] is not None:
            assert 0.0 < rep["act_rate"] <= rep["singleton_rate"]
    mean = sum(covs) / len(covs)
    n_fit = 150  # about half of 300
    assert TARGET - 0.015 <= mean <= TARGET + 1 / (n_fit + 1) + 0.015, mean


def test_matched_mode_rejects_malformed_records():
    import pytest
    rng = random.Random("bad")
    r = draw(rng)
    good = _record(r, "c0")
    test = [_record(draw(rng), "t0")]
    import copy as _copy

    def bad(mutate, where="cal"):
        rec = _copy.deepcopy(good)
        mutate(rec)
        cal, tst = ([rec], test) if where == "cal" else ([good], [rec])
        with pytest.raises(ValueError):
            injection_robust_report(cal, tst, alpha=ALPHA, mode="matched")

    bad(lambda x: x["options"][0].update(lo=x["options"][0]["hi"] + 1.0))          # lo > hi
    bad(lambda x: x["options"].append(dict(x["options"][0])))                      # duplicate option id
    bad(lambda x: x.pop("gold"))                                                    # calibration without gold
    bad(lambda x: x.update(schema="other"))                                         # wrong schema
    bad(lambda x: x["options"][0].update(score=x["options"][0]["hi"] + 1.0))       # score outside its box
    bad(lambda x: x["options"][0].update(lo=float("nan")))                          # non-finite
    bad(lambda x: x.update(gold="nope"))                                            # gold not an option / vetoed
    bad(lambda x: x.update(vetoed=[x["options"][0]["id"]]))                         # vetoed id still an option
    bad(lambda x: x.update(options=[]))                                             # empty options
    bad(lambda x: x.update(decision_id="c0"), where="test")                         # test overlaps calibration
    with pytest.raises(ValueError):
        injection_robust_report([good, _copy.deepcopy(good)], test, alpha=ALPHA, mode="matched")  # duplicate ids
    with pytest.raises(ValueError):
        injection_robust_report([good], test, alpha=ALPHA, mode="bogus")
    # a vetoed gold is legal (scores +inf); a test record may omit gold
    ok = _copy.deepcopy(good)
    ok["vetoed"], ok["gold"], ok["decision_id"] = ["gone"], "gone", "c1"
    rep = injection_robust_report([ok, good], [_record(r, "t9", gold=False)], alpha=ALPHA, mode="matched")
    assert rep["n_with_gold"] == 0 and rep["coverage"] is None


def test_matched_mode_cli(tmp_path):
    import json as _json

    from jevkit import cli
    rng = random.Random("cli")
    cal = [_record(draw(rng), f"c{i}") for i in range(200)]
    test = [_record(draw(rng), f"t{i}") for i in range(50)]
    pc, pt, po = tmp_path / "cal.jsonl", tmp_path / "test.jsonl", tmp_path / "rep.json"
    pc.write_text("\n".join(_json.dumps(r) for r in cal) + "\n", encoding="utf-8")
    pt.write_text("\n".join(_json.dumps(r) for r in test) + "\n", encoding="utf-8")
    assert cli.main(["guarantees", "injection", "--mode", "matched", "--cal-intervals", str(pc), "--test", str(pt),
                     "--alpha", "0.1", "--out", str(po)]) == 0
    rep = _json.loads(po.read_text(encoding="utf-8"))
    assert rep == _json.loads(_json.dumps(injection_robust_report(cal, test, alpha=0.1, mode="matched", delta=0.05)))
