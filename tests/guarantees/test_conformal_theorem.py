"""The jevkit conformal / LTT code against the theorems in docs/guarantees/CONFORMAL.md (synthetic, pure Python)."""
from __future__ import annotations

import hashlib
import math
import random

from jevkit.guarantees import Scored, evaluate_sets, fit_conformal, ltt_select, selective_eval, split_calibration

LABELS = ["a", "b", "c", "d"]


def _draw(rng: random.Random, i: int, cluster: str | None = None) -> Scored:
    z = [rng.gauss(0, 1.2) for _ in LABELS]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    p = {k: v / s for k, v in zip(LABELS, e)}
    u, acc, y = rng.random(), 0.0, LABELS[-1]
    for k in LABELS:
        acc += p[k]
        if u <= acc:
            y = k
            break
    return Scored(f"{cluster or 'r' + str(i)}#{i}", "choice", p, y)


def test_split_conformal_coverage_between_the_theorem_bounds():
    alpha, n, seeds = 0.1, 99, 120
    covs = []
    for s in range(seeds):
        rng = random.Random(f"cov|{s}")
        cal = [_draw(rng, i) for i in range(n)]
        test = [_draw(rng, 10_000 + i) for i in range(100)]
        covs.append(evaluate_sets(test, fit_conformal(cal, alpha, mondrian=False))["all"]["coverage"])
    mean = sum(covs) / seeds
    # E[coverage] in [1 - alpha, 1 - alpha + 1/(n+1)] for continuous scores; SE of the mean here is about 0.004
    assert 1 - alpha - 0.012 <= mean <= 1 - alpha + 1 / (n + 1) + 0.012, mean


def test_ltt_certificate_holds_across_seeds():
    eps, delta, seeds, violations = 0.10, 0.10, 40, 0
    rng_pop = random.Random("pop")
    pop = [_draw(rng_pop, 50_000 + i) for i in range(4000)]
    for s in range(seeds):
        rng = random.Random(f"ltt|{s}")
        cal = [_draw(rng, i) for i in range(800)]
        fit_half, ltt_half, _ = split_calibration(cal, seed=s)
        fit = fit_conformal(fit_half, 0.2, mondrian=False)
        sel = ltt_select(ltt_half, fit, eps, delta)
        if sel["lambda"] is None:
            continue
        ev = selective_eval(pop, fit, sel["lambda"])
        if ev["acted_error"] is not None and ev["acted_error"] > eps + 0.01:  # population risk proxy (n=4000)
            violations += 1
    assert violations / seeds <= delta


def _old_rule(ids, seed=0, fit_frac=0.5):
    fit = set()
    for i in ids:
        u = int(hashlib.sha256(f"{seed}|conformal-ltt|{i}".encode()).hexdigest()[:12], 16) / float(16 ** 12)
        if u < fit_frac:
            fit.add(i)
    return fit


def test_cluster_split_keeps_rows_together_and_default_is_unchanged():
    rng = random.Random(0)
    cal = [_draw(rng, i, cluster=f"row{i // 3}") for i in range(300)]  # 3 decisions per row
    fit, ltt, rec = split_calibration(cal, seed=0)
    assert {r.id for r in fit} == _old_rule([r.id for r in cal])  # default behaviour byte-identical
    straddle = {r.id.split("#")[0] for r in fit} & {r.id.split("#")[0] for r in ltt}
    assert straddle  # the per-decision split DOES put one row's decisions on both sides (discrepancy D1)
    fit_c, ltt_c, rec_c = split_calibration(cal, seed=0, cluster_of=lambda r: r.id.split("#")[0])
    assert not ({r.id.split("#")[0] for r in fit_c} & {r.id.split("#")[0] for r in ltt_c})
    assert rec_c["n_fit_clusters"] + rec_c["n_ltt_clusters"] == 100 and "cluster" in rec_c["rule"]
    assert "cluster" not in rec["rule"]
