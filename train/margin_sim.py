"""Unchanged-candidate (Delta = 0) pass-rate simulations for every G12/G13 criterion (issue #36; LESSONS L-20).

Lesson: margins inside measured noise reject everything. The #36 margins as first proposed passed a genuinely unchanged
candidate ~0.1 % of the time (0.27 x 0.84 x 0.20 x 0.22 x 0.11). Gates are therefore sized by simulation BEFORE they are
adopted, and re-checked at the measured discordance / tau / SE of the first real runs (``train.promotion.resizing``).

Each function simulates paired child/comparator outcomes with the same CI construction the gate uses (the repo's own
cluster bootstrap, ``safety_eval.stats.cluster_weights``, or ``harness.calibration.debiased_ece``) and returns the
fraction of simulated unchanged candidates the criterion ACCEPTS. Defaults are the design values (cohort v2 sizes; the
#36 review's planning assumptions; full-DEV SE projected from job-5cmat). They are assumptions, not measurements.
``tests/test_margin_sim.py`` runs every function at the configured margins at small scale (few simulations) and asserts
>= 0.5; ``python -m train.margin_sim`` prints the table at larger scale.
"""
from __future__ import annotations

import json
import math
import sys
from statistics import NormalDist

import numpy as np

#: job-5cmat paired Index SE 0.752 on the 2,008-row subset, scaled by sqrt(proxy-token ratio 6.067) to full DEV
FULL_DEV_INDEX_SE = 0.752 / math.sqrt(6.067)


def _flip(rng, par, disc, p):
    """Child outcomes with the comparator's marginal rate p and discordance ``disc`` (Delta = 0)."""
    f10 = np.clip(disc / (2 * p), 0, 1) if p > 0 else 0.0
    f01 = np.clip(disc / (2 * (1 - p)), 0, 1) if p < 1 else 0.0
    u = rng.random(par.shape)
    return np.where(par == 1, u >= f10, u < f01).astype(float)


def steer(m_steer: float, n_req: int = 1000, n_tpl: int = 8, n_cl: int = 45, tau: float = 0.0, p: float = 0.5,
          disc: float = 0.10, p_elig: float = 0.6, template_alpha: float = 0.05, n_boot: int = 300,
          n_sims: int = 40, seed: int = 0) -> float:
    """G13 D1-B (1,000 x 8) by default; G12 D1-A = n_req 1800, n_tpl 4. Fixed-template one-way cluster bootstrap;
    accept iff lower <= 0, upper <= m_steer and no template's lower CI > 0 at 1 - template_alpha / k. ``tau``: SD of
    the template x arm effect, centred so the fixed-set difference is exactly 0. An UNCHANGED candidate has tau = 0
    (the default). With tau > 0 single templates genuinely regress and the per-template check rejects them BY DESIGN:
    at tau = 0.02 the pass rate falls to ~0.2-0.4 (``sensitivity``); the re-sizing clause measures tau on the first
    real runs and escalates if the projected pass rate is < 0.8."""
    from safety_eval.stats import cluster_weights

    rng = np.random.default_rng(seed)
    cl = np.repeat(np.arange(n_req) % n_cl, n_tpl)
    tp = np.tile(np.arange(n_tpl), n_req)
    W = cluster_weights([cl.tolist()], n=n_boot, seed=seed)
    a_t = template_alpha / n_tpl
    acc = 0
    for _ in range(n_sims):
        dt = rng.normal(0, tau, n_tpl)
        dt -= dt.mean()
        elig = np.repeat(rng.random(n_req) < p_elig, n_tpl).astype(float)
        par = (rng.random(cl.size) < p).astype(float)
        shift = dt[tp]  # per-unit shift of the child steer probability (the #36 review's simulation)
        f10 = np.clip(disc / (2 * p) + np.maximum(-shift, 0) / p, 0, 1)
        f01 = np.clip(disc / (2 * (1 - p)) + np.maximum(shift, 0) / (1 - p), 0, 1)
        u = rng.random(cl.size)
        child = np.where(par == 1, u >= f10, u < f01).astype(float)
        d = (child - par) * elig

        def ci(mask, level):
            den = W @ (elig * mask)
            r = np.divide(W @ (d * mask), den, out=np.full(den.shape, np.nan), where=den > 0)
            r = r[np.isfinite(r)]
            a = 100 * (1 - level) / 2
            return np.percentile(r, [a, 100 - a])

        lo, hi = ci(np.ones_like(d), 0.95)
        bad_t = any(ci((tp == t).astype(float), 1 - a_t)[0] > 0 for t in range(n_tpl))
        acc += bool(lo <= 0 and hi <= m_steer and not bad_t)
    return acc / n_sims


def false_alarm(m_false_alarm: float, m_refusal: float, n_req: int = 1000, n_cl: int = 45, fa: float = 0.01,
                fa_disc: float = 0.01, refusal: float = 0.0, ref_disc: float = 0.0, n_boot: int = 300,
                n_sims: int = 40, seed: int = 1) -> float:
    """Clean units of the G13 steer set (1,000; G12: 1,800). Parent false-alarm rate ``fa`` (PREREG G2 caps it at 1 %)
    with discordance ``fa_disc``; clean refusals are deterministic per request (control-token refusals), so the
    default refusal discordance is 0."""
    from safety_eval.stats import cluster_weights

    rng = np.random.default_rng(seed)
    cl = np.arange(n_req) % n_cl
    W = cluster_weights([cl.tolist()], n=n_boot, seed=seed)
    tot = W.sum(1)
    acc = 0
    for _ in range(n_sims):
        ok = True
        for rate, dsc, m in ((fa, fa_disc, m_false_alarm), (refusal, ref_disc, m_refusal)):
            par = (rng.random(n_req) < rate).astype(float)
            ch = _flip(rng, par, dsc, rate) if 0 < rate < 1 else par.copy()
            hi = np.percentile((W @ (ch - par)) / tot, 97.5)
            ok &= hi <= m
        acc += ok
    return acc / n_sims


def capability(m_cap: float, se: float = FULL_DEV_INDEX_SE, k_areas: int = 5, area_alpha: float = 0.05,
               n_sims: int = 4000, seed: int = 2) -> float:
    """Full-DEV paired index difference ~ N(0, se) (percentile CI ~ normal at B = 2000); areas ~ N(0, se sqrt(k))
    independent (conservative). Accept iff lower >= -m_cap and no area's upper CI at 1 - area_alpha / k < 0."""
    rng = np.random.default_rng(seed)
    z, za = 1.96, NormalDist().inv_cdf(1 - area_alpha / k_areas / 2)
    d = rng.normal(0, se, n_sims)
    a = rng.normal(0, se * math.sqrt(k_areas), (n_sims, k_areas))
    ok = (d - z * se >= -m_cap) & ~np.any(a + za * se * math.sqrt(k_areas) < 0, axis=1)
    return float(ok.mean())


def debiased_ece_l2(conf, correct, bins: int = 15) -> float:
    """``harness.calibration.debiased_ece(...)["debiased_l2"]`` (used when the kit is importable); this verbatim copy
    of its logic lets the simulation run without the Decision Index kit (equality: tests/test_margin_sim.py)."""
    try:
        from harness.calibration import debiased_ece
    except ImportError:
        conf, correct = np.asarray(conf, float), np.asarray(correct, float)
        n = len(conf)
        if n < 2 * bins:
            bins = max(1, n // 2)
        sq = 0.0
        for idx in np.array_split(np.argsort(conf, kind="stable"), bins):
            nb = len(idx)
            if nb < 2:
                continue
            a, c = correct[idx].mean(), conf[idx].mean()
            sq += nb / n * ((a - c) ** 2 - a * (1 - a) / (nb - 1))
        return math.sqrt(max(sq, 0.0))
    return debiased_ece(conf, correct, bins)["debiased_l2"]


def _ece_arms(rng, n, over=0.0):
    z = rng.normal(2.0, 1.4, n)
    u = rng.random(n)

    def arm(o):
        za = z + rng.normal(0, 0.35, n)
        return np.clip(1 / (1 + np.exp(-za * (1.12 + o))), 0.34, 1), (u < 1 / (1 + np.exp(-za))).astype(float)

    return arm(0.0), arm(over)


def ece(m_ece: float | None, n_fields: int = 33_000, n_clusters: int = 10_200, demonstrated_only: bool = False,
        n_boot: int = 100, n_sims: int = 10, seed: int = 3) -> float:
    """Debiased L2 ECE difference, one-way cluster bootstrap with shared draws (``harness.calibration.debiased_ece``).
    Defaults: full Index DEV (job-5cmat: 5,515 fields / 2,008 rows -> ~33k fields over ~10.2k clusters; synthetic,
    exchangeable, parent debiased L2 ~ 0.02 as measured). ``demonstrated_only``: the retention rule (reject iff lower
    CI > 0), e.g. ``ece(None, n_fields=1420, n_clusters=1420, demonstrated_only=True)``."""
    rng = np.random.default_rng(seed)
    cl = rng.integers(0, n_clusters, n_fields)
    acc = 0
    for _ in range(n_sims):
        (cp, yp), (cc, yc) = _ece_arms(rng, n_fields)
        W = rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), size=n_boot)[:, cl]
        ds = []
        for w in W:
            idx = np.repeat(np.arange(n_fields), w.astype(int))
            ds.append(debiased_ece_l2(cc[idx], yc[idx]) - debiased_ece_l2(cp[idx], yp[idx]))
        lo, hi = np.percentile(ds, [2.5, 97.5])
        acc += bool(not lo > 0) if demonstrated_only else bool(hi <= m_ece)
    return acc / n_sims


def retention_acc(m_ret_acc: float, n_items: int = 1420, n_clusters: int = 1670, p: float = 0.7, disc: float = 0.10,
                  n_boot: int = 300, n_sims: int = 40, seed: int = 4) -> float:
    """Paired decision accuracy on the cohort retention role (~1,420 items over 1,670 records), discordance ``disc``;
    accept iff lower CI >= -m_ret_acc."""
    from safety_eval.stats import cluster_weights

    rng = np.random.default_rng(seed)
    cl = rng.integers(0, n_clusters, n_items)
    W = cluster_weights([cl.tolist()], n=n_boot, seed=seed)
    tot = W.sum(1)
    acc = 0
    for _ in range(n_sims):
        par = (rng.random(n_items) < p).astype(float)
        ch = _flip(rng, par, disc, p)
        lo = np.percentile((W @ (ch - par)) / tot, 2.5)
        acc += bool(lo >= -m_ret_acc)
    return acc / n_sims


def table(margins: dict | None = None, scale: float = 1.0) -> dict[str, float]:
    """Every criterion's unchanged-candidate pass rate at ``margins`` (default ``train.promotion.MARGINS``).
    ``scale`` multiplies the simulation counts (the test uses a small scale)."""
    from train.promotion import MARGINS

    mg = MARGINS if margins is None else margins
    n = lambda k: max(4, int(k * scale))  # noqa: E731
    return {
        "g12_steer_d1a": steer(mg["m_steer"], n_req=1800, n_tpl=4, n_sims=n(40)),
        "g13_steer_d1b": steer(mg["m_steer"], n_req=1000, n_tpl=8, n_sims=n(40)),
        "g12_false_alarm": false_alarm(mg["m_false_alarm"], mg["m_refusal"], n_req=1800, n_sims=n(40)),
        "g13_false_alarm": false_alarm(mg["m_false_alarm"], mg["m_refusal"], n_req=1000, n_sims=n(40)),
        "g13_capability": capability(mg["m_cap"]),
        "g13_ece": ece(mg["m_ece"], n_sims=n(10)),
        "g13_retention_acc": retention_acc(mg["m_ret_acc"], n_sims=n(40)),
        "g13_retention_ece": ece(None, n_fields=1420, n_clusters=1420, demonstrated_only=True, n_boot=200,
                                 n_sims=n(20)),
    }


def sensitivity(margins: dict | None = None, scale: float = 1.0) -> dict[str, float]:
    """Reported, not asserted: steer pass rates when templates move (tau > 0, average Delta = 0)."""
    from train.promotion import MARGINS

    mg = MARGINS if margins is None else margins
    n = max(4, int(40 * scale))
    return {f"{name}_tau{tau}": steer(mg["m_steer"], n_req=nr, n_tpl=nt, tau=tau, n_sims=n)
            for name, nr, nt in (("g12_steer_d1a", 1800, 4), ("g13_steer_d1b", 1000, 8)) for tau in (0.01, 0.02)}


if __name__ == "__main__":
    sc = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    print(json.dumps({"unchanged_candidate": table(scale=sc), "sensitivity_template_shift": sensitivity(scale=sc)},
                     indent=1))
