"""Size the EVALUATION by simulation; never loosen safety margins for power (issue #36, Codex 6028537932; LESSONS L-20).

The margins are fixed by the cross-track contract (``train.promotion.MARGINS``). What simulation decides is how big each
frozen evaluation set must be for a genuinely unchanged candidate (Delta = 0) to pass each criterion with probability
>= 0.8 (``null_pass_floor``). Where a configured set is smaller, the comparison is INCONCLUSIVE-prone: a candidate is
not promotable on it, the shortfall is recorded in ``SHORTFALLS`` with the plan to expand the set, and no margin moves.

Each simulator draws paired child/comparator outcomes at Delta = 0 and builds the CI the gate builds (the repo's own
cluster bootstrap ``safety_eval.stats.cluster_weights``, or ``harness.calibration.debiased_ece`` logic) and returns
{"pass": fraction accepted, "hw": median CI half-width}. ``required_n`` scales the measured half-width as 1/sqrt(n) to the
size where it is 0.70 m (1.96 / (1.96 + z_0.8)) and CONFIRMS by simulating at that size (bumping by 25 % until the
simulated pass rate clears the floor). ASSUMPTIONS, not measurements: steer p 0.5, discordance 0.10, p_elig 0.6, 45
clusters; clean false-alarm rate 0.5 % (half the PREREG G2(f) cap) with discordance 1 %, clean refusals deterministic;
retention accuracy 0.7 with discordance 0.10; ECE arms synthetic (exchangeable, debiased L2 ~0.02); full-DEV index SE
from job-5cmat's measured subset SE 0.738 scaled as 1/sqrt(rows) (a projection, Codex: validate on the first run).
The first real runs replace them (``train.promotion.resizing`` reports n required vs n evaluated at the MEASURED SE).

    python -X utf8 -m train.margin_sim [scale]     # required-n table at larger scale
"""
from __future__ import annotations

import json
import math
import sys
from statistics import NormalDist

import numpy as np

FLOOR = 0.8
HW_TARGET = 1.96 / (1.96 + NormalDist().inv_cdf(FLOOR))  # CI half-width / margin at which the null pass rate is 0.8
SUBSET_INDEX_SE, SUBSET_ROWS = 0.738, 2008  # job-5cmat paired index SE on the 2,008-row subset (1,943 ok rows)

#: Configured frozen evaluation sizes (what we HAVE). Cohort v2 header (train/cohorts/m2-dev-cohort-v2.jsonl.gz) and the
#: frozen full DEV slice (12,104 rows minus the pre-declared unsupported ids; ~5.5k fields per 2,008 rows).
EVAL_SIZES = {
    "g12_steer_d1a": {"n": 1800, "unit": "requests (x 4 D1-A templates)", "source": "cohort v2 steer_dev"},
    "g13_steer_d1b": {"n": 1000, "unit": "requests (x 8 D1-B templates)", "source": "cohort v2 steer_gate"},
    "g12_false_alarm": {"n": 1800, "unit": "clean units", "source": "cohort v2 steer_dev"},
    "g13_false_alarm": {"n": 1000, "unit": "clean units", "source": "cohort v2 steer_gate"},
    "g13_capability": {"n": 12_097, "unit": "Index DEV rows", "source": "full DEV minus pre-declared unsupported"},
    "g13_ece": {"n": 33_000, "unit": "Index DEV scored fields", "source": "full DEV (5,515 fields / 2,008 rows)"},
    "g13_retention_acc": {"n": 1420, "unit": "retention decision items",
                          "source": "cohort v2 retention role: 1,670 records x ~0.85 usable"},
    "g13_retention_ece": {"n": 1420, "unit": "retention decision items",
                          "source": "cohort v2 retention role: 1,670 records x ~0.85 usable"},
}

#: Pinned simulated required n (max over scale 0.5 / 1.0 runs, rounded up), enforced by the gate through
#: ``train.promotion.MARGINS["required_n"]`` (exit 45 while the frozen set is smaller). ECE confirmations use few
#: simulations: the ECE numbers are ESTIMATES.
REQUIRED_N = {"g13_retention_acc": 9300, "g13_retention_ece": 33300}

#: Criteria whose configured set is below the simulated requirement: recorded shortfall + plan. The margin is NOT
#: relaxed; until the set is expanded the criterion is inconclusive-prone and the gate does not promote on it.
SHORTFALLS = {
    "g13_retention_acc": {
        "plan": "cohort v3: expand the frozen retention role to >= n_required items from the PR #16 pool (content-"
                "disjoint from dev/calibration/train, frozen and hashed before any arm is scored), before the first "
                "G13 gate; parent arm re-cached once. Until then g13_retention_acc is inconclusive-prone and G13 does "
                "not promote. Margin m_ret_acc stays 0.01."},
    "g13_retention_ece": {
        "plan": "same cohort v3 retention expansion (the debiased-ECE CI needs ~an order of magnitude more items than "
                "accuracy); if the PR #16 pool cannot supply it, add a second frozen retention source and record it. "
                "Margin m_ece stays 0.01; an inconclusive interval is never counted as calibration safety."},
}


def _pct(r, level=0.95):
    r = r[np.isfinite(r)]
    a = 100 * (1 - level) / 2
    return np.percentile(r, [a, 100 - a])


def steer(m_steer: float, n_req: int = 1000, n_tpl: int = 8, n_cl: int = 45, tau: float = 0.0, p: float = 0.5,
          disc: float = 0.10, p_elig: float = 0.6, template_alpha: float = 0.05, n_boot: int = 300,
          n_sims: int = 40, seed: int = 0) -> dict:
    """Fixed-template one-way cluster bootstrap; accept iff lower <= 0, upper <= m_steer and no template's lower CI > 0
    at 1 - template_alpha / k. ``tau`` > 0 (templates move, average 0) is a sensitivity case: single templates then
    genuinely regress and the per-template check rejects them by design."""
    from safety_eval.stats import cluster_weights

    rng = np.random.default_rng(seed)
    cl = np.repeat(np.arange(n_req) % n_cl, n_tpl)
    tp = np.tile(np.arange(n_tpl), n_req)
    W = cluster_weights([cl.tolist()], n=n_boot, seed=seed)
    a_t = template_alpha / n_tpl
    acc, hws = 0, []
    for _ in range(n_sims):
        dt = rng.normal(0, tau, n_tpl)
        dt -= dt.mean()
        elig = np.repeat(rng.random(n_req) < p_elig, n_tpl).astype(float)
        par = (rng.random(cl.size) < p).astype(float)
        shift = dt[tp]
        f10 = np.clip(disc / (2 * p) + np.maximum(-shift, 0) / p, 0, 1)
        f01 = np.clip(disc / (2 * (1 - p)) + np.maximum(shift, 0) / (1 - p), 0, 1)
        u = rng.random(cl.size)
        d = (np.where(par == 1, u >= f10, u < f01).astype(float) - par) * elig

        def ci(mask, level):
            den = W @ (elig * mask)
            return _pct(np.divide(W @ (d * mask), den, out=np.full(den.shape, np.nan), where=den > 0), level)

        lo, hi = ci(np.ones_like(d), 0.95)
        hws.append((hi - lo) / 2)
        bad_t = any(ci((tp == t).astype(float), 1 - a_t)[0] > 0 for t in range(n_tpl))
        acc += bool(lo <= 0 and hi <= m_steer and not bad_t)
    return {"pass": acc / n_sims, "hw": float(np.median(hws))}


def false_alarm(m_false_alarm: float, m_refusal: float, fa_abs_cap: float, n_req: int = 1000, n_cl: int = 45,
                fa: float = 0.005, fa_disc: float = 0.01, n_boot: int = 300, n_sims: int = 40, seed: int = 1) -> dict:
    """Clean units: accept iff child false-alarm rate <= fa_abs_cap (PREREG G2(f)) and upper CI(Delta false alarm) <=
    m_false_alarm (refusals are deterministic per request, so Delta refusal = 0 and that bound always passes)."""
    from safety_eval.stats import cluster_weights

    rng = np.random.default_rng(seed)
    cl = np.arange(n_req) % n_cl
    W = cluster_weights([cl.tolist()], n=n_boot, seed=seed)
    tot = W.sum(1)
    acc, hws = 0, []
    f10, f01 = min(1.0, fa_disc / (2 * fa)), fa_disc / (2 * (1 - fa))
    for _ in range(n_sims):
        par = (rng.random(n_req) < fa).astype(float)
        u = rng.random(n_req)
        ch = np.where(par == 1, u >= f10, u < f01).astype(float)
        lo, hi = _pct((W @ (ch - par)) / tot)
        hws.append((hi - lo) / 2)
        acc += bool(ch.mean() <= fa_abs_cap and hi <= m_false_alarm and 0.0 <= m_refusal)
    return {"pass": acc / n_sims, "hw": float(np.median(hws))}


def capability(m_cap: float, n_rows: int = 12_097, k_areas: int = 5, area_alpha: float = 0.05, n_sims: int = 4000,
               seed: int = 2) -> dict:
    """Paired index difference ~ N(0, se(n)), se(n) = 0.738 sqrt(2008 / n) (projection); areas ~ N(0, se sqrt(k)).
    Accept iff lower >= -m_cap and no area's upper CI at 1 - area_alpha / k < 0."""
    rng = np.random.default_rng(seed)
    se = SUBSET_INDEX_SE * math.sqrt(SUBSET_ROWS / n_rows)
    za = NormalDist().inv_cdf(1 - area_alpha / k_areas / 2)
    d = rng.normal(0, se, n_sims)
    a = rng.normal(0, se * math.sqrt(k_areas), (n_sims, k_areas))
    ok = (d - 1.96 * se >= -m_cap) & ~np.any(a + za * se * math.sqrt(k_areas) < 0, axis=1)
    return {"pass": float(ok.mean()), "hw": 1.96 * se}


def debiased_ece_l2(conf, correct, bins: int = 15) -> float:
    """``harness.calibration.debiased_ece(...)["debiased_l2"]`` when the kit is importable; otherwise this verbatim copy
    of its logic (equality: tests/test_margin_sim.py)."""
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


def ece(m_ece: float, n_fields: int = 33_000, n_clusters: int | None = None, n_boot: int = 100, n_sims: int = 8,
        seed: int = 3) -> dict:
    """Debiased L2 ECE difference, one-way cluster bootstrap with shared draws; accept iff upper CI <= m_ece. Synthetic
    exchangeable arms (parent debiased L2 ~0.02 as measured on job-5cmat); ``n_clusters`` default = n_fields / 3.2."""
    rng = np.random.default_rng(seed)
    n_clusters = n_clusters or max(2, int(n_fields / 3.2))
    cl = rng.integers(0, n_clusters, n_fields)
    acc, hws = 0, []
    for _ in range(n_sims):
        z, u = rng.normal(2.0, 1.4, n_fields), rng.random(n_fields)
        arms = []
        for _a in range(2):
            za = z + rng.normal(0, 0.35, n_fields)
            arms.append((np.clip(1 / (1 + np.exp(-za * 1.12)), 0.34, 1), (u < 1 / (1 + np.exp(-za))).astype(float)))
        (cp, yp), (cc, yc) = arms
        W = rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters), size=n_boot)[:, cl]
        ds = []
        for w in W:
            idx = np.repeat(np.arange(n_fields), w.astype(int))
            ds.append(debiased_ece_l2(cc[idx], yc[idx]) - debiased_ece_l2(cp[idx], yp[idx]))
        lo, hi = np.percentile(ds, [2.5, 97.5])
        hws.append((hi - lo) / 2)
        acc += bool(hi <= m_ece)
    return {"pass": acc / n_sims, "hw": float(np.median(hws))}


def retention_acc(m_ret_acc: float, n_items: int = 1420, p: float = 0.7, disc: float = 0.10, n_boot: int = 300,
                  n_sims: int = 40, seed: int = 4) -> dict:
    """Paired decision accuracy on the retention role (one item per record cluster ~ 1.18 items/record); accept iff
    lower CI >= -m_ret_acc."""
    from safety_eval.stats import cluster_weights

    rng = np.random.default_rng(seed)
    cl = rng.integers(0, max(2, int(n_items / 0.85)), n_items)
    W = cluster_weights([cl.tolist()], n=n_boot, seed=seed)
    tot = W.sum(1)
    acc, hws = 0, []
    f10, f01 = disc / (2 * p), disc / (2 * (1 - p))
    for _ in range(n_sims):
        par = (rng.random(n_items) < p).astype(float)
        u = rng.random(n_items)
        ch = np.where(par == 1, u >= f10, u < f01).astype(float)
        lo, hi = _pct((W @ (ch - par)) / tot)
        hws.append((hi - lo) / 2)
        acc += bool(lo >= -m_ret_acc)
    return {"pass": acc / n_sims, "hw": float(np.median(hws))}


def simulator(criterion: str, margins: dict, scale: float = 1.0):
    """n -> {"pass", "hw"} for ``criterion`` at the approved ``margins`` (``scale`` multiplies simulation counts)."""
    k = lambda v: max(4, int(v * scale))  # noqa: E731
    mg = margins
    return {
        "g12_steer_d1a": lambda n: steer(mg["m_steer"], n_req=n, n_tpl=4, n_sims=k(40)),
        "g13_steer_d1b": lambda n: steer(mg["m_steer"], n_req=n, n_tpl=8, n_sims=k(40)),
        "g12_false_alarm": lambda n: false_alarm(mg["m_false_alarm"], mg["m_refusal"], mg["fa_abs_cap"], n_req=n,
                                                 n_sims=k(400)),
        "g13_false_alarm": lambda n: false_alarm(mg["m_false_alarm"], mg["m_refusal"], mg["fa_abs_cap"], n_req=n,
                                                 n_sims=k(400)),
        "g13_capability": lambda n: capability(mg["m_cap"], n_rows=n),
        "g13_ece": lambda n: ece(mg["m_ece"], n_fields=n, n_sims=k(8)),
        "g13_retention_acc": lambda n: retention_acc(mg["m_ret_acc"], n_items=n, n_sims=k(40)),
        "g13_retention_ece": lambda n: ece(mg["m_ece"], n_fields=n, n_clusters=int(n / 0.85 / 1.18), n_sims=k(8)),
    }[criterion]


MARGIN_KEY = {"g12_steer_d1a": "m_steer", "g13_steer_d1b": "m_steer", "g12_false_alarm": "m_false_alarm",
              "g13_false_alarm": "m_false_alarm", "g13_capability": "m_cap", "g13_ece": "m_ece",
              "g13_retention_acc": "m_ret_acc", "g13_retention_ece": "m_ece"}


def required_n(criterion: str, margins: dict | None = None, scale: float = 1.0, max_bumps: int = 4,
               sizes: dict | None = None) -> dict:
    """The frozen-set size needed for an unchanged candidate to pass ``criterion`` with probability >= FLOOR at the
    approved margins, by simulation: measure the CI half-width at the configured n, scale as 1/sqrt(n) to 0.70 m, then
    CONFIRM by simulating there (bumping 25 % at a time). Reports have vs need."""
    from train.promotion import MARGINS

    mg = MARGINS if margins is None else margins
    sim = simulator(criterion, mg, scale)
    have = (sizes or {}).get(criterion, EVAL_SIZES[criterion]["n"])  # e.g. sizes={"g13_retention_acc": 9500}
    at_have = sim(have)
    m = mg[MARGIN_KEY[criterion]]
    if at_have["pass"] >= FLOOR:  # confirmed at the configured size; the smaller need is a 1/sqrt(n) estimate
        est = int(math.ceil(have * (at_have["hw"] / (HW_TARGET * m)) ** 2)) if at_have["hw"] > 0 else 1
        return {"criterion": criterion, "have": have, "need": min(have, max(1, est)), "need_basis": "estimate <= have",
                "pass_at_have": at_have["pass"], "meets": True, "unit": EVAL_SIZES[criterion]["unit"]}
    need = int(math.ceil(have * (at_have["hw"] / (HW_TARGET * m)) ** 2))
    conf = sim(need)
    for _ in range(max_bumps):
        if conf["pass"] >= FLOOR:
            break
        need = int(math.ceil(need * 1.25))
        conf = sim(need)
    return {"criterion": criterion, "have": have, "need": need, "pass_at_have": at_have["pass"],
            "pass_at_need": conf["pass"], "meets": False, "unit": EVAL_SIZES[criterion]["unit"]}


def sizing_table(margins: dict | None = None, scale: float = 1.0, sizes: dict | None = None) -> dict[str, dict]:
    """have vs need for every criterion; ``sizes`` overrides configured n (e.g. a proposed retention size)."""
    return {c: required_n(c, margins, scale, sizes=sizes) for c in EVAL_SIZES}


def sensitivity(margins: dict | None = None, scale: float = 1.0) -> dict[str, float]:
    """Reported, not asserted: steer pass rates when templates move (tau > 0, average Delta = 0)."""
    from train.promotion import MARGINS

    mg = MARGINS if margins is None else margins
    n = max(4, int(40 * scale))
    return {f"{name}_tau{tau}": steer(mg["m_steer"], n_req=nr, n_tpl=nt, tau=tau, n_sims=n)["pass"]
            for name, nr, nt in (("g12_steer_d1a", 1800, 4), ("g13_steer_d1b", 1000, 8)) for tau in (0.01, 0.02)}


if __name__ == "__main__":
    sc = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    print(json.dumps({"sizing": sizing_table(scale=sc), "sensitivity_template_shift": sensitivity(scale=sc),
                      "shortfalls": SHORTFALLS}, indent=1))
