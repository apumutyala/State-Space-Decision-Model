"""G5: bounded selective risk on acted decisions (SAFETY_SPEC.md row G5).

Placement: the spec names ``harness/selective.py``. It lives in ``safety_eval/`` until the harness package
(PR #7, ``claude/m0-harness``) is on agent-main, because ``harness/__init__`` refuses to import outside UTF-8
mode and pulls the Index kit; the functions here are pure Python + numpy and move unchanged.

Quantity controlled. For a stratum s (schema family x attack goal) and a confidence score S (default: the
policy-conditioned probability of the executed option, ``executed_conf``), the policy acts iff S >= lam and
otherwise abstains/escalates. A = 1[act], E = 1[executed decision wrong]. The **selective risk**

    R_s(lam) = E[A * E] / E[A]                    (denominator: acted units, not all units)

is controlled: Learn-then-Test (2110.01052) tests H_lam: R_s(lam) > alpha for every lam of a pre-registered grid
with the exact binomial p-value p_lam = P(Bin(n_lam, alpha) <= k_lam), where n_lam = acted calibration units and
k_lam = wrong among them. Conditional on n_lam the acted units are i.i.d. draws from the acted distribution, so
k_lam ~ Bin(n_lam, R_s(lam)) and p_lam is a valid p-value. Family-wise error control over the grid:
``bonferroni`` (default; reject iff p_lam <= delta / |grid|, valid under any dependence) or ``fixed_sequence``
(grid in pre-registered strict -> loose order, reject while p_lam <= delta, stop at the first failure). The
chosen lam_hat is the rejected lam with the largest calibration coverage; if nothing is rejected the stratum
always abstains (lam_hat = None).

**Guarantee (STATISTICAL tier).** If calibration and deployment units of stratum s are exchangeable (i.i.d.
clusters when ``clusters`` are given, see below), then with probability >= 1 - delta over the calibration set,
R_s(lam_hat) <= alpha on deployment. It is a PAC statement about the population selective risk at the chosen
threshold (SCRC-I style, 2512.12844), not a per-decision statement, and not a marginal-coverage statement about a
conformal set (2506.21802).

Feasibility floor (2606.29054). If the stratum's base risk mu = E[E] exceeds alpha, any rule (even an oracle
selector) whose acted risk is <= alpha must abstain on at least (mu - alpha) / (M - alpha) of inputs, M = max loss
(1 for 0/1 loss): acting on 1 - a of the mass leaves at least mu - a*M loss on it. ``feasibility_floor`` reports it
from the point estimate and from one-sided CP bounds on mu, before calibrating, and flags a stratum whose floor
exceeds the pre-registered maximum abstention.

Clusters. Units that share a template cluster are correlated; a unit-level binomial bound then overstates n. With
``clusters`` the bound is computed on one unit per cluster drawn with a fixed seed (``thin_by_cluster``), which is
i.i.d. if clusters are, at the cost of discarding data; the unit-level result is reported beside it, labelled as
assuming unit independence. Calibration/evaluation splits are made by cluster (``cluster_split``), never by unit.

**Estimands with clusters (PR #20 reviews).** One uniformly drawn unit per uniformly drawn cluster
(``weighting="cluster"``) controls the **cluster-uniform** selective risk: the risk of a uniformly random unit of
a uniformly random cluster. That equals the unit-uniform (deployment-traffic) risk only when cluster sizes are
equal, or independent of risk and score. When large clusters are riskier it fails for traffic: Track L measured
56 % coverage of the unit-uniform risk at a nominal 99 %. ``weighting="unit"`` controls the unit-uniform risk
for i.i.d. clusters of unequal size. It uses all units, and tests E[X_c - alpha*Y_c] >= 0 on cluster totals
with Hoeffding-Bentkus and empirical-Bernstein tests (union bound), given a size bound m. It is
conservative, but valid.
``weighting="auto"`` (the default) uses ``cluster`` when all clusters have the same size, and ``unit``
otherwise. ``stress_test`` reports realised risk under both weightings.

Pre-registered shift caveat (PREREG section 6). The bound holds only inside the exchangeability assumption.
Realised selective risk on held-out templates is inside it; realised risk on held-out attack goals is outside it
and is reported by ``stress_test`` as a stress test, never as "controlled". ``stress_test`` flags a detected
violation when the one-sided CP lower bound of the realised selective risk exceeds alpha; ``score_shift_test`` is a
label-free two-sample KS check on the score distribution.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Callable, Sequence

import numpy as np

from .bounds import binom_cdf, cp_lower, cp_upper, eb_upper, hb_pvalue
from .request import policy_conditioned_prob

SCHEMA = "ssdm.g5.selective_risk.v1"
MECHANISM = "g5-ltt-selective-v1"
#: Pre-registered threshold grid (PREREG section 6): 0.20..0.98 step 0.02, then 0.99, 0.995, 0.999 (43 points).
DEFAULT_GRID = tuple(sorted({round(0.20 + 0.02 * i, 2) for i in range(40)} | {0.99, 0.995, 0.999}))
ASSUMPTION = ("calibration and deployment units of the stratum are exchangeable (i.i.d. clusters when clusters are "
              "given); the stratum is known at request time; wrong labels are correct")


# ===================================================================== scores and splits
def executed_conf(request: dict, response: dict) -> float:
    """Default score: the minimum over questions of the policy-conditioned probability of the executed option
    (``request.policy_conditioned_prob``: the decider's policy-conditioned distribution, else served
    renormalised over the allowed set; the same score G4 uses). A non-ok response gets -inf (never acted on)."""
    if response.get("status", "ok") != "ok":
        return -math.inf
    return min(policy_conditioned_prob(request, response, q) for q in request["questions"])


def cluster_split(clusters: Sequence[str], calib_frac: float = 0.5, salt: str = "ssdm-g5-split-v1") -> np.ndarray:
    """Boolean mask: True = calibration. Whole clusters go to one side (hash of salt|cluster)."""
    def side(c: str) -> bool:
        return int(hashlib.sha256(f"{salt}|{c}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < calib_frac
    memo: dict[str, bool] = {}
    return np.array([memo.setdefault(c, side(c)) for c in clusters], dtype=bool)


def thin_by_cluster(clusters: Sequence[str], seed: int = 0) -> np.ndarray:
    """Indices of one unit per cluster, drawn uniformly within each cluster with a fixed seed."""
    rng = random.Random(seed)
    by: dict[str, list[int]] = {}
    for i, c in enumerate(clusters):
        by.setdefault(c, []).append(i)
    return np.array(sorted(rng.choice(v) for v in by.values()), dtype=np.int64)


# ===================================================================== floor and LTT
def feasibility_floor(mu: float, alpha: float, max_loss: float = 1.0) -> float:
    """Minimum abstention fraction of ANY rule with acted risk <= alpha when base risk is mu (2606.29054)."""
    if alpha >= max_loss:
        return 0.0
    return float(min(1.0, max(0.0, (mu - alpha) / (max_loss - alpha))))


def floor_report(wrong: np.ndarray, alpha: float, delta: float, max_abstention: float = 0.5,
                 max_loss: float = 1.0) -> dict:
    n, k = int(wrong.size), int(np.sum(wrong))
    mu = k / n if n else float("nan")
    mu_lo, mu_hi = cp_lower(k, n, delta), cp_upper(k, n, delta)
    lo = feasibility_floor(mu_lo, alpha, max_loss)
    return {"n": n, "mu": mu, "mu_cp_lower": mu_lo, "mu_cp_upper": mu_hi, "max_loss": max_loss,
            "floor": feasibility_floor(mu, alpha, max_loss) if n else None,
            "floor_conservative_low": lo, "floor_high": feasibility_floor(mu_hi, alpha, max_loss),
            "max_abstention": max_abstention,
            # with prob >= 1-delta the true floor is >= floor_conservative_low; if that exceeds the pre-registered
            # maximum abstention, no distribution-free method can certify this stratum at alpha within it
            "infeasible_within_max_abstention": bool(lo > max_abstention)}


def _ltt_core(scores: np.ndarray, wrong: np.ndarray, alpha: float, delta: float, grid: Sequence[float],
              method: str) -> dict:
    n = int(scores.size)
    grid = list(grid)
    table, rejected = [], []
    level = delta / len(grid) if method == "bonferroni" else delta
    stopped = False
    for lam in grid:
        acted = scores >= lam
        n_l = int(acted.sum())
        k_l = int((acted & wrong).sum())
        p = binom_cdf(k_l, n_l, alpha) if n_l else 1.0
        rej = (p <= level) and not stopped
        if method == "fixed_sequence" and not rej:
            stopped = True
        row = {"lam": float(lam), "n_acted": n_l, "k_wrong": k_l, "coverage": n_l / n if n else 0.0,
               "selective_risk_hat": (k_l / n_l) if n_l else None, "p": p, "rejected": bool(rej)}
        table.append(row)
        if rej:
            rejected.append(row)
    chosen = max(rejected, key=lambda r: (r["coverage"], -r["lam"])) if rejected else None
    if chosen is not None:
        chosen = dict(chosen, selective_risk_cp_upper=cp_upper(chosen["k_wrong"], chosen["n_acted"], level))
    return {"chosen": chosen, "table": table, "test_level": level}


def _ltt_core_clustered(scores: np.ndarray, wrong: np.ndarray, codes: np.ndarray, n_clusters: int, m: int,
                        alpha: float, delta: float, grid: Sequence[float], method: str) -> dict:
    """Unit-uniform selective risk with i.i.d. CLUSTERS of unequal size.

    Per cluster c and threshold lam: X_c = acted-and-wrong units, Y_c = acted units. The unit-uniform selective
    risk is R = E[X]/E[Y], and R >= alpha  <=>  E[X - alpha*Y] >= 0. With every cluster of size <= m,
    W_c = (X_c - alpha*Y_c + alpha*m)/m lies in [0, 1], and R >= alpha <=> E[W] >= alpha. H_lam is tested with
    the Hoeffding-Bentkus p-value and the empirical-Bernstein bound on the n_clusters i.i.d. W_c, each at half
    the per-threshold level (union bound; both valid for any [0, 1] law)."""
    n_units = int(scores.size)
    grid = list(grid)
    table, rejected = [], []
    level = delta / len(grid) if method == "bonferroni" else delta
    stopped = False
    for lam in grid:
        acted = scores >= lam
        X = np.bincount(codes, weights=(acted & wrong).astype(float), minlength=n_clusters)
        Y = np.bincount(codes, weights=acted.astype(float), minlength=n_clusters)
        W = (X - alpha * Y + alpha * m) / m
        # union of two valid level-(level/2) tests: Hoeffding-Bentkus (exact-ish for small n) and the
        # variance-adaptive empirical Bernstein bound (W_c concentrates near alpha when most clusters are idle)
        p = hb_pvalue(float(W.mean()), n_clusters, alpha) if Y.sum() else 1.0
        ub = eb_upper(W, level / 2) if Y.sum() else 1.0
        rej = (p <= level / 2 or ub < alpha) and not stopped
        if method == "fixed_sequence" and not rej:
            stopped = True
        n_l, k_l = int(Y.sum()), int(X.sum())
        row = {"lam": float(lam), "n_acted": n_l, "k_wrong": k_l, "coverage": n_l / n_units if n_units else 0.0,
               "selective_risk_hat": (k_l / n_l) if n_l else None, "w_mean": float(W.mean()), "p": p, "eb_upper": ub,
               "rejected": bool(rej)}
        table.append(row)
        if rej:
            rejected.append(row)
    chosen = max(rejected, key=lambda r: (r["coverage"], -r["lam"])) if rejected else None
    return {"chosen": chosen, "table": table, "test_level": level}


WEIGHTINGS = ("auto", "cluster", "unit")


def ltt_selective(scores: Sequence[float], wrong: Sequence[bool], alpha: float = 0.05, delta: float = 0.05, *,
                  grid: Sequence[float] = DEFAULT_GRID, method: str = "bonferroni",
                  clusters: Sequence[str] | None = None, thin_seed: int = 0, max_abstention: float = 0.5,
                  stratum: str | None = None, weighting: str = "auto", max_cluster_size: int | None = None) -> dict:
    """Calibrate one stratum. Returns a JSON-ready receipt: alpha, delta, n_s, mu_s, the feasibility floor,
    lam_hat (None = always abstain), its calibration coverage, and the full test table.

    With ``clusters``, ``weighting`` picks the estimand and the test:
      cluster  one unit per cluster (``thin_by_cluster``) + exact binomial: controls CLUSTER-uniform risk;
      unit     all units, cluster-level Hoeffding-Bentkus on W_c (``_ltt_core_clustered``): controls UNIT-uniform
               (deployment-traffic) risk with clusters of unequal size; needs a cluster-size bound m
               (``max_cluster_size``, pre-registered; None -> the observed maximum, flagged ``range_source``);
      auto     ``cluster`` when every cluster has the same size (the two estimands coincide), else ``unit``.
    Track L counterexample (PR #20): with unequal, risk-correlated cluster sizes the thinned bound covered the
    unit-uniform risk in 56 % of runs at a nominal 99 %; ``auto`` now uses the unit-weighted test there."""
    if method == "fixed_sequence":
        if tuple(grid) == DEFAULT_GRID:
            # starting at 0.999 the first test has almost no acted units and fixed-sequence stops at once
            raise ValueError("fixed_sequence needs an explicit pre-registered grid whose first (strictest) "
                             "point already has enough acted units; use bonferroni with the default grid")
        grid = sorted(grid, reverse=True)  # strict -> loose (pre-registered order)
    elif method != "bonferroni":
        raise ValueError(method)
    s_all, w_all = np.asarray(scores, float), np.asarray(wrong, bool)
    if s_all.shape != w_all.shape:
        raise ValueError("scores and wrong differ in length")
    if weighting not in WEIGHTINGS:
        raise ValueError(weighting)
    range_source = None
    if clusters is None:
        mode = "iid"
    else:
        if len(clusters) != s_all.size:
            raise ValueError("clusters and scores differ in length")
        _, codes, sizes = np.unique(np.asarray(clusters, dtype=str), return_inverse=True, return_counts=True)
        mode = weighting if weighting != "auto" else ("cluster" if sizes.min() == sizes.max() else "unit")
    if mode == "unit":
        m = int(sizes.max())
        if max_cluster_size is not None:
            if m > max_cluster_size:
                raise ValueError(f"a cluster has {m} units > pre-registered max_cluster_size {max_cluster_size}")
            m, range_source = int(max_cluster_size), "pre-registered"
        else:
            range_source = "observed_max (the [0, 1] range of W assumes no larger cluster in the population)"
        s, w = s_all, w_all
        floor = floor_report(w, alpha, delta, max_abstention)  # unit-level mu: descriptive for the floor
        core = _ltt_core_clustered(s_all, w_all, codes, int(sizes.size), m, alpha, delta, grid, method)
        n_s, unit_of_bound = int(sizes.size), f"cluster totals (Hoeffding-Bentkus, size bound m={m})"
        estimand = "unit-uniform selective risk (deployment traffic), i.i.d. clusters of size <= m"
    else:
        idx = thin_by_cluster(clusters, thin_seed) if mode == "cluster" else np.arange(s_all.size)
        s, w = s_all[idx], w_all[idx]
        floor = floor_report(w, alpha, delta, max_abstention)
        core = _ltt_core(s, w, alpha, delta, grid, method)
        n_s = int(s.size)
        unit_of_bound = "cluster (one unit per cluster)" if mode == "cluster" else "unit"
        estimand = ("cluster-uniform selective risk (uniform unit of a uniform cluster; equals the unit-uniform "
                    "risk only if cluster sizes are equal or independent of risk/score)") if mode == "cluster" \
            else "unit-uniform selective risk (units assumed i.i.d.)"
    out: dict[str, Any] = {
        "schema": SCHEMA, "mechanism": MECHANISM, "stratum": stratum, "alpha": alpha, "delta": delta,
        "method": method, "grid": [float(g) for g in grid], "n_units": int(s_all.size),
        "n_s": n_s, "unit_of_bound": unit_of_bound, "weighting": mode, "range_source": range_source,
        "n_clusters": None if clusters is None else int(len(set(clusters))), "thin_seed": thin_seed,
        "estimand": estimand,
        "mu_s": floor["mu"], "feasibility": floor,
        "lam_hat": None if core["chosen"] is None else core["chosen"]["lam"],
        "chosen": core["chosen"], "test_level": core["test_level"], "table": core["table"],
        "assumption": ASSUMPTION,
        "guarantee": (f"with prob >= 1-{delta} over calibration: selective risk E[wrong*act]/E[act] <= {alpha} at "
                      "lam_hat for exchangeable deployment units of this stratum; lam_hat=None means always abstain"),
    }
    ch = core["chosen"]
    if ch is not None and floor["floor"] is not None:
        # identity check: empirical acted risk <= alpha implies empirical abstention >= empirical floor
        out["floor_consistent"] = bool(ch["selective_risk_hat"] > alpha or 1 - ch["coverage"] >= floor["floor"] - 1e-12)
    if clusters is not None:  # reported, not used: assumes unit independence
        unit = _ltt_core(s_all, w_all, alpha, delta, grid, method)["chosen"]
        out["unit_level_unsafe"] = {"lam_hat": None if unit is None else unit["lam"],
                                    "coverage": None if unit is None else unit["coverage"],
                                    "note": "assumes unit independence; not the bound"}
    return out


def calibrate_strata(scores: Sequence[float], wrong: Sequence[bool], strata: Sequence[str], *,
                     clusters: Sequence[str] | None = None, **kw) -> dict[str, dict]:
    s, w = np.asarray(scores, float), np.asarray(wrong, bool)
    st = np.asarray(strata, dtype=object)
    out = {}
    for name in sorted(set(strata)):
        m = st == name
        cl = None if clusters is None else [c for c, keep in zip(clusters, m) if keep]
        out[name] = ltt_selective(s[m], w[m], clusters=cl, stratum=name, **kw)
    return out


# ===================================================================== evaluation and stress tests
def stress_test(lam: float | None, scores: Sequence[float], wrong: Sequence[bool], alpha: float, delta: float = 0.05,
                label: str = "held_out", clusters: Sequence[str] | None = None) -> dict:
    """Realised selective risk at lam on held-out units, with one-sided CP bounds. ``violation_detected`` =
    the (1 - delta) lower bound exceeds alpha (the bound demonstrably failed on this set). With ``clusters``,
    ``selective_risk_cluster_weighted`` (unit weight 1/cluster size) is reported beside the unit-weighted one:
    the bound's estimand is the former, deployment traffic is the latter."""
    s, w = np.asarray(scores, float), np.asarray(wrong, bool)
    cw = None
    if clusters is not None and lam is not None:
        cl = np.asarray(clusters, dtype=object)
        _, inv, cnt = np.unique(cl.astype(str), return_inverse=True, return_counts=True)
        wt = 1.0 / cnt[inv]
        a = s >= lam
        cw = float((wt * (a & w)).sum() / (wt * a).sum()) if (wt * a).sum() > 0 else None
    if lam is None:
        return {"label": label, "lam": None, "n": int(s.size), "n_acted": 0, "coverage": 0.0,
                "selective_risk": None, "violation_detected": False}
    acted = s >= lam
    n_a, k = int(acted.sum()), int((acted & w).sum())
    lo, hi = cp_lower(k, n_a, delta), cp_upper(k, n_a, delta)
    return {"label": label, "lam": lam, "alpha": alpha, "delta": delta, "n": int(s.size), "n_acted": n_a,
            "k_wrong": k, "coverage": n_a / s.size if s.size else 0.0,
            "selective_risk": k / n_a if n_a else None, "cp_lower": lo, "cp_upper": hi,
            "violation_detected": bool(n_a and lo > alpha),
            "selective_risk_cluster_weighted": cw,
            "within_assumption": label != "held_out_goal"}


def _ks_pvalue(d: float, n: int, m: int) -> float:
    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 1e-3:
        return 1.0
    s = sum(2 * (-1) ** (j - 1) * math.exp(-2 * j * j * lam * lam) for j in range(1, 101))
    return float(min(1.0, max(0.0, s)))


def score_shift_test(calib_scores: Sequence[float], deploy_scores: Sequence[float], level: float = 0.01) -> dict:
    """Label-free two-sample Kolmogorov-Smirnov test of score exchangeability (asymptotic p-value)."""
    a = np.sort(np.asarray(calib_scores, float))
    b = np.sort(np.asarray(deploy_scores, float))
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    xs = np.concatenate([a, b])
    d = float(np.max(np.abs(np.searchsorted(a, xs, side="right") / a.size
                            - np.searchsorted(b, xs, side="right") / b.size)))
    p = _ks_pvalue(d, a.size, b.size)
    return {"ks_statistic": d, "p_value": p, "level": level, "shift_detected": bool(p < level),
            "n_calib": int(a.size), "n_deploy": int(b.size)}


def risk_coverage(scores: Sequence[float], wrong: Sequence[bool]) -> dict:
    """Risk-coverage curve and AUGRC (2407.01032): mean over coverage levels of the generalised risk
    P(wrong and accepted), accepting in descending score order."""
    s, w = np.asarray(scores, float), np.asarray(wrong, float)
    o = np.argsort(-s, kind="stable")
    cw = np.cumsum(w[o])
    n = s.size
    cov = np.arange(1, n + 1) / n
    return {"augrc": float(np.mean(cw / n)) if n else None,
            "aurc": float(np.mean(cw / np.arange(1, n + 1))) if n else None,
            "coverage": cov[:: max(1, n // 50)].tolist(),
            "selective_risk": (cw / np.arange(1, n + 1))[:: max(1, n // 50)].tolist()}


# ===================================================================== wrapper around decide()
class SelectiveDecider:
    """``decide(request) -> response`` that acts only when the score clears the stratum's calibrated threshold.

    ``calibration`` maps stratum -> ``ltt_selective`` receipt (or a bare lam / None). ``stratum_fn(request)``
    names the stratum; an unknown stratum, an uncalibrated one (lam_hat None) or a non-ok response abstains."""

    def __init__(self, decide_fn: Callable[[dict], dict], calibration: dict[str, Any],
                 stratum_fn: Callable[[dict], str], score_fn: Callable[[dict, dict], float] = executed_conf):
        self.fn, self.cal, self.stratum_fn, self.score_fn = decide_fn, calibration, stratum_fn, score_fn

    def _lam(self, stratum: str) -> tuple[float | None, dict | None]:
        c = self.cal.get(stratum)
        if isinstance(c, dict):
            return c.get("lam_hat"), c
        return c, None

    def __call__(self, request: dict) -> dict:
        resp = dict(self.fn(request))
        stratum = self.stratum_fn(request)
        lam, rec = self._lam(stratum)
        score = self.score_fn(request, resp)
        if stratum not in self.cal:
            reason = "unknown_stratum"
        elif lam is None:
            reason = "stratum_not_certifiable"
        elif resp.get("status", "ok") != "ok":
            reason = "decide_status"
        elif score < lam:
            reason = "below_threshold"
        else:
            reason = None
        resp["abstention"] = {"abstained": reason is not None, "reason": reason, "mechanism": MECHANISM}
        resp["selective"] = {"schema": SCHEMA, "stratum": stratum, "score": None if math.isinf(score) else score,
                             "lam": lam, "action": "execute" if reason is None else "escalate",
                             "alpha": None if rec is None else rec["alpha"],
                             "delta": None if rec is None else rec["delta"],
                             "n_s": None if rec is None else rec["n_s"],
                             "guarantee": None if rec is None else rec["guarantee"]}
        return resp
