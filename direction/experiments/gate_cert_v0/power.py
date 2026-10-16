"""Power of the complete-gate certificate under alternative preregistered designs (CPU simulation, no model calls).

The v0 attempt issued no certificate even though the student's measured acted error sat 1.5–3.5 points under
ε = 0.05. This simulation asks why, and which design a v1 preregistration should use. It uses the same test as the
certifier: ``jevkit.guarantees.crr_betting_pvalue`` on cluster residuals Y_c = w_c (L_c − ε A_c), clusters of 3
decisions, cap 1.

Scenario "student" takes each operating point's coverage and acted risk from the v0 calibration curve (manipulation
threshold 1.0). Decisions act independently with that coverage, and acted decisions are wrong with that risk.
Scenario "unsafe" raises every point's risk to 0.06 > ε to check that false certificates stay at or below δ.

Designs (all at δ = 0.05):
- bonferroni36: the v0 design, every point at δ/36 (the 30 off-axis points are assumed never certifiable).
- bonferroni6: only the 6 points on the manipulation-off axis, at δ/6.
- fit_order: a fit split of 60 clusters estimates each point's margin. Points are tested in that order at full δ
  (fixed sequence) and testing stops at the first non-rejection.
- fit_single: the fit split picks one point, tested at full δ.

    python -X utf8 direction/experiments/gate_cert_v0/power.py   # writes power.json
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from jevkit.guarantees import CRR_BET_CAP, crr_betting_pvalue  # noqa: E402

EPS, DELTA, SIZE = 0.05, 0.05, 3
POINTS = {  # uncertainty threshold: (coverage, acted risk), v0 student calibration curve
    0.02: (0.1907, 0.0185), 0.05: (0.2931, 0.0145), 0.10: (0.4626, 0.0198),
    0.20: (0.6398, 0.0298), 0.30: (0.7698, 0.0367), 0.50: (0.9371, 0.0543),
}
FIT_CLUSTERS, SIMS = 60, 300


def simulate(n: int, coverage: float, risk: float, rng: random.Random) -> list[tuple[int, int]]:
    """Per cluster (acts, losses) for one operating point."""
    out = []
    for _ in range(n):
        acts = sum(rng.random() < coverage for _ in range(SIZE))
        out.append((acts, sum(rng.random() < risk for _ in range(acts))))
    return out


def _betting_pvalue_var0(y: list[float], level: float, var0: float) -> float:
    """``crr_betting_pvalue`` with a different initial variance. Sensitivity only; the certifier uses 1/4."""
    n, bmax = len(y), CRR_BET_CAP / (1.0 - EPS)
    c0 = 2.0 * math.log(1.0 / level) / n
    s_mu, s_var, logk, best = 0.0, var0, 0.0, 0.0
    for i, yi in enumerate(y, start=1):
        mu = s_mu / (i - 1) if i > 1 else 0.0
        logk += math.log1p(-min(bmax, math.sqrt(c0 * i / s_var)) * yi)
        best = max(best, logk)
        s_var += (yi - mu) ** 2
        s_mu += yi
    return min(1.0, math.exp(-best))


def pvalue(clusters: list[tuple[int, int]], level: float, var0: float = 0.25) -> float:
    w = min(1.0, 1.0 / SIZE)
    y = [w * (losses - EPS * acts) for acts, losses in clusters]
    return crr_betting_pvalue(y, EPS, level)[0] if var0 == 0.25 else _betting_pvalue_var0(y, level, var0)


def margin(clusters: list[tuple[int, int]]) -> float:
    """Fit-split score: estimated acts × (ε − risk) per cluster; larger is easier to certify."""
    acts = sum(a for a, _ in clusters)
    losses = sum(l for _, l in clusters)
    return (EPS * acts - losses) / len(clusters)


def run_design(design: str, n: int, scenario: dict, rng: random.Random, var0: float = 0.25) -> tuple[bool, float]:
    """(certified?, coverage of the selected point) for one simulated calibration study."""
    data = {u: simulate(n, cov, risk, rng) for u, (cov, risk) in scenario.items()}
    if design in ("bonferroni36", "bonferroni6"):
        level = DELTA / (36 if design == "bonferroni36" else len(scenario))
        passed = [u for u in scenario if pvalue(data[u], level, var0) <= level]
        return (bool(passed), max((scenario[u][0] for u in passed), default=0.0))
    fit = {u: simulate(FIT_CLUSTERS, cov, risk, rng) for u, (cov, risk) in scenario.items()}
    order = sorted(scenario, key=lambda u: -margin(fit[u]))
    if design == "fit_single":
        order = order[:1]
    certified = []
    for u in order:  # fixed sequence at full δ; stop at the first non-rejection
        if pvalue(data[u], DELTA, var0) > DELTA:
            break
        certified.append(u)
    return (bool(certified), max((scenario[u][0] for u in certified), default=0.0))


def main() -> None:
    rng = random.Random(20261016)
    unsafe = {u: (cov, 0.06) for u, (cov, _) in POINTS.items()}
    designs = ("bonferroni36", "bonferroni6", "fit_order", "fit_single")
    results = {"settings": {"epsilon": EPS, "delta": DELTA, "cluster_size": SIZE, "sims": SIMS,
                            "fit_clusters": FIT_CLUSTERS, "points": {str(k): v for k, v in POINTS.items()}},
               "power": {}, "false_certification": {}}
    for n in (472, 1000, 2000):
        for design in designs:
            runs = [run_design(design, n, POINTS, rng) for _ in range(SIMS)]
            certified = [cov for ok, cov in runs if ok]
            results["power"][f"{design}@{n}"] = {
                "certified_rate": len(certified) / SIMS,
                "mean_coverage_when_certified": sum(certified) / len(certified) if certified else None}
    for design in designs:
        runs = [run_design(design, 1000, unsafe, rng) for _ in range(SIMS)]
        results["false_certification"][f"{design}@1000"] = sum(ok for ok, _ in runs) / SIMS
    # Does the 1/4 initial variance cost power? Same simulated data (same seed) with a near-oracle prior.
    results["prior_sensitivity"] = {}
    for var0 in (0.25, 0.01):
        paired = random.Random(7)
        runs = [run_design("bonferroni6", 472, POINTS, paired, var0) for _ in range(SIMS)]
        results["prior_sensitivity"][f"bonferroni6@472 var0={var0}"] = sum(ok for ok, _ in runs) / SIMS
    (HERE / "power.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(f"{'design':14s} {'n':>5s} {'P(certify)':>10s} {'coverage|cert':>13s}")
    for key, v in results["power"].items():
        design, n = key.split("@")
        cov = v["mean_coverage_when_certified"]
        print(f"{design:14s} {n:>5s} {v['certified_rate']:10.2f} {cov if cov is None else round(cov, 3)!s:>13s}")
    print("false certification at risk 0.06 (n=1000):", results["false_certification"])
    print("initial-variance sensitivity:", results["prior_sensitivity"])


if __name__ == "__main__":
    main()
