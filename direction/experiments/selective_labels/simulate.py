"""SIM: paired Bernoulli worlds, selective labels, and delayed cluster closure.

Run only the preregistered grid. Per-world results stay in the private output
folder; the compact report has aggregates only. No models or datasets used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from direction.experiments.selective_labels.betting import betting_paths, closure_times, normalized_ipw

EPSILON, DELTA, SIZE, COVERAGE, REVIEW_RISK = 0.05, 0.05, 3, 0.64, 0.01
WORLDS, BOOTSTRAPS, SEED = 2000, 2000, 20261016
HORIZONS, RATES, RISKS = (500, 1000, 2000), (0.02, 0.05, 0.1, 0.2), (0.05, 0.08, 0.01, 0.03)
FAMILIES = ("independent", "shared_fate")


def rate_ci(values):
    """95% Wilson Monte Carlo interval; nondegenerate at 0/n and n/n."""
    n, k, z = len(values), int(np.count_nonzero(values)), 1.959963984540054
    p, denom = k / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [p, max(0.0, center - half), min(1.0, center + half)]


def mean_ci(values, weights):
    """Resample independent world blocks, preserving all within-world clusters."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return None
    numerator = weights @ np.where(finite, values, 0)
    denominator = weights @ finite.astype(float)
    bootstrap = numerator[denominator > 0] / denominator[denominator > 0]
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return [float(values[finite].mean()), float(low), float(high)]


def summarize(path, coverage, risk, labels, audit_labels, weights):
    passed, due = path["p"] <= DELTA, path["p_due"] <= DELTA
    return {"certificate": rate_ci(passed), "certificate_due": rate_ci(due),
            "coverage": mean_ci(coverage, weights),
            "certified_coverage": [COVERAGE * v for v in rate_ci(passed)],
            "certified_coverage_due": [COVERAGE * v for v in rate_ci(due)],
            "risk_estimate": mean_ci(risk, weights), "labels": mean_ci(labels, weights),
            "audit_labels": mean_ci(audit_labels, weights), "closed_clusters_due": mean_ci(path["closed"], weights),
            "first_cluster_if_certified": mean_ci(np.where(passed, path["first"], np.nan), weights),
            "first_wall_if_certified": mean_ci(np.where(passed, path["wall"], np.nan), weights)}


def generate(family: str, rng, risk: float):
    shape = (WORLDS, max(HORIZONS), SIZE)
    acts = rng.random(shape, dtype=np.float32) < COVERAGE
    loss_shape = shape if family == "independent" else (*shape[:2], 1)
    acted_loss = rng.random(loss_shape, dtype=np.float32) < risk
    review_loss = rng.random(loss_shape, dtype=np.float32) < REVIEW_RISK
    losses = np.where(acts, acted_loss, review_loss)
    audit_uniform = rng.random(shape, dtype=np.float32)
    # Loss-dependent delay is deliberately adversarial to completion-order sorting.
    delays = rng.integers(0, 21, size=shape, dtype=np.int16) + 80 * losses.astype(np.int16)
    return acts, losses, audit_uniform, delays


def run(output: Path, freeze: str):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise FileExistsError("refuse to overwrite a frozen evaluation; use a new output directory")
    report = {"kind": "SIM", "freeze_commit": freeze,
              "settings": {"epsilon": EPSILON, "delta": DELTA, "size": SIZE, "coverage": COVERAGE,
                           "review_risk": REVIEW_RISK, "worlds": WORLDS, "bootstrap_world_blocks": BOOTSTRAPS,
                           "seed": SEED, "horizons": HORIZONS, "audit_rates": RATES, "acted_risks": RISKS,
                           "families": FAMILIES, "delay": "UniformInteger(0,20) + 80*loss; FIFO closure",
                           "single_frozen_gate": True},
              "environment": {"python": platform.python_version(), "numpy": np.__version__},
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (HERE / "betting.py", HERE / "simulate.py")}, "cells": []}
    for fi, family in enumerate(FAMILIES):
        for ri, risk in enumerate(RISKS):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, fi, ri]))
            acts, losses, uniforms, delays = generate(family, rng, risk)
            brng = np.random.default_rng(np.random.SeedSequence([SEED, fi, ri, 999]))
            weights = brng.multinomial(WORLDS, np.full(WORLDS, 1 / WORLDS), size=BOOTSTRAPS).astype(float)
            for n in HORIZONS:
                a, loss, delay = acts[:, :n], losses[:, :n], delays[:, :n]
                counts, reviewed = a.sum(axis=2), (~a).sum(axis=2)
                coverage = counts.sum(axis=1) / (SIZE * n)
                full_loss, review_loss = (a & loss).sum(axis=2), ((~a) & loss).sum(axis=2)
                full = betting_paths((full_loss - EPSILON * counts) / SIZE, epsilon=EPSILON, level=DELTA,
                                     horizon=n, closure=closure_times(np.ones_like(a), delay))
                review = betting_paths((review_loss - EPSILON * reviewed) / SIZE, epsilon=EPSILON,
                                       level=DELTA, horizon=n, closure=closure_times(~a, delay))
                totals = counts.sum(axis=1)
                review_labels = reviewed.sum(axis=1)
                complete_summary = summarize(full, coverage, full_loss.sum(axis=1) / totals,
                                             np.full(WORLDS, SIZE * n), totals, weights)
                review_summary = summarize(review, coverage, review_loss.sum(axis=1) / review_labels,
                                           review_labels, np.zeros(WORLDS), weights)
                for pi in RATES:
                    audited = a & (uniforms[:, :n] < pi)
                    observed = np.where(audited, loss, np.nan)
                    y = normalized_ipw(a, audited, observed, pi, epsilon=EPSILON, floor=pi)
                    path = betting_paths(y, epsilon=pi * EPSILON, level=DELTA, horizon=n,
                                         closure=closure_times((~a) | audited, delay))
                    audit_labels = audited.sum(axis=(1, 2))
                    estimate = (audited & loss).sum(axis=(1, 2)) / (pi * totals)
                    passed, due = path["p"] <= DELTA, path["p_due"] <= DELTA
                    cell = {"family": family, "risk": risk, "n": n, "pi": pi,
                            "complete": complete_summary, "review_only_naive": review_summary,
                            "ipw": summarize(path, coverage, estimate, audit_labels + review_labels,
                                             audit_labels, weights),
                            "paired_certificate_differences": {
                                "ipw_minus_complete": mean_ci(passed.astype(float) - (full["p"] <= DELTA), weights),
                                "ipw_minus_review": mean_ci(passed.astype(float) - (review["p"] <= DELTA), weights),
                                "delayed_minus_flushed": mean_ci(due.astype(float) - passed, weights)}}
                    report["cells"].append(cell)
                    # Independent replication-level records only; never committed.
                    np.savez_compressed(output / f"{family}-q{risk}-n{n}-pi{pi}.npz",
                                        ipw_p=path["p"], ipw_p_due=path["p_due"], complete_p=full["p"],
                                        review_p=review["p"], coverage=coverage, estimated_risk=estimate,
                                        audit_labels=audit_labels, review_labels=review_labels,
                                        first=path["first"], first_wall=path["wall"], closed=path["closed"])
            print(f"{family} risk={risk}: finished paired 12-cell grid", flush=True)
    report["resources"] = {"wall_seconds": time.monotonic() - started,
                           "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                           "paid_compute_usd": 0}
    # One compact aggregate cell per line rather than thousands of pretty-print lines.
    cells = report.pop("cells")
    prefix = json.dumps(report, separators=(",", ":"), allow_nan=False)[:-1]
    text = prefix + ',"cells":[\n' + ',\n'.join(json.dumps(c, separators=(",", ":"), allow_nan=False)
                                               for c in cells) + '\n]}\n'
    (output / "results.json").write_text(text, encoding="utf-8")
    print(json.dumps(report["resources"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-commit", required=True)
    args = parser.parse_args()
    run(args.output, args.freeze_commit)
