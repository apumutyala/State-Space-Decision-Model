"""Frozen SIM sweep; write schedule-level records only to SSDM_PRIVATE.

Run as a module in a no-network sandbox; see README for the resource-limited command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time

import numpy as np

from .sim import CONTROLS, POLICIES, SEED, controls, schedule, simulate, validate_host

RHOS = (.25, .50, .75, .95, 1.10)
CAPACITIES = (16, 64)
TIMINGS = ("sustained", "burst")
IDENTITIES = ("authenticated", "sybil")
RATES = (0, .5, 1, 2, 4)
NAMES = POLICIES + CONTROLS
METRICS = ("completion", "harmful_effects", "p95_wait", "starvation", "held_benign",
           "review_work", "reviews_started", "rejected_harmful", "max_waiting")
B = 4000
HERE = Path(__file__).resolve().parent


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def vector(stats: dict) -> list:
    return [stats[m] if stats[m] is not None else np.nan for m in METRICS]


def run(cv: int, destination: Path, development: bool) -> None:
    started = time.monotonic()
    reps, n = (5, 100) if development else (100, 1000)
    data = np.full((len(RHOS), 2, 2, 2, 5, reps, len(NAMES), len(METRICS)), np.nan)
    validation = validate_host()
    for ri, rho in enumerate(RHOS):
        for qi, capacity in enumerate(CAPACITIES):
            for ai, rate in enumerate(RATES):
                for ti, timing in enumerate(TIMINGS):
                    if rate == 0 and ti == 1:
                        data[ri, qi, ti, :, ai] = data[ri, qi, 0, :, ai]
                        continue
                    for rep in range(reps):
                        jobs = schedule(rho, cv, rep, rate, timing, "authenticated", n=n, development=development)
                        fifo = simulate(jobs, "fifo", capacity)
                        reserved = simulate(jobs, "reserved", capacity)
                        base = {"fifo": fifo, "reserved": reserved, **controls(jobs, fifo)}
                        for ii, identity in enumerate(IDENTITIES):
                            if ii and rate == 0:
                                data[ri, qi, ti, ii, ai, rep] = data[ri, qi, ti, 0, ai, rep]
                                continue
                            if ii:
                                jobs = schedule(rho, cv, rep, rate, timing, identity, n=n, development=development)
                            stats = {**base, "fair": simulate(jobs, "fair", capacity),
                                     "cap": simulate(jobs, "cap", capacity)}
                            for pi, policy in enumerate(NAMES):
                                data[ri, qi, ti, ii, ai, rep, pi] = vector(stats[policy])
                            assert all(stats[p]["harmful_effects"] == 0 for p in POLICIES)
        print(json.dumps({"cv": cv, "rho_done": rho, "seconds": round(time.monotonic() - started, 2)}), flush=True)
    sensitivity = np.full((5, reps, len(POLICIES), len(METRICS)), np.nan)
    if cv == 1:
        for ai, rate in enumerate(RATES):
            for rep in range(reps):
                jobs = schedule(.75, 1, rep, rate, "sustained", "entitlement_breach", n=n, development=development)
                for pi, policy in enumerate(POLICIES):
                    sensitivity[ai, rep, pi] = vector(simulate(jobs, policy, 16))
    prefix = "development" if development else "confirmation"
    path = destination / f"{prefix}-cv{cv}.npz"
    np.savez_compressed(path, data=data, sensitivity=sensitivity)
    manifest = {"label": "SIM", "seed": SEED, "development": development, "cv": cv,
                "replicates": reps, "benign_per_schedule": n, "cells": 200,
                "wall_seconds": time.monotonic() - started,
                "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "python": platform.python_version(), "numpy": np.__version__,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "kernel_validation": validation}
    dump(destination / f"{prefix}-cv{cv}.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "kernel_validation"}), flush=True)


def finite(value):
    if isinstance(value, np.ndarray):
        return [finite(x) for x in value]
    return round(float(value), 6) if np.isfinite(value) else None


def describe(matrix: np.ndarray, weights: np.ndarray) -> dict:
    """matrix = independent schedule x policy x metric; share all resamples."""
    keep = np.isfinite(matrix)
    flat = np.nan_to_num(matrix).reshape(len(matrix), -1)
    denom = weights @ keep.reshape(len(matrix), -1)
    boot = np.divide(weights @ flat, denom, out=np.full_like(denom, np.nan), where=denom > 0)
    boot = boot.reshape(len(weights), *matrix.shape[1:])
    count = keep.sum(axis=0)
    means = np.divide(np.nan_to_num(matrix).sum(axis=0), count,
                      out=np.full(matrix.shape[1:], np.nan), where=count > 0)
    output = {}
    for pi, policy in enumerate(NAMES[:matrix.shape[1]]):
        item = {}
        for metric in ("completion", "p95_wait", "starvation", "held_benign", "review_work"):
            mi = METRICS.index(metric)
            samples = boot[:, pi, mi]
            item[metric] = finite(np.r_[means[pi, mi], np.quantile(samples[np.isfinite(samples)], [.025, .975])]) if np.any(np.isfinite(samples)) else [None] * 3
        item["harmful_effects_sum"] = int(matrix[:, pi, 1].sum())
        diff = boot[:, pi, 0] - boot[:, 0, 0]
        item["paired_completion_minus_fifo"] = finite(np.r_[means[pi, 0] - means[0, 0], np.quantile(diff, [.025, .975])])
        item["paired_family_ci"] = finite(np.quantile(diff, [.05 / 6, 1 - .05 / 6]))
        item["paired_sd"] = finite(np.std(matrix[:, pi, 0] - matrix[:, 0, 0], ddof=1))
        output[policy] = item
    return output


def attack_cost(curves: np.ndarray, weights: np.ndarray) -> dict:
    """timing x rate x schedule x policy; censor instead of interpolating."""
    point = curves.mean(axis=2)
    boot = np.einsum("bs,tras->btra", weights, curves.transpose(0, 1, 3, 2))
    answer = {}
    for pi, policy in enumerate(NAMES):
        # Infinite/unsafe are trivial; always-hold has a zero service denominator.
        if policy in CONTROLS:
            continue
        eligible = point[:, 1:, pi] <= point[:, :1, pi] * .5
        possible = np.argwhere(eligible)
        cost = min((RATES[ai + 1] for _, ai in possible), default=np.inf)
        samples = np.full(len(weights), np.inf)
        for ai, rate in enumerate(RATES[1:], start=1):
            halves = (boot[:, :, ai, pi] <= boot[:, :, 0, pi] * .5).any(axis=1)
            samples[halves] = np.minimum(samples[halves], rate)
        encode = lambda x: float(x) if np.isfinite(x) else ">4 tested"
        quantiles = np.quantile(samples, [.025, .975], method="inverted_cdf")
        answer[policy] = {"requests_per_benign": encode(cost), "ci95": [encode(x) for x in quantiles],
                          "bootstrap_censored_fraction": float(np.isinf(samples).mean())}
    return answer


def summarize(destination: Path, output: Path) -> None:
    started = time.monotonic()
    arrays, manifests = [], []
    for cv in (0, 1, 2):
        arrays.append(np.load(destination / f"confirmation-cv{cv}.npz"))
        manifests.append(json.loads((destination / f"confirmation-cv{cv}.json").read_text()))
    # CV x rho x storage x timing x identity x rate x schedule x policy x metric.
    data = np.stack([x["data"] for x in arrays])
    rng = np.random.default_rng(SEED + 900)
    weights = rng.multinomial(100, np.full(100, .01), size=B) / 100
    rows, selected = [], []
    for ci, cv in enumerate((0, 1, 2)):
        for ri, rho in enumerate(RHOS):
            for qi, capacity in enumerate(CAPACITIES):
                for ti, timing in enumerate(TIMINGS):
                    for ii, identity in enumerate(IDENTITIES):
                        for ai, rate in enumerate(RATES):
                            key = {"cv": cv, "rho": rho, "capacity": capacity, "timing": timing,
                                   "identity": identity, "attack_requests_per_benign": rate}
                            summary = describe(data[ci, ri, qi, ti, ii, ai], weights)
                            row = {**key, "policies": summary}
                            rows.append(row)
                            anchor = ci == 1 and ri == 2 and qi == 0 and ti == 0
                            service = ri == 2 and qi == 0 and ai == 4
                            utilization = ci == 1 and qi == 0 and ti == 0 and ai == 4
                            storage = ci == 1 and ri == 2 and ti == 0 and ai == 4
                            if anchor or service or utilization or storage:
                                selected.append(row)
    with (destination / "all_cells.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    costs = []
    for ci, cv in enumerate((0, 1, 2)):
        for ri, rho in enumerate(RHOS):
            for qi, capacity in enumerate(CAPACITIES):
                for ii, identity in enumerate(IDENTITIES):
                    costs.append({"cv": cv, "rho": rho, "capacity": capacity, "identity": identity,
                                  "attacker_principals": 1 if ii == 0 else 32,
                                  "policies": attack_cost(data[ci, ri, qi, :, ii, :, :, :, 0], weights)})
    dump(destination / "all_costs.json", costs)
    # Prespecified anchor sensitivity; no model/policy selection after outcomes.
    breach = [{"attack_requests_per_benign": rate,
               "policies": describe(arrays[1]["sensitivity"][ai], weights)} for ai, rate in enumerate(RATES)]
    summaries = {}
    for ii, identity in enumerate(IDENTITIES):
        subset = data[:, :, :, :, ii, 1:]
        # Align the 100 independent seed blocks, preserving all within-seed pairing.
        pooled = np.nan_to_num(subset).mean(axis=(0, 1, 2, 3, 4))
        # Restore undefined waits for always-hold rather than presenting fake zeros.
        pooled[:, NAMES.index("always_hold"), METRICS.index("p95_wait")] = np.nan
        summaries[identity] = describe(pooled, weights)
    result = {"label": "SIM", "method_commit": "72fc51a7", "seed": SEED,
              "cells": 600, "schedules_per_cell": 100, "benign_per_schedule": 1000,
              "bootstrap_resamples": B, "policies": NAMES,
              "resource_match": {"reviewers": 4, "queue_capacities": CAPACITIES},
              "interval_order": ["mean", "lo95", "hi95"],
              "family_interval": "98.333% paired, three comparisons within cell only",
              "runs": [{k: v for k, v in m.items() if k != "kernel_validation"} for m in manifests],
              "kernel_validation": manifests[0]["kernel_validation"],
              "harmful_fail_closed_effects": int(data[..., :4, 1].sum()),
              "pooled_attacked_cells": summaries, "selected_cells": selected,
              "anchor_attack_cost": [c for c in costs if c["cv"] == 1 and c["rho"] == .75 and c["capacity"] == 16],
              "entitlement_breach": breach,
              "summary_wall_seconds": time.monotonic() - started}
    dump(output, result)
    print(json.dumps({"cells": len(rows), "selected_cells": len(selected),
                      "output_bytes": output.stat().st_size, "seconds": time.monotonic() - started}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=int, choices=(0, 1, 2))
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    args = parser.parse_args()
    destination = Path(os.environ["SSDM_PRIVATE"]) / "research-review-queue"
    destination.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        summarize(destination, args.output)
    elif args.cv is not None:
        run(args.cv, destination, args.development)
    else:
        parser.error("supply --cv or --summarize")


if __name__ == "__main__":
    main()
