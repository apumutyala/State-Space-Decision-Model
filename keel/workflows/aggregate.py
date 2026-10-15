"""Task-cluster percentile bootstrap; no per-episode or dataset text is published."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def cluster_ratio(rows, numerator, denominator=None, *, draws=10000, seed=20261015):
    """Resample user tasks, carrying all their injection goals together.

    The point estimate is a pooled ratio, not a mean of task-specific ratios.
    ``denominator=None`` estimates a mean over episodes. Zero-proposal resamples
    have hold rate zero, matching the zero-review-load convention.
    """
    if not rows:
        raise ValueError("cannot estimate an empty population")
    clusters = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        key = row["user_task"]
        clusters[key][0] += float(row[numerator])
        clusters[key][1] += float(row[denominator]) if denominator else 1.0
    values = np.array([clusters[k] for k in sorted(clusters)], dtype=float)
    indices = np.random.default_rng(seed).integers(0, len(values), size=(draws, len(values)))
    samples = values[indices].sum(axis=1)
    boot = np.divide(samples[:, 0], samples[:, 1], out=np.zeros(draws), where=samples[:, 1] != 0)
    total = values.sum(axis=0)
    lower, upper = np.quantile(boot, [.025, .975])
    return {"estimate": float(total[0] / total[1]) if total[1] else 0.0,
            "ci95": [float(lower), float(upper)], "numerator": float(total[0]),
            "denominator": float(total[1]), "episodes": len(rows), "clusters": len(values)}


def summarize(rows, *, draws=10000, seed=20261015):
    ids = [row["episode_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate episode IDs: do not silently pool reruns")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["suite"], row["arm"])].append(row)
    results = []
    for (suite, arm), group in sorted(groups.items()):
        result = {"suite": suite, "arm": arm, "episodes": len(group),
                  "errors": sum(row["error"] is not None for row in group),
                  "dispatch_errors": sum(row["dispatch_errors"] for row in group),
                  "completion_truncations": sum(row["completion_truncations"] for row in group)}
        for attacked, prefix in [(False, "benign"), (True, "attacked")]:
            subset = [row for row in group if row["attacked"] == attacked]
            metrics = {"utility": ("utility", None), "hold_rate": ("held_calls", "proposed_calls"),
                       "review_episode": ("review_episode", None), "latency_s": ("latency_s", None),
                       "warm_gpu_cost_usd": ("warm_gpu_cost_usd", None)}
            if attacked:
                metrics["targeted_asr"] = ("targeted_asr", None)
            for name, (numerator, denominator) in metrics.items():
                result[f"{prefix}_{name}"] = cluster_ratio(subset, numerator, denominator, draws=draws, seed=seed)
        results.append(result)
    return {"bootstrap": {"method": "percentile", "cluster": "user_task within suite",
                           "draws": draws, "seed": seed, "level": .95}, "results": results}


def format_stat(stat, *, percent=True):
    scale = 100 if percent else 1
    v = stat["estimate"] * scale
    lo, hi = [x * scale for x in stat["ci95"]]
    return f"{v:.1f} [{lo:.1f}, {hi:.1f}]"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", required=True, action="append", help="Repeat for disjoint supplemental arms.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for path in args.episodes for line in Path(path).read_text().splitlines() if line.strip()]
    aggregate = summarize(rows)
    Path(args.output).write_text(json.dumps(aggregate, indent=2) + "\n")
    print("| Suite | Arm | Benign utility | Attacked utility | Targeted ASR | Hold benign | Hold attacked |")
    print("|---|---|---:|---:|---:|---:|---:|")
    keys = ("benign_utility", "attacked_utility", "attacked_targeted_asr", "benign_hold_rate", "attacked_hold_rate")
    for result in aggregate["results"]:
        print("| " + " | ".join([result["suite"], result["arm"], *[format_stat(result[k]) for k in keys]]) + " |")
    print("\n| Suite | Arm | Benign seconds | Attacked seconds | Benign warm GPU $ | Attacked warm GPU $ |")
    print("|---|---|---:|---:|---:|---:|")
    for result in aggregate["results"]:
        times = [format_stat(result[f"{condition}_latency_s"], percent=False) for condition in ("benign", "attacked")]
        costs = [f"{result[f'{condition}_warm_gpu_cost_usd']['estimate']:.4f}" for condition in ("benign", "attacked")]
        print("| " + " | ".join([result["suite"], result["arm"], *times, *costs]) + " |")


if __name__ == "__main__":
    main()
