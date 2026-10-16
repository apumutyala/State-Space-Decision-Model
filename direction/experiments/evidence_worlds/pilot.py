"""The frozen 200-world synthetic CPU audit; all per-world output stays private."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import resource
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from keel.card.metrics import cluster_rate
from keel.card.worlds import (
    GRAMMARS, VARIANTS, VERSION, card_units, constraint_answer, decision_request,
    generate_worlds, transition_answer,
)
from keel.paths import private_root
from keel.protocol import render

SEED = 20261016
BOOT_SEED = 61016
N_BOOT = 10000


def rate(values, clusters, eligible=None):
    result = cluster_rate(values, clusters, eligible=eligible, n_boot=N_BOOT, seed=BOOT_SEED)
    return {key: result[key] for key in ("rate", "ci95", "n_success", "n_eligible", "n_clusters")}


def wilson(success, n):
    z = 1.959963984540054
    p = success / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0, center - radius), min(1, center + radius)]


def summarize(rows):
    clusters = [row["world_id"] for row in rows]
    balance = {v: rate([r["gold"][v] == "yes" for r in rows], clusters) for v in VARIANTS}
    disagreement = rate([r["gold"] != r["constraint"] for r in rows], clusters)
    disagreement["wilson_world_ci95"] = wilson(disagreement["n_success"], len(rows))
    outcomes = ("clean", "factual", "policy", "execute_directive")
    pairs = list(itertools.combinations(outcomes, 2))
    collisions = {f"{a}={b}": rate([r["outcomes"][a] == r["outcomes"][b] for r in rows], clusters)
                  for a, b in pairs}
    collisions["all_six_pairs"] = rate(
        [r["outcomes"][a] == r["outcomes"][b] for r in rows for a, b in pairs],
        [r["world_id"] for r in rows for _ in pairs])
    for metric in collisions.values():
        # Boundary bootstrap intervals are degenerate. A world-level Bernoulli
        # bound supplements the individual contrasts, not the six-pair average.
        if metric["n_eligible"] == len(rows):
            metric["wilson_world_ci95"] = wilson(metric["n_success"], len(rows))
    controls = {}
    for control in ("ignore_all_evidence", "follow_last_directive_else_oracle"):
        correct = [{v: r["controls"][control][v] == r["gold"][v] for v in VARIANTS} for r in rows]
        clean_correct = [c["clean"] for c in correct]
        result = {v: rate([c[v] for c in correct], clusters) for v in VARIANTS}
        result["joint"] = rate([all(c.values()) for c in correct], clusters)
        result["joint"]["wilson_world_ci95"] = wilson(result["joint"]["n_success"], len(rows))
        result["mean_accuracy"] = rate([c[v] for c in correct for v in VARIANTS],
                                       [r["world_id"] for r in rows for _ in VARIANTS])
        result["attack_steering_given_clean"] = rate(
            [r["controls"][control]["directive"] == r["outcomes"]["execute_directive"] for r in rows],
            clusters, clean_correct)
        for variant in ("factual", "policy"):
            result[f"{variant}_given_clean"] = rate([c[variant] for c in correct], clusters, clean_correct)
        controls[control] = result
    differences = np.asarray([
        sum(r["controls"]["follow_last_directive_else_oracle"][v] == r["gold"][v] for v in VARIANTS) / 4
        - sum(r["controls"]["ignore_all_evidence"][v] == r["gold"][v] for v in VARIANTS) / 4 for r in rows])
    rng = np.random.default_rng(BOOT_SEED)
    draws = differences[rng.integers(len(rows), size=(N_BOOT, len(rows)))].mean(axis=1)
    return {"n_worlds": len(rows), "oracle_disagreement": disagreement, "yes_balance": balance,
            "collisions": collisions, "controls": controls,
            "paired_control_mean_difference": {"rate": float(differences.mean()),
                                                "ci95": np.quantile(draws, [.025, .975]).tolist()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=private_root() / "research-evidence-worlds")
    args = parser.parse_args()
    output = args.output.resolve()
    private = private_root().resolve()
    if not output.is_relative_to(private) or output == private:
        parser.error("per-world outputs must be in a slice directory below SSDM_PRIVATE")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    population = generate_worlds(seed=SEED, per_grammar=50)
    rows = []
    digest = hashlib.sha256()
    with (output / "worlds.jsonl").open("w", encoding="utf-8") as stream:
        for world in population:
            gold = {c.variant: transition_answer(c) for c in world.cases}
            independent = {c.variant: constraint_answer(c) for c in world.cases}
            directive = world.case("directive").directive
            controls = {"ignore_all_evidence": dict.fromkeys(VARIANTS, "no"),
                        "follow_last_directive_else_oracle": {**gold, "directive": directive}}
            row = {"world_id": world.world_id, "grammar": world.cases[0].grammar, "gold": gold,
                   "constraint": independent, "controls": controls,
                   "outcomes": {**{v: gold[v] for v in ("clean", "factual", "policy")},
                                "execute_directive": directive}}
            rows.append(row)
            requests = {v: decision_request(world, v).to_json() for v in VARIANTS}
            serialized = json.dumps({**row, "world": asdict(world), "requests": requests},
                                    sort_keys=True, separators=(",", ":")) + "\n"
            digest.update(serialized.encode())
            stream.write(serialized)
    # Fail closed before publishing a measurement contract with conflicting labels.
    if any(r["gold"] != r["constraint"] for r in rows):
        raise RuntimeError("oracle disagreement: inspect private worlds.jsonl; no summary published")
    units = [unit for world in population for unit in card_units(world)]
    report = {"status": "SIM: finite synthetic contract audit; no model inference", "version": VERSION,
              "seed": SEED, "per_grammar": 50, "worlds": len(population), "decisions": 4 * len(population),
              "card_units": len(units), "n_boot": N_BOOT, "bootstrap_seed": BOOT_SEED,
              "cluster": "base world, not variant; fixed four-grammar mixture",
              "data_sha256": digest.hexdigest(),
              "groups": {g: summarize([r for r in rows if g == "all" or r["grammar"] == g])
                         for g in ("all", *GRAMMARS)}}
    example = decision_request(population[0], "policy")
    (output / "example.json").write_text(json.dumps({"request": example.to_json(),
        "gold": transition_answer(population[0].case("policy")), "prompt": render(example).prompt}, indent=2) + "\n")
    report["runtime"] = {"wall_s": time.perf_counter() - started,
                         "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         "python": platform.python_version(), "numpy": np.__version__}
    (output / "summary.json").write_text(json.dumps(report, separators=(",", ":"), sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "worlds": len(population), "oracle_disagreements": 0,
                      "data_sha256": report["data_sha256"], "runtime": report["runtime"]}, indent=2))


if __name__ == "__main__":
    main()
