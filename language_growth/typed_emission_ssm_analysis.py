"""Post-run paired task-cluster statistics; never trains or selects a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .typed_emission_ssm import CONFIG, KINDS, sha, write

BOOTSTRAP = {"seed": 3197, "replicates": 2000,
             "unit": "task cluster; all three fixed seeds and both members remain together",
             "interval": "95% percentile; conditional on these seeds and reused development family"}


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP["seed"])
    draws = rng.choice(values, (BOOTSTRAP["replicates"], len(values)), replace=True).mean(1)
    return {"estimate": float(values.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist()}


def analyze(out):
    receipt = json.loads((out / "receipt.json").read_text())
    raw = json.loads((out / "predictions.json").read_text())
    assert receipt["raw_predictions_sha256"] == sha(out / "predictions.json")
    clusters = {}
    for seed, observations in raw.items():
        for observation in observations:
            clusters.setdefault(observation["cluster"], {}).setdefault(seed, []).append(observation)
    assert all(set(by_seed) == set(raw) and all(len(pair) == 2 for pair in by_seed.values()) for by_seed in clusters.values())
    report = {}
    for kind in ("all", *KINDS):
        cohort = [by_seed for by_seed in clusters.values()
                  if kind == "all" or next(iter(by_seed.values()))[0]["gold"]["kind"] == kind]
        statistics = {"unique_task_clusters": len(cohort), "seed_task_pairs": len(cohort) * len(raw)}
        for metric in ("target_exact", "kernel_accepted", "whole_answer_correct"):
            means, both, deltas = [], [], []
            for by_seed in cohort:
                means.append(np.mean([r[metric] for pair in by_seed.values() for r in pair]))
                both.append(np.mean([all(r[metric] for r in pair) for pair in by_seed.values()]))
                deltas.append(np.mean([int(pair[1][metric]) - int(pair[0][metric]) for pair in by_seed.values()]))
            statistics[metric] = {"marginal": interval(means), "both_members": interval(both),
                                  "partner_minus_original": interval(deltas)}
        report[kind] = statistics
    rejected = [r for rows in raw.values() for r in rows if not r["kernel_accepted"]]
    accepted_wrong = [r for rows in raw.values() for r in rows if r["kernel_accepted"] and not r["whole_answer_correct"]]
    compare = [r for rows in raw.values() for r in rows if r["gold"]["kind"] == "compare"]
    failures = {"kernel_rejected": len(rejected), "rejection_reasons": {reason: sum(r["kernel_reason"] == reason for r in rejected) for reason in sorted({r["kernel_reason"] for r in rejected})},
                "accepted_but_wrong_question": len(accepted_wrong),
                "accepted_but_wrong_question_by_kind": {kind: sum(r["gold"]["kind"] == kind for r in accepted_wrong) for kind in KINDS},
                "comparison_structure_exact": sum(r["structure_exact"] for r in compare),
                "comparison_count": len(compare),
                "comparison_relation_correct_with_correct_structure": sum(r["relation_correct"] and r["structure_exact"] for r in compare),
                "comparison_structure_wrong_kernel_accepted": sum(not r["structure_exact"] and r["kernel_accepted"] for r in compare)}
    source = Path(__file__).with_name("typed_emission_ssm.py")
    shutil.copyfile(source, out / source.name)
    shutil.copyfile(__file__, out / Path(__file__).name)
    analysis = {"bootstrap": BOOTSTRAP, "paired_statistics": report, "failure_decomposition": failures,
                "source_hashes": {p.name: sha(p) for p in (source, Path(__file__))},
                "source_receipt_sha256": sha(out / "receipt.json"),
                "family_provenance": "New generator seeds 9101/9102 and newly authored paired records for this run; reused development_train names/ranges/templates. No development_validation, audit, or held-out data, and no held-out generalization claim.",
                "whole_answer_definition": "Exact complete semantic target (including operand order and relation) AND accepted typed program AND checked rendering matching reference. Runtime also asserts equivalence of semantic exactness and accepted matching rendering in every observation.",
                "scope": CONFIG["scope"]}
    write(out / "paired_analysis.json", analysis)
    print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    analyze(parser.parse_args().out)
