"""Frozen transfer result analysis and portable raw-artifact preservation."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from . import typed_emission_ssm as core

BOOTSTRAP = {"seed": 3197, "replicates": 2000,
    "unit": "Unique task cluster, preserving all three fixed model seeds and both members",
    "interval": "95% percentile, conditional on these fixed seeds and synthetic development families"}


def interval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP["seed"])
    samples = rng.choice(values, (BOOTSTRAP["replicates"], len(values)), replace=True).mean(1)
    return {"estimate": float(values.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist()}


def analyze(out, tracked, baseline):
    receipt = json.loads((out / "receipt.json").read_text())
    raw = json.loads((out / "predictions.json").read_text())
    assert core.sha(out / "predictions.json") == receipt["raw_predictions_sha256"]
    clusters = {}
    for seed, rows in raw.items():
        for row in rows:
            clusters.setdefault(row["cluster"], {}).setdefault(seed, []).append(row)
    assert len(clusters) == 180
    assert all(set(by_seed) == set(raw) and all(len(pair) == 2 for pair in by_seed.values()) for by_seed in clusters.values())
    statistics = {}
    for kind in ("all", *core.KINDS):
        cohort = [by_seed for by_seed in clusters.values()
                  if kind == "all" or next(iter(by_seed.values()))[0]["gold"]["kind"] == kind]
        stats = {"unique_clusters": len(cohort), "seed_task_pairs": len(cohort) * len(raw)}
        for metric in ("target_exact", "kernel_accepted", "whole_answer_correct"):
            means, both, deltas = [], [], []
            for by_seed in cohort:
                means.append(np.mean([r[metric] for pair in by_seed.values() for r in pair]))
                both.append(np.mean([all(r[metric] for r in pair) for pair in by_seed.values()]))
                deltas.append(np.mean([int(pair[1][metric]) - int(pair[0][metric]) for pair in by_seed.values()]))
            stats[metric] = {"marginal": interval(means), "both_members": interval(both),
                             "partner_minus_original": interval(deltas)}
        statistics[kind] = stats
    all_rows = [r for rows in raw.values() for r in rows]
    rejected = [r for r in all_rows if not r["kernel_accepted"]]
    wrong = [r for r in all_rows if r["kernel_accepted"] and not r["whole_answer_correct"]]
    comparison = [r for r in all_rows if r["gold"]["kind"] == "compare"]
    comparison_pairs = [pair for by_seed in clusters.values() for pair in by_seed.values()
                        if pair[0]["gold"]["kind"] == "compare"]
    swap = {"seed_task_pairs": len(comparison_pairs),
            "gold_relation_changes": sum(p[0]["gold"]["relation"] != p[1]["gold"]["relation"] for p in comparison_pairs),
            "gold_equal_pairs": sum(p[0]["gold"]["relation"] == "equal" for p in comparison_pairs),
            "both_predict_compare": sum(all(r["predicted"]["kind"] == "compare" for r in p) for p in comparison_pairs),
            "predicted_relation_changes": sum(p[0]["predicted"].get("relation") != p[1]["predicted"].get("relation") for p in comparison_pairs),
            "both_answers_correct": sum(all(r["whole_answer_correct"] for r in p) for p in comparison_pairs),
            "both_relation_labels_correct": sum(all(r["relation_correct"] for r in p) for p in comparison_pairs),
            "both_structures_correct": sum(all(r["structure_exact"] for r in p) for p in comparison_pairs),
            "structure_prediction_unchanged": sum({k:v for k,v in p[0]["predicted"].items() if k != "relation"} ==
                                                  {k:v for k,v in p[1]["predicted"].items() if k != "relation"} for p in comparison_pairs),
            "operation_prediction_unchanged": sum(p[0]["predicted"]["kind"] == p[1]["predicted"]["kind"] for p in comparison_pairs),
            "correct_both_inequality_answers": sum(p[0]["gold"]["relation"] != "equal" and all(r["whole_answer_correct"] for r in p) for p in comparison_pairs),
            "correct_both_equality_answers": sum(p[0]["gold"]["relation"] == "equal" and all(r["whole_answer_correct"] for r in p) for p in comparison_pairs)}
    failures = {"kernel_rejected": len(rejected),
                "rejection_reasons": {reason: sum(r["kernel_reason"] == reason for r in rejected) for reason in sorted({r["kernel_reason"] for r in rejected})},
                "accepted_wrong_question": len(wrong),
                "accepted_wrong_question_by_kind": {k: sum(r["gold"]["kind"] == k for r in wrong) for k in core.KINDS},
                "comparison_wrong_operation": sum(not r["operation_correct"] for r in comparison),
                "comparison_correct_structure": sum(r["structure_exact"] for r in comparison),
                "comparison_correct_relation_on_correct_structure": sum(r["structure_exact"] and r["relation_correct"] for r in comparison)}
    checkpoint_proof = {}
    for seed in core.CONFIG["seeds"]:
        current_path = out / f"seed-{seed}.pt"
        baseline_path = baseline / f"seed-{seed}.pt"
        current = torch.load(current_path, weights_only=True, map_location="cpu")["state_dict"]
        previous = torch.load(baseline_path, weights_only=True, map_location="cpu")["state_dict"]
        assert current.keys() == previous.keys()
        assert all(torch.equal(current[name], previous[name]) for name in current)
        checkpoint_proof[str(seed)] = {"every_tensor_bitwise_equal": True, "tensor_count": len(current),
                                       "current_sha256": core.sha(current_path), "original_sha256": core.sha(baseline_path)}
    report = {"bootstrap": BOOTSTRAP, "statistics": statistics, "failures": failures,
              "value_swap_response": swap, "training_reproduction": checkpoint_proof,
              "source_receipt_sha256": core.sha(out / "receipt.json"),
              "scope": "Random-initialized train-to-disjoint-validation synthetic development transfer; not official holdout or parent retention; no tuning"}
    core.write(out / "transfer_analysis.json", report)
    tracked.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for filename, expected in (("predictions.json", receipt["raw_predictions_sha256"]),
                               ("frozen_rows.json", receipt["data_sha256"])):
        source = (out / filename).read_bytes()
        assert hashlib.sha256(source).hexdigest() == expected
        destination = tracked / (filename + ".gz")
        destination.write_bytes(gzip.compress(source, compresslevel=9, mtime=0))
        assert gzip.decompress(destination.read_bytes()) == source
        manifest[destination.name] = {"compressed_sha256": core.sha(destination), "decompressed_sha256": expected,
                                      "compressed_bytes": destination.stat().st_size, "decompressed_bytes": len(source)}
    core.write(tracked / "raw_artifact_manifest.json", manifest)
    for filename in ("registration.json", "receipt.json", "transfer_analysis.json"):
        shutil.copyfile(out / filename, tracked / filename)
    for filename in ("typed_emission_ssm.py", "typed_emission_ssm_transfer.py", Path(__file__).name):
        shutil.copyfile(Path(__file__).with_name(filename), out / filename)
    print(json.dumps({"failures": failures, "value_swap": swap, "statistics": statistics,
                      "training_reproduction": checkpoint_proof, "artifacts": manifest}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tracked", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    analyze(args.out, args.tracked, args.baseline)
