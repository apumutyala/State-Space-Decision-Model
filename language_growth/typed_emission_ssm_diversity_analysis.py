"""Paired analysis against saved transfer predictions; no extra fitting."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from . import typed_emission_ssm as core

BOOTSTRAP = {"seed": 3197, "replicates": 2000,
    "unit": "Unique task cluster, preserving both conditions, all three fixed model seeds, and both paired members",
    "interval": "95% paired percentile; conditional on these model seeds and synthetic development families"}


def ci(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP["seed"])
    draws = rng.choice(values, (BOOTSTRAP["replicates"], len(values)), replace=True).mean(1)
    return {"estimate": float(values.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist()}


def failures(rows):
    rejected = [r for r in rows if not r["kernel_accepted"]]
    wrong = [r for r in rows if r["kernel_accepted"] and not r["whole_answer_correct"]]
    return {"rejected": len(rejected),
        "reject_reasons": {reason: sum(r["kernel_reason"] == reason for r in rejected) for reason in sorted({r["kernel_reason"] for r in rejected})},
        "accepted_wrong_target": len(wrong),
        "accepted_wrong_target_by_kind": {k: sum(r["gold"]["kind"] == k for r in wrong) for k in core.KINDS},
        "wrong_target_total": sum(not r["target_exact"] for r in rows),
        "comparison_wrong_operation": sum(r["gold"]["kind"] == "compare" and not r["operation_correct"] for r in rows),
        "comparison_structure_exact": sum(r["gold"]["kind"] == "compare" and r["structure_exact"] for r in rows)}


def swaps(raw):
    pairs = []
    for seed, rows in raw.items():
        clusters = {}
        for row in rows:
            if row["gold"]["kind"] == "compare":
                clusters.setdefault(row["cluster"], []).append(row)
        pairs += list(clusters.values())
    assert len(pairs) == 180 and all(len(p) == 2 for p in pairs)
    return {"seed_task_pairs": len(pairs),
        "required_relation_changes": sum(p[0]["gold"]["relation"] != p[1]["gold"]["relation"] for p in pairs),
        "equality_pairs": sum(p[0]["gold"]["relation"] == "equal" for p in pairs),
        "both_predict_compare": sum(all(r["predicted"]["kind"] == "compare" for r in p) for p in pairs),
        "predicted_relation_changes": sum(p[0]["predicted"].get("relation") != p[1]["predicted"].get("relation") for p in pairs),
        "both_structures_exact": sum(all(r["structure_exact"] for r in p) for p in pairs),
        "both_answers_correct": sum(all(r["whole_answer_correct"] for r in p) for p in pairs),
        "both_relation_labels_correct": sum(all(r["relation_correct"] for r in p) for p in pairs),
        "both_inequality_answers_correct": sum(p[0]["gold"]["relation"] != "equal" and all(r["whole_answer_correct"] for r in p) for p in pairs),
        "both_equality_answers_correct": sum(p[0]["gold"]["relation"] == "equal" and all(r["whole_answer_correct"] for r in p) for p in pairs),
        "structure_unchanged": sum({k:v for k,v in p[0]["predicted"].items() if k != "relation"} ==
                                  {k:v for k,v in p[1]["predicted"].items() if k != "relation"} for p in pairs)}


def analyze(out, tracked):
    receipt = json.loads((out / "receipt.json").read_text())
    registration = json.loads((out / "registration.json").read_text())
    assert core.sha(out / "predictions.json") == receipt["raw_predictions_sha256"]
    assert core.sha(out / "baseline_predictions.json") == registration["baseline_predictions_sha256"]
    current = json.loads((out / "predictions.json").read_text())
    baseline = json.loads((out / "baseline_predictions.json").read_text())
    matched = {}
    by_seed = {}
    for seed in current:
        before = {r["id"]: r for r in baseline[seed]}
        after = {r["id"]: r for r in current[seed]}
        assert before.keys() == after.keys()
        for ident, row in after.items():
            old = before[ident]
            assert (old["gold"], old["cluster"], old["member"]) == (row["gold"], row["cluster"], row["member"])
            matched.setdefault(row["cluster"], []).append((old, row))
        by_seed[seed] = {kind: {"count": len(rows),
            "baseline_correct": sum(b["whole_answer_correct"] for b,a in rows),
            "current_correct": sum(a["whole_answer_correct"] for b,a in rows),
            "improved": sum(not b["whole_answer_correct"] and a["whole_answer_correct"] for b,a in rows),
            "regressed": sum(b["whole_answer_correct"] and not a["whole_answer_correct"] for b,a in rows)}
            for kind in ("all", *core.KINDS)
            for rows in [[(before[i], after[i]) for i in after if kind == "all" or after[i]["gold"]["kind"] == kind]]}
    assert len(matched) == 180 and all(len(rows) == 6 for rows in matched.values())
    statistics = {}
    for kind in ("all", *core.KINDS):
        clusters = [pairs for pairs in matched.values() if kind == "all" or pairs[0][1]["gold"]["kind"] == kind]
        stats = {"unique_clusters": len(clusters)}
        for metric in ("target_exact", "kernel_accepted", "whole_answer_correct", "operation_correct", "structure_exact"):
            before = [np.mean([b[metric] for b,a in pairs]) for pairs in clusters]
            after = [np.mean([a[metric] for b,a in pairs]) for pairs in clusters]
            stats[metric] = {"baseline": ci(before), "current": ci(after),
                             "paired_delta": ci(np.asarray(after) - np.asarray(before))}
        both_before, both_after = [], []
        for pairs in clusters:
            assert len(pairs) == 6
            both_before.append(np.mean([all(pairs[i+j][0]["whole_answer_correct"] for j in (0,1)) for i in (0,2,4)]))
            both_after.append(np.mean([all(pairs[i+j][1]["whole_answer_correct"] for j in (0,1)) for i in (0,2,4)]))
        stats["both_members_correct"] = {"baseline": ci(both_before), "current": ci(both_after),
            "paired_delta": ci(np.asarray(both_after) - np.asarray(both_before))}
        statistics[kind] = stats
    report = {"bootstrap": BOOTSTRAP, "statistics": statistics, "per_seed": by_seed,
        "failures": {"baseline": failures([r for rows in baseline.values() for r in rows]),
                     "current": failures([r for rows in current.values() for r in rows])},
        "value_swap": {"baseline": swaps(baseline), "current": swaps(current)},
        "scope": "One preregistered six-template training-only wording intervention; iterative synthetic development only, no official holdout or parent retention",
        "analysis_source_sha256": core.sha(__file__)}
    core.write(out / "paired_analysis.json", report)
    tracked.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in ("predictions.json", "frozen_rows.json", "baseline_predictions.json"):
        original = (out / name).read_bytes()
        destination = tracked / (name + ".gz")
        destination.write_bytes(gzip.compress(original, compresslevel=9, mtime=0))
        assert gzip.decompress(destination.read_bytes()) == original
        manifest[destination.name] = {"compressed_sha256": core.sha(destination),
            "decompressed_sha256": hashlib.sha256(original).hexdigest(),
            "compressed_bytes": destination.stat().st_size, "decompressed_bytes": len(original)}
    core.write(tracked / "raw_artifact_manifest.json", manifest)
    for name in ("registration.json", "receipt.json", "baseline_receipt.json", "paired_analysis.json"):
        shutil.copyfile(out / name, tracked / name)
    for name in ("typed_emission_ssm.py", "typed_emission_ssm_diversity.py", Path(__file__).name):
        shutil.copyfile(Path(__file__).with_name(name), out / name)
    print(json.dumps({"per_seed": by_seed, "failures": report["failures"], "value_swap": report["value_swap"],
                      "whole_answer_statistics": {k: {m:v for m,v in s.items() if m in ("whole_answer_correct", "both_members_correct")} for k,s in statistics.items()},
                      "artifacts": manifest}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tracked", required=True, type=Path)
    args = parser.parse_args()
    analyze(args.out, args.tracked)
