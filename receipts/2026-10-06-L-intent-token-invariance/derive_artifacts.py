"""Preserve initial weights and paired exploratory statistics; never train/infer."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

VARIANT = Path("C:/Users/apumu/research/CHLabs/ssdm-data/l0-intent-token-invariance-20261006")
BASELINE = Path("C:/Users/apumu/research/CHLabs/ssdm-data/l0-transparent-cpu-validation-20261006")
DESTINATION = Path(__file__).resolve().parent
BOOTSTRAP_SEED = 1729
BOOTSTRAP_SAMPLES = 10000


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path, data):
    payload = (canonical(data) + "\n").encode("ascii")
    if path.exists() and path.read_bytes() != payload:
        raise ValueError("refuse replacing different preserved artifact")
    path.write_bytes(payload)


def interval(values):
    return np.quantile(values, [0.025, 0.975]).tolist()


def main():
    receipt = json.loads((VARIANT / "receipt.json").read_text())
    prior = json.loads((BASELINE / "receipt.json").read_text())
    if receipt["evaluation_tasks_sha256"] != prior["evaluation_tasks_sha256"]:
        raise ValueError("paired worlds must be identical")
    tasks = [json.loads(line) for line in (VARIANT / "evaluation_tasks.jsonl").read_text().splitlines()]
    task_ids = [task["task_id"] for task in tasks]
    if len(set(task_ids)) != len(task_ids) or len(task_ids) != 300:
        raise ValueError("expected 300 unique paired tasks")
    kinds = {task["task_id"]: task["training_only"]["kind"] for task in tasks}
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(task_ids),
                                                             (BOOTSTRAP_SAMPLES, len(task_ids)))
    all_deltas, runs, preserved = [], [], []
    for seed in receipt["config"]["seeds"]:
        model = json.loads((VARIANT / f"model-{seed}.json").read_text())
        # Exact initial array construction in fit(), before any optimizer step.
        initial = {**model, "weights": np.random.default_rng(seed).normal(0, 0.01, len(model["features"])).tolist()}
        expected_revision = "transparent-cpu:" + hashlib.sha256(
            canonical({"model": initial, "numerical": True}).encode()).hexdigest()
        new_raw = json.loads((VARIANT / f"predictions-{seed}.json").read_text())
        old_raw = json.loads((BASELINE / f"predictions-{seed}.json").read_text())
        if not new_raw["initial_calls"] or any(call["response"]["ssdm"]["diagnostics"]["revision_id"]
                                              != expected_revision for call in new_raw["initial_calls"]):
            raise ValueError("derived initial weights do not match measured raw revision")
        name = f"initial-model-{seed}.json"
        save_new(VARIANT / name, initial)
        save_new(DESTINATION / name, initial)
        preserved.append({"seed": seed, "filename": name, "sha256": sha(VARIANT / name),
                          "raw_initial_revision_id": expected_revision,
                          "verified_against_all_initial_call_revisions": True})
        new = {row["task_id"]: row["task_correct"] for row in new_raw["report"]["observations"]}
        old = {row["task_id"]: row["task_correct"] for row in old_raw["report"]["observations"]}
        if set(new) != set(old) or set(new) != set(task_ids):
            raise ValueError("paired task identity mismatch")
        paired = [{"task_id": tid, "kind": kinds[tid], "baseline_correct": old[tid],
                   "variant_correct": new[tid], "delta": int(new[tid]) - int(old[tid])} for tid in task_ids]
        deltas = np.array([row["delta"] for row in paired], dtype=np.float64)
        all_deltas.append(deltas)
        boot = deltas[indices].mean(axis=1)
        runs.append({"seed": seed, "task_count": len(task_ids), "paired_delta": float(deltas.mean()),
                     "paired_bootstrap_95_percentile_ci": interval(boot),
                     "improved_tasks": int(np.sum(deltas > 0)), "regressed_tasks": int(np.sum(deltas < 0)),
                     "by_kind": {kind: {"count": sum(row["kind"] == kind for row in paired),
                                        "delta_sum": sum(row["delta"] for row in paired if row["kind"] == kind)}
                                 for kind in ("read", "explain", "compare")},
                     "baseline_raw_sha256": sha(BASELINE / f"predictions-{seed}.json"),
                     "variant_raw_sha256": sha(VARIANT / f"predictions-{seed}.json"),
                     "paired_tasks": paired})
    # The same worlds recur in all three seeds. Resample 300 task clusters,
    # averaging across seeds within each task; never treat 900 pairs as IID.
    per_task_mean = np.array(all_deltas).mean(axis=0)
    summary = {"schema": "ssdm.exploratory_paired_token_filter.v1", "runs": runs,
               "pooled": {"unique_task_clusters": len(task_ids), "model_initializations": len(runs),
                          "paired_delta": float(per_task_mean.mean()),
                          "paired_bootstrap_95_percentile_ci": interval(per_task_mean[indices].mean(axis=1))},
               "bootstrap": {"seed": BOOTSTRAP_SEED, "samples": BOOTSTRAP_SAMPLES,
                             "method": "paired task-ID percentile bootstrap; same draws for seeds; pooled task-cluster mean"},
               "evaluation_tasks_sha256": receipt["evaluation_tasks_sha256"],
               "training_records_sha256": receipt["train_records_sha256"],
               "initial_checkpoints": preserved, "variant_fit_source_sha": receipt["source_sha"],
               "analysis_script_sha256": sha(Path(__file__)),
               "command": [sys.executable, str(Path(__file__).resolve())],
               "scope": "exploratory iterative development diagnostic selected after baseline failures; reused validation cohort retired from confirmatory use; bootstrap does not account for feature selection or untested domain shift; no retraining/inference"}
    save_new(DESTINATION / "paired_analysis.json", summary)
    save_new(VARIANT / "paired_analysis.json", summary)
    print(canonical({"pooled": summary["pooled"], "initial_checkpoints": preserved,
                     "analysis_sha256": sha(DESTINATION / "paired_analysis.json")}))


if __name__ == "__main__":
    main()
