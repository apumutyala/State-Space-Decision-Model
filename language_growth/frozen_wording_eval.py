"""Evaluate exactly three PR #65 frozen checkpoints; no fitting or tuning API."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

from .transparent_cpu import LinearOptionBackend, FEATURE_VERSION, canonical, sha, np, LoggedProvider
from decide.adapter import Decider
from decide.calibration import OptionCountTemperature
from .free_running_dev import run_development_tasks
from .kernel import check_program
from .realize import render_checked
from .paraphrase_data import generate_paraphrase_tasks, paraphrase_manifest

MODEL_HASHES = {
    17: "fb7c5b1e64111291a10d0733dce85856ca3ec9cd30e886ab499c732b5f34649d",
    29: "59ed7ff5d4906b028df50a99613053b4abd2dda0b526712e82b285289f22ddd0",
    43: "1b84914605f97167e330306957a55df030577387c7a3f4e6abd092434303ef22",
}
MODEL_DIR = Path("receipts/2026-10-06-L-intent-token-invariance")
BOOTSTRAP_SEED, BOOTSTRAP_SAMPLES = 1729, 10000


def save_compressed(path, payload):
    encoded = payload.encode("ascii")
    path.write_bytes(gzip.compress(encoded, mtime=0))
    return {"filename": path.name, "sha256": sha(path),
            "decompressed_sha256": hashlib.sha256(encoded).hexdigest()}


class LoggedFrozenBackend(LinearOptionBackend):
    def __init__(self, model, deadline):
        super().__init__(model)
        self.deadline, self.scores = deadline, []

    def score(self, request, check_deadline):
        self.deadline()
        result = super().score(request, check_deadline)
        self.scores.append({"logits": result.logits,
                            "feature_contributions": result.extras["feature_contributions"]})
        return result


def evaluate(tasks, model, deadline):
    """Actual Decider/planner/kernel/renderer rollout; labels consumed afterwards."""
    before = canonical(model)
    backend = LoggedFrozenBackend(model, deadline)
    calibration = OptionCountTemperature()
    provider = LoggedProvider(Decider(backend, calibration=calibration))
    outcomes, calls, slot_rows = [], [], []
    for task in tasks:
        deadline()
        provider.calls.clear()
        backend.scores.clear()
        report = run_development_tasks([task], provider, model=backend.name,
            parent_revision=backend.revision()["revision_id"],
            expected_served_config={"calibration": calibration.identity(),
                "policy_masking": "none (served covers every supplied option)"})
        observation = report["observations"][0]
        training = task["training_only"]
        reference = check_program(training["program"], training["host"], training["catalog"])
        observation["exact_program_correct"] = observation["program"] == training["program"]
        observation["exact_text_correct"] = observation["text"] == render_checked(reference, training["host"], training["catalog"])
        observation.update(kind=training["kind"], template_id=training["template_id"])
        outcomes.append(observation)
        if len(provider.calls) != len(backend.scores):
            raise ValueError("all measured calls must retain scores")
        selected = {}
        for call, scores in zip(provider.calls, backend.scores):
            qid = next(iter(call["request"]["questions"]))
            slot = qid.rsplit("-", 1)[1]
            selected[slot] = call["response"].get("answers", {}).get(qid, {}).get("choice")
            calls.append({"task_id": task["task_id"], **call, **scores})
        # Missing gold-required slots count as incorrect, never disappear from
        # the denominator. Unexpected actual slots are retained separately.
        expected = {t["slot"]: t["option_id"] for t in training["targets"]}
        for slot, gold in expected.items():
            slot_rows.append({"task_id": task["task_id"], "template_id": training["template_id"],
                "slot": slot, "gold": gold, "selected": selected.get(slot),
                "reached": slot in selected, "correct": selected.get(slot) == gold})
        observation["unexpected_slots"] = {k: v for k, v in selected.items() if k not in expected}
    if canonical(model) != before or backend.weights.tolist() != model["weights"]:
        raise ValueError("frozen model mutated")
    return {"observations": outcomes, "calls": calls, "slot_outcomes": slot_rows}


def group_counts(rows, key):
    return {group: {"correct": sum(r["task_correct"] for r in rows if r[key] == group),
                    "exact_program_correct": sum(r["exact_program_correct"] for r in rows if r[key] == group),
                    "exact_text_correct": sum(r["exact_text_correct"] for r in rows if r[key] == group),
                    "total": sum(r[key] == group for r in rows)}
            for group in sorted({r[key] for r in rows})}


def bootstrap(tasks, raw_runs):
    ids = [t["task_id"] for t in tasks]
    matrix = np.array([[int(r["task_correct"]) for r in raw["observations"]] for raw in raw_runs], dtype=np.float64)
    if any([r["task_id"] for r in raw["observations"]] != ids for raw in raw_runs):
        raise ValueError("seed/task pairing mismatch")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    def estimate(indices):
        values = matrix[:, indices]
        draws = rng.integers(0, len(indices), (BOOTSTRAP_SAMPLES, len(indices)))
        pooled = values.mean(axis=0)
        interval = lambda x: np.quantile(x, [0.025, 0.975]).tolist()
        return {"unique_task_clusters": len(indices), "accuracy": float(pooled.mean()),
            "task_cluster_bootstrap_95_percentile_ci": interval(pooled[draws].mean(axis=1)),
            "by_seed": {str(seed): {"accuracy": float(values[i].mean()),
                "task_cluster_bootstrap_95_percentile_ci": interval(values[i][draws].mean(axis=1))}
                for i, seed in enumerate(MODEL_HASHES)}}
    return {"aggregate": estimate(list(range(len(tasks)))),
        "by_template": {template: estimate([i for i, t in enumerate(tasks) if t["training_only"]["template_id"] == template])
                        for template in sorted({t["training_only"]["template_id"] for t in tasks})},
        "seed": BOOTSTRAP_SEED, "samples": BOOTSTRAP_SAMPLES,
        "method": "paired across frozen model seeds by task ID; average seeds within task before resampling task clusters; same draws for each seed",
        "limitations": "descriptive conditional intervals for authored synthetic tasks; no baseline delta or causal wording effect; boundary intervals may be degenerate; excludes generation/feature-selection uncertainty"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-seed", type=int, default=161803)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output must be new; preserve previous evidence")
    start = time.perf_counter()
    def deadline():
        if time.perf_counter()-start >= 300:
            raise TimeoutError("300-second CPU ceiling")
    # Verify all frozen identities before generating wording or invoking model.
    models = {}
    for seed, digest in MODEL_HASHES.items():
        path = MODEL_DIR / f"model-{seed}.json"
        if sha(path) != digest:
            raise ValueError("frozen checkpoint hash mismatch")
        model = json.loads(path.read_text())
        if model["feature_version"] != FEATURE_VERSION or model["training_seed"] != seed:
            raise ValueError("frozen feature or seed mismatch")
        models[seed] = model
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    source_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    tasks = generate_paraphrase_tasks(args.task_seed)
    manifest = paraphrase_manifest(tasks)
    args.output.mkdir(parents=True)
    task_artifact = save_compressed(args.output/"evaluation_tasks.jsonl.gz", "".join(canonical(t)+"\n" for t in tasks))
    runs, raw_runs = [], []
    for seed, model in models.items():
        run_start = time.perf_counter()
        raw = evaluate(tasks, model, deadline)
        raw_runs.append(raw)
        artifact = save_compressed(args.output/f"predictions-{seed}.json.gz", canonical(raw)+"\n")
        slots = raw["slot_outcomes"]
        runs.append({"seed": seed, "model_sha256": MODEL_HASHES[seed],
            "seconds": time.perf_counter()-run_start, "raw_artifact": artifact,
            "total": len(tasks), "task_correct": sum(r["task_correct"] for r in raw["observations"]),
            "exact_program_correct": sum(r["exact_program_correct"] for r in raw["observations"]),
            "exact_text_correct": sum(r["exact_text_correct"] for r in raw["observations"]),
            "kernel_accepted": sum(r["kernel_accepted"] for r in raw["observations"]),
            "statuses": dict(Counter(r["status"] for r in raw["observations"])),
            "by_kind": group_counts(raw["observations"], "kind"),
            "by_template": group_counts(raw["observations"], "template_id"),
            "slots": {slot: {"correct": sum(r["correct"] for r in slots if r["slot"] == slot),
                "reached": sum(r["reached"] for r in slots if r["slot"] == slot),
                "total": sum(r["slot"] == slot for r in slots)} for slot in sorted({r["slot"] for r in slots})}})
        if sha(MODEL_DIR/f"model-{seed}.json") != MODEL_HASHES[seed]:
            raise ValueError("checkpoint file changed")
    analysis = bootstrap(tasks, raw_runs)
    deadline()
    receipt = {"schema": "ssdm.frozen_wording_eval.v1", "source_sha": source_sha,
        "source_dirty": source_dirty, "command": [sys.executable, "-m", "language_growth.frozen_wording_eval", *sys.argv[1:]],
        "dependency": {"pr": 65, "branch": "codex/l0-intent-token-invariance", "base_sha": "ca62adc67775e2de1019ec4bcdc6686a55c43cb0"},
        "data_manifest": manifest, "task_artifact": task_artifact,
        "source_hashes": {name: sha(Path(name)) for name in ("language_growth/paraphrase_data.py", "language_growth/frozen_wording_eval.py", "language_growth/transparent_cpu.py", "language_growth/planner.py", "language_growth/kernel.py", "language_growth/realize.py")},
        "config": {"task_seed": args.task_seed, "seeds": list(MODEL_HASHES), "feature_version": FEATURE_VERSION,
            "weights_frozen": True, "retraining": False, "tuning": False, "numerical_features": True,
            "repair": False, "choice_format": "production", "max_seconds": 300},
        "hardware": {"device": "cpu", "gpu_used": False, "processor": platform.processor(),
            "platform": platform.platform(), "python": sys.version, "numpy": np.__version__, "thread_limit": 2},
        "cost": {"incremental_paid_usd": 0, "rate_billed_usd": 0}, "runs": runs, "bootstrap": analysis,
        "aggregate": {"task_correct_median": statistics.median(r["task_correct"] for r in runs),
            "task_correct_min": min(r["task_correct"] for r in runs), "task_correct_max": max(r["task_correct"] for r in runs),
            "seconds_median": statistics.median(r["seconds"] for r in runs),
            "seconds_min": min(r["seconds"] for r in runs), "seconds_max": max(r["seconds"] for r in runs)},
        "elapsed_seconds": time.perf_counter()-start,
        "scope": "single frozen option-filter winner on newly authored synthetic development wording; engineered lexical grounding and Decimal-sign features; no official/Index held-out evidence, shared-parent retention, raw arithmetic, causal mediation, or L0 promotion"}
    (args.output/"receipt.json").write_text(canonical(receipt)+"\n", encoding="ascii", newline="\n")
    print(canonical({"aggregate": receipt["aggregate"], "bootstrap": analysis["aggregate"],
        "task_sha256": manifest["sample_sha256"], "elapsed_seconds": receipt["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
