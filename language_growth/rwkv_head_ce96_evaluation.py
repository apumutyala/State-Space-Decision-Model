"""Fixed final-checkpoint reporting for the 96-update exposure ablation.

No optimizer, checkpoint loading or checkpoint selection belongs here.
Earlier measurements are read from committed JSON; current labels remain in
the teacher-forced evaluator, outside unchanged model requests.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATHS = {
    "learning": "receipts/2026-10-06-L-rwkv-head-learning-run1/receipt.json",
    "fit_rows": "receipts/2026-10-06-L-rwkv-head-fit-run1/rows.json",
    "old_recipe": "receipts/2026-10-06-L-rwkv-head-learning-run1/registration.json",
    "old_train_config": "receipts/2026-10-06-L-rwkv-head-learning-run1/train-config.json",
}


def references():
    result = {name: json.loads((ROOT / path).read_text(encoding="utf-8"))
              for name, path in REFERENCE_PATHS.items()}
    result["sha256"] = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                        for path in REFERENCE_PATHS.values()}
    return result


def exposure_contract(data, config):
    from language_growth.training_records import export_training_records
    from language_growth.rwkv_head_learning import digest
    old = references()["old_recipe"]
    if {k: v for k, v in config.items() if k != "steps"} != {
            k: v for k, v in old["config"].items() if k != "steps"}:
        raise ValueError("exposure ablation changed another training setting")
    if old["config"]["steps"] != 24 or config["steps"] != 96:
        raise ValueError("fixed 24-to-96 step comparison required")
    if digest(data) != old["corpus_sha256"]:
        raise ValueError("exposure ablation changed the corpus")
    records = export_training_records(data["development_train"])
    ids = [f"{r['id']}/{next(iter(r['labels']))}" for r in records]
    if len(ids) != 64 or len(set(ids)) != 64 or config["batch_size"] != 2:
        raise ValueError("exactly 64 unique slots and batch 2 required")
    if len(ids) % config["batch_size"] or config["steps"] * config["batch_size"] != 3 * len(ids):
        raise ValueError("runner cannot replay exactly three complete data passes")
    return {"train_slots": 64, "updates": 96, "batch_size": 2, "slot_exposures": 192,
            "complete_passes": 3, "dropped_tail": 0, "prior_exposures": 48,
            "ordered_item_ids_sha256": digest(ids),
            "sampler": "unchanged Trainer._order/_next_batch; randperm(seed*100003+epoch)"}


def initial_parity(current, prior):
    fields = ("task_id", "status", "reason", "kernel_accepted", "task_correct", "program", "text")
    projected = lambda rows: [{key: row[key] for key in fields} for row in rows]
    if projected(current) != projected(prior):
        raise ValueError("initial head failed to reproduce the prior step-0 outputs")


def training_config_parity(config):
    expected = references()["old_train_config"]
    expected["max_steps"] = 96
    if config != expected:
        raise ValueError("full trainer recipe changed beyond max_steps 24 to 96")


def teacher_forced(data, backend, out, phase):
    from language_growth.rwkv_head_fit_audit import rows_for, evaluate_rows, aggregate
    from language_growth.rwkv_head_learning import write
    import torch
    with torch.inference_mode():
        rows = evaluate_rows(rows_for(data), backend, out / f"{phase}-raw.jsonl")
    result = {"rows": rows, "summary": aggregate(rows), "selection": "one fixed checkpoint; no optimizer"}
    write(out / f"{phase}.json", result)
    return result


def paired_slots(current, prior, metric, split, *, samples=2000, seed=9203):
    """Resample worlds with all their slots; failed-row CE stays undefined."""
    key = lambda r: (r["task_id"], r["qid"])
    a = {key(r): r for r in current if r["split"] == split}
    b = {key(r): r for r in prior if r["split"] == split}
    if not a or set(a) != set(b):
        raise ValueError("teacher-forced slot identities do not pair exactly")
    groups = {}
    for k in sorted(a):
        left, right = a[k].get(metric), b[k].get(metric)
        groups.setdefault(k[0], []).append(None if left is None or right is None else float(left) - float(right))
    failed = sum(value is None for group in groups.values() for value in group)
    result = {"n_slots": len(a), "n_worlds": len(groups), "failed_pairs": failed,
              "estimate": None, "ci95": None, "metric": metric,
              "direction": "current-minus-prior; CE improvement is negative"}
    if failed:
        return result
    worlds = sorted(groups)
    flatten = lambda ids: [value for world in ids for value in groups[world]]
    values = flatten(worlds)
    result["estimate"] = sum(values) / len(values)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        sampled = flatten([rng.choice(worlds) for _ in worlds])
        draws.append(sum(sampled) / len(sampled))
    draws.sort()
    result["ci95"] = [draws[int(.025 * samples)], draws[int(.975 * samples)]]
    return result


def comparison_report(before, after, teacher_before, teacher_after, prior):
    from language_growth.rwkv_head_learning import bootstrap
    current = {r["task_id"]: r for r in after}
    free = {}
    for name, rows in (("step0", prior["learning"]["before"]), ("step24", prior["learning"]["after"])):
        reference = {r["task_id"]: r for r in rows}
        if set(current) != set(reference):
            raise ValueError("free-running world identities do not pair exactly")
        ids = sorted(current)
        free[name] = {metric: bootstrap([reference[k][metric] for k in ids],
                                       [current[k][metric] for k in ids])
                      for metric in ("task_correct", "kernel_accepted")}
    teacher = {}
    for name, rows in (("step0", teacher_before["rows"]), ("step24", prior["fit_rows"])):
        teacher[name] = {split: {metric: paired_slots(teacher_after["rows"], rows, metric, split)
                                for metric in ("correct", "ce")}
                         for split in ("development_train", "development_validation")}
    return {"free_running": free, "teacher_forced": teacher,
            "scope": "one exposure change on reused synthetic development worlds; no capability qualification"}
