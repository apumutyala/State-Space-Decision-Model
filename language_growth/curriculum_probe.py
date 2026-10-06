"""Frozen, bounded CPU curriculum comparison; transfer outcomes never tune training."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time

os.environ.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2",
                  MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
BASE_SHA = "115c4535ce9db716d3dde70ad000d206fdd2a4b0"
OLD_RECORD_HASH = "eec306a03ba3e9791fb74b76ead1991e35f162085c33167a5ea6ace4404eb56c"
FROZEN_CHECKPOINT_HASH = "0252d53cdebb6d1d0074542b375cc375badd3ee43c1ba06ce6712aa0f5eab0b4"
SEEDS = (17, 29, 43)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def write(path, value):
    path.write_bytes(encoded(value))


def rows_bytes(rows):
    return b"".join(encoded(row) for row in rows)


def save_rows(path, rows):
    raw = rows_bytes(rows)
    path.write_bytes(gzip.compress(raw, mtime=0))
    return {"file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "jsonl_sha256": hashlib.sha256(raw).hexdigest(), "rows": len(rows)}


def load_rows(path, expected=None):
    raw = gzip.decompress(path.read_bytes())
    if expected and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"frozen data mismatch: {path}")
    return [json.loads(line) for line in raw.splitlines()]


def freeze(out, retired_spec):
    from language_growth.development_data import generate_development_tasks, development_manifest
    from language_growth.training_records import export_training_records
    from language_growth.curriculum_spec import CURRICULUM_SPLITS
    # Read only source constants for an exclusion audit; never load retired tasks.
    retired_raw = retired_spec.read_bytes()
    constants = {node.targets[0].id: ast.literal_eval(node.value)
                 for node in ast.parse(retired_raw).body if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in ("TEMPLATES", "NAMES", "WARRANTY_RANGE", "PRICE_CENTS_RANGE")}
    retired_templates = {x for v in constants["TEMPLATES"].values() for x in v}
    tokenize = lambda templates: set(re.findall(r"[a-z]+", " ".join(templates).lower()))
    retirement_audit = {"source_sha256": hashlib.sha256(retired_raw).hexdigest(),
                        "source_path": str(retired_spec), "tasks_loaded": 0, "split": "development_paraphrase"}
    for split, spec in CURRICULUM_SPLITS.items():
        templates = {x for v in spec["templates"].values() for x in v}
        overlap = sorted(templates & retired_templates)
        if overlap:
            raise ValueError("retired exact wording overlap")
        retirement_audit[split] = {"exact_template_overlap": overlap,
            "shared_tokens": sorted(tokenize(templates) & tokenize(retired_templates))}
    transfer_spec = CURRICULUM_SPLITS["development_transfer_v2"]
    if set(transfer_spec["names"]) & set(constants["NAMES"]):
        raise ValueError("retired names overlap transfer")
    for current, retired in (("warranty_range", "WARRANTY_RANGE"), ("price_cents_range", "PRICE_CENTS_RANGE")):
        if transfer_spec[current][0] <= constants[retired][1]:
            raise ValueError("retired value range overlaps transfer")
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "dependency": {"pr": 64, "sha": BASE_SHA}, "files": {},
                "preregistration": {"seeds": list(SEEDS), "steps": 400, "batch": 32, "width": 64,
                    "optimizer": "Adam lr=0.003; clip_grad_norm=1.0; no schedule",
                    "training_sampling": "uniform slot records with replacement",
                    "training_order": "all six models finish before any transfer evaluation",
                    "paired_ci": "10000 stratified task-cluster bootstrap draws; fixed seed 8128",
                    "training_seed": 271828, "transfer_seed": 141421,
                    "runtime_cap_seconds": 300, "available_ram_floor_bytes": 3 * 1024**3,
                    "scope": "novel synthetic templates with shared vocabulary; retired PR67 tasks excluded; no tuning after outcomes"},
                "retirement_audit": retirement_audit}
    for split, count, seed in (("development_curriculum", 5000, 271828),
                               ("development_transfer_v2", 300, 141421)):
        tasks = generate_development_tasks(seed, count, split=split)
        records = export_training_records(tasks, split=split)
        manifest[split] = development_manifest(tasks)
        manifest[split]["task_kind_counts"] = dict(Counter(t["training_only"]["kind"] for t in tasks))
        manifest[split]["template_counts"] = dict(Counter(t["training_only"]["template_id"] for t in tasks))
        for suffix, rows in (("tasks", tasks), ("records", records)):
            name = f"{split}-{suffix}.jsonl.gz"
            manifest["files"][name] = save_rows(out / name, rows)
    write(out / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


def counts(rows, keys, outcome):
    result = {}
    for row in rows:
        key = "/".join(str(row[k]) for k in keys)
        entry = result.setdefault(key, {"correct": 0, "total": 0})
        entry["total"] += 1
        entry["correct"] += bool(row[outcome])
    return result


def paired_ci(baseline, candidate, metric, *, slots=False):
    """Resample whole tasks within kind; all dependent slots stay in the cluster."""
    import numpy as np
    if len(baseline) != len(candidate) or any(a["task_id"] != b["task_id"] for a, b in zip(baseline, candidate)):
        raise ValueError("paired task identities differ")
    delta = np.array([float(b[metric]) - float(a[metric]) for a, b in zip(baseline, candidate)])
    groups = [np.array([i for i, r in enumerate(baseline) if r["kind"] == kind])
              for kind in ("read", "explain", "compare")]
    denominator = np.array([r["slot_total"] if slots else 1 for r in baseline])
    rng = np.random.default_rng(8128)
    indices = np.concatenate([rng.choice(g, (10000, len(g)), replace=True) for g in groups], axis=1)
    draws = delta[indices].sum(1) / denominator[indices].sum(1)
    return {"difference": float(delta.sum() / denominator.sum()),
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "draws": 10000, "unit": "task cluster stratified by kind", "seed": 8128}


def experiment(data, old_records, out):
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=False)
    import psutil
    minimum = [psutil.virtual_memory().available]
    stopped = threading.Event()

    def check():
        available = psutil.virtual_memory().available
        minimum[0] = min(minimum[0], available)
        elapsed = time.monotonic() - start
        if available < 3 * 1024**3 or elapsed > 300:
            write(out / "aborted.json", {"reason": "ram_or_deadline", "available_bytes": available,
                                        "elapsed_seconds": elapsed})
            os._exit(124)

    def watchdog():
        while not stopped.wait(0.25):
            check()

    check()
    threading.Thread(target=watchdog, daemon=True).start()
    import torch
    from language_growth.tiny_ssm_slots import TinySlotSSM, TinySSMBackend, byte_batch, model_text, state_hash
    from language_growth.free_running_dev import run_development_tasks
    from decide.adapter import Decider
    from decide.calibration import OptionCountTemperature
    from train.guard import SplitGuard

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        raise RuntimeError("GPU visible")
    manifest_raw = (data / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest["preregistration"]["seeds"] != list(SEEDS):
        raise ValueError("predeclared seeds differ")
    files = manifest["files"]
    curriculum = load_rows(data / "development_curriculum-records.jsonl.gz",
                          files["development_curriculum-records.jsonl.gz"]["jsonl_sha256"])
    transfer = load_rows(data / "development_transfer_v2-tasks.jsonl.gz",
                        files["development_transfer_v2-tasks.jsonl.gz"]["jsonl_sha256"])
    transfer_records = load_rows(data / "development_transfer_v2-records.jsonl.gz",
                        files["development_transfer_v2-records.jsonl.gz"]["jsonl_sha256"])
    old_raw = old_records.read_bytes()
    if hashlib.sha256(old_raw).hexdigest() != OLD_RECORD_HASH:
        raise ValueError("PR64 old training records mismatch")
    old = [json.loads(line) for line in old_raw.splitlines()]
    del old_raw
    guard = SplitGuard.synthetic_only("new curriculum plus pinned PR64 baseline; no PR67 or Index")
    for row in old + curriculum:
        guard.check(row)
    if {r["id"] for r in old + curriculum} & {r["id"] for r in transfer_records}:
        raise ValueError("training/evaluation overlap")

    def prepare(rows):
        prepared = []
        for row in rows:
            context, ids, descriptions = model_text(row["request"])
            qid = next(iter(row["labels"]))
            prepared.append((context, ids, descriptions, ids.index(row["labels"][qid])))
        return prepared

    prepared_old, prepared_curriculum = prepare(old), prepare(curriculum)
    eval_rows = prepare(transfer_records)

    def batch(model, rows):
        descriptions = [desc for row in rows for desc in row[2]]
        owners = torch.tensor([i for i, row in enumerate(rows) for _ in row[2]])
        positions = torch.tensor([j for row in rows for j in range(len(row[2]))])
        scores = model(byte_batch([r[0] for r in rows]), byte_batch(descriptions), owners)
        matrix = scores.new_full((len(rows), 4), -1e9)
        matrix[owners, positions] = scores
        return matrix, torch.tensor([r[3] for r in rows])

    trained = []
    for seed in SEEDS:
        for arm, rows in (("old_cohort", prepared_old), ("curriculum", prepared_curriculum)):
            check()
            torch.manual_seed(seed)
            model = TinySlotSSM(64)
            initial = state_hash(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
            generator = torch.Generator().manual_seed(seed)
            model.train()
            history = []
            train_start = time.monotonic()
            for step in range(400):
                check()
                selection = torch.randint(len(rows), (32,), generator=generator).tolist()
                logits, target = batch(model, [rows[i] for i in selection])
                loss = torch.nn.functional.cross_entropy(logits, target)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                history.append({"step": step + 1, "loss": float(loss.detach())})
            name = f"{arm}-seed{seed}"
            torch.save(model.state_dict(), out / f"{name}.pt")
            write(out / f"{name}-history.json", history)
            info = {"name": name, "seed": seed, "arm": arm, "initial_state_sha256": initial,
                    "trained_state_sha256": state_hash(model), "training_seconds": time.monotonic() - train_start,
                    "records": len(rows)}
            trained.append(info)
            write(out / "training-progress.json", trained)
            print(json.dumps({"trained": name, "elapsed_seconds": time.monotonic() - start}), flush=True)
            del model, optimizer
    # All training is complete before a transfer score is computed or printed.
    frozen = Path("receipts/2026-10-06-l-tiny-ssm-slots-run1/trained.pt")
    if hashlib.sha256(frozen.read_bytes()).hexdigest() != FROZEN_CHECKPOINT_HASH:
        raise ValueError("PR64 frozen checkpoint mismatch")
    models = [{"name": "pr64_frozen_seed20261006", "seed": 20261006,
               "arm": "pr64_frozen", "checkpoint": str(frozen)}] + trained
    metadata = {t["task_id"]: {"kind": t["training_only"]["kind"],
                               "template": t["training_only"]["template_id"]} for t in transfer}
    results = {}
    calibration = OptionCountTemperature()
    config = {"calibration": calibration.identity(), "policy_masking": "none (served covers every supplied option)"}
    for info in models:
        check()
        name = info["name"]
        model = TinySlotSSM(64)
        checkpoint = Path(info.get("checkpoint", out / f"{name}.pt"))
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        model.eval()
        slot_rows = []
        with torch.no_grad():
            for offset in range(0, len(eval_rows), 32):
                check()
                part = eval_rows[offset:offset + 32]
                logits, _ = batch(model, part)
                for j, (record, row) in enumerate(zip(transfer_records[offset:offset + 32], part)):
                    qid = next(iter(record["labels"]))
                    prediction = int(logits[j].argmax())
                    slot_rows.append({"task_id": record["id"], **metadata[record["id"]],
                        "slot": qid.rsplit("-", 1)[1], "qid": qid, "correct": prediction == row[3],
                        "gold_description": row[2][row[3]], "prediction_description": row[2][prediction],
                        "option_ids": row[1], "logits": logits[j, :len(row[1])].tolist()})
        save_rows(out / f"{name}-teacher-slots.jsonl.gz", slot_rows)
        backend = TinySSMBackend(model, check)
        report = run_development_tasks(transfer, Decider(backend, calibration=calibration),
                    model=backend.name, parent_revision=backend.revision()["revision_id"],
                    expected_served_config=config)
        task_rows = []
        free_slots = []
        calls = []
        cursor = 0
        for example, observation in zip(transfer, report["observations"]):
            check()
            tid = example["task_id"]
            if observation["task_id"] != tid or observation["provider_calls"] is None:
                raise ValueError("missing task/call accounting")
            predicted = {}
            for call in backend.predictions[cursor:cursor + observation["provider_calls"]]:
                call.update(task_id=tid, **metadata[tid])
                qid = next(iter(call["request"]["questions"]))
                slot = qid.rsplit("-", 1)[1]
                selected = max(range(len(call["logits"])), key=lambda i: call["logits"][i])
                predicted[slot] = call["descriptions"][selected]
                calls.append(call)
            cursor += observation["provider_calls"]
            for request, target in zip(example["model_inputs"], example["training_only"]["targets"]):
                question = next(iter(request["questions"].values()))
                gold = question["criteria"][target["option_id"]]
                free_slots.append({"task_id": tid, **metadata[tid], "slot": target["slot"],
                    "gold_description": gold, "prediction_description": predicted.get(target["slot"]),
                    "correct": predicted.get(target["slot"]) == gold})
            teacher = [row for row in slot_rows if row["task_id"] == tid]
            task_rows.append({**observation, **metadata[tid],
                "exact_program_correct": observation["program"] == example["training_only"]["program"],
                "slot_correct": sum(r["correct"] for r in teacher), "slot_total": len(teacher),
                "free_slot_correct": sum(r["correct"] for r in free_slots if r["task_id"] == tid)})
        if cursor != len(backend.predictions):
            raise ValueError("unassigned free-running calls")
        save_rows(out / f"{name}-free-calls.jsonl.gz", calls)
        save_rows(out / f"{name}-free-slots.jsonl.gz", free_slots)
        save_rows(out / f"{name}-tasks.jsonl.gz", task_rows)
        summary = {"semantic_correct": report["task_correct"], "total": report["total"],
            "exact_program_correct": sum(r["exact_program_correct"] for r in task_rows),
            "kernel_accepted": report["kernel_accepted"],
            "teacher_slot_correct": sum(r["correct"] for r in slot_rows), "teacher_slot_total": len(slot_rows),
            "free_slot_correct": sum(r["correct"] for r in free_slots), "free_slot_total": len(free_slots),
            "teacher_by_slot": counts(slot_rows, ["slot"], "correct"),
            "teacher_by_template_slot": counts(slot_rows, ["template", "slot"], "correct"),
            "free_by_template_slot": counts(free_slots, ["template", "slot"], "correct"),
            "semantic_by_kind": counts(task_rows, ["kind"], "task_correct"),
            "semantic_by_template": counts(task_rows, ["template"], "task_correct"),
            "exact_by_template": counts(task_rows, ["template"], "exact_program_correct"),
            "kernel_by_template": counts(task_rows, ["template"], "kernel_accepted"),
            "status_counts": dict(Counter(r["status"] for r in task_rows))}
        results[name] = {**info, "state_sha256": state_hash(model), "summary": summary, "rows": task_rows}
    comparisons = {}
    for seed in SEEDS:
        candidate = results[f"curriculum-seed{seed}"]["rows"]
        for old_name in (f"old_cohort-seed{seed}", "pr64_frozen_seed20261006"):
            baseline = results[old_name]["rows"]
            comparisons[f"curriculum-seed{seed}_minus_{old_name}"] = {
                key: paired_ci(baseline, candidate, key, slots=key.endswith("slot_correct"))
                for key in ("task_correct", "exact_program_correct", "kernel_accepted", "slot_correct", "free_slot_correct")}
    receipt = {"scope": "three seeds, synthetic development transfer only; no held-out, safety or qualification claim",
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dependency": {"pr": 64, "sha": BASE_SHA}, "preregistration": manifest["preregistration"],
        "command": [sys.executable, "-X", "utf8", "-m", "language_growth.curriculum_probe", *sys.argv[1:]],
        "frozen_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(), "data_hashes": files,
        "old_train_records_sha256": OLD_RECORD_HASH, "pr64_checkpoint_file_sha256": FROZEN_CHECKPOINT_HASH,
        "results": {k: {key: value for key, value in v.items() if key != "rows"} for k, v in results.items()},
        "paired_comparisons": comparisons, "hardware": platform.processor(), "python": platform.python_version(),
        "torch": torch.__version__, "device": "cpu", "threads": torch.get_num_threads(),
        "parameter_count": sum(p.numel() for p in model.parameters()), "paid_usd": 0,
        "minimum_available_bytes": minimum[0], "elapsed_seconds": time.monotonic() - start,
        "guard": guard.identity(), "artifacts_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},
        "limitations": ["slot supervision samples records uniformly rather than tasks",
            "task-cluster CIs condition on training seeds; three seeds are not a population uncertainty estimate",
            "transfer wording is synthetic and development-only; no outcome-dependent tuning",
            "curriculum changes wording diversity; matched baseline shares model initialization and training bounds"]}
    write(out / "receipt.json", receipt)
    stopped.set()
    print(json.dumps({"elapsed_seconds": receipt["elapsed_seconds"],
                      "results": {k: v["summary"]["semantic_correct"] for k, v in results.items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--retired-spec", type=Path, required=True)
    p = sub.add_parser("run")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--old-records", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args.out, args.retired_spec)
    else:
        experiment(args.data, args.old_records, args.out)


if __name__ == "__main__":
    main()
