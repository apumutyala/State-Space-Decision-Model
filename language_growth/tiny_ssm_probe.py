"""Bounded, synthetic-only CPU training and planner evaluation receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

START = time.monotonic()
os.environ.update({"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "2",
                   "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--limit-tasks", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    import psutil
    stopped = threading.Event()
    minimum = [psutil.virtual_memory().available]

    def check():
        available = psutil.virtual_memory().available
        minimum[0] = min(minimum[0], available)
        if available < 3 * 1024**3 or time.monotonic() - START > 300:
            dump(args.out / "aborted.json", {"reason": "ram_or_deadline", "available_bytes": available,
                                          "elapsed_seconds": time.monotonic() - START})
            os._exit(124)

    def watchdog():
        while not stopped.wait(0.25):
            check()

    check()
    threading.Thread(target=watchdog, daemon=True).start()
    import torch
    from language_growth.development_data import generate_development_tasks, development_manifest
    from language_growth.training_records import export_training_records
    from language_growth.tiny_ssm_slots import TinySlotSSM, TinySSMBackend, byte_batch, model_text, state_hash
    from language_growth.free_running_dev import run_development_tasks
    from decide.adapter import Decider
    from decide.calibration import OptionCountTemperature
    from train.guard import SplitGuard

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        raise RuntimeError("GPU visible")
    raw = (args.cohort / "training_records.jsonl").read_bytes()
    raw_hash = hashlib.sha256(raw).hexdigest()
    if raw_hash != "eec306a03ba3e9791fb74b76ead1991e35f162085c33167a5ea6ace4404eb56c":
        raise ValueError("shared cohort hash mismatch")
    records = [json.loads(line) for line in raw.splitlines()]
    if args.limit_tasks:
        records = records[:next((i for i, r in enumerate(records)
                   if int(r["id"].rsplit("-", 1)[1]) >= args.limit_tasks), len(records))]
    del raw
    guard = SplitGuard.synthetic_only("pinned shared L0 development_train cohort only")
    for record in records:
        guard.check(record)
    validation = generate_development_tasks(314159, 300, split="development_validation")
    validation_records = export_training_records(validation, split="development_validation")
    validation_manifest = development_manifest(validation)
    if validation_manifest["sample_sha256"] != "d34df20c1448998791071833cc9154589b526e2ce0a68e7f98e307028d0a827f":
        raise ValueError("paired validation hash mismatch")
    dump(args.out / "validation_manifest.json", validation_manifest)

    def prepare(rows):
        prepared = []
        for row in rows:
            context, ids, desc = model_text(row["request"])
            qid = next(iter(row["labels"]))
            prepared.append((context, ids, desc, ids.index(row["labels"][qid])))
        return prepared

    train_rows, eval_rows = prepare(records), prepare(validation_records)

    def batch(rows):
        descriptions = [desc for row in rows for desc in row[2]]
        owners = torch.tensor([i for i, row in enumerate(rows) for _ in row[2]])
        position = torch.tensor([j for row in rows for j in range(len(row[2]))])
        scores = model(byte_batch([r[0] for r in rows]), byte_batch(descriptions), owners)
        matrix = scores.new_full((len(rows), 4), -1e9)
        matrix[owners, position] = scores
        return matrix, torch.tensor([r[3] for r in rows])

    model = TinySlotSSM(args.width)
    initial_hash = state_hash(model)
    torch.save(model.state_dict(), args.out / "initial.pt")
    calibration = OptionCountTemperature()
    config = {"calibration": calibration.identity(),
              "policy_masking": "none (served covers every supplied option)"}

    def evaluate(arm, dataset, prepared, free_run):
        rows = []
        by_slot = {}
        model.eval()
        with torch.no_grad():
            for start in range(0, len(prepared), args.batch):
                check()
                part = prepared[start:start + args.batch]
                logits, gold = batch(part)
                predictions = logits.argmax(1)
                for j, (item, row) in enumerate(zip(dataset[start:start + args.batch], part)):
                    qid = next(iter(item["labels"]))
                    slot = qid.rsplit("-", 1)[1]
                    correct = int(predictions[j]) == row[3]
                    counts = by_slot.setdefault(slot, {"correct": 0, "total": 0})
                    counts["correct"] += correct
                    counts["total"] += 1
                    rows.append({"id": item["id"], "qid": qid, "option_ids": row[1],
                                 "descriptions": row[2], "gold_id": row[1][row[3]],
                                 "prediction_id": row[1][int(predictions[j])], "correct": correct,
                                 "logits": logits[j, :len(row[1])].tolist()})
        dump(args.out / f"{arm}-slots.json", rows)
        result = {"by_slot": by_slot, "correct": sum(r["correct"] for r in rows), "total": len(rows)}
        if free_run:
            backend = TinySSMBackend(model, check)
            provider = Decider(backend, calibration=calibration)
            report = run_development_tasks(validation, provider, model=backend.name,
                         parent_revision=backend.revision()["revision_id"], expected_served_config=config)
            dump(args.out / f"{arm}-planner.json", report)
            dump(args.out / f"{arm}-calls.json", backend.predictions)
            result["free_running"] = {key: report[key] for key in ("total", "task_correct", "kernel_accepted")}
            result["failure_reasons"] = {reason: sum(row["reason"] == reason for row in report["observations"])
                                        for reason in sorted({row["reason"] for row in report["observations"]
                                                              if row["reason"] is not None})}
        print(json.dumps({"arm": arm, **result}), flush=True)
        return result

    initial = evaluate("initial-validation", validation_records, eval_rows, True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    generator = torch.Generator().manual_seed(args.seed)
    history = []
    trained_at = time.monotonic()
    model.train()
    for step in range(args.steps):
        check()
        selection = torch.randint(len(train_rows), (args.batch,), generator=generator).tolist()
        logits, target = batch([train_rows[i] for i in selection])
        loss = torch.nn.functional.cross_entropy(logits, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        history.append({"step": step + 1, "loss": float(loss.detach())})
        if (step + 1) % 50 == 0:
            print(json.dumps(history[-1] | {"elapsed": time.monotonic() - START}), flush=True)
    training_seconds = time.monotonic() - trained_at
    final_hash = state_hash(model)
    torch.save(model.state_dict(), args.out / "trained.pt")
    dump(args.out / "history.json", history)
    train_result = evaluate("trained-train", records, train_rows, False)
    trained = evaluate("trained-validation", validation_records, eval_rows, True)
    artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file()}
    receipt = {"scope": "single seed, synthetic development only; no Index, held-out, safety or L0 qualification claim",
               "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "command": [sys.executable, "-m", "language_growth.tiny_ssm_probe", *sys.argv[1:]],
               "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
               "seed": args.seed, "hardware": platform.processor(), "python": platform.python_version(),
               "torch": torch.__version__, "threads": torch.get_num_threads(), "device": "cpu", "paid_usd": 0,
               "runtime_cap_seconds": 300, "minimum_available_bytes": minimum[0],
               "training_seconds": training_seconds, "elapsed_seconds": time.monotonic() - START,
               "train_records": len(records), "train_cohort_sha256": raw_hash,
               "guard": guard.identity(), "initial_state_sha256": initial_hash, "trained_state_sha256": final_hash,
               "parameter_count": sum(p.numel() for p in model.parameters()),
               "initial_validation": initial, "trained_train": train_result, "trained_validation": trained,
               "artifacts_sha256": artifacts, "runs": 1, "spread": "unavailable: one seeded run"}
    dump(args.out / "receipt.json", receipt)
    stopped.set()
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
