"""Learned CPU linear option scorer with explicitly engineered semantic features.

No pretrained model, audit data, Torch, CUDA or downloads. The numerical sign
and lexical grounding are host-side features, not learned raw-text reasoning.
The option ranking and sign-to-relation mapping are fitted from supervision.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time

# Pin before NumPy import. This module performs CPU arithmetic only.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import numpy as np

from decide.adapter import Decider
from decide.backends.base import BackendUnsupported, Plan, ScoreResult
from decide.calibration import OptionCountTemperature
from .development_data import generate_development_tasks, development_manifest
from .free_running_dev import run_development_tasks
from .training_records import export_training_records

FEATURE_VERSION = "lexical-entity-position-field-match-decimal-sign-v1"
FACT = re.compile(r"Fact (\S+) field (\S+)\Z")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def feature_rows(request, *, numerical=True):
    """Consumes only the permitted request, never labels or reference programs.

    Option IDs, request IDs, task IDs and context hashes are not features.
    Earlier selected operands are model proposals, including during rollout.
    """
    state = json.loads(request["state"])
    if state["schema"] != "ssdm.track_l_state.v0.1":
        raise ValueError("unsupported state schema")
    if len(request["questions"]) != 1:
        raise ValueError("one slot per request required")
    qid, q = next(iter(request["questions"].items()))
    slot = qid.rsplit("-", 1)[1]
    if slot not in ("operation", "fact", "left", "right", "relation"):
        raise ValueError("unsupported slot")
    task = state["task_input"].lower()
    tokens = set(words(task))
    records = {r["record_ref"]: r for r in state["records"]}
    mentions = sorted((match.start(), ref) for ref, rec in records.items()
                      for match in re.finditer(r"\b" + re.escape(rec["entity_ref"].lower()) + r"\b", task))
    sign = None
    if slot == "relation" and numerical:
        selected = {p["slot"]: p["description"] for p in state["proposed_slot_choices"]}
        left, right = FACT.fullmatch(selected["left"]), FACT.fullmatch(selected["right"])
        if not left or not right:
            raise ValueError("unsupported operand")
        a = records[left[1]]["fields"][left[2]]
        b = records[right[1]]["fields"][right[2]]
        if (a["value_type"] not in ("integer", "decimal") or b["value_type"] not in ("integer", "decimal")
                or a["unit"] != b["unit"] or a["temporal_scope"] != b["temporal_scope"]):
            raise ValueError("unsupported comparison")
        av, bv = Decimal(str(a["value"])), Decimal(str(b["value"]))
        sign = str(int(av > bv) - int(av < bv))
    result = []
    for oid, description in q["criteria"].items():
        f = {}
        if slot == "operation":
            head = description.split()[0].lower()
            for token in tokens | {"BIAS"}:
                f[f"operation/{head}/{token}"] = 1.0
        elif slot in ("fact", "left", "right"):
            match = FACT.fullmatch(description)
            if not match or match[1] not in records or match[2] not in records[match[1]]["fields"]:
                raise ValueError("unsupported fact description")
            ref, field = match.groups()
            field_match = set(words(field)) <= tokens
            f[f"{slot}/field_match"] = float(field_match)
            f[f"{slot}/first_mention"] = float(bool(mentions) and mentions[0][1] == ref)
            f[f"{slot}/last_mention"] = float(bool(mentions) and mentions[-1][1] == ref)
            f[f"{slot}/field_and_first"] = float(field_match and bool(mentions) and mentions[0][1] == ref)
            f[f"{slot}/field_and_last"] = float(field_match and bool(mentions) and mentions[-1][1] == ref)
        else:
            if not description.startswith("Relation "):
                raise ValueError("unsupported relation description")
            relation = description.split()[1]
            f[f"relation/{relation}/bias"] = 1.0
            if sign is not None:
                f[f"relation/{relation}/sign={sign}"] = 1.0
        result.append((oid, f))
    return qid, slot, result


class LinearOptionBackend:
    name = "transparent_cpu_linear"

    def __init__(self, model, *, numerical=True):
        self.model = model
        self.numerical = numerical
        self.index = {f: i for i, f in enumerate(model["features"])}
        self.weights = np.asarray(model["weights"], dtype=np.float64)
        self.identity = hashlib.sha256(canonical({"model": model, "numerical": numerical}).encode()).hexdigest()

    def revision(self):
        return {"revision_id": "transparent-cpu:" + self.identity, "backend": self.name,
                "feature_version": FEATURE_VERSION, "numerical_features": self.numerical}

    def plan(self, request):
        try:
            feature_rows(request, numerical=self.numerical)
        except (ValueError, KeyError, TypeError):
            raise BackendUnsupported("unsupported_features", "request outside baseline feature domain") from None
        # This structured scorer has no tokenizer or token inference inputs.
        return Plan(inference_calls=1, max_call_input_tokens=0, total_input_tokens=0)

    def score(self, request, check_deadline):
        check_deadline()
        qid, _, rows = feature_rows(request, numerical=self.numerical)
        contributions = {oid: {key: float(self.weights[self.index[key]] * value)
                               for key, value in f.items() if key in self.index} for oid, f in rows}
        logits = {oid: float(sum(values.values())) for oid, values in contributions.items()}
        return ScoreResult({qid: logits}, inference_calls=1, total_input_tokens=0,
                           extras={"feature_contributions": contributions,
                                   "attribution_scope": "linear accounting, not causal mediation"})


def fit(records, seed, epochs, learning_rate):
    extracted = []
    vocabulary = set()
    for record in records:
        qid, _, rows = feature_rows(record["request"])
        gold = record["labels"][qid]
        extracted.append((rows, gold))
        vocabulary.update(key for _, f in rows for key in f)
    features = sorted(vocabulary)
    index = {f: i for i, f in enumerate(features)}
    x = np.zeros((len(records), 4, len(features)), dtype=np.float64)
    mask = np.zeros((len(records), 4), dtype=bool)
    targets = np.zeros((len(records), 4), dtype=np.float64)
    for i, (rows, gold) in enumerate(extracted):
        for j, (oid, f) in enumerate(rows):
            mask[i, j] = True
            targets[i, j] = oid == gold
            for key, value in f.items():
                x[i, j, index[key]] = value
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.01, len(features))
    initial = w.copy()
    history = []
    for epoch in range(epochs):
        logits = np.einsum("nkf,f->nk", x, w)
        logits[~mask] = -np.inf
        logits -= np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        loss = float(-np.sum(targets * np.log(np.maximum(probs, 1e-300))) / len(records))
        gradient = np.einsum("nk,nkf->f", probs - targets, x) / len(records) + 1e-4 * w
        w -= learning_rate * gradient
        if not np.isfinite(w).all() or not np.isfinite(loss):
            raise ValueError("nonfinite optimization")
        if epoch in (0, epochs - 1):
            history.append({"epoch": epoch, "cross_entropy": loss})
    base = {"schema": "ssdm.transparent_cpu_linear.v1", "feature_version": FEATURE_VERSION,
            "features": features, "training_seed": seed}
    return {**base, "weights": w.tolist()}, {**base, "weights": initial.tolist()}, history


def slot_predictions(records, backend):
    rows = []
    for record in records:
        qid, slot, _ = feature_rows(record["request"])
        scores = backend.score(record["request"], lambda: None)
        logits = scores.logits[qid]
        selected = max(sorted(logits), key=logits.get)
        rows.append({"task_id": record["id"], "slot": slot, "qid": qid, "selected": selected,
                     "gold": record["labels"][qid], "correct": selected == record["labels"][qid],
                     "logits": logits, "feature_contributions": scores.extras["feature_contributions"]})
    return rows


def counts(rows):
    return {slot: {"correct": sum(r["correct"] for r in rows if r["slot"] == slot),
                   "total": sum(r["slot"] == slot for r in rows)} for slot in sorted({r["slot"] for r in rows})}


class LoggedProvider:
    def __init__(self, provider):
        self.provider, self.calls = provider, []

    def decide(self, request):
        response = self.provider.decide(request)
        self.calls.append({"request": request, "response": response})
        return response


def rollout(tasks, backend):
    calibration = OptionCountTemperature()
    provider = LoggedProvider(Decider(backend, calibration=calibration))
    report = run_development_tasks(tasks, provider, model=backend.name,
                                  parent_revision=backend.revision()["revision_id"],
                                  expected_served_config={"calibration": calibration.identity(),
                                    "policy_masking": "none (served covers every supplied option)"})
    return report, provider.calls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-seed", type=int, default=314159)
    parser.add_argument("--eval-count", type=int, default=300)
    parser.add_argument("--eval-split", choices=("development_train", "development_validation"),
                        default="development_train")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2.0)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("output must be new; preserve previous artifacts")
    receipt = json.loads((args.cohort / "receipt.json").read_text())
    if receipt["split"] != "development_train" or receipt["allow_one_repair"]:
        raise ValueError("only the development_train one-shot cohort is permitted")
    records_path = args.cohort / "training_records.jsonl"
    if sha(records_path) != receipt["hashes"]["training_records_sha256"]:
        raise ValueError("cohort hash mismatch")
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    if len(records) != receipt["counts"]["records"] or any(
            not r["id"].startswith(f"synthetic-development_train-{receipt['seed']}-") for r in records):
        raise ValueError("cohort records outside declared domain")
    if args.eval_seed == receipt["seed"]:
        raise ValueError("fresh development seed required")
    if args.epochs < 1 or not 0 < args.learning_rate <= 10 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("invalid training config")
    tasks = generate_development_tasks(args.eval_seed, args.eval_count, split=args.eval_split, choice_format="mixed")
    evaluation = export_training_records(tasks, split=args.eval_split)
    args.output.mkdir(parents=True)
    (args.output / "evaluation_tasks.jsonl").write_text("".join(canonical(t) + "\n" for t in tasks), encoding="ascii", newline="\n")
    runs = []
    for seed in args.seeds:
        started = time.perf_counter()
        model, initial, history = fit(records, seed, args.epochs, args.learning_rate)
        duration = time.perf_counter() - started
        model_path = args.output / f"model-{seed}.json"
        model_path.write_text(canonical(model) + "\n", encoding="ascii", newline="\n")
        # Score restored checkpoint, not an unsaved training object.
        backend = LinearOptionBackend(json.loads(model_path.read_text()))
        predictions = slot_predictions(evaluation, backend)
        pre = slot_predictions(evaluation, LinearOptionBackend(initial))
        no_sign = slot_predictions(evaluation, LinearOptionBackend(model, numerical=False))
        report, calls = rollout(tasks, backend)
        initial_report, initial_calls = rollout(tasks, LinearOptionBackend(initial))
        raw = {"slot_predictions": predictions, "initial_slot_predictions": pre,
               "without_numeric_sign_slot_predictions": no_sign, "report": report, "calls": calls,
               "initial_report": initial_report, "initial_calls": initial_calls}
        raw_path = args.output / f"predictions-{seed}.json"
        raw_path.write_text(canonical(raw) + "\n", encoding="ascii", newline="\n")
        runs.append({"seed": seed, "history": history, "fit_seconds": duration,
                     "model_sha256": sha(model_path), "raw_predictions_sha256": sha(raw_path),
                     "slots": counts(predictions), "initial_slots": counts(pre),
                     "without_numeric_sign_slots": counts(no_sign),
                     "free_running_statuses": dict(Counter(row["status"] for row in report["observations"])),
                     "free_running": {k: report[k] for k in ("total", "kernel_accepted", "task_correct")},
                     "initial_free_running": {k: initial_report[k] for k in ("total", "kernel_accepted", "task_correct")}})
    summary = {"schema": "ssdm.transparent_cpu_receipt.v1", "command": [sys.executable, "-m", "language_growth.transparent_cpu", *sys.argv[1:]],
               "cwd": str(Path.cwd()), "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
               "hardware": {"processor": platform.processor(), "platform": platform.platform(), "python": sys.version,
                             "numpy": np.__version__, "device": "cpu", "gpu_used": False, "thread_limit": 2},
               "cost": {"incremental_paid_usd": 0, "rate_billed_usd": 0},
               "train_records_sha256": sha(records_path), "cohort_receipt_sha256": sha(args.cohort / "receipt.json"),
               "evaluation_manifest": development_manifest(tasks), "evaluation_tasks_sha256": sha(args.output / "evaluation_tasks.jsonl"),
               "config": {"epochs": args.epochs, "learning_rate": args.learning_rate, "l2": 1e-4, "seeds": args.seeds,
                          "feature_version": FEATURE_VERSION, "evaluation_split": args.eval_split}, "runs": runs,
               "aggregate": {"task_correct_median": statistics.median(r["free_running"]["task_correct"] for r in runs),
                             "task_correct_min": min(r["free_running"]["task_correct"] for r in runs),
                             "task_correct_max": max(r["free_running"]["task_correct"] for r in runs),
                             "fit_seconds_median": statistics.median(r["fit_seconds"] for r in runs),
                             "fit_seconds_min": min(r["fit_seconds"] for r in runs),
                             "fit_seconds_max": max(r["fit_seconds"] for r in runs)},
               "scope": f"learned ranking over engineered lexical grounding and exact decimal-sign features; fresh iterative {args.eval_split} worlds; no untouched held-out claim, shared-parent retention, causal mediation or L0 promotion claim"}
    (args.output / "receipt.json").write_text(canonical(summary) + "\n", encoding="ascii", newline="\n")
    print(canonical(summary))


if __name__ == "__main__":
    main()
