"""One evaluation-only teacher-forced fit/transfer audit; default is preparation.

Real parent/checkpoint access occurs only in the authenticated --execute worker.
No optimizer, fitting, rollout, calibration fit or data generation is performed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import contextlib
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import secrets
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from language_growth import rwkv_head_learning as shared

PRIOR = Path("C:/Users/apumu/research/CHLabs/ssdm-data/rwkv-head-learning-run1-20261006")
CHECKPOINT = PRIOR / "checkpoint-after/ckpt-step0000024.pt"
CHECKPOINT_SHA256 = "3a896a00dc06fdd79aa8656f5272fcd4307038633efb0e0d642773280affe13c"
INPUT_HASHES = {
    "corpus.json": "086dff24e00bcd6979669b752a290439aba5edf946441fa1c17c5375858c2f12",
    "registration.json": "952c44a84e6140488179027d39e8a92297b355d91b31f3fde05d5975aa46dc15",
    "receipt.json": "950f1bc202b55f01e79953d07505757368eceede2ab5c95939dcb4b3ae867ecd",
    "train-config.json": "c59db7602b6a0623842c3e76cb43d61988e2ca9e6e580e959f8b7d5f60aaf82a",
}
FROZEN_REGISTRATION = ROOT / "receipts/2026-10-06-L-rwkv-head-fit-prepared/registration.json"
CONFIG = {"requests": {"development_train": 64, "development_validation": 32},
          "threads": 2, "timeout_seconds": 900, "min_start_bytes": 8 * 1024**3,
          "min_running_bytes": 3 * 1024**3, "checkpoint_step": 24,
          "strong_fit_accuracy": 0.8}
SCOPE = "one final-checkpoint teacher-forced synthetic development diagnostic; no capability qualification"
PLAN = "plans/codex/L0_RWKV_HEAD_FIT_AUDIT.md"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def event(raw, name, **fields):
    from language_growth.l0_qwen_cpu_probe import _write_event
    _write_event(raw, name, **fields)


def rows_for(data):
    """Keep current gold labels outside the unmodified, earlier-gold-prefix request."""
    from language_growth.training_records import export_training_records
    rows = []
    for split, count in CONFIG["requests"].items():
        examples = data[split]
        records = export_training_records(examples, split=split)
        kinds = {e["task_id"]: e["training_only"]["kind"] for e in examples}
        if len(records) != count:
            raise ValueError(f"expected exactly {count} {split} slot rows")
        for rec in records:
            qid, label = next(iter(rec["labels"].items()))
            rows.append({"split": split, "kind": kinds[rec["id"]], "task_id": rec["id"],
                         "qid": qid, "slot": qid.split("-", 2)[-1],
                         "qtype": rec["request"]["questions"][qid]["type"],
                         "request": deepcopy(rec["request"]), "label": label})
    return rows


def preparation(out):
    """Read only pinned JSON inputs and source; never inspect checkpoint/snapshot."""
    out.mkdir(parents=True, exist_ok=False)
    inputs = {}
    for name, expected in INPUT_HASHES.items():
        content = (PRIOR / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"original input drift: {name}")
        inputs[name] = json.loads(content)
    data, old = inputs["corpus.json"], inputs["registration.json"]
    if shared.digest(data) != old["corpus_sha256"]:
        raise ValueError("original canonical corpus mismatch")
    for split, expected in old["split_sha256"].items():
        if shared.digest(data[split]) != expected:
            raise ValueError(f"original split mismatch: {split}")
    rows_for(data)
    shared.write(out / "corpus.json", data)
    shared.write(out / "prior-receipt.json", inputs["receipt.json"])
    shared.write(out / "prior-train-config.json", inputs["train-config.json"])
    sources = sorted(str(p.relative_to(ROOT)).replace("\\", "/")
                     for folder in ("language_growth", "heads", "train", "decide", "third_party")
                     for p in (ROOT / folder).rglob("*.py") if "receipts" not in p.parts)
    sources.append(PLAN)
    registration = {
        "schema": "ssdm.rwkv-head-fit-audit.v1", "status": "prepared_not_run", "scope": SCOPE,
        "base_commit": "5499fe754bf2c3d9fc43a87e07890f5df6845fb8",
        "prior_run_source": "d8b3be06b9f4e8f142d86c082ce9e69818a41d8f", "config": CONFIG,
        "input_sha256": INPUT_HASHES, "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": CHECKPOINT_SHA256, "expected_parent": old["expected_parent"],
        "corpus_sha256": shared.digest(data), "split_sha256": old["split_sha256"],
        "source_sha256": {name: sha(ROOT / name) for name in sources},
        "software": {name: importlib.metadata.version(name) for name in old["software"]},
        "hypothesis": "The 24-update head has not yet fit the supervised mapping; diagnose fit before more training.",
        "primary": "Exact accuracy and unrounded CE on all 64 training slots, with 32 validation slots as transfer diagnostic.",
        "selection": "Exactly one preserved final checkpoint; one fixed pass; no fitting or selection.",
        "failures": "Every row remains in accuracy denominator; failed-row CE is undefined, never zero or silently omitted.",
        "approval": "Preparation does not authorize checkpoint or parent loading.",
    }
    shared.write(out / "registration.json", registration)
    return registration


def verify_frozen(registration):
    expected = json.loads(FROZEN_REGISTRATION.read_text(encoding="utf-8"))
    if registration != expected:
        raise ValueError("committed head-fit preregistration drift")


def aggregate(rows):
    groups = {}
    for row in rows:
        for kind, slot, qtype in (("all", "all", "all"), (row["kind"], "all", "all"),
                                 ("all", row["slot"], "all"), ("all", "all", row["qtype"]),
                                 (row["kind"], row["slot"], row["qtype"])):
            key = (row["split"], kind, slot, qtype)
            groups.setdefault(key, []).append(row)
    result = []
    for (split, kind, slot, qtype), group in sorted(groups.items()):
        ok = [r for r in group if r["status"] == "ok"]
        n, n_ok = len(group), len(ok)
        correct = sum(r["correct"] for r in group)
        ce = sum(r["ce"] for r in ok) / n_ok if n_ok else None
        result.append({"split": split, "kind": kind, "slot": slot, "qtype": qtype,
                       "n": n, "n_ok": n_ok, "n_failed": n - n_ok, "correct": correct,
                       "accuracy_all": correct / n, "ce": ce if n_ok == n else None,
                       "ce_on_successful_rows": ce, "ce_coverage": n_ok / n})
    return result


def evaluate_rows(rows, backend, raw, check_deadline=lambda: None):
    """Exactly one backend scoring call per request; failures do not trigger retries."""
    observations = []
    for row in rows:
        identity = {k: v for k, v in row.items() if k not in ("request", "label")}
        event(raw, "request", **identity, request=row["request"])
        record = {**identity, "label": row["label"], "status": "failed", "correct": False,
                  "selected": None, "ce": None}
        try:
            check_deadline()
            plan = backend.plan(row["request"])
            if plan.max_call_input_tokens > 2048:
                raise ValueError("request exceeds pinned 2048-token context")
            score = backend.score(row["request"], check_deadline)
            # A malformed/nonfinite response must remain representable in JSON.
            event(raw, "response", **identity, score_repr=repr(score.logits), inference_calls=score.inference_calls)
            offered = row["request"]["questions"][row["qid"]]["criteria"]
            if score.inference_calls != 1 or set(score.logits) != {row["qid"]}:
                raise ValueError("expected exactly one question and scoring forward")
            logits = score.logits[row["qid"]]
            if set(logits) != set(offered) or not all(math.isfinite(v) for v in logits.values()):
                raise ValueError("missing options or nonfinite log-probabilities")
            maximum = max(logits.values())
            log_z = maximum + math.log(sum(math.exp(v - maximum) for v in logits.values()))
            if abs(log_z) > 2e-5:
                raise ValueError("head scores must be normalized uncalibrated log-probabilities")
            selected = max(sorted(logits), key=logits.get)
            record.update(status="ok", selected=selected, correct=selected == row["label"],
                          ce=-logits[row["label"]], log_probabilities=logits,
                          input_tokens=score.total_input_tokens, inference_calls=score.inference_calls)
        except Exception as exc:
            record.update(exception_type=type(exc).__name__, message=str(exc))
        observations.append(record)
        event(raw, "row_result", **record)
    return observations


def interpretation(summary, prior):
    totals = {r["split"]: r for r in summary if (r["kind"], r["slot"], r["qtype"]) == ("all", "all", "all")}
    train, dev = totals["development_train"], totals["development_validation"]
    if train["n_failed"] or dev["n_failed"]:
        diagnosis = "execution_failures_no_fit_transfer_classification"
    elif train["accuracy_all"] < CONFIG["strong_fit_accuracy"]:
        diagnosis = "mapping_not_strongly_fit_representation_vs_optimization_unresolved"
    elif dev["accuracy_all"] < CONFIG["strong_fit_accuracy"]:
        diagnosis = "fit_transfer_gap_on_reused_development_worlds"
    else:
        diagnosis = "strong_teacher_forced_slots_do_not_establish_free_running_capability"
    return {"diagnosis": diagnosis, "strong_fit_threshold": CONFIG["strong_fit_accuracy"],
            "prior_free_running": {phase: {"useful": sum(r["task_correct"] for r in prior[phase]),
                                           "kernel_accepted": sum(r["kernel_accepted"] for r in prior[phase]),
                                           "n": len(prior[phase])} for phase in ("before", "after")},
            "limits": "Teacher forcing, unequal per-kind slot counts and prior validation reuse prevent capability qualification or causal separation of optimization and representation."}


def restore_head(model, checkpoint, train_items, train_config):
    """Restore only pinned head tensors; never construct Trainer or an optimizer."""
    import torch
    from train.lineage import lineage, diff
    if sha(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("final checkpoint bytes differ from pin")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if sha(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("final checkpoint changed while loading")
    if (saved["step"] != CONFIG["checkpoint_step"] or saved["config"] != train_config
            or saved["init_seed"] != train_config["seed"] or saved["lora_modules"]):
        raise ValueError("checkpoint step/config/init/LoRA mismatch")
    differences = diff(saved["lineage"], lineage(model.backbone, train_items))
    if differences:
        raise ValueError("checkpoint lineage mismatch: " + ",".join(differences))
    current = {n: p for n, p in model.named_parameters() if n.startswith(("head.", "typed."))}
    if set(saved["trainable"]) != set(current):
        raise ValueError("checkpoint head tensor keys mismatch")
    for name, tensor in current.items():
        value = saved["trainable"][name]
        if (not isinstance(value, torch.Tensor) or value.shape != tensor.shape
                or value.dtype != tensor.dtype or not torch.isfinite(value).all()):
            raise ValueError("checkpoint head tensor mismatch: " + name)
    with torch.no_grad():
        for name, tensor in current.items():
            tensor.copy_(saved["trainable"][name])
    model.eval().requires_grad_(False)


def _run(out):
    import psutil
    shared.assert_environment()
    if psutil.virtual_memory().available < CONFIG["min_start_bytes"]:
        raise RuntimeError("worker requires 8 GiB available before Torch/load")
    registration = json.loads((out / "registration.json").read_text())
    verify_frozen(registration)
    for name, expected in registration["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("source drift: " + name)
    for name, expected in registration["software"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("software drift: " + name)
    data = json.loads((out / "corpus.json").read_text())
    if shared.digest(data) != registration["corpus_sha256"]:
        raise ValueError("prepared corpus drift")
    # Input JSON copied for convenience is still checked against pinned originals.
    for name, expected in INPUT_HASHES.items():
        if sha(PRIOR / name) != expected:
            raise ValueError("original input drift: " + name)
    prior = json.loads((PRIOR / "receipt.json").read_text())
    train_config = json.loads((PRIOR / "train-config.json").read_text())
    if sha(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("final checkpoint bytes differ from pin")
    from language_growth.rwkv7_cpu_parent import inspect_snapshot, load_cpu_parent
    evidence = inspect_snapshot()
    shared.write(out / "parent-evidence.json", evidence)
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available() or psutil.virtual_memory().available < CONFIG["min_running_bytes"]:
        raise RuntimeError("offline CPU/running RAM guard failed before load")
    plugin, parent, loaded = load_cpu_parent()
    if loaded != evidence:
        raise ValueError("snapshot evidence changed while loading")
    from language_growth.rwkv_decision_head import rwkv_cpu_parent_encoder, rwkv_cpu_parent_identity, rwkv_decision_backend
    from train.data import items_from_records
    from train.guard import SplitGuard
    from language_growth.training_records import export_training_records
    backend = rwkv_decision_backend(parent, rwkv_cpu_parent_encoder(plugin),
        source_identity=rwkv_cpu_parent_identity(plugin, evidence), seed=train_config["seed"],
        head=train_config["head"], head_kwargs=train_config["head_kwargs"])
    items = items_from_records(export_training_records(data["development_train"]), backend.encode,
        template=backend.template, layout=backend.layout,
        guard=SplitGuard.synthetic_only("existing frozen head-fit development corpus"))
    restore_head(backend.model, CHECKPOINT, items, train_config)
    frozen = shared.state_digest(parent)
    if frozen != prior["full_parent_state_sha256"]:
        raise ValueError("loaded parent does not match original run full state")
    from heads.decide_backend import DecisionModelBackend
    backend = DecisionModelBackend(backend.model, backend.encode, template=backend.template,
                                  layout=backend.layout, checkpoint={"sha256": CHECKPOINT_SHA256})
    started = time.monotonic()
    def deadline():
        if time.monotonic() - started >= CONFIG["timeout_seconds"]:
            raise TimeoutError("evaluation deadline")
    with torch.inference_mode():
        observations = evaluate_rows(rows_for(data), backend, out / "raw.jsonl", deadline)
    if shared.state_digest(parent) != frozen:
        raise ValueError("parent state changed during evaluation")
    summary = aggregate(observations)
    shared.write(out / "rows.json", observations)
    shared.write(out / "summary.json", summary)
    shared.write(out / "receipt.json", {"scope": SCOPE, "registration_sha256": sha(out / "registration.json"),
        "checkpoint_sha256": CHECKPOINT_SHA256, "full_parent_state_sha256": frozen,
        "parent_unchanged": True, "revision": backend.revision(), "rows_sha256": sha(out / "rows.json"),
        "summary": summary, "interpretation": interpretation(summary, prior),
        "requests_attempted": len(observations), "cost_usd": 0, "optimizer_steps": 0})


def child(out, connection, launcher_pid, secret):
    shared.authorize_worker(connection, launcher_pid, secret)
    with (out / "stdout.txt").open("w", encoding="utf-8") as stdout, (out / "stderr.txt").open("w", encoding="utf-8") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                _run(out)
            except Exception as exc:
                event(out / "raw.jsonl", "worker_failure", exception_type=type(exc).__name__, message=str(exc))
                raise


def launch(out):
    import psutil
    from language_growth.l0_qwen_cpu_probe import _offline_env
    registration = preparation(out)
    verify_frozen(registration)
    raw = out / "raw.jsonl"
    ram = psutil.virtual_memory().available
    if ram < CONFIG["min_start_bytes"]:
        event(raw, "launcher_refused", available_bytes=ram, required_bytes=CONFIG["min_start_bytes"])
        return 125
    event(raw, "launcher_start", config=CONFIG, worker="internal inherited-pipe worker")
    ctx = multiprocessing.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    secret = secrets.token_bytes(32)
    process = ctx.Process(target=child, args=(out, receive, os.getpid(), secret), daemon=True)
    old = dict(os.environ)
    try:
        os.environ.update(_offline_env())
        process.start()
    finally:
        os.environ.clear()
        os.environ.update(old)
    receive.close()
    handle = shared._ProcessHandle(process)
    try:
        send.send_bytes(secret)
        send.close()
        return shared.supervise(handle, raw, available=lambda: psutil.virtual_memory().available)
    finally:
        send.close()
        if handle.poll() is None:
            handle.kill()
            handle.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    owns = not out.exists()
    try:
        if args.execute:
            return launch(out)
        preparation(out)
        print(json.dumps({"status": "prepared_not_run", "out": str(out), "scope": SCOPE}))
        return 0
    except Exception as exc:
        if owns and out.is_dir():
            event(out / "raw.jsonl", "failure", exception_type=type(exc).__name__, message=str(exc))
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
