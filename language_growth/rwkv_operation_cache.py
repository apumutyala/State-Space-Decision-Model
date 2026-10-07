"""Prepared operation-only CPU diagnostic. Default/dry-run never import Torch."""
from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
import gzip
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import random
import secrets
import sys
import time

from language_growth.operation_cache import digest, operation_records

ROOT = Path(__file__).resolve().parents[1]
PLAN = "plans/codex/L0_OPERATION_ONLY_CACHE.md"
PREPARED = ROOT / "receipts/2026-10-06-L-operation-only-cache-prepared"
CORPUS = PREPARED / "corpus.json.gz"
CORPUS_SHA = "cf70a3e1abd8a8e5b1c2a7efe93ff0f2c0392982b1ef33dec11d60bd81143466"
CONFIG = {"seed": 20261006, "steps": 96, "passes": 8, "batch_size": 2,
          "head": "qend_axes", "loss": "ce_only", "axis_loss_weight": 0,
          "optimizer": "AdamW", "lr": .001, "weight_decay": 0., "max_grad_norm": 1.,
          "threads": 2, "min_start_bytes": 6*1024**3, "min_running_bytes": 3*1024**3,
          "timeout_seconds": 1200, "bootstrap_seed": 9203, "bootstrap_samples": 2000}
SCOPE = "prepared reused-development operation-only head diagnostic; no L0 qualification"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def corpus():
    # Corpus pins use the existing campaign's JSON serialization.
    data = json.loads(gzip.decompress(CORPUS.read_bytes()))
    if digest(data) != CORPUS_SHA:
        raise ValueError("frozen corpus drift")
    return data


def registration():
    from language_growth.rwkv7_cpu_parent import MODEL_ID, SNAPSHOT, PINNED_HASHES
    data = corpus()
    records = operation_records(data)
    sources = sorted(str(p.relative_to(ROOT)).replace("\\", "/")
        for folder in ("language_growth", "heads", "train", "decide", "third_party")
        for p in (ROOT / folder).rglob("*.py")) + [PLAN]
    return {"schema": "ssdm.operation-only-cache.v1", "status": "prepared_not_run",
        "base_commit": "018585548104dfbcb9d65a22214ec51405529711", "scope": SCOPE, "config": CONFIG,
        "corpus_sha256": CORPUS_SHA,
        "operation_records_sha256": {split: digest(rows) for split, rows in records.items()},
        "operation_row_ids": {split: [r["id"]+"/slot-001-operation" for r in rows]
                              for split, rows in records.items()},
        "counts": {"train_worlds": 24, "dev_worlds": 12, "train_rows": 24, "dev_rows": 12},
        "expected_parent": {"model_id": MODEL_ID, "snapshot": SNAPSHOT.name, "file_sha256": PINNED_HASHES},
        "runtime": {"interpreter": str(Path(sys.executable).resolve()), "python": sys.version,
                    "device": "cpu", "cuda_visible_devices": "-1", "cuda_available_required": False},
        "memory_basis": {"reference": "PR71 same-parent CPU run2", "required_start_bytes": 6*1024**3,
            "observed_start_available_bytes": 7554498560, "observed_minimum_available_bytes": 6093643776,
            "estimated_additional_operation_cache_bytes": 442368,
            "shared_parent": True, "feature_rows_per_call": 1, "stock_max_batch_rows": 2, "max_context": 2048,
            "residual_risk": "repeated longer sequence/high-water and unrelated host processes; retain runtime guard and owned-child cap"},
        "software": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "safetensors", "numpy", "psutil")},
        "source_sha256": {name: sha(ROOT/name) for name in sources},
        "cache_contract": {"features": ["q_end", "e", "option_mask", "ks", "qtypes"],
            "float_dtype": "detached CPU FP32", "labels": "separate targets; absent from cache and identity",
            "identity": "row ID/token IDs/spans/order/template/layout version + parent/encoder"},
        "arms": {"stock": "unchanged HFReadoutBackend (primary baseline)",
                 "zero_update": "same initial q_end checkpoint; operation-only composition (secondary)",
                 "trained": "fixed 96-update checkpoint; operation-only composition"},
        "primary": "paired trained-minus-stock useful answer and acceptance, all 12 dev worlds",
        "secondary": "paired trained-minus-zero-update; operation accuracy and downstream attribution; predeclared read/explain/compare paired CIs",
        "selection": "one seed, fixed checkpoint; no tuning; CE only; eight complete passes",
        "report": ["train/dev operation accuracy", "fit CE", "useful answers", "kernel acceptance",
                   "downstream slots", "all failures/wrong-task/false-relation", "cluster paired CI",
                   "parent direct accuracy and exact equality", "frozen backbone", "cost_usd=0"],
        "decision": {"strong_train_operation_fit": .8, "strict_useful_gain_required": True,
                     "read_non_regression_required": True,
                     "direct_equality_required": True, "complete_12_worlds_required": True,
                     "adoption": False},
        "execution_dependencies": "PR71/PR85/PR110 merged in pinned base; tiny-model proofs, independent PR review, fresh >=6 GiB start guard; no Index/heldout or paid spend",
        "preparation_scope": "This implementation task prepares/dry-runs only; the standing goal authorizes gated CPU work. No paid/GPU authority."}


def verify_frozen(value):
    if value != json.loads((PREPARED/"registration.json").read_text(encoding="utf-8")):
        raise ValueError("frozen operation registration drift")


def preparation(out):
    value = registration()
    if (PREPARED/"registration.json").exists():
        verify_frozen(value)
    out.mkdir(parents=True, exist_ok=False)
    write(out/"registration.json", value)
    write(out/"operation-records.json", operation_records(corpus()))
    return value


def budget_check(started, available):
    if time.monotonic()-started >= CONFIG["timeout_seconds"]:
        raise TimeoutError("1200-second operation diagnostic cap")
    if available < CONFIG["min_running_bytes"]:
        raise MemoryError("3 GiB running guard")


def lineage(entries, parent, encoder, template, targets):
    if len(entries) != len(targets):
        raise ValueError("one target per cache row required")
    return {"config": CONFIG, "parent": parent, "encoder": encoder, "template": template,
            "cache_keys": [e["key"] for e in entries], "corpus_sha256": CORPUS_SHA,
            "targets_sha256": digest([{"row_id": e["identity"]["row_id"], "label": label}
                                      for e, label in zip(entries, targets)])}


def train_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if k.startswith(("head.", "typed."))}


def save_checkpoint(path, model, opt, step, identity):
    import torch
    torch.save({"schema": "operation-cache-head-v1", "lineage": identity, "step": step,
                "trainable": train_state(model), "optimizer": opt.state_dict()}, path)


def restore_checkpoint(path, model, opt, identity, step):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state.get("schema") != "operation-cache-head-v1" or state.get("lineage") != identity
            or state.get("step") != step):
        raise ValueError("checkpoint lineage/step drift")
    expected = train_state(model)
    if set(state["trainable"]) != set(expected) or any(
            state["trainable"][k].shape != v.shape or state["trainable"][k].dtype != v.dtype
            for k, v in expected.items()):
        raise ValueError("checkpoint head shape drift")
    model.load_state_dict(state["trainable"], strict=False)
    opt.load_state_dict(state["optimizer"])


def fit_cached(model, entries, labels, opt, check=lambda: None):
    import torch
    from language_growth.operation_cache import cached_forward
    if len(entries) != 24 or len(labels) != 24:
        raise ValueError("24 operation targets required")
    if any(p.requires_grad for p in model.backbone.parameters()):
        raise ValueError("backbone must remain frozen")
    history, orders = [], []
    model.head.train()
    model.typed.train()
    for epoch in range(8):
        order = list(range(24))
        random.Random(CONFIG["seed"]+epoch).shuffle(order)
        orders.append([entries[i]["key"] for i in order])
        for cursor in range(0, 24, 2):
            check()
            ids = order[cursor:cursor+2]
            opt.zero_grad(set_to_none=True)
            lp = cached_forward(model, [entries[i] for i in ids])
            loss = torch.nn.functional.nll_loss(lp, torch.tensor([labels[i] for i in ids]))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head_parameters(), CONFIG["max_grad_norm"])
            opt.step()
            history.append({"step": len(history)+1, "epoch": epoch, "loss": float(loss.detach())})
    if len(history) != 96:
        raise ValueError("eight-pass update count drift")
    return history, orders


def fit_report(model, entries, labels):
    import torch
    from language_growth.operation_cache import cached_forward
    rows = []
    model.eval()
    with torch.no_grad():
        for entry, label in zip(entries, labels):
            lp = cached_forward(model, [entry])[0]
            rows.append({"row_id": entry["identity"]["row_id"], "correct": int(lp.argmax()) == label,
                         "ce": float(-lp[label]), "prediction": int(lp.argmax()), "target": label})
    return {"rows": rows, "n": len(rows), "accuracy": sum(r["correct"] for r in rows)/len(rows),
            "ce": sum(r["ce"] for r in rows)/len(rows)}


def bootstrap(before, after, *, seed, samples):
    if not before or len(before) != len(after):
        raise ValueError("paired worlds required")
    delta = [int(b)-int(a) for a, b in zip(before, after)]
    rng = random.Random(seed)
    draws = sorted(sum(rng.choice(delta) for _ in delta)/len(delta) for _ in range(samples))
    return {"n_worlds": len(delta), "estimate": sum(delta)/len(delta),
            "ci95": [draws[int(.025*samples)], draws[int(.975*samples)]],
            "unit": "independent generated world/task", "method": "paired percentile bootstrap",
            "seed": seed, "samples": samples}


def paired(current, baseline, *, expected_worlds=12):
    a, b = {r["task_id"]: r for r in current}, {r["task_id"]: r for r in baseline}
    if len(a) != expected_worlds or len(current) != len(a) or len(baseline) != len(b) or set(a) != set(b):
        raise ValueError("complete paired worlds required")
    return {metric: bootstrap([b[k][metric] for k in sorted(a)], [a[k][metric] for k in sorted(a)],
                              seed=CONFIG["bootstrap_seed"], samples=CONFIG["bootstrap_samples"])
            for metric in ("task_correct", "kernel_accepted")}


def paired_by_kind(current, baseline, examples):
    # Kinds are evaluator-only and never reach the serving router or cache key.
    groups = {kind: {e["task_id"] for e in examples if e["training_only"]["kind"] == kind}
              for kind in ("read", "explain", "compare")}
    if any(len(ids) != 4 for ids in groups.values()):
        raise ValueError("predeclared four worlds per kind required")
    paired(current, baseline)
    return {kind: paired([r for r in current if r["task_id"] in ids],
                         [r for r in baseline if r["task_id"] in ids], expected_worlds=4)
            for kind, ids in groups.items()}


def downstream(rows):
    """Keep all failures and missing slots visible; no conditional denominator."""
    observed = [slot for row in rows for slot in row["slot_attribution"] if slot["slot"] != "operation"]
    return {"worlds": len(rows), "failed_worlds": sum(r["status"] != "accepted" for r in rows),
            "accepted_wrong_task": sum(r["kernel_accepted"] and not r["task_correct"] for r in rows),
            "observed_later_slots": len(observed),
            "correct_later_slots": sum(s["matches_reference_slot"] for s in observed),
            "missing_reference_slots": [{"task_id": r["task_id"], "slots": r["missing_reference_slots"]} for r in rows],
            "wrong_relation_slots": sum(s["slot"] == "relation" and not s["matches_reference_slot"] for s in observed)}


# Bounded execution and attribution helpers kept local to avoid unmerged experiments.
def assert_environment():
    from language_growth.l0_qwen_cpu_probe import _offline_env
    required = _offline_env()
    keys = ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    if any(os.environ.get(k) != required[k] for k in keys):
        raise RuntimeError("offline CPU/2-thread environment required")


def stop_reason(elapsed, available):
    if elapsed >= CONFIG["timeout_seconds"]:
        return "timeout", 124
    if available < CONFIG["min_running_bytes"]:
        return "memory_guard", 125
    return None


def supervise(child, raw, *, available, now=time.monotonic, sleep=time.sleep):
    """Monitor only the child handle created by this launcher; never global kill."""
    from language_growth.l0_qwen_cpu_probe import _write_event
    try:
        started = now()
        minimum = available()
        while True:
            ram = available()
            minimum = min(minimum, ram)
            elapsed = now() - started
            result = child.poll()
            if result is not None:
                _write_event(raw, "launcher_exit", exit_code=result, elapsed_seconds=elapsed,
                             minimum_available_bytes=minimum)
                return result
            stop = stop_reason(elapsed, ram)
            if stop:
                child.kill()
                child.wait()
                _write_event(raw, "launcher_stopped", reason=stop[0], elapsed_seconds=elapsed,
                             minimum_available_bytes=minimum)
                return stop[1]
            sleep(0.25)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


class _ProcessHandle:
    def __init__(self, process):
        self.process = process
    def poll(self):
        return self.process.exitcode
    def kill(self):
        self.process.kill()
    def wait(self):
        self.process.join()
        return self.process.exitcode


def authorize_worker(connection, launcher_pid, expected_secret):
    parent = multiprocessing.parent_process()
    if (parent is None or parent.pid != launcher_pid or not parent.is_alive()
            or os.getppid() != launcher_pid or not connection.poll(2)):
        raise RuntimeError("internal worker requires owned launcher and inherited pipe")
    supplied = connection.recv_bytes(32)
    connection.close()
    if len(expected_secret) != 32 or not secrets.compare_digest(supplied, expected_secret):
        raise RuntimeError("internal worker authentication failed")


class Capture:
    def __init__(self, provider, raw, phase):
        self.provider, self.raw, self.phase = provider, raw, phase
        self.calls = []

    def decide(self, request):
        from language_growth.l0_qwen_cpu_probe import _write_event
        row = {"request": deepcopy(request), "response": None}
        self.calls.append(row)
        _write_event(self.raw, "request", phase=self.phase, request=request)
        try:
            response = self.provider.decide(request)
        except Exception as exc:
            _write_event(self.raw, "provider_exception", phase=self.phase,
                         request_id=request.get("ssdm", {}).get("request_id"),
                         question_ids=list(request.get("questions", {})),
                         exception_type=type(exc).__name__, message=str(exc))
            raise
        row["response"] = deepcopy(response)
        _write_event(self.raw, "response", phase=self.phase, response=response)
        return response


def evaluate_tasks(backend, decider, examples, raw, phase):
    from language_growth.free_running_dev import run_development_tasks
    report = []
    config = {"calibration": decider.calibration.identity(),
              "policy_masking": "none (served covers every supplied option)"}
    for example in examples:
        capture = Capture(decider, raw, phase)
        observation = run_development_tasks([example], capture, model=example["model_inputs"][0]["model"],
            parent_revision=backend.revision()["revision_id"], expected_served_config=config)["observations"][0]
        # Gold descriptions stay in this evaluator after the free-running call.
        reference = {}
        for request, target in zip(example["model_inputs"], example["training_only"]["targets"]):
            q = next(iter(request["questions"].values()))
            reference[target["slot"]] = q["criteria"][target["option_id"]]
        errors = []
        for call in capture.calls:
            qid, q = next(iter(call["request"]["questions"].items()))
            slot = qid.split("-", 2)[-1]
            answer = (call["response"] or {}).get("answers", {}).get(qid, {}).get("choice")
            actual = q["criteria"].get(answer) if isinstance(answer, str) else None
            errors.append({"slot": slot, "selected": answer, "selected_description": actual,
                           "expected_description": reference.get(slot),
                           "matches_reference_slot": actual == reference.get(slot)})
        report.append({**observation, "slot_attribution": errors,
                       "missing_reference_slots": sorted(set(reference) - {e["slot"] for e in errors})})
    write(raw.parent / f"{phase}.json", report)
    return report


def state_digest(module):
    """All parameters AND registered buffers; includes the independent LM head."""
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def run(out):
    import psutil
    assert_environment()
    if psutil.virtual_memory().available < CONFIG["min_start_bytes"]:
        raise MemoryError("6 GiB start guard before Torch/checkpoint load")
    frozen = json.loads((out/"registration.json").read_text())
    verify_frozen(frozen)
    if registration() != frozen:
        raise ValueError("runtime source/software/corpus drift")
    started = time.monotonic()
    check = lambda: budget_check(started, psutil.virtual_memory().available)
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only guard")
    check()
    from language_growth.rwkv7_cpu_parent import load_cpu_parent
    from language_growth.rwkv_decision_head import (rwkv_decision_backend,
        rwkv_cpu_parent_encoder, rwkv_cpu_parent_identity)
    from language_growth.operation_cache import capture, cached_forward, OperationOnlyBackend
    from train.data import items_from_records
    from train.guard import SplitGuard
    from decide.backends.hf_readout import HFReadoutBackend
    from decide.adapter import Decider
    from heads.decide_backend import DecisionModelBackend
    plugin, parent, evidence = load_cpu_parent()
    check()
    parent_sha = state_digest(parent)
    encode = rwkv_cpu_parent_encoder(plugin)
    parent_id = rwkv_cpu_parent_identity(plugin, evidence)
    backend = rwkv_decision_backend(parent, encode, source_identity=parent_id,
                                   head="qend_axes", seed=CONFIG["seed"])
    model = backend.model
    model.backbone.eval().requires_grad_(False)
    records = operation_records(corpus())
    items, caches, labels = {}, {}, {}
    for split, rows in records.items():
        items[split] = items_from_records(rows, encode, template=backend.template, layout=backend.layout,
            guard=SplitGuard.synthetic_only("frozen operation-only development corpus"))
        caches[split], labels[split] = [], []
        for item in items[split]:
            check()
            if len(item.row.ids) > 2048:
                raise ValueError("bounded CPU context exceeded")
            entry = capture(model, item, parent=parent_id, encoder=encode.identity(), template=backend.template.identity())
            with torch.no_grad():
                live = model([item.row]).log_probs
                cached = cached_forward(model, [entry])
                torch.testing.assert_close(cached, live, rtol=1e-5, atol=1e-6)
            caches[split].append(entry)
            labels[split].append(item.label)
    identity = lineage(caches["development_train"], parent_id, encode.identity(), backend.template.identity(), labels["development_train"])
    feature_identity = {k: v for k, v in identity.items() if k != "targets_sha256"}
    torch.save({"schema": "operation-feature-cache-v1", "lineage": feature_identity, "splits": caches}, out/"features.pt")
    write(out/"targets.json", {s: [{"row_id": i.id, "label": i.label} for i in group] for s, group in items.items()})
    opt = torch.optim.AdamW(model.head_parameters(), lr=CONFIG["lr"], weight_decay=0.)
    save_checkpoint(out/"initial-head.pt", model, opt, 0, identity)
    stock = HFReadoutBackend(plugin, "jrt-rwkv-g1-v1", max_batch_rows=2)
    stock_decider = Decider(stock, backend_max_input_tokens=plugin.max_context)
    data = corpus()
    def evaluate(condition, scorer):
        check()
        # Check the global cap between every real backend plan and score too.
        class Bounded:
            name = scorer.name
            def revision(self): return scorer.revision()
            def plan(self, request):
                check()
                return scorer.plan(request)
            def score(self, request, deadline):
                check()
                result = scorer.score(request, lambda: (check(), deadline()))
                check()
                return result
        provider = Decider(Bounded(), backend_max_input_tokens=plugin.max_context)
        return evaluate_tasks(scorer, provider, data["development_validation"], out/"raw.jsonl", condition)
    def learned(condition):
        fresh = DecisionModelBackend(model, encode, template=backend.template, layout=backend.layout,
                                     checkpoint={"sha256": sha(out/("initial-head.pt" if condition == "zero_update" else "trained-head.pt"))})
        return OperationOnlyBackend(fresh, stock)
    from language_growth.paired_comparison import direct_comparison_request, _target_relation, DIRECT_QID
    direct_requests = [(e, direct_comparison_request(e, model=plugin.model_id,
        parent_revision=stock_decider.revision()["revision_id"])) for e in data["development_validation"]
        if e["training_only"]["kind"] == "compare"]
    def direct():
        results = []
        for example, request in direct_requests:
            check()
            response = stock_decider.decide(request)
            answer = response.get("answers", {}).get(DIRECT_QID, {})
            choice = answer.get("choice")
            correct = request["questions"][DIRECT_QID]["criteria"].get(choice) == "Relation " + _target_relation(example)
            results.append({"task_id": example["task_id"], "status": response.get("ssdm", {}).get("status"),
                "answers": response.get("answers"), "served_probabilities": response.get("ssdm", {}).get("served_probabilities"),
                "correct": correct})
        return results
    before_direct = direct()
    baseline = evaluate("stock", stock)
    zero_fit = {s: fit_report(model, caches[s], labels[s]) for s in caches}
    zero = evaluate("zero_update", learned("zero_update"))
    history, orders = fit_cached(model, caches["development_train"], labels["development_train"], opt, check)
    save_checkpoint(out/"trained-head.pt", model, opt, 96, identity)
    saved_head = digest({k: v.tolist() for k, v in train_state(model).items()})
    restore_checkpoint(out/"trained-head.pt", model, opt, identity, 96)
    if saved_head != digest({k: v.tolist() for k, v in train_state(model).items()}):
        raise ValueError("checkpoint strict reload drift")
    trained_fit = {s: fit_report(model, caches[s], labels[s]) for s in caches}
    treatment = evaluate("trained", learned("trained"))
    after_direct = direct()
    if before_direct != after_direct or state_digest(parent) != parent_sha:
        raise ValueError("parent direct-path/backbone drift")
    comparison = {"trained_minus_stock": paired(treatment, baseline),
                  "trained_minus_zero_update": paired(treatment, zero)}
    kind_comparison = {"trained_minus_stock": paired_by_kind(treatment, baseline, data["development_validation"]),
                       "trained_minus_zero_update": paired_by_kind(treatment, zero, data["development_validation"])}
    write(out/"receipt.json", {"schema": "ssdm.operation-only-cache-result.v1", "status": "complete",
        "registration_sha256": sha(out/"registration.json"), "lineage": identity,
        "features_sha256": sha(out/"features.pt"), "initial_head_sha256": sha(out/"initial-head.pt"),
        "trained_head_sha256": sha(out/"trained-head.pt"), "parent_sha256": parent_sha,
        "stock": baseline, "zero_update": zero, "trained": treatment,
        "fit_before": zero_fit, "fit_after": trained_fit, "history": history, "orders": orders,
        "downstream": {"stock": downstream(baseline), "zero_update": downstream(zero), "trained": downstream(treatment)},
        "paired": comparison, "paired_by_kind": kind_comparison,
        "direct_before": before_direct, "direct_after": after_direct,
        "direct_equal": True, "frozen_backbone": True, "cost_usd": 0, "adopted": False,
        "eligible_for_review": trained_fit["development_train"]["accuracy"] >= .8 and
            comparison["trained_minus_stock"]["task_correct"]["estimate"] > 0 and
            kind_comparison["trained_minus_stock"]["read"]["task_correct"]["estimate"] >= 0 and
            kind_comparison["trained_minus_stock"]["read"]["kernel_accepted"]["estimate"] >= 0,
        "elapsed_seconds": time.monotonic()-started})


def worker(out, pipe, launcher_pid, secret):
    authorize_worker(pipe, launcher_pid, secret)
    with (out/"stdout.txt").open("w") as stdout, (out/"stderr.txt").open("w") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                run(out)
            except Exception as exc:
                write(out/"failure.json", {"status": "failed", "type": type(exc).__name__, "message": str(exc), "cost_usd": 0})
                raise


def launch(out):
    import psutil
    from language_growth.l0_qwen_cpu_probe import _offline_env
    preparation(out)
    if psutil.virtual_memory().available < CONFIG["min_start_bytes"]:
        write(out/"failure.json", {"status": "refused", "reason": "6 GiB start guard", "cost_usd": 0})
        return 125
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    secret = secrets.token_bytes(32)
    process = context.Process(target=worker, args=(out, receive, os.getpid(), secret), daemon=True)
    old = dict(os.environ)
    try:
        os.environ.update(_offline_env())
        process.start()
    finally:
        os.environ.clear()
        os.environ.update(old)
    receive.close()
    handle = _ProcessHandle(process)
    try:
        send.send_bytes(secret)
        send.close()
        return supervise(handle, out/"raw.jsonl", available=lambda: psutil.virtual_memory().available)
    finally:
        send.close()
        if handle.poll() is None:
            handle.kill()
            handle.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.execute:
        return launch(out)
    preparation(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
