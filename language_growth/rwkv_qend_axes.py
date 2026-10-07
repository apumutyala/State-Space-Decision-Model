"""Matched CPU q_end head-port ablation; default prepares only, never loads weights.

D-owned additive head/trainer changes require Claude cross-review. These are
explicitly supervised head axes, never emergent or backbone-causal semantics.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from language_growth import rwkv_head_learning as guard
from language_growth.qend_axis_targets import attach_axis_targets, targets_by_item
from language_growth.rwkv_head_ce96_evaluation import references

PLAN = "plans/codex/L0_RWKV_QEND_AXES.md"
PREPARED = ROOT / "receipts/2026-10-06-L-rwkv-qend-axes-prepared"
ARMS = {"control": 0.0, "axes": 1.0}
CONFIG = {"seed": 20261006, "steps": 32, "batch_size": 2, "lr_head": 0.001,
          "head": "qend_axes", "head_kwargs": {}, "lora_rank": 0,
          "loss": {"arm": "ce", "label_smoothing": 0.0, "brier_weight": 0.0, "rps_weight": 0.0},
          "threads": 2, "min_start_bytes": 12 * 1024**3, "min_running_bytes": 3 * 1024**3,
          "timeout_seconds": 1200, "intervention_seed": 9321}
CORPUS = ROOT / "receipts/2026-10-06-L-rwkv-head-ce-96-prepared-v2/corpus.json.gz"
PRIOR96 = ROOT / "receipts/2026-10-06-L-rwkv-head-ce-96-partial-run1/partial_receipt.json"
SCOPE = "one matched explicitly supervised head-port ablation; reused development data; no backbone-causal semantics or L0 qualification"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus():
    data = json.loads(gzip.decompress(CORPUS.read_bytes()))
    if guard.digest(data) != "cf70a3e1abd8a8e5b1c2a7efe93ff0f2c0392982b1ef33dec11d60bd81143466":
        raise ValueError("frozen ablation corpus changed")
    return data


def preparation(out, arm):
    if arm not in ARMS:
        raise ValueError("unknown matched arm")
    out.mkdir(parents=True, exist_ok=False)
    from language_growth.rwkv7_cpu_parent import MODEL_ID, SNAPSHOT, PINNED_HASHES
    data = corpus()
    targets = {split: targets_by_item(examples) for split, examples in data.items()}
    if {split: len(rows) for split, rows in targets.items()} != {"development_train": 64, "development_validation": 32}:
        raise ValueError("fixed 64/32 slot corpus required")
    guard.write(out / "corpus.json", data)
    guard.write(out / "axis-targets.json", targets)
    sources = sorted(str(path.relative_to(ROOT)).replace("\\", "/")
        for folder in ("language_growth", "heads", "train", "decide", "third_party")
        for path in (ROOT / folder).rglob("*.py") if "receipts" not in path.parts) + [PLAN]
    registration = {"schema": "ssdm.rwkv-qend-axes.v1", "status": "prepared_not_run",
        "base_commit": "0200ee29d0d7a7987a9128b629f32fa298d30b41", "scope": SCOPE,
        "claude_cross_review_required": True, "arm": arm, "axis_loss_weight": ARMS[arm],
        "config": CONFIG, "corpus_sha256": guard.digest(data),
        "split_sha256": {split: guard.digest(rows) for split, rows in data.items()},
        "axis_targets_sha256": guard.digest(targets),
        "expected_parent": {"model_id": MODEL_ID, "snapshot": SNAPSHOT.name, "file_sha256": PINNED_HASHES},
        "source_sha256": {name: sha(ROOT / name) for name in sources},
        "software": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "safetensors", "numpy", "psutil")},
        "reference_sha256": references()["sha256"], "prior96_sha256": sha(PRIOR96),
        "hypothesis": "Axis-supervised q_end head improves checked useful answers over its identical CE-only control.",
        "primary": "Paired complete useful-answer accuracy and kernel acceptance: lambda1 minus lambda0; same12 reused worlds.",
        "selection": "One seed,32 fixed updates/one complete pass, fixed final checkpoint; no tuning/selection.",
        "decision_rule": {"fit_accuracy_min": 0.8, "fit_correct_min_of_64": 52,
                          "require_complete_arms": True, "require_strict_useful_gain": True,
                          "require_no_read_regression": True, "axis_metrics_alone_never_capability": True},
        "interventions": {"axes": 6, "values": [-1, 0, 1],
            "random_control": "same port-perturbation norm, fixed seeded direction",
            "output_only_control": "affine replay on output logits, original axes unchanged; equal by construction",
            "claim_limit": "explicit head readout only; no causal superiority or backbone semantics"},
        "approval": "Preparation grants no parent/checkpoint load, fit, GPU, spend or PR."}
    guard.write(out / "registration.json", registration)
    if (PREPARED / f"registration-{arm}.json").is_file():
        verify_frozen(registration)
    return registration


def verify_frozen(registration):
    path = PREPARED / f"registration-{registration['arm']}.json"
    if registration != json.loads(path.read_text()):
        raise ValueError("committed qend-arm preregistration drift")


def make_trainer(backend, data, arm):
    from language_growth.training_records import export_training_records
    from train.data import items_from_records
    from train.guard import SplitGuard
    from train.losses import LossConfig
    from train.trainer import TrainConfig, Trainer
    split_guard = SplitGuard.synthetic_only("frozen qend-axis matched development corpus")
    items = items_from_records(export_training_records(data["development_train"]), backend.encode,
        template=backend.template, layout=backend.layout, guard=split_guard)
    items = attach_axis_targets(items, data["development_train"])
    if len(items) != 64 or any(len(item.row.ids) > 2048 for item in items):
        raise ValueError("exact64 bounded-context training slots required")
    cfg = TrainConfig(head="qend_axes", lora_rank=0, seed=CONFIG["seed"], batch_size=2,
        max_steps=32, lr_head=CONFIG["lr_head"], loss=LossConfig(**CONFIG["loss"]), axis_loss_weight=ARMS[arm])
    trainer = Trainer(backend.model, cfg, items, guard=split_guard)
    if any(p.requires_grad for p in backend.model.backbone.parameters()):
        raise ValueError("matched experiment must freeze every backbone parameter")
    return trainer


def port_sweep(port, seed):
    """Numeric final-head controls; no candidate text/ID semantics or text output."""
    import torch
    z, basis, bias = port["z"], port["basis"], port["bias"]
    baseline = torch.einsum("bd,bkd->bk", z, basis) + bias
    generator = torch.Generator().manual_seed(seed)
    result = []
    for axis in range(6):
        for value in (-1., 0., 1.):
            clamped = z.clone()
            clamped[:, axis] = value
            delta = clamped - z
            direction = torch.randn(z.shape, generator=generator)
            direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
            random_delta = direction * delta.norm(dim=1, keepdim=True)
            clamp_logits = torch.einsum("bd,bkd->bk", clamped, basis) + bias
            random_logits = torch.einsum("bd,bkd->bk", z + random_delta, basis) + bias
            output_only = baseline + torch.einsum("bd,bkd->bk", delta, basis)
            result.append({"axis": axis, "value": value,
                "clamp_norm": float(delta.norm()), "random_norm": float(random_delta.norm()),
                "baseline_index": baseline.argmax(-1).tolist(), "clamp_index": clamp_logits.argmax(-1).tolist(),
                "random_index": random_logits.argmax(-1).tolist(), "output_only_index": output_only.argmax(-1).tolist(),
                "clamp_logits": clamp_logits.tolist(), "random_logits": random_logits.tolist(),
                "output_only_logits": output_only.tolist(),
                "affine_replay_max_abs_error": float((output_only-clamp_logits).abs().max()),
                "output_only_axes_unchanged": True})
    return result


def final_fit(data, backend, out):
    from language_growth.rwkv_head_fit_audit import rows_for, evaluate_rows, aggregate
    from language_growth.l0_qwen_cpu_probe import _write_event
    import torch
    sidecars = {identity: target for examples in data.values() for identity, target in targets_by_item(examples).items()}
    observed, sweeps = [], []
    for index, row in enumerate(rows_for(data)):
        capture = []
        def hook(module, args, output):
            x = args[0]
            capture.append({"z": output.semantic_axes.detach().cpu().clone(),
                "basis": module.option_projection(x.e).detach().cpu().clone(),
                "bias": output.bias.detach().cpu().clone()})
        handle = backend.model.head.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                result = evaluate_rows([row], backend, out / "teacher-after-raw.jsonl")[0]
        finally:
            handle.remove()
        identity = f"{row['task_id']}/{row['qid']}"
        result["axis_target"] = list(sidecars[identity])  # evaluator only, after scoring
        if result["status"] == "ok" and len(capture) == 1:
            port = capture[0]
            result["semantic_axes"] = port["z"][0].tolist()
            result["axis_mse"] = float((port["z"][0] - torch.tensor(sidecars[identity])).square().mean())
            if row["split"] == "development_validation":
                sweeps.append({"task_id": row["task_id"], "qid": row["qid"],
                    "scope": "supervised stateless head port; no backbone intervention",
                    "conditions": port_sweep(port, CONFIG["intervention_seed"] + index)})
        else:
            result["semantic_axes"], result["axis_mse"] = None, None
            result["axis_diagnostic_failure"] = True
        observed.append(result)
        _write_event(out / "teacher-after-raw.jsonl", "axis_observation", **result)
    fit = {"rows": observed, "summary": aggregate(observed)}
    for summary in fit["summary"]:
        group = [row for row in observed if row["split"] == summary["split"] and
                 all(summary[key] == "all" or row[key] == summary[key] for key in ("kind", "slot", "qtype"))]
        values = [row["axis_mse"] for row in group if row["axis_mse"] is not None]
        summary["axis_mse_coverage"] = len(values) / len(group)
        summary["axis_mse"] = sum(values) / len(values) if len(values) == len(group) else None
    guard.write(out / "teacher-after.json", fit)
    guard.write(out / "axis-sweep.json", sweeps)
    return fit, sweeps


def launch(out, arm):
    import psutil
    from language_growth.l0_qwen_cpu_probe import _offline_env, _write_event
    registration = preparation(out, arm)
    verify_frozen(registration)
    available = psutil.virtual_memory().available
    if available < CONFIG["min_start_bytes"]:
        _write_event(out / "raw.jsonl", "launcher_refused", available_bytes=available,
                     required_bytes=CONFIG["min_start_bytes"])
        return 125
    ctx = multiprocessing.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    secret = secrets.token_bytes(32)
    process = ctx.Process(target=worker, args=(out, arm, receive, os.getpid(), secret), daemon=True)
    old = dict(os.environ)
    try:
        os.environ.update(_offline_env())
        process.start()
    finally:
        os.environ.clear()
        os.environ.update(old)
    receive.close()
    handle = guard._ProcessHandle(process)
    try:
        send.send_bytes(secret)
        send.close()
        result = guard.supervise(handle, out / "raw.jsonl", available=lambda: psutil.virtual_memory().available)
        return guard.recover_stopped_run(out, result)
    finally:
        send.close()
        if handle.poll() is None:
            handle.kill()
            handle.wait()


def worker(out, arm, connection, parent_pid, secret):
    guard.authorize_worker(connection, parent_pid, secret)
    with (out / "stdout.txt").open("w") as stdout, (out / "stderr.txt").open("w") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                run(out, arm)
            except Exception as exc:
                from language_growth.l0_qwen_cpu_probe import _write_event
                _write_event(out / "raw.jsonl", "worker_failure", exception_type=type(exc).__name__, message=str(exc))
                raise


def run(out, arm):
    import psutil
    guard.assert_environment()
    if psutil.virtual_memory().available < CONFIG["min_start_bytes"]:
        raise RuntimeError("qend worker requires12 GiB available before Torch/load")
    registration = json.loads((out / "registration.json").read_text())
    verify_frozen(registration)
    if registration["arm"] != arm:
        raise ValueError("arm identity drift")
    for path, expected in registration["source_sha256"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("source drift: " + path)
    for name, version in registration["software"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError("software drift: " + name)
    data = json.loads((out / "corpus.json").read_text())
    targets = {split: targets_by_item(examples) for split, examples in data.items()}
    if guard.digest(data) != registration["corpus_sha256"] or guard.digest(targets) != registration["axis_targets_sha256"]:
        raise ValueError("corpus/axis-sidecar drift")
    if guard.digest(json.loads((out / "axis-targets.json").read_text())) != registration["axis_targets_sha256"]:
        raise ValueError("saved axis-sidecar artifact drift")
    if references()["sha256"] != registration["reference_sha256"] or sha(PRIOR96) != registration["prior96_sha256"]:
        raise ValueError("reference result drift")
    from language_growth.rwkv7_cpu_parent import load_cpu_parent
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available() or psutil.virtual_memory().available < CONFIG["min_running_bytes"]:
        raise RuntimeError("CPU/running memory guard failed before parent load")
    plugin, parent, evidence = load_cpu_parent()
    from language_growth.rwkv_decision_head import (rwkv_cpu_parent_encoder, rwkv_cpu_parent_identity,
        rwkv_cpu_parent_decider, rwkv_decision_backend)
    from heads.decide_backend import DecisionModelBackend
    from language_growth.l0_qwen_cpu_probe import _write_event
    guard.write(out / "parent-evidence.json", evidence)
    backend = rwkv_decision_backend(parent, rwkv_cpu_parent_encoder(plugin),
        source_identity=rwkv_cpu_parent_identity(plugin, evidence), head="qend_axes", seed=CONFIG["seed"])
    trainer = make_trainer(backend, data, arm)
    initial_head_sha = head_digest(backend.model)
    parent_sha = guard.state_digest(parent)
    order = [trainer.train_items[i].id for i in trainer._order(0)]
    guard.write(out / "train-config.json", trainer.cfg.to_json())
    checkpoint0 = trainer.save(out / "checkpoint-before")
    for _ in range(32):
        loss = trainer.train_step()
        _write_event(out / "raw.jsonl", "train_step", step=trainer.step, loss=loss, arm=arm)
    if trainer.epoch != 0 or trainer.cursor != 64:
        raise ValueError("matched arm did not complete exactly one data pass")
    checkpoint = trainer.save(out / "checkpoint-after")
    guard.strict_restore(trainer, checkpoint)
    history = list(trainer.history)
    del trainer
    backend.model.eval().requires_grad_(False)
    final_backend = DecisionModelBackend(backend.model, backend.encode, template=backend.template,
        layout=backend.layout, checkpoint={"sha256": sha(checkpoint), "arm": arm})
    after = checked_answers(final_backend, rwkv_cpu_parent_decider(final_backend, plugin),
                            data["development_validation"], out)
    fit, sweeps = final_fit(data, final_backend, out)
    if guard.state_digest(parent) != parent_sha:
        raise ValueError("full parent state changed")
    prior = references()["learning"]
    prior96 = json.loads(PRIOR96.read_text())
    comparison = {name: free_comparison(after, saved["after"])
                  for name, saved in (("step24", prior), ("step96_partial", prior96))}
    guard.write(out / "receipt.json", {"schema": "ssdm.rwkv-qend-axes-result.v1", "status": "complete",
        "arm": arm, "axis_loss_weight": ARMS[arm], "config": CONFIG, "scope": SCOPE,
        "registration_sha256": sha(out / "registration.json"), "source_sha256": registration["source_sha256"],
        "corpus_sha256": registration["corpus_sha256"], "axis_targets_sha256": registration["axis_targets_sha256"],
        "initial_head_sha256": initial_head_sha, "full_parent_state_sha256": parent_sha,
        "expected_training_order_sha256": guard.digest(order), "checkpoint_before_sha256": sha(checkpoint0),
        "checkpoint_after_sha256": sha(checkpoint), "after": after, "fit": fit,
        "axis_sweep_count": len(sweeps), "previous_free_running": comparison, "history": history,
        "cost_usd": 0, "adopted": False, "claude_cross_review_required": True})


def head_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith(("head.", "typed.")):
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def checked_answers(backend, provider, examples, out):
    return guard.evaluate(backend, provider, examples, out / "raw.jsonl", "after")


def free_comparison(current, previous):
    a, b = {r["task_id"]: r for r in current}, {r["task_id"]: r for r in previous}
    if len(a) != 12 or set(a) != set(b):
        raise ValueError("complete identical12-world free-running evidence required")
    ids = sorted(a)
    return {metric: guard.bootstrap([b[k][metric] for k in ids], [a[k][metric] for k in ids])
            for metric in ("task_correct", "kernel_accepted")}


def compare_arms(control_out, axes_out, destination):
    """JSON-only paired screening; never selects/checkpoints or adopts a model."""
    control = json.loads((Path(control_out) / "receipt.json").read_text())
    axes = json.loads((Path(axes_out) / "receipt.json").read_text())
    if [(r["status"], r["arm"], r["axis_loss_weight"]) for r in (control, axes)] != [
            ("complete", "control", 0.0), ("complete", "axes", 1.0)]:
        raise ValueError("complete control/axes arm receipts required")
    for key in ("config", "source_sha256", "corpus_sha256", "axis_targets_sha256", "initial_head_sha256",
                "full_parent_state_sha256", "expected_training_order_sha256"):
        if control[key] != axes[key]:
            raise ValueError("unmatched arm identity: " + key)
    paired = free_comparison(axes["after"], control["after"])
    read_ids = {e["task_id"] for e in corpus()["development_validation"] if e["training_only"]["kind"] == "read"}
    read = lambda report: sum(r["task_correct"] for r in report["after"] if r["task_id"] in read_ids)
    fitted = next(r for r in axes["fit"]["summary"] if
        (r["split"], r["kind"], r["slot"], r["qtype"]) == ("development_train", "all", "all", "all"))
    eligible = fitted["n_failed"] == 0 and fitted["accuracy_all"] >= .8 and paired["task_correct"]["estimate"] > 0 and read(axes) >= read(control)
    result = {"scope": SCOPE, "paired": paired, "lambda1_fit": fitted,
        "read_correct": {"control": read(control), "axes": read(axes)},
        "eligible_for_review": eligible, "adopted": False,
        "limit": "screening only; axes are supervised head ports, not backbone-causal semantics"}
    with Path(destination).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    try:
        if args.execute and args.dry_run:
            raise ValueError("dry-run and execute are mutually exclusive")
        if args.execute:
            return launch(out, args.arm)
        preparation(out, args.arm)
        print(json.dumps({"status": "prepared_not_run", "arm": args.arm, "out": str(out)}))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
