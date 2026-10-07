"""Prepared, guarded CPU head-learning diagnostic; real execution needs review.

The default CLI freezes synthetic inputs and exits without importing Torch or
opening checkpoint files. --execute is reserved for explicit run approval.
This is an exploratory surrogate, never an L0 gate or D-parent retention test.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import contextlib
import hashlib
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import secrets
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FROZEN_REGISTRATION = ROOT / "receipts/2026-10-06-L-rwkv-head-ce-96-prepared-v2/registration.json"
PLAN = "plans/codex/L0_RWKV_HEAD_CE_96.md"
CONFIG = {"seed": 20261006, "train_seed": 9201, "validation_seed": 9202,
          "train_count": 24, "validation_count": 12, "steps": 96,
          "batch_size": 2, "lr_head": 0.001, "head": "snapshot",
          "head_kwargs": {"channels": 4, "max_snapshots": 2},
          "lora_rank": 0, "threads": 2, "timeout_seconds": 1200,
          "loss": {"arm": "ce", "label_smoothing": 0.0, "brier_weight": 0.0, "rps_weight": 0.0},
          "min_start_bytes": 8 * 1024**3, "min_running_bytes": 3 * 1024**3,
          "bootstrap_seed": 9203, "bootstrap_samples": 2000}
DEPENDENCIES = {"agent-main": "0c8eaf7276dd7680aedd7d7b3b7d6814eada5f41",
                "PR71": "14aa0daa486804ca82bf332dd0469717cc7a040b",
                "PR85": "e9447e64e97fac8a2e0b3377acdacc4e10d343e9"}
SCOPE = "exploratory frozen RWKV7-0.1B surrogate; not L0 qualification or D-parent retention"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def corpus():
    from language_growth.development_data import generate_development_tasks
    result = {}
    for split, seed, count in (("development_train", CONFIG["train_seed"], CONFIG["train_count"]),
                               ("development_validation", CONFIG["validation_seed"], CONFIG["validation_count"])):
        examples = generate_development_tasks(seed, count, choice_format="production", split=split)
        counts = {kind: sum(e["training_only"]["kind"] == kind for e in examples)
                  for kind in ("read", "explain", "compare")}
        if set(counts.values()) != {count // 3}:
            raise ValueError("stratification drift")
        result[split] = examples
    train_states = {e["model_inputs"][0]["state"] for e in result["development_train"]}
    if train_states & {e["model_inputs"][0]["state"] for e in result["development_validation"]}:
        raise ValueError("train/dev worlds overlap")
    return result


def preparation(out):
    """Freeze inputs without inspecting/loading real weights or importing Torch."""
    out.mkdir(parents=True, exist_ok=False)
    from language_growth.rwkv7_cpu_parent import MODEL_ID, SNAPSHOT, PINNED_HASHES
    data = corpus()
    from language_growth.rwkv_head_ce96_evaluation import exposure_contract, references, operational_budget
    exposure = exposure_contract(data, CONFIG)
    prior = references()
    budget = operational_budget()
    write(out / "corpus.json", data)
    # Freeze the complete local executable closure, including losses/layout/
    # adapter helpers and the vendored model. Receipts are data, not runtime code.
    sources = sorted(str(p.relative_to(ROOT)).replace("\\", "/")
                     for folder in ("language_growth", "heads", "train", "decide", "third_party")
                     for p in (ROOT / folder).rglob("*.py") if "receipts" not in p.parts)
    sources.append(PLAN)
    registration = {"schema": "ssdm.rwkv-head-learning.v1", "status": "prepared_not_run",
                    "scope": SCOPE, "config": CONFIG, "dependencies": DEPENDENCIES,
                    "expected_parent": {"model_id": MODEL_ID, "snapshot": SNAPSHOT.name,
                                        "file_sha256": PINNED_HASHES},
                    "corpus_sha256": digest(data),
                    "split_sha256": {k: digest(v) for k, v in data.items()},
                    "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
                    "software": {name: importlib.metadata.version(name) for name in
                                 ("torch", "transformers", "safetensors", "numpy", "psutil")},
                    "variant_base": "bb2eadd798b5d5ae317524767b37115e9a8df168",
                    "training_delta": "Only optimizer steps 24 to 96; same initialization, sampler, corpus, parent, head, loss and seed.",
                    "operational_delta": budget,
                    "exposure": exposure, "reference_sha256": prior["sha256"],
                    "hypothesis": "Three complete passes (96 updates) improve train-slot fit and possibly useful free-running answers over the weak 24-step head.",
                    "selection": "One seed, one recipe, final step only; no validation-based selection.",
                    "primary": "Final teacher-forced train/dev exact accuracy and CE; free-running useful answers and kernel acceptance versus step 0 and step 24, with world-cluster paired intervals.",
                    "claim_rule": "Report all outcomes; tiny exploratory cohort cannot qualify a level.",
                    "approval": "Real checkpoint load and fitting are not authorized by preparation."}
    write(out / "registration.json", registration)
    return registration


def assert_environment():
    from language_growth.l0_qwen_cpu_probe import _offline_env
    required = _offline_env()
    keys = ("CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    if any(os.environ.get(k) != required[k] for k in keys):
        raise RuntimeError("offline CPU/2-thread environment required")


def verify_frozen(registration):
    frozen = json.loads(FROZEN_REGISTRATION.read_text(encoding="utf-8"))
    for key in ("config", "dependencies", "expected_parent", "corpus_sha256", "split_sha256", "source_sha256", "software", "variant_base", "training_delta", "operational_delta", "exposure", "reference_sha256"):
        if registration[key] != frozen[key]:
            raise ValueError(f"committed preregistration drift: {key}")


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


def launch(out):
    import psutil
    from language_growth.l0_qwen_cpu_probe import _offline_env, _write_event
    registration = preparation(out)
    raw = out / "raw.jsonl"
    ram = psutil.virtual_memory().available
    if ram < CONFIG["min_start_bytes"]:
        _write_event(raw, "launcher_refused", available_bytes=ram, required_bytes=CONFIG["min_start_bytes"])
        return 125
    verify_frozen(registration)
    _write_event(raw, "launcher_start", config=CONFIG, worker="internal inherited-pipe worker")
    ctx = multiprocessing.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    secret = secrets.token_bytes(32)  # memory/anonymous pipe only; never write or log
    process = ctx.Process(target=child, args=(out, receive, os.getpid(), secret), daemon=True)
    env = _offline_env()
    old = dict(os.environ)
    try:
        os.environ.update(env)
        process.start()
    finally:
        os.environ.clear()
        os.environ.update(old)
    receive.close()
    handle = _ProcessHandle(process)
    try:
        send.send_bytes(secret)
        send.close()
        return supervise(handle, raw, available=lambda: psutil.virtual_memory().available)
    finally:
        send.close()
        if handle.poll() is None:
            handle.kill()
            handle.wait()


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


def trainer_for(backend, examples, *, steps=None):
    from train.data import items_from_records
    from train.guard import SplitGuard
    from train.trainer import TrainConfig, Trainer
    from train.losses import LossConfig
    from language_growth.training_records import export_training_records
    guard = SplitGuard.synthetic_only("RWKV head-only generated development-train worlds")
    records = export_training_records(examples)
    items = items_from_records(records, backend.encode, template=backend.template,
                               layout=backend.layout, guard=guard)
    if any(len(i.row.ids) > 2048 for i in items):
        raise ValueError("training row exceeds pinned CPU context")
    cfg = TrainConfig(head=CONFIG["head"], head_kwargs=CONFIG["head_kwargs"],
                      lora_rank=0, seed=CONFIG["seed"], batch_size=CONFIG["batch_size"],
                      loss=LossConfig(**CONFIG["loss"]),
                      lr_head=CONFIG["lr_head"], max_steps=CONFIG["steps"] if steps is None else steps)
    trainer = Trainer(backend.model, cfg, items, guard=guard)
    if any(p.requires_grad for p in backend.model.backbone.parameters()):
        raise ValueError("backbone unexpectedly trainable")
    if any(not n.startswith(("head.", "typed.")) for n in trainer.trainable_state()):
        raise ValueError("only existing head/typed-output parameters may train")
    return trainer


def strict_restore(trainer, checkpoint):
    """Strengthen Trainer.load's partial-copy behavior for this fixed experiment."""
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    current = trainer.trainable_state()
    if set(saved["trainable"]) != set(current):
        raise ValueError("checkpoint trainable key mismatch")
    for key, value in current.items():
        other = saved["trainable"][key]
        if other.shape != value.shape or other.dtype != value.dtype or not torch.isfinite(other).all():
            raise ValueError("checkpoint tensor mismatch")
    trainer.load(checkpoint, strict_config=True, strict_lineage=True)


def bootstrap(before, after):
    import random
    if not before or len(before) != len(after):
        raise ValueError("paired worlds required")
    delta = [int(b) - int(a) for a, b in zip(before, after)]
    rng = random.Random(CONFIG["bootstrap_seed"])
    values = sorted(sum(rng.choice(delta) for _ in delta) / len(delta)
                    for _ in range(CONFIG["bootstrap_samples"]))
    return {"n_worlds": len(delta), "estimate": sum(delta) / len(delta),
            "ci95": [values[int(.025 * len(values))], values[int(.975 * len(values))]],
            "unit": "independent generated world/task", "method": "paired percentile bootstrap"}


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


def evaluate(backend, decider, examples, raw, phase):
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


def child(out, connection, launcher_pid, secret):
    """No public CLI entry point; authenticate inherited launcher IPC first."""
    authorize_worker(connection, launcher_pid, secret)
    with (out / "stdout.txt").open("w", encoding="utf-8") as stdout, (out / "stderr.txt").open("w", encoding="utf-8") as stderr:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                _run(out)
            except Exception as exc:
                from language_growth.l0_qwen_cpu_probe import _write_event
                _write_event(out / "raw.jsonl", "worker_failure", exception_type=type(exc).__name__, message=str(exc))
                raise


def state_digest(module):
    """All parameters AND registered buffers; includes the independent LM head."""
    h = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _run(out):
    import platform
    import psutil
    assert_environment()
    if psutil.virtual_memory().available < CONFIG["min_start_bytes"]:
        raise RuntimeError("child requires 8 GiB available before Torch/load")
    registration = json.loads((out / "registration.json").read_text(encoding="utf-8"))
    verify_frozen(registration)
    data = json.loads((out / "corpus.json").read_text(encoding="utf-8"))
    if registration["config"] != CONFIG or registration["corpus_sha256"] != digest(data):
        raise ValueError("frozen registration/corpus mismatch")
    for name, expected in registration["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"source drift: {name}")
    for name, version in registration["software"].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"software drift: {name}")
    from language_growth.rwkv7_cpu_parent import inspect_snapshot, load_cpu_parent, sha
    evidence = inspect_snapshot()
    write(out / "parent-evidence.json", evidence)
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available() or psutil.virtual_memory().available < CONFIG["min_running_bytes"]:
        raise RuntimeError("CPU or running RAM guard failed before load")
    plugin, parent, loaded_evidence = load_cpu_parent()
    if loaded_evidence != evidence:
        raise ValueError("snapshot evidence drifted during load")
    from language_growth.rwkv_decision_head import (rwkv_cpu_parent_encoder, rwkv_cpu_parent_identity,
        rwkv_cpu_parent_decider, rwkv_decision_backend)
    from heads.decide_backend import DecisionModelBackend
    identity = rwkv_cpu_parent_identity(plugin, evidence)
    backend = rwkv_decision_backend(parent, rwkv_cpu_parent_encoder(plugin), source_identity=identity,
        seed=CONFIG["seed"], head=CONFIG["head"], head_kwargs=CONFIG["head_kwargs"])
    trainer = trainer_for(backend, data["development_train"])
    from language_growth.rwkv_head_ce96_evaluation import training_config_parity
    training_config_parity(trainer.cfg.to_json())
    write(out / "train-config.json", trainer.cfg.to_json())
    raw = out / "raw.jsonl"
    frozen_sha256 = state_digest(parent)
    before = evaluate(backend, rwkv_cpu_parent_decider(backend, plugin),
                      data["development_validation"], raw, "before")
    from language_growth.rwkv_head_ce96_evaluation import (references, initial_parity,
        teacher_forced, comparison_report, exposure_contract)
    prior = references()
    if prior["sha256"] != registration["reference_sha256"]:
        raise ValueError("prior measurement artifacts drifted")
    initial_parity(before, prior["learning"]["before"])
    exposure_contract(data, CONFIG)
    teacher_before = teacher_forced(data, backend, out, "teacher-before")
    # Independent unchanged LM-head baseline, same direct requests before/after.
    from decide.adapter import Decider
    from decide.backends.hf_readout import HFReadoutBackend
    from language_growth.paired_comparison import direct_comparison_request
    direct = Decider(HFReadoutBackend(plugin, "jrt-rwkv-g1-v1", max_batch_rows=2),
                     backend_max_input_tokens=plugin.max_context)
    requests = [direct_comparison_request(e, model=plugin.model_id,
        parent_revision=direct.revision()["revision_id"]) for e in data["development_validation"]
        if e["training_only"]["kind"] == "compare"]
    parent.eval()
    def baseline(phase):
        answers = []
        for request in requests:
            response = Capture(direct, raw, phase).decide(request)
            if response.get("ssdm", {}).get("status") != "ok" or set(response.get("answers", {})) != set(request["questions"]):
                raise ValueError("direct baseline failed; no parity claim possible")
            answers.append({"answers": response["answers"],
                            "served_probabilities": response["ssdm"]["served_probabilities"]})
        return answers
    direct_before = baseline("direct_before")
    checkpoint0 = trainer.save(out / "checkpoint-before")
    for _ in range(CONFIG["steps"]):
        loss = trainer.train_step()
        from language_growth.l0_qwen_cpu_probe import _write_event
        _write_event(raw, "train_step", step=trainer.step, loss=loss)
    checkpoint = trainer.save(out / "checkpoint-after")
    strict_restore(trainer, checkpoint)
    history = deepcopy(trainer.history)
    del trainer  # post-evaluation has no optimizer or checkpoint selection
    backend.model.eval().requires_grad_(False)
    if state_digest(parent) != frozen_sha256:
        raise ValueError("frozen full parent state changed (parameters, buffers or LM head)")
    # New backend identity after head mutation; cached revisions are immutable.
    after_backend = DecisionModelBackend(backend.model, backend.encode, template=backend.template,
        layout=backend.layout, checkpoint={"sha256": sha(checkpoint)})
    after = evaluate(after_backend, rwkv_cpu_parent_decider(after_backend, plugin),
                     data["development_validation"], raw, "after")
    teacher_after = teacher_forced(data, after_backend, out, "teacher-after")
    comparisons = comparison_report(before, after, teacher_before, teacher_after, prior)
    write(out / "comparisons.json", comparisons)
    parent.eval()
    direct_after = baseline("direct_after")
    if direct_before != direct_after:
        raise ValueError("unchanged direct baseline drifted")
    if state_digest(parent) != frozen_sha256:
        raise ValueError("full parent state changed during post-evaluation")
    write(out / "receipt.json", {"scope": SCOPE, "identity": identity,
        "parent_evidence_sha256": sha(out / "parent-evidence.json"), "registration_sha256": sha(out / "registration.json"),
        "checkpoint_before_sha256": sha(checkpoint0), "checkpoint_after_sha256": sha(checkpoint),
        "full_parent_state_sha256": frozen_sha256,
        "strict_reload": True, "frozen_backbone_unchanged": True, "direct_baseline_parity": True,
        "direct_before": direct_before, "direct_after": direct_after,
        "paired_usefulness": bootstrap([r["task_correct"] for r in before], [r["task_correct"] for r in after]),
        "before": before, "after": after, "history": history, "cost_usd": 0,
        "training_exposure": registration["exposure"],
        "teacher_forced_before": teacher_before, "teacher_forced_after": teacher_after,
        "comparisons": comparisons, "initial_checkpoint_parity": True,
        "hardware": {"device": "existing local CPU", "platform": platform.platform(),
                     "processor": platform.processor(), "threads": torch.get_num_threads()},
        "software": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "safetensors", "psutil")}})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="real-load mode; requires explicit run approval")
    mode.add_argument("--dry-run", action="store_true", help="default: freeze corpus/registration only")
    args = parser.parse_args()
    out = args.out.resolve()
    owns_output = not out.exists()
    try:
        if args.execute:
            return launch(out)
        preparation(out)
        print(json.dumps({"status": "prepared_not_run", "out": str(out), "scope": SCOPE}))
        return 0
    except Exception as exc:
        if out.is_dir() and owns_output:
            from language_growth.l0_qwen_cpu_probe import _write_event
            _write_event(out / "raw.jsonl", "failure", kind=type(exc).__name__, message=str(exc))
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
