"""Exploratory paired stock-Qwen CPU diagnostic; initial evidence only.

Run with .venv-gpu/Scripts/python.exe -I this-file --out a-new-directory.
The owned child is killed at 300 s or below 3 GiB available RAM. CPU/offline
environment is established before importing Torch. No GPU or paid resource.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from language_growth.l0_qwen_cpu_probe import (
    MIN_RUNNING_AVAILABLE, MIN_START_AVAILABLE, MODEL_ID, SNAPSHOT,
    TIMEOUT_SECONDS, _LoggedDecider, _check_dependencies, _offline_env, _sha_task, _write_event,
)

SEED = 271828
INDICES = [2, 5]


def child(out):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or os.environ.get("HF_HUB_OFFLINE") != "1":
        raise RuntimeError("offline CPU guard missing")
    import psutil
    if psutil.virtual_memory().available < MIN_START_AVAILABLE:
        raise RuntimeError("less than 8 GiB RAM before Torch")
    versions = _check_dependencies()
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        raise RuntimeError("CUDA must be unavailable")
    from decide.adapter import Decider
    from decide.backends.hf_readout import HFReadoutBackend, load_qwen35
    from decide.calibration import OptionCountTemperature
    from language_growth.development_data import generate_development_tasks
    from language_growth.free_running_dev import _initial_task, score_development_task
    from language_growth.planner import plan_checked_answer

    raw = out / "raw.jsonl"
    tasks = generate_development_tasks(SEED, 6, choice_format="production",
                                       split="development_validation")
    tasks = [tasks[index] for index in INDICES]
    _write_event(raw, "config", seed=SEED, indices=INDICES, software=versions,
                 split="development_validation", task_sha256=[_sha_task(task) for task in tasks],
                 hardware="Windows CPU", billed_rate=0, cost_usd=0,
                 snapshot=str(SNAPSHOT), dtype="float32", batch_rows=2,
                 source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                 source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (ROOT / "language_growth/planner.py", Path(__file__))})
    plugin = load_qwen35(SNAPSHOT, device="cpu", dtype="float32", model_id=MODEL_ID)
    calibration = OptionCountTemperature()
    parent = Decider(HFReadoutBackend(plugin, "jrt-chatml-v1", max_batch_rows=2),
                     calibration=calibration, backend_max_input_tokens=plugin.max_context)
    revision = parent.revision()["revision_id"]
    _write_event(raw, "loaded", parent_revision=revision, model_revision=parent.revision())
    config = {"calibration": calibration.identity(),
              "policy_masking": "none (served covers every supplied option)"}
    logged = _LoggedDecider(parent, raw)
    results = []
    # Counterbalance mode order between tasks. Every request is stateless.
    for index, task in enumerate(tasks):
        for mode in (("reference", "grounded") if index == 0 else ("grounded", "reference")):
            _write_event(raw, "arm_start", task_id=task["task_id"], candidate_presentation=mode)
            training = task["training_only"]
            started = time.monotonic()
            result = plan_checked_answer(_initial_task(task), training["host"], training["catalog"],
                                         logged, model=MODEL_ID, parent_revision=revision,
                                         expected_served_config=config, candidate_presentation=mode)
            results.append((task, mode, result, time.monotonic() - started))
            _write_event(raw, "arm_result", task_id=task["task_id"], candidate_presentation=mode,
                         result=asdict(result), elapsed_seconds=results[-1][-1])
    # Labels and reference programs are consulted only after every provider call.
    observations = []
    for task, mode, result, elapsed in results:
        targets = {item["slot"]: item["option_id"] for item in task["training_only"]["targets"]}
        selected = {step["slot"]: step["selected_id"] for step in result.trace if "slot" in step}
        slots = {slot: {"expected_id": target, "selected_id": selected.get(slot),
                        "correct": selected.get(slot) == target} for slot, target in targets.items()}
        observations.append({"task_id": task["task_id"], "candidate_presentation": mode,
                             "task_correct": score_development_task(task, result), "slots": slots,
                             "status": result.status, "reason": result.reason,
                             "provider_calls": result.provider_calls, "forward_calls": result.forward_calls,
                             "elapsed_seconds": elapsed})
    report = {"scope": "exploratory; two development failures used to design the intervention; no qualification claim",
              "seed": SEED, "indices": INDICES, "observations": observations,
              "repeat_count": 1, "median_and_spread": "not estimated; one paired deterministic diagnostic"}
    for mode in ("reference", "grounded"):
        rows = [row for row in observations if row["candidate_presentation"] == mode]
        report[mode] = {"task_correct": sum(row["task_correct"] for row in rows), "tasks": len(rows),
                        "kernel_accepted": sum(row["status"] == "accepted" for row in rows),
                        "slots_correct": sum(slot["correct"] for row in rows for slot in row["slots"].values()),
                        "slots_total": sum(len(row["slots"]) for row in rows)}
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_event(raw, "summary", report=report)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.child:
        return child(out)
    import psutil
    available = psutil.virtual_memory().available
    if available < MIN_START_AVAILABLE:
        raise RuntimeError(f"available_bytes={available} < min_start_bytes={MIN_START_AVAILABLE}; refusing CPU model load")
    out.mkdir(parents=True, exist_ok=False)
    raw = out / "raw.jsonl"
    _write_event(raw, "launch", command=[sys.executable, "-I", str(Path(__file__).resolve()), "--out", str(out)],
                 timeout_seconds=TIMEOUT_SECONDS, min_start_available_bytes=MIN_START_AVAILABLE,
                 min_running_available_bytes=MIN_RUNNING_AVAILABLE)
    proc = subprocess.Popen([sys.executable, "-I", str(Path(__file__).resolve()), "--out", str(out), "--child"],
                            env=_offline_env(), cwd=ROOT)
    started = time.monotonic()
    try:
        while proc.poll() is None:
            available = psutil.virtual_memory().available
            if time.monotonic() - started > TIMEOUT_SECONDS or available < MIN_RUNNING_AVAILABLE:
                _write_event(raw, "guard_stop", available_bytes=available,
                             elapsed_seconds=time.monotonic() - started)
                return 124
            time.sleep(0.5)
        _write_event(raw, "exit", exit_code=proc.returncode, elapsed_seconds=time.monotonic() - started)
        return proc.returncode
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
