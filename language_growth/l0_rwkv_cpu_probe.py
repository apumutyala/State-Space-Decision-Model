"""One guarded cached attention-free RWKV CPU smoke and exploratory paired task."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from language_growth.l0_qwen_cpu_probe import _offline_env, _write_event, _LoggedDecider, _sha_task
from language_growth.rwkv7_cpu_parent import MODEL_ID, SNAPSHOT, inspect_snapshot, sha

MIN_START_AVAILABLE = 6 * 1024**3
MIN_RUNNING_AVAILABLE = 3 * 1024**3
TIMEOUT_SECONDS = 300


def _task():
    from language_growth.development_data import generate_development_tasks
    task = generate_development_tasks(271828, 3, choice_format="production",
                                      split="development_train", allow_one_repair=False)[-1]
    if task["training_only"]["kind"] != "compare" or not task["development_only"]:
        raise RuntimeError("pinned synthetic development comparison changed")
    return task


def _child(raw: Path) -> int:
    import psutil
    if any(os.environ.get(name) != value for name, value in
           (("CUDA_VISIBLE_DEVICES", "-1"), ("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"))):
        raise RuntimeError("offline CPU environment is not pinned")
    available = psutil.virtual_memory().available
    if available < MIN_START_AVAILABLE:
        raise RuntimeError("RWKV child requires >=6 GiB available before Torch")
    evidence = inspect_snapshot()
    task = _task()
    evidence["run_source_sha256"] = {name: sha(ROOT / name) for name in
        ("language_growth/l0_rwkv_cpu_probe.py", "language_growth/paired_comparison.py",
         "language_growth/planner.py", "language_growth/development_data.py",
         "decide/backends/hf_readout.py", "decide/prompt.py", "decide/adapter.py")}
    (raw.parent / "preflight.json").write_text(json.dumps(evidence, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    (raw.parent / "task.json").write_text(json.dumps(task, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    _write_event(raw, "task_pinned", task_id=task["task_id"], task_sha256=_sha_task(task),
                 split="development_train", seed=271828, task=task,
                 snapshot=str(SNAPSHOT), scope="exploratory_attention_free_CPU_parent",
                 software={name: importlib.metadata.version(name) for name in
                           ("torch", "transformers", "safetensors", "psutil")})
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CUDA visible despite guard")
    if psutil.virtual_memory().available < MIN_RUNNING_AVAILABLE:
        raise RuntimeError("RWKV child requires >=3 GiB available before weight load")
    from language_growth.rwkv7_cpu_parent import load_cpu_parent
    _write_event(raw, "load_start")
    plugin, model, _ = load_cpu_parent()
    # Actual checkpoint smoke: portable chunked prefill versus token-serial eager.
    ids = torch.tensor([plugin.encode("The sky is blue. 2 + 2 =", True)], device="cpu")
    with torch.inference_mode():
        chunked = model(input_ids=ids, use_cache=False).logits
        model.config.wkv_implementation = "eager"
        eager = model(input_ids=ids, use_cache=False).logits
        model.config.wkv_implementation = "chunked"
    delta = float((chunked-eager).abs().max())
    if not torch.isfinite(chunked).all() or not torch.isfinite(eager).all() or delta > 0.005:
        raise RuntimeError(f"checkpoint CPU smoke failed: max_abs_delta={delta}")
    torch.save({"input_ids": ids, "chunked_logits": chunked, "eager_logits": eager}, raw.parent / "smoke.pt")
    _write_event(raw, "cpu_smoke_pass", tokens=ids.tolist(), max_abs_logits_delta=delta,
                 chunked_argmax=chunked.argmax(-1).tolist(), eager_argmax=eager.argmax(-1).tolist(),
                 tensor_sha256=sha(raw.parent / "smoke.pt"))
    from decide.adapter import Decider
    from decide.backends.hf_readout import HFReadoutBackend
    from decide.calibration import OptionCountTemperature
    from language_growth.paired_comparison import run_paired_comparisons
    calibration = OptionCountTemperature()
    parent = Decider(HFReadoutBackend(plugin, "jrt-rwkv-g1-v1", max_batch_rows=2),
                     calibration=calibration, backend_max_input_tokens=2048)
    revision = parent.revision()["revision_id"]
    _write_event(raw, "load_complete", parent_revision=revision, revision=parent.revision())
    report = run_paired_comparisons([task], _LoggedDecider(parent, raw), model=MODEL_ID,
        parent_revision=revision, expected_served_config={"calibration": calibration.identity(),
        "policy_masking": "none (served covers every supplied option)"})
    report["source"] = "learned_cached_attention_free_RWKV7_checkpoint_under_reviewed_CPU_conversion"
    report["qualification"] = "CPU smoke dependency only; one synthetic development training task; no strong parent claim"
    if report["total"] != 1 or report["pairs"][0]["same_initial_state"] is not True:
        raise RuntimeError("paired initial state identity failed")
    _write_event(raw, "paired_report", report=report)
    return 0


def _parent(raw: Path) -> int:
    import psutil
    available = psutil.virtual_memory().available
    raw.parent.mkdir(parents=True, exist_ok=False)
    if available < MIN_START_AVAILABLE:
        _write_event(raw, "launcher_refused", available_bytes=available,
                     min_start_available_bytes=MIN_START_AVAILABLE)
        raise RuntimeError(f"RWKV CPU probe requires >=6 GiB available; found {available / 1024**3:.2f}")
    command = [sys.executable, "-I", "-X", "utf8", str(Path(__file__).resolve()), "--child", "--raw", str(raw)]
    _write_event(raw, "launcher_start", command=command, timeout_seconds=TIMEOUT_SECONDS,
                 min_start_available_bytes=MIN_START_AVAILABLE, min_running_available_bytes=MIN_RUNNING_AVAILABLE,
                 available_bytes=available, threads=2, CUDA_VISIBLE_DEVICES="-1",
                 resource_justification="191034624 FP32 parameters=0.712 GiB; <=2x2048 hidden activations; no full-prompt vocab logits in scoring")
    child = None
    stopped = False
    with (raw.parent / "stdout.txt").open("w", encoding="utf-8") as stdout, (raw.parent / "stderr.txt").open("w", encoding="utf-8") as stderr:
        try:
            child = subprocess.Popen(command, env=_offline_env(), stdout=stdout, stderr=stderr)
            started = time.monotonic()
            minimum = available
            while True:
                available = psutil.virtual_memory().available
                minimum = min(minimum, available)
                elapsed = time.monotonic() - started
                result = child.poll()
                if result is not None:
                    _write_event(raw, "launcher_exit", exit_code=result, elapsed_seconds=elapsed,
                                 minimum_available_bytes=minimum)
                    return result
                if elapsed >= TIMEOUT_SECONDS or available < MIN_RUNNING_AVAILABLE:
                    reason = "timeout" if elapsed >= TIMEOUT_SECONDS else "memory_guard"
                    child.kill()
                    child.wait()
                    stopped = True
                    _write_event(raw, "launcher_stopped", reason=reason, elapsed_seconds=elapsed,
                                 minimum_available_bytes=minimum)
                    return 124 if reason == "timeout" else 125
                time.sleep(0.5)
        finally:
            if child is not None and not stopped and child.poll() is None:
                child.kill()
                child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    raw = args.raw.resolve()
    if raw.name != "raw.jsonl":
        parser.error("--raw must name raw.jsonl in a fresh directory")
    try:
        return _child(raw) if args.child else _parent(raw)
    except Exception as exc:
        if raw.parent.is_dir():
            _write_event(raw, "failure", exception_type=type(exc).__name__, message=str(exc))
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
