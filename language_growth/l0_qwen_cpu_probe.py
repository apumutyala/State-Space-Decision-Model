"""One offline, bounded stock-Qwen CPU diagnostic on development validation.

Preparation only until PR #37 is merged and a resource owner clears the run.
The checkpoint is an inherited hybrid Qwen parent, not attention-free evidence.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SNAPSHOT = Path(
    "C:/Users/apumu/research/CHLabs/ssdm-data/hf/"
    "models--Qwen--Qwen3.5-0.8B/snapshots/"
    "2fc06364715b967f1860aea9cf38778875588b17"
)
SEED = 271828
COUNT = 3
TIMEOUT_SECONDS = 300
MIN_START_AVAILABLE = 8 * 1024**3
MIN_RUNNING_AVAILABLE = 3 * 1024**3
MODEL_ID = "Qwen/Qwen3.5-0.8B"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_event(path: Path, event: str, **fields: Any) -> None:
    row = {"event": event, "at_utc": _now(), **fields}
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _task() -> dict[str, Any]:
    from language_growth.development_data import generate_development_tasks

    tasks = generate_development_tasks(SEED, COUNT, choice_format="production",
                                       split="development_validation")
    task = tasks[-1]
    if (task["task_id"] != "synthetic-development_validation-271828-2"
            or task["training_only"]["kind"] != "compare"
            or task["development_split"] != "development_validation"):
        raise RuntimeError("pinned development comparison changed")
    return task


def _sha_task(task: dict[str, Any]) -> str:
    blob = json.dumps(task, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(blob).hexdigest()


def _check_dependencies() -> dict[str, str]:
    if not (ROOT / "language_growth" / "paired_comparison.py").is_file():
        raise RuntimeError("PR #37 dependency missing: language_growth/paired_comparison.py")
    for name in ("config.json", "tokenizer.json", "model.safetensors-00001-of-00001.safetensors"):
        if not (SNAPSHOT / name).is_file():
            raise RuntimeError(f"pinned local snapshot missing {name}")
    versions = {name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "accelerate", "psutil")}
    major_minor = tuple(int(part) for part in versions["transformers"].split(".")[:2])
    if major_minor < (5, 17):
        raise RuntimeError("transformers >= 5.17 is required for the Qwen3.5 readout")
    return versions


def _offline_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "-1", "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
    return env


class _LoggedDecider:
    def __init__(self, parent: Any, raw: Path):
        self.parent, self.raw = parent, raw
        self.calls = 0

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        call = self.calls
        _write_event(self.raw, "provider_request", call=call, request=request)
        try:
            response = self.parent.decide(request)
        except Exception as exc:
            _write_event(self.raw, "provider_exception", call=call,
                         exception_type=type(exc).__name__, message=str(exc)[:500])
            raise
        _write_event(self.raw, "provider_response", call=call, response=response)
        return response


def _child(raw: Path) -> int:
    # The launcher sets these before interpreter startup. Recheck before any
    # import that could initialize a model runtime.
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"
            or os.environ.get("HF_HUB_OFFLINE") != "1"
            or os.environ.get("TRANSFORMERS_OFFLINE") != "1"):
        raise RuntimeError("offline CPU environment is not pinned")
    import psutil
    available = psutil.virtual_memory().available
    if available < MIN_START_AVAILABLE:
        raise RuntimeError(f"CPU probe child requires >=8 GiB available RAM before Torch; found {available / 1024**3:.2f} GiB")
    versions = _check_dependencies()
    task = _task()
    _write_event(raw, "task_pinned", task_id=task["task_id"], task_sha256=_sha_task(task),
                 seed=SEED, split="development_validation", choice_format="production",
                 software=versions, model_snapshot=str(SNAPSHOT), available_bytes=available,
                 scope="hybrid_diagnostic")

    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CUDA is visible despite CPU-only guard")
    from decide.adapter import Decider
    from decide.backends.hf_readout import HFReadoutBackend, load_qwen35
    from decide.calibration import OptionCountTemperature
    from language_growth.paired_comparison import run_paired_comparisons

    _write_event(raw, "load_start")
    plugin = load_qwen35(SNAPSHOT, device="cpu", dtype="float32", model_id=MODEL_ID)
    backend = HFReadoutBackend(plugin, "jrt-chatml-v1", max_batch_rows=2)
    calibration = OptionCountTemperature()
    parent = Decider(backend, calibration=calibration,
                     backend_max_input_tokens=plugin.max_context)
    revision = parent.revision()["revision_id"]
    _write_event(raw, "load_complete", parent_revision=revision,
                 backend=backend.name, calibration=calibration.identity())
    report = run_paired_comparisons(
        [task], _LoggedDecider(parent, raw), model=MODEL_ID,
        parent_revision=revision,
        expected_served_config={"calibration": calibration.identity(),
                                "policy_masking": "none (served covers every supplied option)"})
    if report["total"] != 1 or report["pairs"][0]["same_initial_state"] is not True:
        raise RuntimeError("paired denominator or initial state identity failed")
    _write_event(raw, "paired_report", report=report)
    return 0


def _parent(raw: Path) -> int:
    import psutil

    versions = _check_dependencies()
    task = _task()  # validate the one intended task before starting a child
    available = psutil.virtual_memory().available
    if available < MIN_START_AVAILABLE:
        raise RuntimeError(f"CPU probe requires >=8 GiB available RAM; found {available / 1024**3:.2f} GiB")
    raw.parent.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-I", str(ROOT / "l0_qwen_cpu_probe.py"),
               "--child", "--raw", str(raw)]
    _write_event(raw, "launcher_start", command=command, timeout_seconds=TIMEOUT_SECONDS,
                 min_start_available_bytes=MIN_START_AVAILABLE,
                 min_running_available_bytes=MIN_RUNNING_AVAILABLE,
                 available_bytes=available, task_id=task["task_id"],
                 task_sha256=_sha_task(task), software=versions)
    child = subprocess.Popen(command, env=_offline_env())
    stopped = False
    try:
        started = time.monotonic()
        while True:
            result = child.poll()
            if result is not None:
                _write_event(raw, "launcher_exit", exit_code=result,
                             elapsed_seconds=round(time.monotonic() - started, 3))
                return result
            elapsed = time.monotonic() - started
            available = psutil.virtual_memory().available
            if elapsed >= TIMEOUT_SECONDS or available < MIN_RUNNING_AVAILABLE:
                reason = "timeout" if elapsed >= TIMEOUT_SECONDS else "memory_guard"
                child.kill()  # only this launcher's child; never another process
                child.wait()
                stopped = True
                _write_event(raw, "launcher_stopped", reason=reason,
                             elapsed_seconds=round(elapsed, 3), available_bytes=available)
                return 124 if reason == "timeout" else 125
            time.sleep(0.5)
    finally:
        if not stopped:
            try:
                running = child.poll() is None
            except Exception:
                running = True  # a failed monitor must not leave this child running
            if running:
                try:
                    child.kill()
                except ProcessLookupError:
                    pass
                child.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path,
                        help="raw.jsonl in a fresh output directory")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    raw = args.raw.resolve()
    if raw.name != "raw.jsonl":
        parser.error("--raw must name raw.jsonl")
    try:
        return _child(raw) if args.child else _parent(raw)
    except Exception as exc:
        if args.child and raw.parent.is_dir():
            _write_event(raw, "child_failure", exception_type=type(exc).__name__, message=str(exc)[:500])
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
