"""Own CUDA training image; merged HF checkpoints are written to keel-results."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import modal

from keel.runtime.modal_app import HF_CACHE, hf_cache, results_vol

APP = modal.App("keel-distill-train")
training_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential", "git")
    .pip_install("torch==2.14.1", "transformers==5.14.1", "peft==0.21.2", "accelerate==1.12.0",
                 "safetensors", "huggingface_hub", "ninja", "packaging", "wheel",
                 "flash-linear-attention==0.5.2")
    .env({"MAX_JOBS": "4", "TOKENIZERS_PARALLELISM": "false", "CC": "gcc", "CXX": "g++"})
    .run_commands("pip install --no-build-isolation causal-conv1d==1.7.0")
    .add_local_python_source("keel", "decide")
)


@APP.function(image=training_image, gpu="H100", timeout=14400,
              volumes={HF_CACHE: hf_cache, "/results": results_vol}, memory=65536, cpu=8)
def train_remote(rows: list[dict], run: str, smoke: bool = False, batch_size: int = 4) -> dict:
    from keel.distill.train_student import train

    if not run or "/" in run or run in (".", ".."):
        raise ValueError("run must be one path component")
    output = f"/results/students/{run}"
    result = train(rows, output, smoke=smoke, batch_size=batch_size)
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / ("smoke.json" if smoke else "train.json")).write_text(json.dumps(result, indent=2) + "\n")
    results_vol.commit()
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--labels")
    p.add_argument("--smoke-sources")
    p.add_argument("--run", default="distill-v0-20261015")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    a = p.parse_args()
    from keel.distill.sources import guard_units, read_rows
    if bool(a.labels) == bool(a.smoke_sources):
        p.error("supply exactly one of --labels or --smoke-sources")
    if a.smoke_sources:
        from keel.distill.triplets import variants
        from keel.protocol import DecisionRequest, render
        rows = []
        for source in read_rows(a.smoke_sources)[:8]:
            u = variants(source)[0]
            options = render(DecisionRequest.from_json(u["request"])).option_ids
            u["teacher_probs"] = {k: float(k == u["gold"]) for k in options}
            rows.append(u)
    else:
        rows = read_rows(a.labels)
    start = time.time()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    guard_report = guard_units(rows, out.with_suffix(".guard-input.jsonl.gz"))
    out.with_suffix(".guards.json").write_text(json.dumps(guard_report, indent=2) + "\n")
    ledger = {"date": "2026-10-15", "function": "train_remote", "status": "running", "smoke": bool(a.smoke_sources)}
    try:
        with modal.enable_output(), APP.run():
            result = train_remote.remote(rows, a.run, bool(a.smoke_sources), a.batch_size)
        ledger.update(result)
        ledger["status"] = "complete"
        print(json.dumps(ledger, indent=2))
    finally:
        if ledger["status"] == "running":
            ledger["status"] = "failed"
        ledger["local_wall_s"] = time.time() - start
        # Image-build CPU time is included in this conservative upper estimate.
        ledger["gpu_minutes_upper_estimate"] = ledger["local_wall_s"] / 60
        ledger["usd_upper_estimate"] = ledger["local_wall_s"] * 3.95 / 3600
        out.write_text(json.dumps(ledger, indent=2) + "\n")
        with out.with_suffix(".attempts.jsonl").open("a") as log:
            log.write(json.dumps(ledger) + "\n")


if __name__ == "__main__":
    main()
