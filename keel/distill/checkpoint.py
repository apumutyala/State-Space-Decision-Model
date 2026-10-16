"""Preserve the pinned multimodal processor assets in merged text-trained models."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

# These are the processor assets present in the pinned Qwen3.5-4B repository.
# Saving model weights and AutoTokenizer alone does not export either preprocessor.
PROCESSOR_FILES = ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja")


def copy_processor_assets(snapshot: str | Path, output: str | Path) -> dict[str, str]:
    source, destination = Path(snapshot), Path(output)
    for name in PROCESSOR_FILES:
        if not (source / name).is_file():
            raise FileNotFoundError(f"pinned processor asset missing: {source / name}")
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in PROCESSOR_FILES:
        shutil.copyfile(source / name, destination / name)
        hashes[name] = hashlib.sha256((destination / name).read_bytes()).hexdigest()
    return hashes


def save_processor_assets(output: str | Path) -> dict:
    from huggingface_hub import snapshot_download
    from keel.runtime.modal_app import MODELS

    model, revision = MODELS["qwen3.5-4b"]
    snapshot = snapshot_download(model, revision=revision, allow_patterns=list(PROCESSOR_FILES))
    return {"model": model, "revision": revision, "files_sha256": copy_processor_assets(snapshot, output)}


def main() -> None:
    """Bounded full-checkpoint load check on four private training-source rows."""
    import argparse
    import json
    import time

    import modal
    from keel.distill.sources import read_rows
    from keel.distill.teacher import argmax
    from keel.distill.triplets import request_of
    from keel.runtime.budget import approved_run
    from keel.runtime.vllm_judge import APP, JudgeServer, judge

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sources", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-usd", type=float, required=True,
                        help="cap for this run, within the owner's approved budget")
    args = parser.parse_args()
    rows = read_rows(args.train_sources)[:4]
    if len(rows) != 4:
        raise ValueError("checkpoint smoke requires four training-source rows")
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"date": "2026-10-15", "function": "runtime.judge: merged checkpoint smoke",
              "checkpoint": args.checkpoint, "train_source_rows": 4, "mode": "trust",
              "think": False, "status": "running", "remote_timeout_s": 900}
    start = time.monotonic()
    try:
        with modal.enable_output(), approved_run(APP, label="distill.checkpoint_smoke", cap_usd=args.max_usd):
            server = JudgeServer.with_options(timeout=900, scaledown_window=60)(model=args.checkpoint)
            results = judge([request_of(row) for row in rows], args.checkpoint,
                            mode="trust", think=False, server=server)
        if len(results) != 4:
            raise ValueError("checkpoint smoke returned an incomplete batch")
        for result in results:
            argmax(result["probs"])
        output.with_suffix(".outputs.json").write_text(json.dumps(results) + "\n")
        report.update(status="complete", valid_distributions=True, runtime=results[0]["meta"])
    finally:
        if report["status"] == "running":
            report["status"] = "failed"
        report["wall_s"] = time.monotonic() - start
        report["gpu_minutes_upper_estimate"] = report["wall_s"] / 60
        report["usd_upper_estimate"] = report["wall_s"] * 3.95 / 3600
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
