"""Select three declared LoRA recipes on synthetic dev, then train one full union."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import modal

from keel.distill.assembly_v1 import CANDIDATES, assert_disjoint, mix, select_candidate, validate_source
from keel.distill.modal_train import training_image
from keel.distill.sources import guard_units, read_rows
from keel.distill.v1_pipeline import RATE, spending
from keel.runtime.modal_app import HF_CACHE, hf_cache, results_vol

APP = modal.App("keel-distill-v1-train")


@APP.function(image=training_image, gpu="H100", timeout=7200, memory=65536, cpu=8,
              volumes={HF_CACHE: hf_cache, "/results": results_vol})
def train_recipe(rows: list[dict], dev: list[dict], config: dict, run: str, export: bool):
    from keel.distill.train_student import train
    if not run or "/" in run or run in {".", ".."}:
        raise ValueError("run must be one path component")
    result = train(rows, f"/results/students/{run}", rank=config["rank"], epochs=config["epochs"],
                   dev_rows=dev, export=export, max_length=4096, batch_size=4, accumulation=4)
    result["name"] = config["name"]
    if export:
        path = Path(f"/results/students/{run}/train.json")
        path.write_text(json.dumps(result, indent=2))
        results_vol.commit()
    return result


def main():
    from keel.paths import private_root
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["select", "final"])
    p.add_argument("--out", default=str(private_root() / "distill_v1"))
    p.add_argument("--run", default="distill-v1-20261015")
    p.add_argument("--pilot-rows", type=int, default=1536)
    p.add_argument("--cap", type=float, default=35)
    a = p.parse_args()
    out = Path(a.out)
    rows, dev = read_rows(out / "train.jsonl.gz"), read_rows(out / "dev.jsonl.gz")
    for r in rows + dev:
        validate_source(r)
    assert_disjoint(rows, dev)
    results = []
    configs = CANDIDATES if a.phase == "select" else [json.loads((out / "selection.json").read_text())["selected"]]
    with modal.enable_output(), APP.run():
        for config in configs:
            dest = out / (f"selection-{config['name']}.json" if a.phase == "select" else "train-result.json")
            if dest.exists():
                results.append(json.loads(dest.read_text()))
                continue
            train_rows = mix(rows, config["capability_fraction"], limit=a.pilot_rows if a.phase == "select" else None)
            guard = guard_units(train_rows, out / f"{a.phase}-{config['name']}-guard-input.jsonl.gz")
            (out / f"{a.phase}-{config['name']}-guards.json").write_text(json.dumps(guard, indent=2))
            if spending(out) > a.cap - 4:
                raise RuntimeError("insufficient reserved training/evaluation budget")
            start = time.time()
            cost = {"date": "2026-10-15", "function": "train_recipe", "phase": a.phase,
                    "config": config, "status": "failed"}
            try:
                timeout = min(7200, int((a.cap - spending(out) - 2) / RATE))
                if timeout < 60:
                    raise RuntimeError("insufficient budget for a bounded training call")
                result = train_recipe.with_options(timeout=timeout).remote(
                    train_rows, dev, config, a.run, a.phase == "final")
                dest.write_text(json.dumps(result, indent=2))
                results.append(result)
                cost["status"] = "complete"
            finally:
                cost["wall_s"] = time.time() - start
                cost["gpu_minutes_upper_estimate"] = cost["wall_s"] / 60
                cost["usd_upper_estimate"] = cost["wall_s"] * RATE
                with (out / "spend.jsonl").open("a") as f:
                    f.write(json.dumps(cost) + "\n")
    if a.phase == "select":
        best = select_candidate(results)
        selected = next(c for c in CANDIDATES if c["name"] == best["name"])
        report = {"selected": selected, "pilot_rows": a.pilot_rows, "selection": "synthetic balanced-source dev loss",
                  "public_used": False, "candidates": results}
        (out / "selection.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
