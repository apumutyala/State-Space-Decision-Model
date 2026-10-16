"""Load a merged student through the shared runtime on train-source requests only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from keel.distill.sources import read_rows
from keel.distill.teacher import argmax
from keel.distill.v1_pipeline import RATE
from keel.paths import private_root


def main():
    import modal
    from keel.protocol import DecisionRequest
    from keel.runtime.vllm_judge import APP, JudgeServer, judge

    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", default=str(private_root() / "distill_v1"))
    a = p.parse_args()
    out = Path(a.out)
    rows = read_rows(out / "train.jsonl.gz")
    requests = [DecisionRequest.from_json(r["request"]) for r in rows[:2] + [r for r in rows if r["kind"] == "capability"][:2]]
    start = time.time()
    cost = {"date": datetime.now(timezone.utc).date().isoformat(), "function": "runtime.vllm_judge train-source smoke",
            "status": "failed"}
    try:
        with modal.enable_output(), APP.run():
            server = JudgeServer.with_options(timeout=900, scaledown_window=30)(model=a.model)
            results = judge(requests, a.model, mode="trust", think=False, server=server)
        if len(results) != 4:
            raise ValueError("checkpoint smoke batch incomplete")
        for r in results:
            argmax(r["probs"])
        (out / "checkpoint-smoke-outputs.json").write_text(json.dumps(results))
        report = {"checkpoint": a.model, "train_source_rows": 4, "valid_distributions": True,
                  "mode": "trust", "think": False, "runtime": results[0]["meta"]}
        (out / "checkpoint-smoke.json").write_text(json.dumps(report, indent=2))
        cost["status"] = "complete"
        print(json.dumps(report, indent=2))
    finally:
        cost["wall_s"] = time.time() - start
        cost["gpu_minutes_upper_estimate"] = cost["wall_s"] / 60
        cost["usd_upper_estimate"] = cost["wall_s"] * RATE
        with (out / "spend.jsonl").open("a") as f:
            f.write(json.dumps(cost) + "\n")


if __name__ == "__main__":
    main()
