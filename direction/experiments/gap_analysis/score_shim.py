"""Score a judge on public-231 in one pass with the H2O shim prompt, byte-identical to reasoning_headroom/rh_modal.py.

EVAL-ONLY: nothing is fitted or selected. Scoring runs through ``keel.runtime.vllm_judge`` on Modal. Predictions use the
shim post-processing (``score_rh.answer``, T 0.8, yes/no floor 0.801) and go to
``$SSDM_PRIVATE/gap_analysis/<tag>.nothink.shim.preds.jsonl``.

    python -X utf8 direction/experiments/gap_analysis/score_shim.py \
        --model /results/students/distill-v0-20261015/merged --tag student
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "direction" / "experiments" / "reasoning_headroom"))

from jevkit import items as IT  # noqa: E402
from keel.paths import private_root  # noqa: E402
from keel.runtime.budget import approved_run  # noqa: E402
from keel.runtime.vllm_judge import APP, score  # noqa: E402
from rh_modal import chat, user_turn  # noqa: E402
from score_rh import answer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="MODELS key or /results/... checkpoint")
    ap.add_argument("--tag", action="append", required=True, help="output tag, one per --model")
    ap.add_argument("--max-usd", type=float, required=True, help="cap for this run, within the owner's approved budget")
    a = ap.parse_args()
    if len(a.model) != len(a.tag):
        ap.error("give one --tag per --model")
    pub = IT.load_public()
    rows = [(it, *user_turn(it.to_dict())) for it in pub]
    jobs = [{"id": it.id, "prompt": chat(u, False), "labels": labels} for it, u, _, labels in rows]
    out_dir = private_root() / "gap_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    with approved_run(APP, label="gap_analysis.score_shim", cap_usd=a.max_usd, gpus=len(a.model)):
        calls = [score.spawn(m, jobs) for m in a.model]
        results = [c.get() for c in calls]
    for tag, res in zip(a.tag, results):
        missing = 0
        with open(out_dir / f"{tag}.nothink.shim.preds.jsonl", "w", encoding="utf-8") as w:
            for (it, _, names, _), r in zip(rows, res["results"]):
                lp = r["samples"][0]["logprobs"]
                present = [x for x in lp if x is not None]
                missing += len(lp) - len(present)
                lp = [x if x is not None else min(present) - 1.0 for x in lp]
                w.write(json.dumps({"id": it.id, "status": "ok", "answer": answer(it.type, names, lp, 0.8, 0.801)}) + "\n")
        meta = {k: v for k, v in res.items() if k != "results"}
        (out_dir / f"{tag}.meta.json").write_text(json.dumps({**meta, "missing_labels": missing}, indent=1))
        print(tag, meta["model"], "missing labels:", missing, "load_s %.0f wall_s %.1f" % (meta["load_s"], meta["wall_s"]))


if __name__ == "__main__":
    main()
