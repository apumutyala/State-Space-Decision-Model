"""Smoke check of the Keel judge runtime on a few public JevBench items (EVAL-ONLY; nothing is fitted).

    SSDM_DATA=~/ssdm/data python -X utf8 -m keel.runtime.smoke_judge --n 4 --model qwen3.5-4b

Prints, per item and arm (one-pass / think), the gold, the decision and its probability.
"""
from __future__ import annotations

import argparse

from jevkit import items as IT

from ..protocol import decision_of, from_jevbench_item
from .vllm_judge import APP, JudgeServer, judge


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--mode", default="trust", choices=["trust", "base"])
    a = ap.parse_args()
    pub = [it for it in IT.load_public() if it.type in ("choice", "noul")][: a.n]
    reqs = [from_jevbench_item({"id": it.id, "state": it.state, "question": it.question}) for it in pub]
    with APP.run():
        srv = JudgeServer(model=a.model)  # one model load serves both arms
        one = judge(reqs, a.model, mode=a.mode, think=False, server=srv)
        thk = judge(reqs, a.model, mode=a.mode, think=True, n_samples=2, server=srv)
    for it, o, t in zip(pub, one, thk):
        d1, d2 = decision_of(o["probs"], it.type), decision_of(t["probs"], it.type)
        print(f"{it.id:28s} gold={it.expected!s:12s} one-pass={d1:12s} think={d2:12s} "
              f"think_tokens={t['think_tokens']}")
    print("meta:", {k: v for k, v in one[0]["meta"].items() if k != "results"})


if __name__ == "__main__":
    main()
