"""Gated deliberation on the probe outputs: answer in one pass when the System-1 confidence clears tau, otherwise think.

EVAL-ONLY diagnostic on public-231 (no fitting: tau is swept and every point is reported; nothing is selected here).
Also the budget approximation: think answer if the trace finished within B tokens, else the one-pass answer.
Speed uses jevkit's timed set (open standard items); per-item latency = measured nothink latency, or measured think
latency on timed items. Cost = input x $0.03/M + think tokens x $0.15/M (the board's Qwen3.5-4B reference).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, ".")
from jevkit import items as IT  # noqa: E402
from jevkit import scoring as SC  # noqa: E402
from keel.paths import private_root  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_rh import answer  # noqa: E402

RAW = private_root() / "experiments" / "reasoning_headroom" / "raw"

PIN, POUT = 0.03, 0.15


def conf(ans):
    if ans["type"] == "noul":
        return max(ans["noul"], 1 - ans["noul"])
    return max(ans["probabilities"].values())


def run(model: str) -> None:
    r = json.load(open(RAW / f"{model}.json"))
    pub = IT.load_public()
    a0 = {x["id"]: x for x in r["arms"]["nothink"]["items"]}
    a1 = {x["id"]: x for x in r["arms"]["think"]["items"]}
    lat0, lat1 = r["latency"]["nothink"], r["latency"]["think"]
    timed = set(lat0)

    def evaluate(choose_think):  # id -> bool
        answers, cost, lats, nthink = {}, 0.0, [], 0
        for iid, x0 in a0.items():
            t = choose_think(iid)
            x = a1[iid] if t else x0
            answers[iid] = answer(x["type"], x["names"], x["logprobs"], 0.8, 0.801)
            cost += x0["input_tokens"] * PIN / 1e6 + (a1[iid]["output_tokens"] * POUT / 1e6 if t else 0.0)
            nthink += t
            if iid in timed:
                lats.append((lat0[iid] + lat1[iid]) if t else lat0[iid])  # gate: one pass first, then think
        tab = SC.build_table(pub, answers)
        agg = SC.aggregate(tab)
        i, c = agg["intelligence"]["I"], agg["calibration"]["score"]
        lats.sort()
        p50 = SC.percentile(lats, 0.5)
        p95 = SC.percentile(lats, 0.95)
        s = SC.speed(p50, p95, self_hosted=True)["score"]
        usd1k = cost / len(a0) * 1000
        k = SC.cost_score(usd1k)
        return dict(I=i, C=c, S=s, K=k, A=SC.composite(i, c, s, k, "A"), cap=(i + c) / 2, think=nthink / len(a0),
                    usd1k=usd1k, p50a=2 * p50 + 0.15, p95a=2 * p95 + 0.15)

    print(f"\n== {model}: gate = think iff one-pass confidence < tau (shim-calibrated)")
    print(f"{'tau':>6} {'think%':>7} {'I':>6} {'C':>6} {'S':>6} {'K':>6} {'CompA':>6} {'Cap':>6} {'$/1k':>8} {'p50adj':>7} {'p95adj':>7}")
    ans0 = {iid: answer(x["type"], x["names"], x["logprobs"], 0.8, 0.801) for iid, x in a0.items()}
    for tau in (0.0, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01):
        m = evaluate(lambda iid: conf(ans0[iid]) < tau)
        print(f"{tau:6.2f} {100*m['think']:6.1f}% {m['I']:6.1f} {m['C']:6.1f} {m['S']:6.1f} {m['K']:6.1f} {m['A']:6.1f} "
              f"{m['cap']:6.1f} {m['usd1k']:8.4f} {m['p50a']:7.3f} {m['p95a']:7.3f}")
    print(f"-- budget approximation: think answer iff the trace closed within B tokens (else one-pass answer)")
    for b in (64, 128, 256, 512, 1024, 2048, 3072):
        m = evaluate(lambda iid: a1[iid]["output_tokens"] <= b and not a1[iid].get("truncated"))
        print(f"B={b:5d} {100*m['think']:6.1f}% I {m['I']:5.1f} C {m['C']:5.1f} Cap {m['cap']:5.1f}")


if __name__ == "__main__":
    for mm in sys.argv[1:]:
        run(mm)
