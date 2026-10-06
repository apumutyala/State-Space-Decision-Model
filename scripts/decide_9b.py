"""Apply the pre-registered decision rule (configs/m1_9b_qual.json: decision_rule) to the 9B probe results.

python scripts/decide_9b.py <9b receipt dir>   -> <dir>/decision.json (exit 0 always; the verdict is data)
"""
from __future__ import annotations

import json
import os
import sys


def decide(R, cfg):
    rule = cfg["decision_rule"]
    stopped = R.get("stopped")
    steps = R.get("progressive", {}).get("steps", [])
    s2 = steps[0] if steps else None
    tok_s = (s2 or {}).get("steady_tok_s") or (s2 or {}).get("train_tok_s")
    if stopped and stopped.get("tok_s"):
        tok_s = stopped["tok_s"]
    peaks = [x for x in [R.get("stage1", {}).get("peak_gb"), (s2 or {}).get("peak_gb"),
                         (stopped or {}).get("peak_gb")] if x]
    peak = max(peaks) if peaks else None
    mfu = tok_s * rule["flop_per_token_stage2"] / rule["h100_peak_flops"] if tok_s else None
    out = {"stage2_steady_tok_s": tok_s, "stage2_mfu": mfu, "peak_gb": peak, "stopped": stopped,
           "stage1_steady_tok_s": R.get("stage1", {}).get("steady_tok_s"), "rules": rule["rules"]}
    if stopped and stopped.get("exit_code") in (3, 4):
        v = "fold"
    elif mfu is None or peak is None:
        v = "no-measurement (stage 2 did not run): do not spend further; diagnose from logs"
    elif peak > 76 or mfu < 0.125:
        v = "fold"
    elif mfu >= 0.25 and peak <= 60:
        v = "open-$40-one-layer-probe-spend-issue"
    else:
        v = "re-price-lean-plan"
    out["verdict"] = v
    return out


def main(d):
    R = json.load(open(os.path.join(d, "results.json")))
    cfg = json.load(open(os.path.join(d, "config.json")))
    out = decide(R, cfg)
    json.dump(out, open(os.path.join(d, "decision.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
