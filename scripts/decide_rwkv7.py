"""Decisions for jobs/rwkv7 (stage A readout, stage B LoRA pilot). Pre-registered; validators used by finalize.

python scripts/decide_rwkv7.py readout <work_out_dir> <decision.json>     (exit 9 if invalid)
python scripts/decide_rwkv7.py lora    <cell.json>    <decision.json>     (exit 9 if invalid)

Stage A verdict "readout-measured": the subset Index score (with CI), ECE, steer-smoke summary and the memory probe all
exist and are numeric. It is a measurement, not a pass/fail on quality: the parent choice is made by comparing it with
the Q9-AF probe arms on the same subset (jobs/DECISION_TREE.md). Stage B verdicts "pilot-pass" / "pilot-fail" follow
train.run_cell.pass_rule (jobs/rwkv7/pilot_cell.json: pass_rule); both are valid measured outcomes.
"""
from __future__ import annotations

import json
import math
import os
import sys

READOUT_VERDICTS = ("readout-measured",)
LORA_VERDICTS = ("pilot-pass", "pilot-fail")


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _load(p):
    try:
        return json.load(open(p))
    except Exception:  # noqa: BLE001
        return None


def decide_readout(work_out):
    score = _load(os.path.join(work_out, "index", "score.json")) or {}
    ece = _load(os.path.join(work_out, "index", "ece.json"))
    steer = _load(os.path.join(work_out, "steer", "summary.json"))
    probe = _load(os.path.join(work_out, "probe.json")) or {}
    idx = (score.get("slice_index") or {}).get("index") or {}
    d = {"index": idx, "index_rows": score.get("rows"), "ece_present": ece is not None, "steer_present": steer is not None,
         "probe_peak_gb": probe.get("peak_gb"), "prefill": probe.get("prefill"),
         "state_bytes_per_seq": probe.get("state_bytes_per_seq")}
    ok = all(_num(idx.get(k)) for k in ("point", "lo", "hi")) and ece is not None and steer is not None \
        and _num(probe.get("peak_gb"))
    d["verdict"] = "readout-measured" if ok else "no-measurement"
    return d


def decide_lora(cell):
    tr = (cell or {}).get("train") or {}
    v = tr.get("verdict") or {}
    d = {"smoke": (cell or {}).get("smoke"), "ladder": (cell or {}).get("ladder"), "steps": tr.get("steps"),
         "pass_rule_result": v}
    if not (cell or {}).get("smoke", {}).get("pass") or not isinstance(v.get("pass"), bool):
        d["verdict"] = "no-measurement"
    else:
        d["verdict"] = "pilot-pass" if v["pass"] else "pilot-fail"
    return d


def valid_readout(d):
    return isinstance(d, dict) and d.get("verdict") in READOUT_VERDICTS and _num((d.get("index") or {}).get("point"))


def valid_lora(d):
    if not isinstance(d, dict) or d.get("verdict") not in LORA_VERDICTS:
        return False
    v = d.get("pass_rule_result") or {}
    return _num(v.get("loss_first")) and _num(v.get("peak_gb")) and isinstance(d.get("steps"), int) and d["steps"] > 0


def main(kind, src, out):
    d = decide_readout(src) if kind == "readout" else decide_lora(_load(src))
    json.dump(d, open(out, "w"), indent=1)
    print(json.dumps({"verdict": d["verdict"]}))
    return 0 if (valid_readout(d) if kind == "readout" else valid_lora(d)) else 9


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
