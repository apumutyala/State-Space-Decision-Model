"""Apply the pre-registered probe9b success rule (configs/probe9b_h100.json: success_rule) to the probe's measurements.

python scripts/decide_probe9b.py <pipeline results.json> <index compare.json or -> <config.json> <decision.json>
(exit 9 if no valid decision could be made, else 0). A rejected gate is a VALID measured outcome (quality-fail),
not a job failure: the job still exits 0 with a decision.
"""
from __future__ import annotations

import json
import math
import os
import sys

VERDICTS = ("proceed-all-8", "retrieval-fail", "quality-fail")
REQUIRED = ("ppl_ratio_ctx1024", "ppl_ratio_ctx4096", "mmlu_drop_pts", "mqar_drop", "niah_mean_drop",
            "index_diff_point", "index_diff_lo")


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def checks(R, cmp, rule):
    """-> (gate, quality checks, retrieval checks, numbers). Missing inputs give None values (never a pass)."""
    steps = (R.get("progressive") or {}).get("steps") or []
    s = steps[0] if steps else {}
    ge = s.get("gate_eval") or {}
    gate = {"layer": s.get("layer"), "pass": s.get("gate_pass"), "accepted": s.get("accepted"),
            "step_delta_nll": s.get("step_delta_nll"), "delta_nll": ge.get("delta_nll"),
            "hidden_cos": ge.get("hidden_cos"), "kl": ge.get("kl"),
            "importance_order": (R.get("importance") or {}).get("order")}
    st = R.get("probe_stock") or {}
    cv = R.get(f"probe_{s.get('probe_tag')}") if s.get("probe_tag") else None
    cv = cv or {}
    num = {}
    q, r = {}, {}
    for ctx, mx in rule["ppl_ratio_max"].items():
        a, b = (cv.get("ppl") or {}).get(ctx, {}).get("ppl"), (st.get("ppl") or {}).get(ctx, {}).get("ppl")
        num[f"ppl_ratio_{ctx}"] = a / b if _num(a) and _num(b) and b > 0 else None
        q[f"ppl_{ctx}"] = num[f"ppl_ratio_{ctx}"] is not None and num[f"ppl_ratio_{ctx}"] <= mx
    a, b = (cv.get("mmlu") or {}).get("acc"), (st.get("mmlu") or {}).get("acc")
    num["mmlu_drop_pts"] = 100 * (b - a) if _num(a) and _num(b) else None
    q["mmlu"] = num["mmlu_drop_pts"] is not None and num["mmlu_drop_pts"] <= rule["mmlu_drop_max_pts"]
    d = ((cmp or {}).get("diff") or {}).get("index") or {}
    num["index_diff_point"], num["index_diff_lo"], num["index_diff_hi"] = d.get("point"), d.get("lo"), d.get("hi")
    q["index"] = (_num(d.get("point")) and _num(d.get("lo")) and d["point"] >= rule["index_diff_point_min"]
                  and d["lo"] >= rule["index_diff_lo_min"])
    a, b = (cv.get("mqar") or {}).get("mean_acc"), (st.get("mqar") or {}).get("mean_acc")
    num["mqar_drop"] = b - a if _num(a) and _num(b) else None
    r["mqar"] = num["mqar_drop"] is not None and num["mqar_drop"] <= rule["mqar_drop_max"]
    a, b = (cv.get("niah") or {}).get("mean_acc"), (st.get("niah") or {}).get("mean_acc")
    num["niah_mean_drop"] = b - a if _num(a) and _num(b) else None
    r["niah_mean"] = num["niah_mean_drop"] is not None and num["niah_mean_drop"] <= rule["niah_mean_drop_max"]
    per = {}
    for k, v in (st.get("niah") or {}).items():
        if k.startswith("L") and isinstance(v, dict):
            c = (cv.get("niah") or {}).get(k, {}).get("acc")
            per[k] = v["acc"] - c if _num(c) and _num(v.get("acc")) else None
    num["niah_length_drop"] = per
    r["niah_per_length"] = bool(per) and all(x is not None and x <= rule["niah_length_drop_max"] for x in per.values())
    return gate, q, r, num


def decide(R, cmp, cfg):
    rule = cfg["success_rule"]
    gate, q, r, num = checks(R, cmp, rule)
    out = {"gate": gate, "quality_checks": q, "retrieval_checks": r, "numbers": num, "rule": rule,
           "stopped": R.get("stopped")}
    if R.get("stopped"):
        out["verdict"] = f"stopped (exit {R['stopped'].get('exit_code')}): no verdict"
    elif gate["pass"] is False and rule.get("gate_must_pass", True):
        out["verdict"] = "quality-fail"
    elif gate["pass"] is not True:
        out["verdict"] = "no-measurement: the conversion step did not run"
    elif [k for k in REQUIRED if not _num(num.get(k))]:
        out["verdict"] = f"no-measurement: missing {[k for k in REQUIRED if not _num(num.get(k))]}"
    elif not all(q.values()):
        out["verdict"] = "quality-fail"
    elif not all(r.values()):
        out["verdict"] = "retrieval-fail"
    else:
        out["verdict"] = "proceed-all-8"
    return out


def valid_decision(d):
    """A decision the probe job may PASS on: a known verdict computed from real measurements."""
    if not isinstance(d, dict) or d.get("verdict") not in VERDICTS:
        return False
    g = d.get("gate") or {}
    if g.get("pass") is False:  # a rejected layer: the gate numbers themselves are the measurement
        return d["verdict"] == "quality-fail" and all(_num(g.get(k)) for k in ("step_delta_nll", "delta_nll", "hidden_cos"))
    n = d.get("numbers") or {}
    return g.get("pass") is True and all(_num(n.get(k)) for k in REQUIRED)


def main(results_json, compare_json, config_json, out_json):
    R = json.load(open(results_json))
    cmp = json.load(open(compare_json)) if compare_json != "-" and os.path.exists(compare_json) else None
    out = decide(R, cmp, json.load(open(config_json)))
    json.dump(out, open(out_json, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("verdict", "gate", "numbers")}, indent=1))
    return 0 if valid_decision(out) else 9


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
