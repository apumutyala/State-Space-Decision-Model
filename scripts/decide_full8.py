"""Apply the PRE-REGISTERED full-8 rule (jobs/full8/rule.json) to the full 8-layer Q9-AF conversion's measurements.

python scripts/decide_full8.py <pipeline results.json> <index compare.json or -> <rule.json> <decision.json>
(exit 9 if no valid decision could be made, else 0).

Binding: the verdict is computed from rule.json only. Everything under "report" (the cost-of-attention-freedom curve,
per-layer gate numbers, stage-2 receipts, the MMLU noise note) is reported and never changes the verdict.

Before any verdict the gate record is re-checked independently: the 8 steps must be the 8 attention layers in the
recorded ascending-KL importance order, every step's pass/fail must re-compute from its own gate numbers with the
rule's constants, every step must be enforced (accepted == passed), each step's baseline must be the last accepted
step's nll (or the teacher's), and the final converted list must be exactly the accepted layers. Any mismatch is
"no-measurement" (exit 9): a gate that did not run as pre-registered cannot authorise anything.
"""
from __future__ import annotations

import json
import math
import os
import sys

VERDICTS = ("q9af-backbone", "retrieval-fail", "quality-fail", "partial-conversion", "no-conversion")
REQUIRED = ("ppl_ratio_ctx1024", "ppl_ratio_ctx4096", "mmlu_drop_pts", "mqar_drop", "niah_mean_drop",
            "index_diff_point", "index_diff_lo")
TOL = 1e-9


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def gate_audit(R, rule):
    """-> (problems: list[str], steps summary, converted list). Re-derives every gate verdict from recorded numbers."""
    g = rule["gate"]
    attn = list(rule["attention_layers"])
    prob = []
    imp = R.get("importance") or {}
    order = imp.get("order") or []
    per = imp.get("per_layer") or {}
    if sorted(order) != sorted(attn):
        prob.append(f"importance order {order} is not a permutation of the attention layers {attn}")
    elif [per.get(str(i), {}).get("kl") for i in order] != sorted(per.get(str(i), {}).get("kl") for i in order):
        prob.append("importance order is not ascending single-swap KL")
    prog = R.get("progressive") or {}
    steps = prog.get("steps") or []
    if len(steps) != len(attn):
        prob.append(f"{len(steps)} of {len(attn)} layers attempted")
    teacher = None
    accepted, rows = [], []
    for k, s in enumerate(steps):
        ge = s.get("gate_eval") or {}
        if not all(_num(ge.get(x)) for x in ("nll_student", "nll_teacher", "delta_nll", "hidden_cos")):
            prob.append(f"step {k}: gate numbers missing")
            continue
        teacher = ge["nll_teacher"] if teacher is None else teacher
        if k < len(order) and s.get("layer") != order[k]:
            prob.append(f"step {k}: layer {s.get('layer')} != importance order {order[k]}")
        last_acc = next((r for r in reversed(rows) if r["accepted"]), None)
        base_want = last_acc["nll_student"] if last_acc else teacher
        if not _num(s.get("baseline_nll")) or abs(s["baseline_nll"] - base_want) > 1e-6:
            prob.append(f"step {k}: baseline {s.get('baseline_nll')} != last accepted / teacher nll {base_want}")
        step_d = ge["nll_student"] - base_want
        cum_d = ge["nll_student"] - ge["nll_teacher"]
        passed = (step_d <= g["max_step_delta_nll"] + TOL and cum_d <= g["max_cum_delta_nll"] + TOL
                  and ge["hidden_cos"] >= g["min_hidden_cos"] - TOL)
        if abs(cum_d - ge["delta_nll"]) > 1e-6:
            prob.append(f"step {k}: delta_nll {ge['delta_nll']} != nll_student - nll_teacher {cum_d}")
        if s.get("policy") != g["policy"]:
            prob.append(f"step {k}: policy {s.get('policy')} != {g['policy']}")
        if s.get("gate_pass") is not passed:
            prob.append(f"step {k}: recorded gate_pass {s.get('gate_pass')} != re-computed {passed}")
        if s.get("accepted") is not passed:
            prob.append(f"step {k}: accepted {s.get('accepted')} but the enforced gate says {passed}")
        if passed:
            accepted.append(s.get("layer"))
        if list(s.get("converted_after") or []) != accepted:
            prob.append(f"step {k}: converted_after {s.get('converted_after')} != accepted so far {accepted}")
        rows.append({"k": k, "layer": s.get("layer"), "accepted": passed, "nll_student": ge["nll_student"],
                     "step_delta_nll": step_d, "delta_nll": cum_d, "kl": ge.get("kl"), "hidden_cos": ge["hidden_cos"],
                     "probe_tag": s.get("probe_tag"), "train_tokens": s.get("train_tokens"),
                     "steady_tok_s": s.get("steady_tok_s"), "mfu": s.get("mfu"), "peak_gb": s.get("peak_gb")})
    if list(prog.get("converted") or []) != accepted:
        prob.append(f"converted {prog.get('converted')} != accepted layers {accepted}")
    return prob, rows, accepted


def probe_numbers(st, cv, cmp):
    """Final model (cv) minus stock (st): the same numbers as the probe's 511e8dd rule. None = missing."""
    num = {}
    for ctx in ("ctx1024", "ctx4096"):
        a, b = (cv.get("ppl") or {}).get(ctx, {}).get("ppl"), (st.get("ppl") or {}).get(ctx, {}).get("ppl")
        num[f"ppl_ratio_{ctx}"] = a / b if _num(a) and _num(b) and b > 0 else None
    a, b = (cv.get("mmlu") or {}).get("acc"), (st.get("mmlu") or {}).get("acc")
    num["mmlu_drop_pts"] = 100 * (b - a) if _num(a) and _num(b) else None
    d = ((cmp or {}).get("diff") or {}).get("index") or {}
    num["index_diff_point"], num["index_diff_lo"], num["index_diff_hi"] = d.get("point"), d.get("lo"), d.get("hi")
    a, b = (cv.get("mqar") or {}).get("mean_acc"), (st.get("mqar") or {}).get("mean_acc")
    num["mqar_drop"] = b - a if _num(a) and _num(b) else None
    a, b = (cv.get("niah") or {}).get("mean_acc"), (st.get("niah") or {}).get("mean_acc")
    num["niah_mean_drop"] = b - a if _num(a) and _num(b) else None
    per = {}
    for k, v in (st.get("niah") or {}).items():
        if k.startswith("L") and isinstance(v, dict):
            c = (cv.get("niah") or {}).get(k, {}).get("acc")
            per[k] = v["acc"] - c if _num(c) and _num(v.get("acc")) else None
    num["niah_length_drop"] = per
    return num


def checks(num, rule):
    q, r = rule["quality"], rule["retrieval"]
    qc = {f"ppl_{ctx}": _num(num.get(f"ppl_ratio_{ctx}")) and num[f"ppl_ratio_{ctx}"] <= mx
          for ctx, mx in q["ppl_ratio_max"].items()}
    qc["mmlu"] = _num(num.get("mmlu_drop_pts")) and num["mmlu_drop_pts"] <= q["mmlu_drop_max_pts"]
    qc["index"] = (_num(num.get("index_diff_point")) and _num(num.get("index_diff_lo"))
                   and num["index_diff_point"] >= q["index_diff_point_min"] and num["index_diff_lo"] >= q["index_diff_lo_min"])
    per = num.get("niah_length_drop") or {}
    rc = {"mqar": _num(num.get("mqar_drop")) and num["mqar_drop"] <= r["mqar_drop_max"],
          "niah_mean": _num(num.get("niah_mean_drop")) and num["niah_mean_drop"] <= r["niah_mean_drop_max"],
          "niah_per_length": bool(per) and all(_num(x) and x <= r["niah_length_drop_max"] for x in per.values())}
    return {k: bool(v) for k, v in qc.items()}, {k: bool(v) for k, v in rc.items()}


def curve(R, rows, st):
    """Cost-of-attention-freedom curve (reported, never binding): one point per accepted layer, plus stock."""
    def pt(kept, p, layer=None):
        p = p or {}
        return {"attention_layers_kept": kept, "added_layer": layer,
                "ppl1k": ((p.get("ppl") or {}).get("ctx1024") or {}).get("ppl"),
                "ppl4k": ((p.get("ppl") or {}).get("ctx4096") or {}).get("ppl"),
                "mmlu": (p.get("mmlu") or {}).get("acc"), "mqar": (p.get("mqar") or {}).get("mean_acc"),
                "niah": (p.get("niah") or {}).get("mean_acc")}
    out = [pt(8, st)]
    n = 0
    for r in rows:
        if r["accepted"]:
            n += 1
            out.append(pt(8 - n, R.get(f"probe_{r['probe_tag']}") if r["probe_tag"] else None, r["layer"]))
    return out


def decide(R, cmp, rule):
    prob, rows, conv = gate_audit(R, rule)
    st = R.get("probe_stock") or {}
    last = next((r for r in reversed(rows) if r["accepted"]), None)
    cv = (R.get(f"probe_{last['probe_tag']}") if last and last["probe_tag"] else None) or {}
    if last and cv and sorted(int(k) for k in (cv.get("modes") or {})) != sorted(conv):
        prob.append(f"final probe {last['probe_tag']} modes {sorted(cv.get('modes') or {})} != converted {conv}")
    num = probe_numbers(st, cv, cmp)
    qc, rc = checks(num, rule)
    n = len(conv)
    out = {"rule_version": rule["version"], "n_converted": n, "converted": conv,
           "attention_layers_kept": [i for i in rule["attention_layers"] if i not in conv],
           "attention_free": n == len(rule["attention_layers"]),
           "gate_audit_problems": prob, "quality_checks": qc, "retrieval_checks": rc, "numbers": num,
           "stopped": R.get("stopped"), "rule": {k: rule[k] for k in ("gate", "quality", "retrieval")}}
    missing = [k for k in REQUIRED if not _num(num.get(k))]
    if R.get("stopped"):
        out["verdict"] = f"stopped (exit {R['stopped'].get('exit_code')}): no verdict"
    elif prob:
        out["verdict"] = f"no-measurement: gate record {prob[0]}"
    elif n == 0:
        out["verdict"] = "no-conversion"
    elif missing:
        out["verdict"] = f"no-measurement: missing {missing}"
    elif n < len(rule["attention_layers"]):
        out["verdict"] = "partial-conversion"
        out["hybrid_name"] = f"Q9-H{len(rule['attention_layers']) - n}"
    elif not all(qc.values()):
        out["verdict"] = "quality-fail"
    elif not all(rc.values()):
        out["verdict"] = "retrieval-fail"
    else:
        out["verdict"] = "q9af-backbone"
    out["next"] = rule["verdicts"].get(out["verdict"], rule["no_verdict"])
    out["report"] = report(R, rows, st, num)  # reported only: computed AFTER the verdict, never read by it
    return out


def report(R, rows, st, num):
    try:
        mm = (st.get("mmlu") or {}).get("acc")
        n_mmlu = (st.get("mmlu") or {}).get("n") or 285
        se = math.sqrt(2 * mm * (1 - mm) / n_mmlu) * 100 if _num(mm) else None  # unpaired upper bound on the diff SE
        return {"binding": False, "curve": curve(R, rows, st), "steps": rows,
                "mmlu_diff_se_upper_pts": se,
                "mmlu_noise_limited": bool(_num(num.get("mmlu_drop_pts")) and se and 3.0 < num["mmlu_drop_pts"] <= 3.0 + 2 * se),
                "throughput": R.get("throughput"), "timing": R.get("timing")}
    except Exception as e:  # noqa: BLE001
        return {"binding": False, "error": repr(e)[:300]}


def valid_decision(d):
    """A decision the full-8 job may PASS on: a known verdict computed from real, audited measurements."""
    if not isinstance(d, dict) or d.get("verdict") not in VERDICTS or d.get("gate_audit_problems"):
        return False
    if d["verdict"] == "no-conversion":
        return d.get("n_converted") == 0
    n = d.get("numbers") or {}
    if not all(_num(n.get(k)) for k in REQUIRED):
        return False
    return (d["verdict"] == "partial-conversion") == (not d.get("attention_free")) and d.get("n_converted", 0) > 0


def main(results_json, compare_json, rule_json, out_json):
    R = json.load(open(results_json))
    cmp = json.load(open(compare_json)) if compare_json != "-" and os.path.exists(compare_json) else None
    out = decide(R, cmp, json.load(open(rule_json, encoding="utf-8")))
    json.dump(out, open(out_json, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("verdict", "n_converted", "converted", "numbers", "gate_audit_problems")},
                     indent=1))
    return 0 if valid_decision(out) else 9


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
