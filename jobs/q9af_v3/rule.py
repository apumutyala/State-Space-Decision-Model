"""Pre-registered gate of jobs/q9af_v3 (rule q9af-v3-rule-v1). Committed with the job, before any v3 run or data.

    python -m jobs.q9af_v3.rule decide  <params_B.json> <results_B.json> <export.json> <v2_export.json|->
                                        <compare_v3_vs_ref.json> <compare_v3_vs_stock.json> <compare_v2_vs_stock.json|->
                                        <decision.json>
    python -m jobs.q9af_v3.rule stagea  <results_A.json> <gen_results.json> <export_A.json> [<gates.json>] <decision.json>
    python -m jobs.q9af_v3.rule gate    <results_A.json> <k> <compare_stage_k_vs_stock.json|-> <gate.json>
                                        (exit 0 CONTINUE, 20 STOP, 21 no measurement)
    python -m jobs.q9af_v3.rule sens    <results_S.json> <decision.json>

THE GATE (job B; jobs/q9af_v3/README.md): v3 is PROMOTED over v2 iff the paired proxy600 Index difference
(v3 converted - REFERENCE), both arms through the SAME generic v3w LM-logit readout (PR #236: family q9af-probe-v3w,
readout identity V3W_IDENTITY) in the same job, edition 0.2.1, bootstrap B 2000 seed 0 (the frozen
harness/score.py + bootstrap via ``jobs.common.dev_subset compare``), has a CI LOWER bound > PROMOTE_LO (0), AND (recall
redesign, after v2 folded on recall) v3's final MQAR and NIAH mean-accuracy drops vs stock are each <= CUM_MAX_DROP
(0.20; v2 lost 0.59 / 0.525). Else the verdict is REPORT (numbers reported, nothing promoted). REFERENCE = v2's converted model (job-7ji99's
gdn_converted.pt, an all-8 attention-free export) when job_params_B pins it (``v2.artifact_sha256`` set), else the
stock Qwen3.5-9B under the same readout. The reference is fixed by the committed params, never chosen after a run.
Held-out KL (4k mix, 32k, teacher-generated completions) is REPORTED, never gated: v2's lesson (Q9-H2 had KL 0.094 and
lost 26 Index points) is that KL alone cannot promote anything.

Stage A (job A) has no verdict of its own: ``stage_a`` checks that the run produced a complete, attention-free stage-A
export with its numbers (verdict STAGE_A_DONE), which job B consumes.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

RULE_VERSION = "q9af-v3-rule-v1"
STAGE_A_VERSION = "q9af-v3-stageA-v1"
ATTENTION_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31]
PROMOTE_LO = 0.0
PROXY_IDS_SHA256 = "7d42c398336e0934e4fba8d7d78bda07d0073967fc92a8af7b3a14253a9185e6"
V3W_IDENTITY = "2e4907ee4302d9d196052fc04c5618e88fc8058446644715345eaad93a3a2543"
BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 0
REFERENCES = ("v2", "stock")
VERDICTS = ("NO_MEASUREMENT", "PROMOTE", "REPORT")
# ---- recall gates (pre-registered with the recall redesign, after v2's fold: MQAR drop 0.59, NIAH drop 0.525 for 8/8)
#: the cheap sensitivity job: a single converted layer passes iff its MQAR AND NIAH mean-accuracy drops vs stock are both
#: <= SENS_MAX_DROP (absolute). GO iff >= SENS_GO_LAYERS of the 8 pass; NO_GO iff none does; else PARTIAL.
SENS_VERSION = "q9af-v3-sens-v1"
SENS_MAX_DROP = 0.05
SENS_GO_LAYERS = 6
SENS_VERDICTS = ("NO_MEASUREMENT", "GO", "PARTIAL", "NO_GO")
#: job A's stage gate, after every conversion stage (2 layers): CONTINUE iff (a) the stage's own MQAR and NIAH drops
#: (vs the previous stage; stage 1 vs stock) are each <= STAGE_MAX_DROP, (b) the cumulative drops vs stock are each
#: <= CUM_MAX_DROP, (c) the proxy600 paired point (converted-so-far - stock, v3w readout) is >= STAGE_INDEX_FLOOR.
#: Else STOP: no further layer is converted and no further GPU time is spent on this conversion (the breaching layers
#: are named; the order already put the recall-hardest layers last).
STAGE_MAX_DROP = 0.05
CUM_MAX_DROP = 0.20
STAGE_INDEX_FLOOR = -10.0
EPS = 1e-9  # float slack on the bounds (a drop of exactly the bound passes)
ACTIONS = {
    "NO_MEASUREMENT": "no verdict: diagnose the run; nothing is promoted",
    "PROMOTE": "v3 replaces v2 as the Q9-AF conversion: pin v3's gdn_converted.pt for jobs/q9af_cell (jobs.q9af_cell.pin "
               "fill) in its own commit; the next cell runs on v3",
    "REPORT": "v3 is not promoted: report the paired deltas (vs reference, vs stock, v2 vs stock) and the KL curves; "
              "the reference (v2, else stock) stays the Q9-AF input",
}


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def expected_reference(params: dict) -> str:
    """The pre-registered reference: v2 iff the committed job_params_B pins v2's export artifact."""
    v2 = params.get("v2") or {}
    return "v2" if isinstance(v2.get("artifact_sha256"), str) and len(v2["artifact_sha256"]) == 64 else "stock"


def _diff(cmp: dict | None) -> dict:
    d = ((cmp or {}).get("diff") or {}).get("index") or {}
    return {"point": d.get("point"), "lo": d.get("lo"), "hi": d.get("hi")}


def _cmp_problems(name: str, cmp: dict | None) -> list[str]:
    if cmp is None:
        return [f"{name}: missing"]
    p = []
    if cmp.get("subset_ids_sha256") != PROXY_IDS_SHA256:
        p.append(f"{name}: not on proxy600 ({cmp.get('subset_ids_sha256')})")
    if cmp.get("B") != BOOTSTRAP_B or cmp.get("seed") != BOOTSTRAP_SEED:
        p.append(f"{name}: bootstrap B={cmp.get('B')} seed={cmp.get('seed')} != {BOOTSTRAP_B}/{BOOTSTRAP_SEED}")
    return p


def _export_ok(e: dict | None) -> bool:
    e = e or {}
    return bool(e.get("ok")) and e.get("mode") == "final" and sorted(e.get("converted") or []) == ATTENTION_LAYERS \
        and bool(e.get("attention_free"))


def _kl(d, *path):
    for k in path:
        d = (d or {}).get(k) if isinstance(d, dict) else None
    return d if _num(d) else None


def decide(params: dict, results: dict, export: dict | None, v2_export: dict | None, cmp_ref: dict | None,
           cmp_stock: dict | None, cmp_v2: dict | None, reference: str | None = None) -> dict:
    ref = reference or expected_reference(params)
    problems = []
    if ref not in REFERENCES or ref != expected_reference(params):
        problems.append(f"reference {ref!r} != the pre-registered {expected_reference(params)!r}")
    if not _export_ok(export):
        problems.append(f"v3 export is not the full attention-free final export ({(export or {}).get('converted')})")
    if ref == "v2" and not _export_ok(v2_export):
        problems.append("v2 reference export is not an attention-free 8-layer final export")
    problems += _cmp_problems("compare v3 vs reference", cmp_ref)
    dref, dstock, dv2 = _diff(cmp_ref), _diff(cmp_stock), _diff(cmp_v2)
    for k in ("point", "lo"):
        if not _num(dref[k]):
            problems.append(f"v3 - {ref} {k} missing")
    num = {"ref": ref, "diff_vs_ref_point": dref["point"], "diff_vs_ref_lo": dref["lo"], "diff_vs_ref_hi": dref["hi"],
           "diff_vs_stock_point": dstock["point"], "diff_vs_stock_lo": dstock["lo"], "diff_vs_stock_hi": dstock["hi"],
           "v2_vs_stock_point": dv2["point"], "v2_vs_stock_lo": dv2["lo"], "v2_vs_stock_hi": dv2["hi"],
           "heldout_kl_mix": _kl(results, "final", "heldout", "mix", "kl"),
           "heldout_kl_32k": _kl(results, "final", "long", "kl"),
           "gen_completion_kl": _kl(results, "final", "gen_heldout", "kl"),
           "entry_heldout_kl_mix": _kl(results, "entry", "heldout", "mix", "kl"),
           "entry_heldout_kl_32k": _kl(results, "entry", "long", "kl"),
           "mqar_drop_vs_stock": _kl(results, "final", "recall_drop_vs_stock", "mqar_drop"),
           "niah_drop_vs_stock": _kl(results, "final", "recall_drop_vs_stock", "niah_drop")}
    for k in ("mqar_drop_vs_stock", "niah_drop_vs_stock"):
        if num[k] is None:
            problems.append(f"{k} missing")
    recall_ok = not problems and num["mqar_drop_vs_stock"] <= CUM_MAX_DROP and num["niah_drop_vs_stock"] <= CUM_MAX_DROP
    if problems:
        v = "NO_MEASUREMENT"
    elif dref["lo"] > PROMOTE_LO and recall_ok:
        v = "PROMOTE"
    else:
        v = "REPORT"
    return {"rule_version": RULE_VERSION, "verdict": v, "action": ACTIONS[v], "problems": problems, "numbers": num,
            "attention_free": _export_ok(export), "reference": ref, "recall_ok": recall_ok,
            "report": {"binding": ["diff_vs_ref_point", "diff_vs_ref_lo", "mqar_drop_vs_stock", "niah_drop_vs_stock"],
                       "gate": f"PROMOTE iff CI lo of (v3 - {ref}) > {PROMOTE_LO} AND the final MQAR and NIAH drops vs "
                               f"stock are each <= {CUM_MAX_DROP}",
                       "long_context": {k: (results.get("long_context") or {}).get(k)
                                        for k in ("tokens", "best", "curve", "windows_used", "long_after")},
                       "onpolicy": {k: (results.get("onpolicy") or {}).get(k)
                                    for k in ("tokens", "best", "curve", "prompts_used", "rollout_completion_tokens")},
                       "candidate": (cmp_ref or {}).get("candidate"), "baseline": (cmp_ref or {}).get("baseline")}}


def valid_decision(d: dict) -> bool:
    """finalize.py validator (job B): a real verdict with its binding numbers present."""
    return (isinstance(d, dict) and d.get("rule_version") == RULE_VERSION and d.get("verdict") in VERDICTS
            and d["verdict"] != "NO_MEASUREMENT" and all(_num(d["numbers"].get(k)) for k in d["report"]["binding"]))


# ----------------------------------------------------------------------------- the cheap sensitivity job (mode S)
def sensitivity(results: dict) -> dict:
    stock = results.get("recall_stock") or {}
    table = results.get("table") or {}
    rows = {k: v for k, v in table.items() if k.startswith("wide_L")}
    problems = []
    if not (_num(stock.get("mqar_mean")) and _num(stock.get("niah_mean"))):
        problems.append("stock recall missing")
    if sorted(int(k[6:]) for k in rows) != ATTENTION_LAYERS:
        problems.append(f"wide rows for {sorted(rows)}")
    out = {}
    for k, r in sorted(rows.items(), key=lambda kv: int(kv[0][6:])):
        ok = _num(r.get("mqar_drop")) and _num(r.get("niah_drop"))
        if not ok:
            problems.append(f"{k}: recall drops missing")
        out[k[6:]] = {"mqar_drop": r.get("mqar_drop"), "niah_drop": r.get("niah_drop"), "kl": r.get("kl"),
                      "pass": bool(ok and r["mqar_drop"] <= SENS_MAX_DROP + EPS and r["niah_drop"] <= SENS_MAX_DROP + EPS)}
    n_ok = sum(v["pass"] for v in out.values())
    ctrl = {}
    for k, r in table.items():
        if k.startswith("base_L") and f"wide_L{k[6:]}" in table:
            w = table[f"wide_L{k[6:]}"]
            ctrl[k[6:]] = {"base_mqar_drop": r.get("mqar_drop"), "base_niah_drop": r.get("niah_drop"),
                           "wide_mqar_drop": w.get("mqar_drop"), "wide_niah_drop": w.get("niah_drop")}
    v = "NO_MEASUREMENT" if problems else ("GO" if n_ok >= SENS_GO_LAYERS else "PARTIAL" if n_ok else "NO_GO")
    act = {"NO_MEASUREMENT": "diagnose; nothing is funded",
           "GO": "submit job A (recall-gated conversion, order ascending recall damage), then the continuation chain",
           "PARTIAL": "decision-needed: the failing layers convert last and job A's stage gate will likely STOP before "
                      "them; fund A only for a hybrid, or raise capacity first",
           "NO_GO": "do not fund the big chain at this capacity / mix: next lever = more state (dk 512 / more heads) or "
                    "another mixer"}[v]
    return {"rule_version": SENS_VERSION, "verdict": v, "action": act, "problems": problems, "layers": out,
            "n_pass": n_ok, "max_drop": SENS_MAX_DROP, "capacity_control": ctrl,
            "stock_recall": {k: stock.get(k) for k in ("mqar_mean", "niah_mean")}}


def valid_sensitivity(d: dict) -> bool:
    return isinstance(d, dict) and d.get("rule_version") == SENS_VERSION and d.get("verdict") in SENS_VERDICTS[1:]


# ----------------------------------------------------------------------------- job A's per-stage gate
def stage_gate(results: dict, k: int, cmp: dict | None) -> dict:
    """CONTINUE / STOP after conversion stage k (1-based), from stage A's results.json and the proxy600 compare of
    the converted-so-far model vs stock (``None`` -> NO_MEASUREMENT, which the runner treats as STOP)."""
    stages = results.get("stages") or {}
    st = stages.get(f"stage{k}") or {}
    stock = results.get("recall_stock") or {}
    prev = (stages.get(f"stage{k - 1}") or {}).get("recall") if k > 1 else None
    prev = prev or {"mqar_mean": stock.get("mqar_mean"), "niah_mean": stock.get("niah_mean")}
    rec = st.get("recall") or {}
    problems = _cmp_problems(f"compare stage{k} vs stock", cmp)
    num = {}
    for m in ("mqar", "niah"):
        a, b, s0 = rec.get(f"{m}_mean"), prev.get(f"{m}_mean"), stock.get(f"{m}_mean")
        if not (_num(a) and _num(b) and _num(s0)):
            problems.append(f"{m} recall missing")
            continue
        num[f"{m}_stage_drop"], num[f"{m}_cum_drop"] = b - a, s0 - a
    d = _diff(cmp)
    num.update({"proxy_point": d["point"], "proxy_lo": d["lo"], "proxy_hi": d["hi"]})
    if not _num(d["point"]):
        problems.append("proxy600 point missing")
    why = []
    if not problems:
        for m in ("mqar", "niah"):
            if num[f"{m}_stage_drop"] > STAGE_MAX_DROP + EPS:
                why.append(f"{m} stage drop {num[m + '_stage_drop']:.3f} > {STAGE_MAX_DROP}")
            if num[f"{m}_cum_drop"] > CUM_MAX_DROP + EPS:
                why.append(f"{m} cumulative drop {num[m + '_cum_drop']:.3f} > {CUM_MAX_DROP}")
        if d["point"] < STAGE_INDEX_FLOOR:
            why.append(f"proxy600 {d['point']:.2f} < {STAGE_INDEX_FLOOR}")
    v = "NO_MEASUREMENT" if problems else ("STOP" if why else "CONTINUE")
    return {"rule_version": RULE_VERSION, "stage": k, "verdict": v, "breaches": why, "problems": problems,
            "numbers": num, "adds": st.get("adds"), "converted": st.get("converted")}


# ----------------------------------------------------------------------------- stage A (job A)
def stage_a(results: dict, gen: dict, export: dict | None, gates: list | None = None) -> dict:
    """gen: phase G's results.json (``{"gen": {...}, ...}``) or its ``gen`` record itself. gates: the runner's
    stage_gate records in order; a final STOP / NO_MEASUREMENT gate makes the record STOPPED (a valid, measured
    outcome: the conversion is not attention-free and nothing is chained)."""
    gen = (gen or {}).get("gen", gen) if isinstance((gen or {}).get("gen"), dict) else gen
    gates = gates or []
    if gates and gates[-1].get("verdict") != "CONTINUE":
        g = gates[-1]
        return {"rule_version": STAGE_A_VERSION, "verdict": "STOPPED", "problems": [], "stopped_at_stage": g["stage"],
                "breaches": g.get("breaches") or g.get("problems"), "gates": gates,
                "numbers": {"heldout_kl_mix": None, "gen_completion_tokens": (gen or {}).get("completion_tokens")},
                "order": (results.get("plan") or {}).get("order"), "export": (export or {}).get("mode"),
                "action": f"STOP rule: layers {g.get('adds')} could not be converted within the recall / proxy600 "
                          "bounds; report the per-stage table; no job B"}
    problems = []
    if not _export_ok(export):
        problems.append(f"stage-A export is not the full attention-free export ({(export or {}).get('converted')})")
    plan = results.get("plan") or {}
    order = sorted(plan.get("order") or [])
    if order != ATTENTION_LAYERS:
        problems.append(f"plan order covers {order}")
    stages = results.get("stages") or {}
    if len(stages) != len(plan.get("stages") or []) or not stages:
        problems.append(f"{len(stages)} of {len(plan.get('stages') or [])} stages ran")
    num = {"heldout_kl_mix": _kl(results, "final", "heldout", "mix", "kl"),
           "gen_completion_kl": _kl(results, "final", "gen_heldout", "kl"),
           "gen_completion_tokens": (gen or {}).get("completion_tokens"), "gen_tok_s": (gen or {}).get("tok_s")}
    for k in ("heldout_kl_mix", "gen_completion_tokens"):
        if not _num(num[k]):
            problems.append(f"{k} missing")
    return {"rule_version": STAGE_A_VERSION, "verdict": "NO_MEASUREMENT" if problems else "STAGE_A_DONE",
            "problems": problems, "numbers": num, "order": plan.get("order"), "sensitivity": plan.get("sensitivity"),
            "stages": {k: {"adds": s.get("adds"), "best": s.get("best"), "tokens": s.get("tokens"),
                           "minutes_budget": s.get("minutes_budget"), "kept_last": s.get("kept_last")}
                       for k, s in stages.items()},
            "action": "package job B from this artifact (python -m jobs.q9af_v3.artifact fill --which stageA)"
            if not problems else "diagnose; nothing to chain"}


def valid_stage_a(d: dict) -> bool:
    if not (isinstance(d, dict) and d.get("rule_version") == STAGE_A_VERSION):
        return False
    if d.get("verdict") == "STOPPED":
        return bool(d.get("gates")) and d.get("stopped_at_stage") is not None
    return d.get("verdict") == "STAGE_A_DONE" and _num((d.get("numbers") or {}).get("heldout_kl_mix"))


def _load(p):
    return None if p == "-" or not Path(p).exists() else json.loads(Path(p).read_text(encoding="utf-8"))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["decide"] and len(argv) == 9:
        params, res, exp, v2e, cr, cs, cv = (_load(p) for p in argv[1:8])
        out = argv[8]
        d = decide(params or {}, res or {}, exp, v2e, cr, cs, cv)
        code = 0 if d["verdict"] != "NO_MEASUREMENT" else 9
    elif argv[:1] == ["stagea"] and len(argv) in (5, 6):
        res, gen, exp = (_load(p) for p in argv[1:4])
        gates = _load(argv[4]) if len(argv) == 6 else None
        out = argv[-1]
        d = stage_a(res or {}, gen or {}, exp, gates)
        code = 0 if d["verdict"] in ("STAGE_A_DONE", "STOPPED") else 9
    elif argv[:1] == ["gate"] and len(argv) == 5:
        res, cmp = _load(argv[1]), _load(argv[3])
        out = argv[4]
        d = stage_gate(res or {}, int(argv[2]), cmp)
        code = {"CONTINUE": 0, "STOP": 20}.get(d["verdict"], 21)
    elif argv[:1] == ["sens"] and len(argv) == 3:
        out = argv[2]
        d = sensitivity(_load(argv[1]) or {})
        code = 0 if d["verdict"] != "NO_MEASUREMENT" else 9
    else:
        print(__doc__.split("\n\n")[1])
        return 2
    Path(out).write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(json.dumps({k: d.get(k) for k in ("verdict", "problems", "numbers", "breaches", "n_pass")}))
    return code


if __name__ == "__main__":
    sys.exit(main())
