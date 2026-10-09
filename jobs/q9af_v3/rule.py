"""Pre-registered gate of jobs/q9af_v3 (rule q9af-v3-rule-v1). Committed with the job, before any v3 run or data.

    python -m jobs.q9af_v3.rule decide  <params_B.json> <results_B.json> <export.json> <v2_export.json|->
                                        <compare_v3_vs_ref.json> <compare_v3_vs_stock.json> <compare_v2_vs_stock.json|->
                                        <decision.json>
    python -m jobs.q9af_v3.rule stagea  <results_A.json> <gen_results.json> <export_A.json> <decision.json>

THE GATE (job B; jobs/q9af_v3/README.md): v3 is PROMOTED over v2 iff the paired proxy600 Index difference
(v3 converted - REFERENCE), both arms through the SAME generic v3w LM-logit readout (PR #236: family q9af-probe-v3w,
readout identity V3W_IDENTITY) in the same job, edition 0.2.1, bootstrap B 2000 seed 0 (the frozen
harness/score.py + bootstrap via ``jobs.common.dev_subset compare``), has a CI LOWER bound > PROMOTE_LO (0). Else the
verdict is REPORT (numbers reported, nothing promoted). REFERENCE = v2's converted model (job-7ji99's
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
           "entry_heldout_kl_32k": _kl(results, "entry", "long", "kl")}
    if problems:
        v = "NO_MEASUREMENT"
    elif dref["lo"] > PROMOTE_LO:
        v = "PROMOTE"
    else:
        v = "REPORT"
    return {"rule_version": RULE_VERSION, "verdict": v, "action": ACTIONS[v], "problems": problems, "numbers": num,
            "attention_free": _export_ok(export), "reference": ref,
            "report": {"binding": ["diff_vs_ref_point", "diff_vs_ref_lo"],
                       "gate": f"PROMOTE iff CI lo of (v3 - {ref}) > {PROMOTE_LO}",
                       "long_context": {k: (results.get("long_context") or {}).get(k)
                                        for k in ("tokens", "best", "curve", "windows_used", "long_after")},
                       "onpolicy": {k: (results.get("onpolicy") or {}).get(k)
                                    for k in ("tokens", "best", "curve", "prompts_used", "rollout_completion_tokens")},
                       "candidate": (cmp_ref or {}).get("candidate"), "baseline": (cmp_ref or {}).get("baseline")}}


def valid_decision(d: dict) -> bool:
    """finalize.py validator (job B): a real verdict with its binding numbers present."""
    return (isinstance(d, dict) and d.get("rule_version") == RULE_VERSION and d.get("verdict") in VERDICTS
            and d["verdict"] != "NO_MEASUREMENT" and all(_num(d["numbers"].get(k)) for k in d["report"]["binding"]))


# ----------------------------------------------------------------------------- stage A (job A)
def stage_a(results: dict, gen: dict, export: dict | None) -> dict:
    """gen: phase G's results.json (``{"gen": {...}, ...}``) or its ``gen`` record itself."""
    gen = (gen or {}).get("gen", gen) if isinstance((gen or {}).get("gen"), dict) else gen
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
    return (isinstance(d, dict) and d.get("rule_version") == STAGE_A_VERSION and d.get("verdict") == "STAGE_A_DONE"
            and _num((d.get("numbers") or {}).get("heldout_kl_mix")))


def _load(p):
    return None if p == "-" or not Path(p).exists() else json.loads(Path(p).read_text(encoding="utf-8"))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["decide"] and len(argv) == 9:
        params, res, exp, v2e, cr, cs, cv = (_load(p) for p in argv[1:8])
        out = argv[8]
        d = decide(params or {}, res or {}, exp, v2e, cr, cs, cv)
        code = 0 if d["verdict"] != "NO_MEASUREMENT" else 9
    elif argv[:1] == ["stagea"] and len(argv) == 5:
        res, gen, exp = (_load(p) for p in argv[1:4])
        out = argv[4]
        d = stage_a(res or {}, gen or {}, exp)
        code = 0 if d["verdict"] == "STAGE_A_DONE" else 9
    else:
        print(__doc__.split("\n\n")[1])
        return 2
    Path(out).write_text(json.dumps(d, indent=1), encoding="utf-8")
    print(json.dumps({k: d[k] for k in ("verdict", "problems", "numbers")}))
    return code


if __name__ == "__main__":
    sys.exit(main())
