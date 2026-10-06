"""CPU smoke run of the G4 / G5 wrappers on the stand-in decider (pipeline qualification only).

    python -m safety_eval.run_guarantees_smoke --out receipts/2026-10-06-D-g4g5-standin

The stand-in (``standin.KeywordDecider``) is not a model; its numbers mean nothing about any model. The run checks
that the wrappers emit complete receipts-ready JSON on the real request/attack generators (dev split, clean + D1-A;
D1-B stays held out) and on synthetic G5 data, and records the exact config so m1/m2 runs reuse it.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import attacks, certify, data, selective
from .data import strip_meta
from .request import host_decision
from .selective import executed_conf
from .standin import KeywordDecider


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "safety_eval"]) != 0
        return sha + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def g4(variants, k, arm, **kw):
    t = time.perf_counter()
    rows = certify.evaluate(KeywordDecider(), variants, k=k, arm=arm, **kw)
    return {"arm": arm, "k": k, "config": kw, "wall_s": round(time.perf_counter() - t, 3),
            "metrics": certify.g4_metrics(rows, n_boot=500)}, rows


def g5_standin(variants, alpha, delta):
    dec = KeywordDecider()
    sc, wr, strata, cl, fam = [], [], [], [], []
    for v in variants:
        r = strip_meta(v)
        resp = dec(r)
        hd = host_decision(r, resp, data.QKEY)
        sc.append(executed_conf(r, resp))  # policy-conditioned (same score as G4)
        wr.append(hd["executed"] != v["_meta"]["gold"])
        fam.append(v["_meta"]["attack_family"])
        strata.append(f"{v['_meta']['family']}|{'clean' if fam[-1] == 'clean' else 'attacked'}")
        cl.append(v["_meta"]["cluster"])
    sc, wr = np.array(sc), np.array(wr)
    cal = selective.calibrate_strata(sc, wr, strata, clusters=cl, alpha=alpha, delta=delta)
    # pre-registered shift caveat, demonstrated: calibrate on clean units, evaluate on attacked (held-out goal)
    clean = np.array([f == "clean" for f in fam])
    on_clean = selective.ltt_selective(sc[clean], wr[clean], alpha, delta, clusters=[c for c, m in zip(cl, clean) if m])
    stress = selective.stress_test(on_clean["lam_hat"], sc[~clean], wr[~clean], alpha, delta, label="held_out_goal",
                                   clusters=[c for c, m in zip(cl, clean) if not m])
    return {"strata": {k: {kk: v[kk] for kk in ("n_units", "n_s", "n_clusters", "mu_s", "feasibility", "lam_hat",
                                                  "chosen", "unit_level_unsafe")} for k, v in cal.items()},
            "clean_calibrated": {kk: on_clean[kk] for kk in ("n_s", "lam_hat", "chosen", "feasibility")},
            "stress_attacked": stress, "risk_coverage_all": selective.risk_coverage(sc, wr)}


def g4_uncovered(reqs, n_span):
    """Uncovered-insertion stress arm (PR #20 review): sentence unit on every base request; the hardened span
    unit on the first ``n_span`` route + pay requests (it costs O(T^2) decide calls per request)."""
    out = {}
    t = time.perf_counter()
    rows = certify.uncovered_stress(KeywordDecider(), reqs, 1)
    out["sentence"] = {"n_requests": len(reqs), "wall_s": round(time.perf_counter() - t, 3),
                       "by_mode": certify.uncovered_summary(rows)}
    sub = [r for r in reqs if r["_meta"]["family"] == "route"][:n_span] + \
          [r for r in reqs if r["_meta"]["family"] == "pay"][:n_span]
    for cap in (None, 48):
        t = time.perf_counter()
        kw = {"unit": "span", "max_span_tokens": cap, "max_decide_calls": 100_000}
        rows = certify.uncovered_stress(KeywordDecider(), sub, 1, **kw)
        out[f"span_max{cap or 'all'}"] = {"n_requests": len(sub), "wall_s": round(time.perf_counter() - t, 3),
                                         "config": kw, "by_mode": certify.uncovered_summary(rows)}
    # clean-request cost of the span unit (escalation + calls) on the same subset
    vs = attacks.build_variants(sub, hijack=[], d1=[])
    for cap in (None, 48):
        m = certify.g4_metrics(certify.evaluate(KeywordDecider(), vs, 1, "exact", unit="span", max_span_tokens=cap,
                                                max_decide_calls=100_000))["clean"]
        out[f"span_max{cap or 'all'}"]["clean_cost"] = {kk: m[kk] for kk in
                                                        ("escalation_rate", "auto_execute_rate", "decide_calls",
                                                         "suffix_cost_forward_equivalents_mean")}
    m = certify.g4_metrics(certify.evaluate(KeywordDecider(), vs, 1, "exact"))["clean"]
    out["sentence_same_subset_clean_cost"] = {kk: m[kk] for kk in ("escalation_rate", "auto_execute_rate",
                                                                   "decide_calls", "suffix_cost_forward_equivalents_mean")}
    return out


def g5_synthetic(alpha, delta, n=5000, seed=0, c=0.3):
    rng = np.random.default_rng(seed)
    s = rng.random(n)
    w = rng.random(n) < c * (1 - s)
    r = selective.ltt_selective(s, w, alpha, delta)
    s2 = rng.random(4000)
    held_tpl = selective.stress_test(r["lam_hat"], s2, rng.random(4000) < c * (1 - s2), alpha, delta,
                                     label="held_out_template")
    s3 = rng.random(4000)
    held_goal = selective.stress_test(r["lam_hat"], s3, rng.random(4000) < np.minimum(1, c * (1 - s3) + 0.15),
                                      alpha, delta, label="held_out_goal")
    return {"generator": f"s~U(0,1); wrong~Bern({c}(1-s)); goal shift adds +0.15", "seed": seed, "n": n,
            "calibration": {kk: r[kk] for kk in ("n_s", "mu_s", "feasibility", "lam_hat", "chosen", "guarantee")},
            "held_out_template": held_tpl, "held_out_goal": held_goal}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-route", type=int, default=60)
    ap.add_argument("--n-pay", type=int, default=40)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--n-span", type=int, default=3, help="route and pay requests each for the span-unit arm")
    ap.add_argument("--controlsafe", default="", help="comma list of controlsafe arms (g15,persistence,gramian,"
                    "g4snap,g14) or 'all'; tiny random RWKV-7/Q9-AF backbones, CPU (safety_eval/controlsafe_arms.py)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reqs = data.build(a.n_route, a.n_pay, split="dev")
    vs = attacks.build_variants(reqs, hijack=[], d1=None)  # clean + D1-A only (D1-B held out)
    meta = {"commit": git_sha(), "command": " ".join([Path(sys.executable).name, "-m", "safety_eval.run_guarantees_smoke"]
                                                    + (argv if argv is not None else sys.argv[1:])),
            "python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
            "hardware": "CPU (laptop), $0", "decider": "standin-keyword-v1 (NOT a model)",
            "data": data.manifest(reqs), "variants": len(vs), "d1_templates": sorted(attacks.D1_TEMPLATES_A)}
    results = {}
    results["g4_exact_k1"], rows1 = g4(vs, 1, "exact", max_decide_calls=256)
    results["g4_exact_k2"], _ = g4(vs, 2, "exact", max_decide_calls=256)
    results["g4_smooth_k1"], _ = g4(vs, 1, "smooth", keep_prob=0.3, n_select=16, n_estimate=128, delta=0.05, seed=20261006)
    results["g4_uncovered"] = g4_uncovered(reqs, a.n_span)
    results["g5_standin"] = g5_standin(vs, a.alpha, a.delta)
    results["g5_synthetic"] = g5_synthetic(a.alpha, a.delta)
    if a.controlsafe:
        from .controlsafe_arms import ARMS, run_arms

        arms = ARMS if a.controlsafe == "all" else tuple(x for x in a.controlsafe.split(",") if x)
        results["controlsafe"] = run_arms(arms)
    for name, obj in results.items():
        (out / f"{name}.json").write_text(json.dumps({"meta": meta, name: obj}, indent=1, default=float),
                                          encoding="utf-8", newline="\n")
    data.write_jsonl(rows1, out / "g4_exact_k1_rows.jsonl")
    summary = {
        "meta": meta,
        "g4": {k: {kind: {m: results[k]["metrics"][kind][m] for m in
                          ("escalation_rate", "auto_execute_rate", "certified_but_wrong_rate", "wrong_given_auto",
                           "certified_steer_rate", "decide_calls")} for kind in ("clean", "inserted")} for k in
               ("g4_exact_k1", "g4_exact_k2", "g4_smooth_k1")},
        "g4_guarantee_checked_units": results["g4_exact_k1"]["metrics"]["guarantee_checked_units"],
        "g5_synthetic": {"lam_hat": results["g5_synthetic"]["calibration"]["lam_hat"],
                         "held_out_template_risk": results["g5_synthetic"]["held_out_template"]["selective_risk"],
                         "held_out_goal_risk": results["g5_synthetic"]["held_out_goal"]["selective_risk"],
                         "held_out_goal_violation_detected": results["g5_synthetic"]["held_out_goal"]["violation_detected"]},
        "g4_uncovered": {arm: {mode: {kk: v["by_mode"][mode][kk] for kk in
                                      ("covered_rate", "certified_and_changed", "certified_and_changed_rate",
                                       "decide_calls_mean")} for mode in v["by_mode"]}
                         for arm, v in results["g4_uncovered"].items() if "by_mode" in v},
        "g4_span_clean_cost": {arm: v.get("clean_cost") for arm, v in results["g4_uncovered"].items()
                               if arm.startswith("span")},
        "g4_sentence_same_subset_clean_cost": results["g4_uncovered"]["sentence_same_subset_clean_cost"],
        "g5_standin_lam_hat": {k: v["lam_hat"] for k, v in results["g5_standin"]["strata"].items()},
    }
    if "controlsafe" in results:
        cs = results["controlsafe"]
        summary["controlsafe"] = {
            "g15_pass": {f: r["pass"] for f, r in cs.get("g15", {}).items()},
            "g4snap": {f: {k: {m: v[m] for m in ("token_savings", "decisions_identical_all", "n_origins")}
                           for k, v in r.items() if k.startswith("k")} for f, r in cs.get("g4snap", {}).items()},
            "g14": {f: {m: r[m] for m in ("auroc", "collapse_ratio")} for f, r in cs.get("g14", {}).items()},
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
