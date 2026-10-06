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
        sc.append(hd["executed_conf"])
        wr.append(hd["executed"] != v["_meta"]["gold"])
        fam.append(v["_meta"]["attack_family"])
        strata.append(f"{v['_meta']['family']}|{'clean' if fam[-1] == 'clean' else 'attacked'}")
        cl.append(v["_meta"]["cluster"])
    sc, wr = np.array(sc), np.array(wr)
    cal = selective.calibrate_strata(sc, wr, strata, clusters=cl, alpha=alpha, delta=delta)
    # pre-registered shift caveat, demonstrated: calibrate on clean units, evaluate on attacked (held-out goal)
    clean = np.array([f == "clean" for f in fam])
    on_clean = selective.ltt_selective(sc[clean], wr[clean], alpha, delta, clusters=[c for c, m in zip(cl, clean) if m])
    stress = selective.stress_test(on_clean["lam_hat"], sc[~clean], wr[~clean], alpha, delta, label="held_out_goal")
    return {"strata": {k: {kk: v[kk] for kk in ("n_units", "n_s", "n_clusters", "mu_s", "feasibility", "lam_hat",
                                                  "chosen", "unit_level_unsafe")} for k, v in cal.items()},
            "clean_calibrated": {kk: on_clean[kk] for kk in ("n_s", "lam_hat", "chosen", "feasibility")},
            "stress_attacked": stress, "risk_coverage_all": selective.risk_coverage(sc, wr)}


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
    results["g5_standin"] = g5_standin(vs, a.alpha, a.delta)
    results["g5_synthetic"] = g5_synthetic(a.alpha, a.delta)
    for name, obj in results.items():
        (out / f"{name}.json").write_text(json.dumps({"meta": meta, name: obj}, indent=1, default=float),
                                          encoding="utf-8", newline="\n")
    data.write_jsonl(rows1, out / "g4_exact_k1_rows.jsonl")
    summary = {
        "meta": meta,
        "g4": {k: {kind: {m: results[k]["metrics"][kind][m] for m in
                          ("escalation_rate", "auto_execute_rate", "certified_but_wrong_rate", "certified_steer_rate",
                           "decide_calls")} for kind in ("clean", "inserted")} for k in
               ("g4_exact_k1", "g4_exact_k2", "g4_smooth_k1")},
        "g4_guarantee_checked_units": results["g4_exact_k1"]["metrics"]["guarantee_checked_units"],
        "g5_synthetic": {"lam_hat": results["g5_synthetic"]["calibration"]["lam_hat"],
                         "held_out_template_risk": results["g5_synthetic"]["held_out_template"]["selective_risk"],
                         "held_out_goal_risk": results["g5_synthetic"]["held_out_goal"]["selective_risk"],
                         "held_out_goal_violation_detected": results["g5_synthetic"]["held_out_goal"]["violation_detected"]},
        "g5_standin_lam_hat": {k: v["lam_hat"] for k, v in results["g5_standin"]["strata"].items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
