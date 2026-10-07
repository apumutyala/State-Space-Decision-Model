"""Static preflight of a feature-cache mode: every phase cap >= (1 + margin_min_frac) x its projection from MEASURED
rates and the EXACT committed volume; schedule = caps + reserve == max_duration; worst case <= hard cap.

    python -m jobs.q9_feature_cache.preflight --mode stock|extra [--out preflight.json]

Arm projection (minutes) = rows/tokenisation on the box (``TOKENIZE_MIN``, measured locally: 10.2-14.2 min per arm for
the real draw + both DEV splits, three arms in parallel on a laptop CPU) + model load (job-5cmat 43 s) + the exact
extraction volume (``volume_plan.json``: positions / tok_s_min + calls x call_s_fitted_5cmat). The runner calls this at
setup (exit 5 on violation); the in-job preflight of jobs/q9_feature_cache/cache.py (exact volume at the same rates,
exit 15) and the live weighted rate guard stay authoritative.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
#: tokenisation + draw + guard checks before the model loads, per arm (measured locally, see docstring)
TOKENIZE_MIN = 14.2
#: phases other than the arms: projection = their expected_minutes (each from a receipt in job_params / JOB_SPEC)


def project(params: dict, volume: dict) -> dict:
    m = params["measured"]
    tok_s, call_s = m["stock_prefill_tok_s_min"], m["call_s_fitted_5cmat"]
    out = {}
    for arm in params["arms"]:
        v = volume["arms"][arm]
        ext = sum(v[s]["positions"] / tok_s + v[s]["calls"] * call_s for s in ("select", "holdout", "train")) / 60
        out[f"arm_{arm}"] = {"extract_min": round(ext, 2), "tokenize_min": TOKENIZE_MIN,
                             "load_min": round(m["model_load_s"] / 60, 2),
                             "projected_min": round(ext + TOKENIZE_MIN + m["model_load_s"] / 60, 2)}
    return out


def check(params: dict, volume: dict) -> dict:
    from jobs.common.costs import worst_case_usd

    mf = params["margin_min_frac"]
    caps, exp = params["phase_caps_minutes"], params["expected_minutes"]
    proj = project(params, volume)
    rows = {}
    for ph, cap in caps.items():
        p = proj[ph]["projected_min"] if ph in proj else exp[ph]
        rows[ph] = {"projected_min": p, "cap_min": cap, "margin_frac": round(cap / p - 1, 3), "ok": cap >= (1 + mf) * p}
    sched = sum(caps.values()) + params["reserve_minutes"]
    rows["schedule"] = {"sum_caps_plus_reserve": sched, "max_duration": params["max_duration_minutes"],
                        "ok": sched == params["max_duration_minutes"]}
    w = worst_case_usd(params)
    rows["worst_case"] = {"usd": w, "hard_cap_usd": params["hard_cap_usd"], "ok": w <= params["hard_cap_usd"]}
    rows["build_timeout"] = {"minutes": params["build_timeout_minutes"], "ok": params["build_timeout_minutes"] >= 60}
    return {"ok": all(r["ok"] for r in rows.values()), "checks": rows, "arms": proj}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="jobs.q9_feature_cache.preflight")
    ap.add_argument("--mode", required=True, choices=("stock", "extra"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    params = json.loads((HERE / f"job_params_{a.mode}.json").read_text(encoding="utf-8"))
    volume = json.loads((HERE / "volume_plan.json").read_text(encoding="utf-8"))
    res = check(params, volume)
    txt = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)
    return 0 if res["ok"] else 5


if __name__ == "__main__":
    sys.exit(main())
