"""Measured-rate preflight of a q9af_cell mode (the runner calls it at setup; exit 5 on any violation, before any GPU
work): every phase cap >= (1 + margin_min_frac) x its projection (jobs/q9af_cell/price.py: job-5cmat's measured 9B
rates, the act-ckpt projection, stated assumptions); caps + reserve == max_duration <= MAX_DURATION_CEILING (org
ceiling 720); worst case <= hard cap; build_timeout >= 60; the projected peak memory at the largest ladder rung fits.

    python -m jobs.q9af_cell.preflight --mode parent|cell_r64|cell_r256 [--out preflight.json]

The live stops stay authoritative on the box: run_cell's ladder + 10-step rate checks (exit 26 when min_steps is out of
reach at the MEASURED step time), the per-phase timeouts (exit 7), and the export pin check (exit 10).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODES = ("parent", "cell_r64", "cell_r256")
MAX_DURATION_CEILING = 720


def check(params: dict, cell_cfg: dict | None) -> dict:
    from jobs.common.costs import worst_case_usd
    from jobs.q9af_cell import price as P

    mf = params["margin_min_frac"]
    proj = P.projections(params, cell_cfg)
    rows = {}
    for ph, cap in params["phase_caps_minutes"].items():
        p = proj[ph]["projected_min"]
        rows[ph] = {"projected_min": round(p, 1), "cap_min": cap, "margin_frac": round(cap / p - 1, 3),
                    "ok": cap >= (1 + mf) * p}
    sched = sum(params["phase_caps_minutes"].values()) + params["reserve_minutes"]
    rows["schedule"] = {"sum_caps_plus_reserve": sched, "max_duration": params["max_duration_minutes"],
                        "ok": sched == params["max_duration_minutes"] <= MAX_DURATION_CEILING}
    w = worst_case_usd(params)
    rows["worst_case"] = {"usd": w, "hard_cap_usd": params["hard_cap_usd"], "ok": w <= params["hard_cap_usd"]}
    rows["build_timeout"] = {"minutes": params["build_timeout_minutes"], "ok": params["build_timeout_minutes"] >= 60}
    if cell_cfg is not None:
        T = max(cell_cfg["t_ladder"])
        mem = P.memory_gb(cell_cfg["lora_rank"], T, K=cell_cfg["max_k"], ckpt=bool(cell_cfg.get("act_ckpt")))
        rows["memory_top_rung"] = {"T": T, **mem, "ok": mem["fits"] and bool(cell_cfg.get("act_ckpt"))}
        c = proj["cell"]
        rows["min_steps_reachable"] = {"min_steps": c["min_steps"], "projected_steps_in_budget": c["steps_in_budget"],
                                       "ok": c["steps_in_budget"] >= 1.3 * (c["min_steps"] or 0)}
    return {"ok": all(r["ok"] for r in rows.values()), "checks": rows, "projections": proj,
            "assumptions": P.ASSUMPTIONS, "measured": P.MEASURED}


def load(mode: str) -> tuple[dict, dict | None]:
    params = json.loads((HERE / f"job_params_{mode}.json").read_text(encoding="utf-8"))
    cfg = json.loads((ROOT / params["cell_config"]).read_text(encoding="utf-8")) if mode != "parent" else None
    return params, cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="jobs.q9af_cell.preflight")
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = check(*load(a.mode))
    txt = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    print(json.dumps({"ok": res["ok"], "failed": [k for k, v in res["checks"].items() if not v["ok"]]}))
    return 0 if res["ok"] else 5


if __name__ == "__main__":
    sys.exit(main())
