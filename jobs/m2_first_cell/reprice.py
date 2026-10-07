"""Re-price the child evals from the box's MEASURED scoring speed, before any training spend (review fix 1).

    python -X utf8 -m jobs.m2_first_cell.reprice --arm-dir <ck>/parent_arm --params jobs/m2_first_cell/job_params.json \
        --tok-s T --call-s C --out reprice.json

The setup preflight prices every eval phase from the env rates (job-tkfaj's probe), which a slower box does not
change. The parent arm runs first and its live rate guard records how long scoring really took
(``rate-parent-<stage>.json``: minutes from the progress total file to the end). factor = measured / projected scoring
minutes over both parent stages; the child evals (same scoring code, ~1.4x the parent's volume) are then projected
at TOK_S / factor and CALL_S x factor against their cap. Exit 15 (stop before training) if over 95 % of it; 0 otherwise.
A parent stage cached by an earlier attempt keeps its rate report in the arm dir; if none exists the factor is
unmeasured and the env-rate preflight stands (recorded, exit 0).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def factor(arm_dir: Path, params: dict, tok_s: float, call_s: float) -> dict:
    from jobs.m2_parent_arm.score_parent import estimate

    meas = proj = 0.0
    used = {}
    for st in ("g12", "index"):
        f = arm_dir / f"rate-parent-{st}.json"
        if not f.exists():
            return {"factor": None, "reason": f"no rate report for parent {st} (cached by an earlier attempt?)"}
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("exit") != 0 or r.get("scoring_started_min") is None:
            return {"factor": None, "reason": f"parent {st} rate report incomplete: {r.get('exit')}"}
        m = r["elapsed_min"] - r["scoring_started_min"]
        p = estimate(tok_s, stages=[st], fixed_min=0, mode="parent", call_s=call_s)["total_minutes"]
        meas += m
        proj += p
        used[st] = {"measured_scoring_min": m, "projected_scoring_min": p}
    return {"factor": meas / proj if proj > 0 else None, "stages": used}


def main(argv=None) -> int:
    from jobs.m2_parent_arm.score_parent import estimate

    ap = argparse.ArgumentParser(prog="python -X utf8 -m jobs.m2_first_cell.reprice")
    ap.add_argument("--arm-dir", required=True, type=Path)
    ap.add_argument("--params", required=True)
    ap.add_argument("--tok-s", type=float, required=True)
    ap.add_argument("--call-s", type=float, required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    p = json.loads(Path(a.params).read_text(encoding="utf-8"))
    f = factor(a.arm_dir, p, a.tok_s, a.call_s)
    cap = p["phase_caps_minutes"]["evals"]
    rec = {**f, "cap_min": cap, "rule": "child evals projected at TOK_S/factor, CALL_S*factor must be <= 0.95 x cap"}
    if f["factor"] is None:
        rec.update(ok=True, note="factor unmeasured: the setup preflight at the env rates stands")
    else:
        k = max(f["factor"], 1e-9)
        est = estimate(a.tok_s / k, stages=["g12", "index"], fixed_min=p["evals_fixed_minutes"]["child"], mode="child",
                       call_s=a.call_s * k)
        rec.update(child_projection=est, ok=est["total_minutes"] <= 0.95 * cap)
    text = json.dumps(rec, indent=1)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    return 0 if rec["ok"] else 15


if __name__ == "__main__":
    sys.exit(main())
