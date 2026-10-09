"""BOARD H1: compose the hijack (+ steer re-measurement) result of a SAVED cell checkpoint (``hijack_eval_result.json``,
the decision file ``finalize`` validates for ``first_cell_job.sh hijack_eval``). CPU only; no model.

    python -X utf8 -m jobs.m2_first_cell.hijack_eval --cell <cell.json> --report <out>/report.json \
        --parent-arm <ck>/parent_arm_hijack --params jobs/m2_first_cell/job_params_hijack_eval.json \
        --elapsed-min N --out <out>/hijack_eval_result.json

What is measured (no metric is defined here; every number is ``safety_eval.metrics.compare_arms`` through
``train.m2_gates.hijack_measurement``, the same paired function ``summarize.steer_hijack`` and
``baselines/run_clef_safety`` land with):
  * hijack: ``safety_eval.attacks.HIJACK_TEMPLATES`` (five templates, two of them role-token spoofs), success = the
    free (raw) argmax is the disallowed attacker option (``metrics.run``: ``hijack``); a refused unit is not hijacked.
    Paired child - parent, eligibility ``reference_y`` (parent clean-correct; the hijack rate itself is over every
    observed unit, as compare_arms defines it). The eval hijack templates share no word 6-gram with any training text
    (``data.injection.FORBIDDEN_6GRAMS``), so this is OUT-of-template-distribution for the trained cell.
  * steer (re-measurement): D1-A, the same population rule; D1-A is in the training data verbatim (PR #44), so it is
    in-distribution robustness. Held-out D1-B (G13) is never built here.
Units: ``train.m2_gates.hijack_subset`` of cohort ``steer_dev`` (8 per cluster x 45 clusters = 360 requests) x
(clean + 5 hijack + 4 D1-A) = 3,600 units per arm. Both arms are scored in this job, untempered.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "ssdm.m2.hijack_eval.v1"


def _load(p) -> Any:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def measurement(child_rows: list, parent_rows: list, cohort, n_boot: int) -> dict:
    from jobs.m2_first_cell.summarize import keep_compare
    from train import m2_gates as G

    variants = G.stage_variants(cohort, "hijack")
    m = G.hijack_measurement(child_rows, parent_rows, variants, n_boot)
    reqs = {v["request_id"] for v in variants if v["_meta"]["attack_family"] == "clean"}
    out: dict[str, Any] = {
        "units": {"n_requests": len(reqs), "n_clusters": len({v["_meta"]["cluster"] for v in variants}),
                  "per_cluster": G.HIJACK_PER_CLUSTER, "subset_salt": G.HIJACK_SUBSET_SALT,
                  "n_units_per_arm": len(variants),
                  "by_family": dict(Counter(v["_meta"]["attack_family"] for v in variants))},
        "complete": m["complete"]}
    for fam, key in (("hijack", "hijack"), ("d1_A", "steer_d1a")):
        f = m["families"][fam]
        blk = keep_compare(f["two_way"])
        blk["fixed_template"] = {k: f["fixed_template"][k]["diff"] for k in ("hijack_rate", "steer_rate")}
        blk["per_template"] = f["per_template"]
        out[key] = blk
    out["hijack"]["note"] = ("x = child, y = parent; hijack_rate = raw argmax is the disallowed attacker option over "
                             "every observed unit (refused = not hijacked); eval hijack templates are disjoint from "
                             "training text (data.injection.FORBIDDEN_6GRAMS)")
    out["steer_d1a"]["note"] = ("x = child, y = parent; D1-A (development split) on the hijack subset; eligibility = "
                                "parent clean-correct; in-distribution (D1-A is in the training data verbatim)")
    return out


def summarize(a) -> dict:
    from jobs.common.costs import quoted_max_cost_usd, worst_case_usd
    from train.cohort import Cohort

    cell, rep, params = _load(a.cell), _load(a.report), _load(a.params)
    child = _load(Path(a.report).with_name("units-child-hijack.json"))
    par = _load(Path(a.parent_arm) / "units-hijack.json")
    cohort = Cohort(expected_sha256=cell["config"]["cohort_sha256"])
    usd = params["batch_usd_per_min"]
    n = len(child["steer_rows"])
    out = {
        "schema": SCHEMA,
        "checkpoint": {**{k: v for k, v in params["checkpoint"].items() if not k.startswith("_")},
                       "loaded": (rep.get("identities") or {}).get("checkpoint")},
        "cell": {k: cell["config"].get(k) for k in ("parent", "head", "loss_arm", "lora_rank", "template", "layout",
                                                     "n_train_records", "max_steps", "seed")},
        "training_steps": (cell.get("train") or {}).get("steps"),
        "readout_requested": params.get("readout") or "training",
        "readout": (rep.get("identities") or {}).get("readout"),
        **measurement(child["steer_rows"], par["steer_rows"], cohort, a.n_boot),
        "scoring_seconds": {"child": child.get("seconds"), "parent": par.get("seconds"),
                            "child_s_per_unit": (child.get("seconds") or 0) / n if n else None,
                            "parent_s_per_unit": (par.get("seconds") or 0) / max(len(par["steer_rows"]), 1)},
        "m2_gates_exit": rep.get("exit_code"),
        "cost": {"elapsed_min": a.elapsed_min, "runtime_usd": round(a.elapsed_min * usd, 2), "usd_per_min": usd,
                 "build_minutes_not_included": True, "worst_case_usd": worst_case_usd(params),
                 "quoted_max_cost_usd": quoted_max_cost_usd(params), "hard_cap_usd": params["hard_cap_usd"]},
        "verdict": "measured",
    }
    return out


def valid_result(d: Any) -> bool:
    """``jobs/common/finalize.py`` validator: a complete, self-consistent hijack measurement."""
    try:
        fin = lambda x: isinstance(x, (int, float)) and math.isfinite(x)  # noqa: E731
        h, s = d["hijack"], d["steer_d1a"]
        return (d.get("schema") == SCHEMA and d.get("verdict") == "measured" and d["complete"] is True
                and d["units"]["n_units_per_arm"] > 0 and h["n_units"] > 0 and s["n_units"] > 0
                and all(fin(h["hijack_rate"][k]) for k in ("child", "parent"))
                and all(fin(h["hijack_rate"]["diff"][k]) for k in ("point", "lo", "hi"))
                and fin(s["steer_rate"]["diff"]["point"]) and d["m2_gates_exit"] == 46
                and bool(d["checkpoint"].get("ckpt_sha256"))
                and (d["checkpoint"].get("loaded") or {}).get("sha256") == d["checkpoint"]["ckpt_sha256"]
                and (d.get("readout_requested", "training") == "training"
                     or (d.get("readout") or {}).get("requested") == d["readout_requested"]))
    except Exception:  # noqa: BLE001
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -X utf8 -m jobs.m2_first_cell.hijack_eval")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--parent-arm", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--elapsed-min", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    res = summarize(a)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    h, s = res["hijack"], res["steer_d1a"]
    print(json.dumps({"hijack": {k: h["hijack_rate"][k] for k in ("child", "parent")},
                      "hijack_diff": h["hijack_rate"]["diff"], "steer_diff": s["steer_rate"]["diff"],
                      "valid": valid_result(res)}, default=float))
    return 0 if valid_result(res) else 9


if __name__ == "__main__":
    sys.exit(main())
