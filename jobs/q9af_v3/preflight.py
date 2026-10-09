"""jobs/q9af_v3 phase projections from MEASURED rates (v2 receipts: job-5cmat, job-2xe5w, job-4j3vu) + the stated
estimates, and the margin rule (the runner calls this at setup; exit 5 if any check fails).

    python -m jobs.q9af_v3.preflight --params jobs/q9af_v3/job_params_A.json --config configs/q9af_v3_h100.json [--out f]

Mode A (job_params_A): G = load + prompt selection (uncapped) + the generation wall clock (capped by gen.max_minutes);
  P = capped (stage 1 + kd_minutes + one step of overshoot per stage) + uncapped (load, probe of 8 single-layer
  held-out evals, per-stage entry + periodic evals, final held-out + generated-completion KL).
Mode B (job_params_B): F = artifact downloads; P = capped (long-context + on-policy wall clocks + one step each) +
  uncapped (load, entry / final evals incl. the 32k windows, per-stage evals, throughput both arms);
  L arms = the v3w readout's exact proxy600 fork volume (jobs/q9af_v3/readout_volume_v3w_proxy600.json) at the measured
  stock prefill rate and the larger per-call cost (jobs.common.readout_cost.project).
Rules: every capped phase holds capped + (1 + m) x uncapped; every other cap >= (1 + m) x projection (L arms against
95 % of the cap, as rate_guard stops there); rate floors <= (1 - m) x the slowest predicted rate; peak <= (1 - m) x the
stop; data budgets (short-region tokens, reserved 32k blocks, on-policy prompts) cover (1.15 x) the predicted use;
caps + reserve == max_duration; worst case <= hard cap.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _evals(minutes: float, every: float) -> int:
    return int(minutes // every) + 2  # periodic points + the stop point + the stage-entry point


def project_a(p: dict, cfg: dict) -> dict:
    m, e = p["measured"], p["estimates_not_measured"]
    B, L, dc, gc = cfg["batch"], cfg["seq_len"], cfg["distill"], cfg["gen"]
    from convert.distill import conversion_plan
    from jobs.common.readout_cost import project as readout

    n_stages = math.ceil(len(cfg["attention_layers"]) / dc["group_size"])
    plan = conversion_plan({i: 1.0 for i in cfg["attention_layers"]}, dc["group_size"], dc["kd_minutes"],
                           dc["floor_minutes"])
    s2 = m["stage2_steady_tok_s_layer3_7gdn"]
    capped = cfg["stage1"]["guard"]["max_minutes"] + dc["kd_minutes"] \
        + (B * L / m["stage1_steady_tok_s"] + n_stages * B * L / s2) / 60
    n_ev = sum(_evals(s["minutes"], dc["eval_every_min"]) for s in plan["stages"])
    n_layers = len(cfg["attention_layers"])
    # one model load + data per stage invocation; the stock recall probe; per layer KL + recall; per stage KL evals
    # + the stage-end recall probe; final held-out + generated-completion KL
    uncapped = (n_stages * m["load_and_data_s"] / 60 + e["recall_eval_min"]
                + n_layers * (e["curve_eval_min_each"] + e["recall_eval_min"])
                + n_ev * e["curve_eval_min_each"] + n_stages * e["recall_eval_min"] + e["final_heldout_min"]
                + e["gen_heldout_eval_min"] + e["teacher_cache_min"])
    g_capped = gc["max_minutes"] + gc["max_new"] * e["gen_step_s"] / 60  # the last batch started before the cap
    g_uncapped = m["load_and_data_s"] / 60 + e["prompt_select_min"] + e["gen_pack_write_min"]
    short_tokens = cfg["mix"]["train_tokens"] - cfg["lc"]["blocks"] * cfg["lc"]["seq_len"]
    gfrac = cfg["mix_fracs"]["generic"]
    need = {"stage1": 1.15 * m["stage1_steady_tok_s"] * 60 * cfg["stage1"]["guard"]["max_minutes"],
            "kd_generic": 1.15 * m["stage2_steady_tok_s_layer23"] * 60 * dc["kd_minutes"] * gfrac}
    v3w = json.loads((ROOT / p["v3w_volume"]).read_text(encoding="utf-8"))
    arm = readout(v3w["totals"], m["stock_prefill_tok_s_min"], max(m["pack_call_s"], m["small_call_s"]), "fork")
    return {"phases": {"setup": m["setup_min"], "weights": m["weights_fetch_s"] / 60, "D": e["D_min"],
                       "X": e["export_X_min"], "I_stock": arm["projected_min"], "S": n_stages * m["S_min"] / 2},
            "arms": {"I_stages": (n_stages, arm["projected_min"])},
            "capped": {"G": (g_capped, g_uncapped), "P": (capped, uncapped)},
            "floors": {"stage1": (cfg["stage1"]["guard"]["min_tok_s"], m["stage1_steady_tok_s"]),
                       "distill": (dc["guard"]["min_tok_s"], s2)},
            "peaks": {"stage1": cfg["stage1"]["guard"]["max_peak_gb"], "distill": dc["guard"]["max_peak_gb"]},
            "data": {"short_region_tokens": (short_tokens, sum(need.values()))},
            "detail": {"n_stage_evals": n_ev, "short_tokens_needed": need, "v3w_arm": arm,
                       "gen_completion_tokens_at_estimate": e["gen_tok_s"] * 60 * gc["max_minutes"]}}


def project_s(p: dict, cfg: dict) -> dict:
    """The cheap sensitivity job: stage 1 + 8 single-layer KD runs + the base-width controls (all capped) and the
    stock / per-layer recall probes + KL evals (uncapped)."""
    m, e, sc = p["measured"], p["estimates_not_measured"], cfg["sens"]
    B, L = cfg["batch"], cfg["seq_len"]
    n = len(cfg["attention_layers"]) + sc["control_layers"]
    s2 = m["stage2_steady_tok_s_layer3"]
    capped = cfg["stage1"]["guard"]["max_minutes"] + n * sc["minutes_per_layer"] \
        + sc["control_layers"] * sc["control_stage1_minutes"] + (n + 1) * B * L / min(s2, m["stage1_steady_tok_s"]) / 60
    uncapped = (m["load_and_data_s"] / 60 + e["recall_eval_min"] + n * (e["recall_eval_min"] + e["curve_eval_min_each"])
                + e["teacher_cache_min"])
    gfrac = cfg["mix_fracs"]["generic"]
    need = 1.15 * (m["stage1_steady_tok_s"] * 60 * cfg["stage1"]["guard"]["max_minutes"]
                   + m["stage2_steady_tok_s_layer23"] * 60 * n * sc["minutes_per_layer"] * gfrac)
    return {"phases": {"setup": m["setup_min"], "weights": m["weights_fetch_s"] / 60, "D": e["D_min"]},
            "arms": {}, "capped": {"P": (capped, uncapped)},
            "floors": {"stage1": (cfg["stage1"]["guard"]["min_tok_s"], m["stage1_steady_tok_s"]),
                       "distill": (cfg["distill"]["guard"]["min_tok_s"], s2)},
            "peaks": {"stage1": cfg["stage1"]["guard"]["max_peak_gb"]},
            "data": {"train_tokens": (cfg["mix"]["train_tokens"], need)},
            "detail": {"single_layer_runs": n}}


def project_b(p: dict, cfg: dict) -> dict:
    from jobs.common.readout_cost import project as readout

    m, e = p["measured"], p["estimates_not_measured"]
    lc, op = cfg["long_context"], cfg["onpolicy"]
    lc_min, op_min = lc["guard"]["max_minutes"], op["guard"]["max_minutes"]
    capped = lc_min + op_min + (lc["batch"] * lc["seq_len"] / e["lc_tok_s"] + e["op_step_s"]) / 60
    n_ev = _evals(lc_min, lc["eval_every_min"]) + _evals(op_min, op["eval_every_min"])
    uncapped = (m["load_and_data_s"] / 60 + 2 * (e["final_heldout_min"] + e["long_eval_min"] + e["gen_heldout_eval_min"])
                + n_ev * e["curve_eval_min_each"] + 2 * m["throughput_s_per_arm"] / 60 + e["teacher_cache_min"])
    v3w = json.loads((ROOT / p["v3w_volume"]).read_text(encoding="utf-8"))
    arm = readout(v3w["totals"], m["stock_prefill_tok_s_min"], max(m["pack_call_s"], m["small_call_s"]), "fork")
    op_steps = op_min * 60 / e["op_step_s"]
    return {"phases": {"setup": m["setup_min"], "weights": m["weights_fetch_s"] / 60, "F": e["fetch_min"],
                       "X": e["export_X_min"], "L_stock": arm["projected_min"], "L_v3": arm["projected_min"],
                       "L_v2": arm["projected_min"], "S": m["S_min"]},
            "arms": {}, "capped": {"P": (capped, uncapped)},
            "floors": {"long_context": (lc["guard"]["min_tok_s"], e["lc_tok_s"]),
                       "onpolicy": (op["guard"]["min_tok_s"], op["rollout_batch"] * e["op_completion_tokens_per_rollout"]
                                    / e["op_step_s"])},
            "peaks": {"long_context": lc["guard"]["max_peak_gb"], "onpolicy": op["guard"]["max_peak_gb"]},
            "data": {"lc_tokens": (cfg["lc"]["blocks"] * cfg["lc"]["seq_len"], 1.15 * e["lc_tok_s"] * 60 * lc_min),
                     "onpolicy_prompts": (cfg["gen"]["onpolicy"]["n_prompts"], 1.15 * op_steps * op["rollout_batch"])},
            "detail": {"v3w_arm": arm, "n_stage_evals": n_ev, "onpolicy_steps": op_steps}}


def check(params: dict, cfg: dict) -> dict:
    mode = params["v3_mode"]
    pr = {"A": project_a, "B": project_b, "S": project_s}[mode](params, cfg)
    mf, caps = params["margin_min_frac"], params["phase_caps_minutes"]
    rows = {}
    for ph, proj in pr["phases"].items():
        usable = caps[ph] * (0.95 if ph.startswith(("L_", "I_")) else 1.0)
        rows[ph] = {"projected_min": round(proj, 2), "cap_min": caps[ph], "ok": usable >= (1 + mf) * proj}
    for ph, (n, proj) in pr["arms"].items():  # n guarded proxy600 arms sharing one cap (rate_guard stops at 95 %)
        rows[ph] = {"arms": n, "projected_min_each": round(proj, 2), "cap_min": caps[ph],
                    "ok": 0.95 * caps[ph] / n >= (1 + mf) * proj}
    for ph, (cp, un) in pr["capped"].items():
        need = cp + (1 + mf) * un
        rows[ph] = {"capped_min": round(cp, 2), "uncapped_min": round(un, 2), "need_min": round(need, 2),
                    "cap_min": caps[ph], "ok": caps[ph] >= need}
    for k, (floor, rate) in pr["floors"].items():
        rows[f"floor_{k}"] = {"min_tok_s": floor, "slowest_rate_tok_s": round(rate), "ok": floor <= (1 - mf) * rate}
    for k, stop in pr["peaks"].items():
        rows[f"peak_{k}"] = {"measured_gb": params["measured"]["peak_gb"], "stop_gb": stop,
                             "ok": params["measured"]["peak_gb"] <= (1 - mf) * stop}
    for k, (have, need) in pr["data"].items():
        rows[f"data_{k}"] = {"have": have, "need": round(need), "ok": have >= need}
    sched = sum(caps.values()) + params["reserve_minutes"]
    rows["schedule"] = {"sum_caps_plus_reserve": sched, "max_duration": params["max_duration_minutes"],
                        "ok": sched == params["max_duration_minutes"]}
    from jobs.common.costs import worst_case_usd

    rows["hard_cap"] = {"worst_case_usd": worst_case_usd(params), "hard_cap_usd": params["hard_cap_usd"],
                        "ok": worst_case_usd(params) <= params["hard_cap_usd"]}
    return {"ok": all(r["ok"] for r in rows.values()), "mode": mode, "checks": rows, "detail": pr["detail"]}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jobs.q9af_v3.preflight")
    ap.add_argument("--params", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs/q9af_v3_h100.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = check(json.loads(Path(a.params).read_text(encoding="utf-8")),
                json.loads(Path(a.config).read_text(encoding="utf-8")))
    txt = json.dumps(res, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)
    return 0 if res["ok"] else 5


if __name__ == "__main__":
    sys.exit(main())
