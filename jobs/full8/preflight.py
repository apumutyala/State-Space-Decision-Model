"""full8 phase projections from MEASURED job-5cmat timings (+ the job-4j3vu layer-3 stage-2 rate) and the margin rule.

    python -m jobs.full8.preflight --params jobs/full8/job_params.json --config configs/full8_h100.json [--out f.json]

Pricing model (every input is a receipt in job_params.json:measured, except X, which is labelled an estimate):
  P = pre-progressive work (load + data, stock probes, stage 1, teacher cache, KL importance; each measured once on
      5cmat and independent of how many layers are converted)
    + 8 x one progressive step: the stage-2 wall-clock cap (identical for every layer: the cap is minutes, so a slower
      later layer trains fewer tokens, not longer) + one optimiser step of overshoot at that layer's predicted rate
      + gate evaluate / checkpoint save (5cmat: progressive - stage 2 - converted probes) + converted probes
    + throughput (stock and converted, both call latencies)
  I arms: the runner's own volume preflight at the measured 5cmat inputs (min prefill tok/s over lengths, the larger
      of the two per-call costs) -- conservative: it projected 22.5 min where 5cmat measured 10.0 / 9.0.
Per-layer stage-2 rates are predicted with convert.memmodel.flops_stage2 at the lower of the two measured stage-2 MFUs
(0.341 on layer 3 / 4j3vu, 0.343 on layer 23 / 5cmat: the FLOP model reproduces both within 1%), for the worst case
(first converted layer 3, all 8 GDNs training) and for 5cmat's measured order (tokens per layer, informational).

Rules (exit 5 if any fails; the runner calls this at setup): every cap >= (1 + margin) x projection; the deadline
covers the start of the 8th layer with margin AND a layer started just before the deadline (with margin) still fits the
P cap; the stage floors are <= (1 - margin) x the slowest predicted rate; the predicted stage-2 peak <= (1 - margin) x
the stop; the phase caps + reserve == max_duration.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N_LAYERS = 8
ORDER_5CMAT = [23, 31, 15, 11, 7, 19, 3, 27]


def stage2_rates(m: dict, order=ORDER_5CMAT, micro_tokens=16384) -> dict:
    from convert import memmodel as mm

    d = mm.QWEN35_9B
    mfu = min(m["stage2_mfu_layer3"], m["stage2_mfu_layer23"])
    worst_f = mm.flops_stage2(d, d.attn_layers, n_train=N_LAYERS)
    out = {"mfu_used": mfu, "worst": {"flop_per_token": worst_f, "tok_s": mfu * mm.H100_PEAK / worst_f,
                                      "peak_gb": mm.stage2_peak(d, micro_tokens, min(d.attn_layers),
                                                                n_attached=N_LAYERS, n_train=N_LAYERS) / 1e9},
           "per_step_5cmat_order": []}
    conv = []
    for i in order:
        conv = conv + [i]
        f = mm.flops_stage2(d, conv, n_train=len(conv))
        out["per_step_5cmat_order"].append({"layer": i, "first_student_layer": min(conv), "flop_per_token": f,
                                            "tok_s": mfu * mm.H100_PEAK / f})
    # FLOP-model check against the two measured layers (same MFU within 1%)
    out["model_check"] = {"layer23_mfu": m["stage2_steady_tok_s_layer23"] * mm.flops_stage2(d, [23]) / mm.H100_PEAK,
                          "layer3_mfu": m["stage2_steady_tok_s_layer3"] * mm.flops_stage2(d, [3]) / mm.H100_PEAK}
    return out


def project(params: dict, cfg: dict) -> dict:
    m = params["measured"]
    L, B = cfg["seq_len"], cfg["batch"]
    s2min = cfg["stage2"]["guard"]["max_minutes"]
    rates = stage2_rates(m)
    pre = (m["load_and_data_s"] + m["stock_probes_s"] + m["stage1_s"] + m["gate_teacher_cache_s"]
           + m["importance_s"]) / 60
    step_overhead = (m["progressive_one_layer_s"] - m["stage2_s"] - m["converted_probes_s"]) / 60  # evaluate + save
    overshoot = B * L / rates["worst"]["tok_s"] / 60  # the guard stops after the step that crosses the cap
    layer = s2min + overshoot + step_overhead + m["converted_probes_s"] / 60
    thr = 2 * m["throughput_s"] / 60  # the converted arm's call latency now runs too (it failed on 5cmat)
    mv = params["estimates_not_measured"]["mmlu_full"]  # A1: full MMLU, stock + final, after the 8th layer
    mmlu = mv["arms"] * (mv["positions_per_arm"] / m["stock_prefill_tok_s_min"] / 60
                         + mv["calls_per_arm"] * m["pack_call_s"] / 60 + mv["tokenise_min_per_arm_est"])
    thr = thr + mmlu  # post-progressive work (never deadline-gated)
    P = pre + N_LAYERS * layer + thr
    from jobs.common.readout_cost import project as readout

    vol = json.loads((ROOT / "jobs/common/readout_volume_qwen3.5.json").read_text(encoding="utf-8"))
    call = max(m["pack_call_s"], m["small_call_s"])
    arm = readout(vol["totals"], m["stock_prefill_tok_s_min"], call, "fork")["projected_min"]
    phases = {"setup": m["setup_min"], "weights": m["weights_fetch_s"] / 60, "P": P,
              "X": params["estimates_not_measured"]["export_X_min"], "I_stock": arm, "I_conv": arm, "S": m["S_min"]}
    measured_like = {**phases, "I_stock": m["I_stock_min"], "I_conv": m["I_conv_min"]}
    return {"phases": phases, "phases_measured_basis": measured_like,
            "P_parts": {"pre_progressive": pre, "per_layer": layer, "per_layer_stage2_cap": s2min,
                        "per_layer_overshoot": overshoot, "per_layer_eval_save": step_overhead,
                        "per_layer_probes": m["converted_probes_s"] / 60, "throughput_and_mmlu_full": thr,
                        "mmlu_full_both_arms": mmlu,
                        "stage2_total": N_LAYERS * (s2min + overshoot)},
            "last_layer_start": pre + (N_LAYERS - 1) * layer, "layer_min": layer, "throughput_min": thr,
            "stage2_rates": rates,
            "stage2_tokens_5cmat_order": [r["tok_s"] * 60 * s2min for r in rates["per_step_5cmat_order"]],
            "estimates_not_measured": ["X (export)", "throughput of the converted arm's call latency (5cmat failed it)"]}


def check(params: dict, cfg: dict) -> dict:
    pr = project(params, cfg)
    mf = params["margin_min_frac"]
    caps = params["phase_caps_minutes"]
    m = params["measured"]
    rows = {}
    for ph, proj in pr["phases"].items():
        usable = caps[ph] * (0.95 if ph.startswith("I_") else 1.0)  # rate_guard stops an Index arm at 95% of its cap
        rows[ph] = {"projected_min": round(proj, 2), "cap_min": caps[ph], "margin_frac": round(usable / proj - 1, 3),
                    "ok": usable >= (1 + mf) * proj}
    dl = params["pipeline_deadline_minutes"]
    late = dl + (1 + mf) * (pr["layer_min"] + pr["throughput_min"])
    rows["P_deadline"] = {"last_layer_start_min": round(pr["last_layer_start"], 2), "deadline_min": dl,
                          "late_start_finish_min": round(late, 2), "P_cap": caps["P"],
                          "ok": dl >= (1 + mf) * pr["last_layer_start"] and late <= caps["P"]}
    w = pr["stage2_rates"]["worst"]
    floors = {"stage1": (cfg["stage1"]["guard"]["min_tok_s"], m["stage1_steady_tok_s"]),
              "stage2": (cfg["stage2"]["guard"]["min_tok_s"], w["tok_s"])}
    for k, (floor, rate) in floors.items():
        rows[f"floor_{k}"] = {"min_tok_s": floor, "slowest_rate_tok_s": round(rate), "ok": floor <= (1 - mf) * rate}
    stop = cfg["stage2"]["guard"]["max_peak_gb"]
    rows["peak"] = {"measured_gb": m["peak_gb"], "predicted_worst_gb": round(w["peak_gb"], 1), "stop_gb": stop,
                    "ok": max(m["peak_gb"], w["peak_gb"]) <= (1 - mf) * stop}
    mc = pr["stage2_rates"]["model_check"]
    rows["flop_model"] = {**{k: round(v, 4) for k, v in mc.items()},
                          "ok": abs(mc["layer23_mfu"] - m["stage2_mfu_layer23"]) < 0.01
                          and abs(mc["layer3_mfu"] - m["stage2_mfu_layer3"]) < 0.01}
    rows["layers"] = {"max_layers": cfg["progressive"]["max_layers"], "ok": cfg["progressive"]["max_layers"] == N_LAYERS}
    sched = sum(caps.values()) + params["reserve_minutes"]
    rows["schedule"] = {"sum_caps_plus_reserve": sched, "max_duration": params["max_duration_minutes"],
                        "ok": sched == params["max_duration_minutes"]}
    return {"ok": all(r["ok"] for r in rows.values()), "checks": rows, **pr}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jobs.full8.preflight")
    ap.add_argument("--params", default=str(ROOT / "jobs/full8/job_params.json"))
    ap.add_argument("--config", default=str(ROOT / "configs/full8_h100.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = check(json.loads(Path(a.params).read_text(encoding="utf-8")),
                json.loads(Path(a.config).read_text(encoding="utf-8")))
    txt = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)
    return 0 if res["ok"] else 5


if __name__ == "__main__":
    sys.exit(main())
