"""probe9b phase projections from MEASURED rates (job-4j3vu, issue #68) and the margin rule (jobs/PERF_AUDIT.md).

    python -m jobs.probe9b.preflight --params jobs/probe9b/job_params.json --config configs/probe9b_h100.json \
        [--out preflight.json]

Every phase is projected from the numbers in job_params.json:measured (each one a receipt from job-4j3vu, art-5ys6j)
plus, where the 9B has not been measured, an estimate that is labelled as such. The rule: every phase cap is at least
(1 + margin_min_frac) x its projection, the pipeline's pre-probe work fits its deadline with the same margin, and
every stop floor is at most (1 - margin_min_frac) x the measured rate, so a healthy run can never trip it. Exit 0 if
all hold, 5 otherwise (the runner calls this at setup; tests/test_probe9b.py checks the committed files).

The in-run checks stay authoritative: align.Guard (exit 3 / 4), the Index preflight from the pipeline's own measured
prefill and per-call cost (jobs.common.rate_guard preflight, exit 15) and the live row-rate guard (exit 15).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
#: evaluate() = student forward + teacher (cached) + fp32 log-softmax of both over the 248k vocab, 512-token chunks.
#: Not separately timed on the 9B (job-4j3vu ran it once on 8 windows inside a 93 s block that also tokenised 20M
#: tokens); 3 forward-equivalents is an estimate, labelled so in the output.
EVAL_FORWARD_EQUIV = 3.0


def probe_volume(cfg: dict) -> dict:
    """Tokens and calls of probes.core.run_all for one arm (upper bounds where the probe pads)."""
    p = cfg["probes"]
    ppl_tok = sum(c * n for c, n in p["ppl"])
    ppl_calls = sum(-(-n // 4) for _, n in p["ppl"])
    niah_n = p["niah_n_per"] * 3  # 3 depths
    niah_tok = niah_n * sum(p["niah_lengths"])
    mqar_calls = p["mqar_n_seq"] * len(p["mqar_pairs"])
    mmlu_calls = 57 * p["mmlu_k"]
    return {"tokens": ppl_tok + niah_tok + mqar_calls * 1024 + mmlu_calls * 512,
            "calls": ppl_calls + niah_n * len(p["niah_lengths"]) + mqar_calls + mmlu_calls}


def project(params: dict, cfg: dict) -> dict:
    m = params["measured"]
    pre = m["stock_prefill_tok_s_4096"]
    call = m["call_s_fitted_rwkv7"]
    L, B = cfg["seq_len"], cfg["batch"]
    s1, s2 = cfg["stage1"], cfg["stage2"]
    pv = probe_volume(cfg)
    probes_min = (pv["tokens"] / pre + pv["calls"] * call) / 60
    n_attn = 8
    evals = n_attn + 2 + 1  # importance (one per layer) + teacher nll + gate; +1 for the teacher cache forward
    eval_min = cfg["gate"]["n_windows"] * L / pre * EVAL_FORWARD_EQUIV / 60
    tok_scale = cfg["data"]["train_max_tokens"] / m["qual_train_tokens"]
    p = {
        "load": m["model_load_s"] / 60,
        "data": m["qual_pre_stage1_upper_min"] * tok_scale,
        "controls_probes": probes_min,
        "stage1": m["stage1_first_step_s"] / 60 + s1["steps"] * B * L / m["stage1_steady_tok_s"] / 60,
        "importance_gate": evals * eval_min,
        "stage2": float(s2["guard"]["max_minutes"]),
        "converted_probes": probes_min,
        "throughput": m["throughput_both_arms_s"] / 60 + 0.5,
    }
    pre_probe = sum(v for k, v in p.items() if k not in ("converted_probes", "throughput"))
    from jobs.common.readout_cost import project as readout

    vol = json.loads((ROOT / "jobs/common/readout_volume_qwen3.5.json").read_text(encoding="utf-8"))
    arm = readout(vol["totals"], pre, call, "fork")["projected_min"] + m["model_load_s"] / 60
    out = {
        "setup": m["qual_setup_min"] + m["subset_rows_min_est"],
        "weights": m["weights_fetch_s"] / 60,
        "P": sum(p.values()),
        "I_stock": arm,
        "I_conv": arm,
        "S": m["subset_scoring_min_est"],
    }
    return {"phases": out, "P_parts": p, "P_pre_probe": pre_probe, "probe_volume": pv,
            "stage2_tokens_at_measured_rate": m["stage2_steady_tok_s"] * 60 * s2["guard"]["max_minutes"],
            "stage1_tokens": s1["steps"] * B * L,
            "estimates_not_measured": ["controls_probes", "converted_probes", "importance_gate (EVAL_FORWARD_EQUIV)",
                                       "I_* per-call cost (RWKV7 fitted 0.098 s)", "setup subset rows", "S"]}


def check(params: dict, cfg: dict) -> dict:
    pr = project(params, cfg)
    mf = params["margin_min_frac"]
    caps = params["phase_caps_minutes"]
    m = params["measured"]
    rows = {}
    for ph, proj in pr["phases"].items():
        cap = caps[ph]
        usable = cap * (0.95 if ph.startswith("I_") else 1.0)  # rate_guard stops an Index arm at 95% of its cap
        rows[ph] = {"projected_min": round(proj, 2), "cap_min": cap, "margin_frac": round(usable / proj - 1, 3),
                    "ok": usable >= (1 + mf) * proj}
    dl = params["pipeline_deadline_minutes"]
    rows["P_deadline"] = {"projected_min": round(pr["P_pre_probe"], 2), "cap_min": dl,
                          "margin_frac": round(dl / pr["P_pre_probe"] - 1, 3),
                          "ok": dl >= (1 + mf) * pr["P_pre_probe"] and dl < caps["P"]}
    floors = {"stage1": (cfg["stage1"]["guard"]["min_tok_s"], m["stage1_steady_tok_s"]),
              "stage2": (cfg["stage2"]["guard"]["min_tok_s"], m["stage2_steady_tok_s"])}
    for k, (floor, rate) in floors.items():
        rows[f"floor_{k}"] = {"min_tok_s": floor, "measured_tok_s": rate, "margin_frac": round(1 - floor / rate, 3),
                              "ok": floor <= (1 - mf) * rate}
    rows["peak"] = {"measured_gb": m["peak_gb"], "stop_gb": cfg["stage2"]["guard"]["max_peak_gb"],
                    "ok": m["peak_gb"] <= (1 - mf) * cfg["stage2"]["guard"]["max_peak_gb"]}
    sched = sum(caps.values()) + params["reserve_minutes"]
    rows["schedule"] = {"sum_caps_plus_reserve": sched, "max_duration": params["max_duration_minutes"],
                        "ok": sched == params["max_duration_minutes"]}
    return {"ok": all(r["ok"] for r in rows.values()), "checks": rows, **pr}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jobs.probe9b.preflight")
    ap.add_argument("--params", default=str(ROOT / "jobs/probe9b/job_params.json"))
    ap.add_argument("--config", default=str(ROOT / "configs/probe9b_h100.json"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = check(json.loads(Path(a.params).read_text(encoding="utf-8")), json.loads(Path(a.config).read_text(encoding="utf-8")))
    txt = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")
    print(txt)
    return 0 if res["ok"] else 5


if __name__ == "__main__":
    sys.exit(main())
