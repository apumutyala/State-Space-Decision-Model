"""Price, memory and schedule model of the Q9-AF cell (jobs/q9af_cell). Every MEASURED input names its receipt; every
other input is an ASSUMPTION, named as such in ``ASSUMPTIONS`` and re-measured on the box before training spend (the
run_cell ladder + 10-step rate checks, exit 26) or scoring spend (the per-phase timeouts).

    python -m jobs.q9af_cell.price            # prints the projection table for every mode (JOB_SPEC.md quotes it)

Training throughput (per optimiser step of ROWS_PER_STEP rows, LoRA + fork head, cache fork, block checkpointing):
    compute_s  = rows x (mean_row_tokens + K x cont_tokens) x flops_per_token / (MFU x H100 peak)
    flops/token = 6 N  (forward 2N + block recompute 2N + input-gradient backward 2N; the frozen base has no weight
                  gradients) + 6 x 2 x lora_params (LoRA forward/recompute/backward, weight grads included)
    overhead_s = rows x CALLS_PER_ROW x PASSES_PER_CALL x small_call_s   (per-call fixed cost: job-5cmat's measured
                  small readout call, applied to each pass of each call: launch-bound floor, conservative)
Memory at prefix length T (GB = 1e9 B), the act-ckpt projection:
    weights bf16 (params_memory x 2) + LoRA fp32 param/grad/Adam (16 B / param)
    + kept block inputs: n_layers x T x d x 2 B x CKPT_KEPT_OVERHEAD          (CPU-measured: 5.0 B vs 4 B fp32 per d)
    + one block's recompute live: NOCKPT_BYTES_PER_TOKEN_LAYER x T             (PREMORTEM P4 anchor, job-mden7)
    + K continuation state copies saved at block inputs: K x n_layers x (S fp32 + conv tail bf16)
  without checkpointing: NOCKPT_BYTES_PER_TOKEN_LAYER x n_layers x T (+ the same weights / LoRA / states).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: job-5cmat (probe9b one-layer 9B, art-2bybh; jobs/full8/job_params.json "measured"; same image stack and revision)
MEASURED = {"source_job": "job-5cmat", "artifact": "art-2bybh",
            "prefill_tok_s_min": 22144.0, "call_s_fitted": 0.0595, "small_call_s": 0.0493, "stage1_mfu": 0.316,
            "stage2_mfu_layer23": 0.3428, "peak_gb_stage": 31.38, "model_load_s": 43.0, "setup_min": 2.4,
            "build_and_start_min": 14.0}
#: the CPU measurement of this PR (tests/test_q9af_act_ckpt.py, receipts/2026-10-07-D-q9af-act-ckpt/)
CPU_ACT_CKPT = {"bytes_per_token_layer_d_ckpt": 5.015625, "bytes_per_token_layer_d_plain": 112.814453125,
                "slope_ratio": 22.49}
#: receipts/2026-10-07-D-cohort-v3/measure.json (Q9 rates): the G13 retention term of ONE arm on cohort v3
V3_RETENTION = {"positions": 22_106_052, "calls": 63_583}
#: jobs/m2_parent_arm/score_parent.py VOLUME_TOKENS / CHILD_EXTRA_TOKENS (upper bounds, cohort v2 roles; the dev and
#: steer roles are unchanged in v3) + the full-DEV Index projection (38.79 M fork positions)
VOLUME = {"dev_records": 3_887_640, "steer_dev_units": 5_378_738, "steer_gate_units": 5_430_015,
          "index_dev_full": 38_787_584, "calibration_fit": 3_393_271, "dev_card": 3_993_020,
          "steer_units_per_stage": 9_000, "dev_record_calls": 2_000, "index_dev_rows": 12_104}

ASSUMPTIONS = {
    "h100_bf16_dense_flops": 989.4e12,
    "train_mfu": 0.30,  # below job-5cmat's measured 0.316 (stage 1: frozen-9B forward + backward)
    "n_compute_params": 6.92e9,  # heads.backbone.BACKBONES["q9af"].params
    "params_memory": 7.94e9,  # BACKBONES["q9af"].params_memory (bf16 weights)
    "n_layers": 32, "d_model": 4096,
    "mean_row_tokens": 619,  # PR #74 2,000-record sample of the 150k set (RWKV tokenizer); Qwen tokenizer assumed equal
    "mean_k": 3.4, "cont_tokens": 4, "rows_per_record": 0.91,
    "calls_per_row": 2, "passes_per_call": 3,  # prefix + continuation call; forward, recompute, backward
    "lora_sum_din_dout_per_rank": 2_490_368,  # sum over the q9af LoRA modules of (din + dout): 77,824 x 32 layers
    "nockpt_bytes_per_token_layer": 0.97e6,  # PREMORTEM P4 (Q9-AF/RWKV7 0.97 x job-mden7's ~1.0 MB)
    "ckpt_kept_overhead": 1.25,  # CPU measured 5.0 B per d per token per layer vs 4 B (fp32 h): x1.25
    "gdn_state_bytes_per_seq_layer": 32 * 128 * 128 * 4 + 8192 * 3 * 2,  # S fp32 + conv tail bf16
    "max_peak_gb": 76.0,
    "tokenize_min": 15.0,  # q9_feature_cache TOKENIZE_MIN 14.2 (24k draw + DEV), rounded up
    "evals_head_slowdown": 1.0,  # the decision-head arm scored at the stock readout's measured rates
}
ROWS_PER_STEP = 8


def lora_params(rank: int) -> int:
    return rank * ASSUMPTIONS["lora_sum_din_dout_per_rank"]


def train_step_s(rank: int, a: dict = ASSUMPTIONS) -> dict:
    n_tok = a["mean_row_tokens"] + a["mean_k"] * a["cont_tokens"]
    fpt = 6 * a["n_compute_params"] + 6 * 2 * lora_params(rank)
    compute = ROWS_PER_STEP * n_tok * fpt / (a["train_mfu"] * a["h100_bf16_dense_flops"])
    over = ROWS_PER_STEP * a["calls_per_row"] * a["passes_per_call"] * MEASURED["small_call_s"]
    return {"compute_s": round(compute, 3), "overhead_s": round(over, 3), "step_s": round(compute + over, 3),
            "tok_s": round(ROWS_PER_STEP * n_tok / (compute + over), 1), "flops_per_token": fpt}


def memory_gb(rank: int, T: int, K: int = 16, ckpt: bool = True, a: dict = ASSUMPTIONS) -> dict:
    w = a["params_memory"] * 2
    lo = lora_params(rank) * 16
    st = K * a["n_layers"] * a["gdn_state_bytes_per_seq_layer"]
    if ckpt:
        act = (a["n_layers"] * T * a["d_model"] * 2 * a["ckpt_kept_overhead"]
               + a["nockpt_bytes_per_token_layer"] * T)
    else:
        act = a["nockpt_bytes_per_token_layer"] * a["n_layers"] * T
    tot = w + lo + st + act
    return {"weights": round(w / 1e9, 2), "lora_adam": round(lo / 1e9, 2), "states": round(st / 1e9, 2),
            "activations": round(act / 1e9, 2), "total": round(tot / 1e9, 2), "fits": tot / 1e9 <= a["max_peak_gb"]}


def max_t(rank: int, ckpt: bool, K: int = 16, a: dict = ASSUMPTIONS) -> int:
    lo, hi = 0, 1 << 20
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if memory_gb(rank, mid, K, ckpt, a)["fits"]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def scoring_min(positions: float, calls: float) -> float:
    return (positions / MEASURED["prefill_tok_s_min"] + calls * MEASURED["call_s_fitted"]) / 60


def arm_minutes(child: bool) -> dict:
    """One arm's G12 + G13 units (cohort v3 retention, full-DEV Index), at the stock readout's measured rates."""
    v = VOLUME
    g12 = scoring_min(v["dev_records"] + v["steer_dev_units"], v["dev_record_calls"] + v["steer_units_per_stage"])
    g13 = (scoring_min(V3_RETENTION["positions"], V3_RETENTION["calls"])
           + scoring_min(v["steer_gate_units"], v["steer_units_per_stage"])
           + scoring_min(v["index_dev_full"], v["index_dev_rows"]))
    extra = scoring_min(v["calibration_fit"] + v["dev_card"], 2 * v["dev_record_calls"]) if child else 0.0
    s = ASSUMPTIONS["evals_head_slowdown"]
    return {"g12": round(g12 * s, 1), "g13": round(g13 * s, 1), "child_extra": round(extra * s, 1),
            "total": round((g12 + g13 + extra) * s, 1)}


def cell_projection(cfg: dict) -> dict:
    """The cell phase: tokenisation + load + smoke + ladder + training (wall-clock capped by max_minutes) + cards."""
    st = train_step_s(cfg["lora_rank"])
    steps_cap = cfg["max_steps"]
    train_min = min(cfg["max_minutes"], steps_cap * st["step_s"] / 60)
    fixed = (ASSUMPTIONS["tokenize_min"] + MEASURED["model_load_s"] / 60 + 3.0  # smoke: 2 rows x 2 modes
             + len(cfg["t_ladder"]) * 2.0  # one optimiser step per rung + reset
             + 2 * scoring_min(VOLUME["dev_card"], VOLUME["dev_record_calls"]))  # step-0 + final eval cards
    return {"fixed_min": round(fixed, 1), "train_min_expected": round(train_min, 1),
            "train_min_cap": cfg["max_minutes"], "projected_min": round(fixed + cfg["max_minutes"], 1),
            "expected_min": round(fixed + train_min, 1), "step": st,
            "steps_in_budget": int(cfg["max_minutes"] * 60 / st["step_s"]),
            "min_steps": cfg.get("min_steps"), "max_step_s_for_min_steps":
                round(cfg["max_minutes"] * 60 / max(1, cfg.get("min_steps") or 1), 2)}


def projections(params: dict, cell_cfg: dict | None) -> dict:
    """Projected minutes per phase of a mode (the preflight compares each cap with them)."""
    e = params["expected_minutes"]
    out = {k: {"projected_min": v, "source": "expected_minutes (receipt in job_params)"} for k, v in e.items()
           if k != "build"}
    if params["mode"] == "parent":
        arm = arm_minutes(child=False)
        out["parent_g12"] = {"projected_min": arm["g12"] + MEASURED["model_load_s"] / 60, "source": "price.arm_minutes"}
        out["parent_g13"] = {"projected_min": arm["g13"] + MEASURED["model_load_s"] / 60, "source": "price.arm_minutes"}
    else:
        c = cell_projection(cell_cfg)
        out["cell"] = {"projected_min": c["projected_min"], "source": "price.cell_projection", **c}
        arm = arm_minutes(child=True)
        out["evals"] = {"projected_min": arm["total"] + ASSUMPTIONS["tokenize_min"] + MEASURED["model_load_s"] / 60,
                        "source": "price.arm_minutes(child) + re-tokenisation + load", **arm}
    return out


def table() -> dict:
    rows = {}
    for mode in ("parent", "cell_r64", "cell_r256"):
        p = json.loads((HERE / f"job_params_{mode}.json").read_text(encoding="utf-8"))
        cfg = json.loads((HERE.parents[1] / p["cell_config"]).read_text(encoding="utf-8")) if p.get("cell_config") else None
        rows[mode] = projections(p, cfg)
    return {"train_step": {r: train_step_s(r) for r in (64, 256)},
            "memory_T4096": {f"r{r}_{'ckpt' if c else 'plain'}": memory_gb(r, 4096, ckpt=c)
                             for r in (64, 256) for c in (True, False)},
            "max_T": {f"r{r}_{'ckpt' if c else 'plain'}": max_t(r, c) for r in (64, 256) for c in (True, False)},
            "arms": {"parent": arm_minutes(False), "child": arm_minutes(True)}, "modes": rows}


if __name__ == "__main__":
    print(json.dumps(table(), indent=1))
    sys.exit(0)
