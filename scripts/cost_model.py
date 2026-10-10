"""9B conversion cost model, anchored on the 0.8B pilot's measured training throughput.

python scripts/cost_model.py receipts/<dir>   -> <dir>/cost_model.json (and prints it)

Method
  FLOP/token per stage, from the pilot's actual recipe:
    stage1 (alignment):  teacher fwd 2N  + student GDN fwd+bwd 6*n_gdn_new           (no backprop through the trunk)
    stage2 (e2e KL):     teacher fwd 2N  + student fwd 2N + recompute fwd 2N (activation checkpointing)
                         + activation-only backward 2N (frozen trunk: no weight grads) + 2N KL-head  ~= 10N
  Achieved TFLOP/s on the laptop (from the measured tok/s) gives the pilot's MFU; the 9B on H100 is priced at
  MFU 25% (pessimistic, pure-PyTorch glue + fla kernels) and 35% (B-review assumption). H100 bf16 dense = 989 TFLOP/s.
"""
from __future__ import annotations

import json
import os
import sys

H100_PEAK = 989e12
RATE_BATCH = 2.97  # $/H100-h, SF Compute batch (LEDGER.md, verified 2026-10-06). HISTORICAL: this script reproduces the
#                   2026-10-06 qualification pricing; current rates (from 2026-10-08, $3.78/h) are in jobs/common/rates.py
LAPTOP_PEAK = 2 * 44e12  # RTX 5070 Ti Laptop: ~44 dense fp16/bf16 TFLOP/s w/ fp32 acc per spec sheets; x2 = fp16 acc (uncertain)

P08 = {"N": 0.75e9, "n_attn": 6, "gdn_new_params": 10.5e6}  # 0.8B text params excl. embeddings ~0.5B; incl ~0.75B
P9 = {"N": 8.5e9, "n_attn": 8, "gdn_new_params": 67e6, "hidden": 4096, "layers": 32}


def flops(stage, p):
    N = p["N"]
    if stage == "stage1":
        return 2 * N + 6 * p["n_attn"] * p["gdn_new_params"]
    return 10 * N


def main(d):
    R = json.load(open(os.path.join(d, "results.json")))
    meas, ach = {}, {}
    if "stage1" in R and R.get("progressive", {}).get("steps"):
        meas["stage1_tok_s"] = R["stage1"]["tok_s"]
        s2 = [s["train_tok_s"] for s in R["progressive"]["steps"]]
        meas["stage2_tok_s_per_step"] = s2
        meas["stage2_tok_s_max"] = max(s2)  # least contended step (GPU shared with another worker)
        ach = {"stage1_TFLOPs": meas["stage1_tok_s"] * flops("stage1", P08) / 1e12,
               "stage2_TFLOPs": meas["stage2_tok_s_max"] * flops("stage2", P08) / 1e12}
    else:
        meas["note"] = ("full pilot did not complete locally; laptop smoke (2x512 tokens/step, shared GPU): stage1 "
                        "~3.5-5.0k tok/s, stage2 ~0.85-0.99k tok/s uncontended -> not used to price the 9B; "
                        "9B priced from FLOP model at assumed MFU")
    plan = {  # tokens for the 9B m1 run (PLAN.md m1: one-layer probe, then all 8 with accept/rollback)
        "one_layer_probe": {"stage1": 50e6, "stage2": 100e6},
        "full_8_layers": {"stage1": 100e6, "stage2": 8 * 100e6},
        "full_8_layers_lean": {"stage1": 50e6, "stage2": 8 * 50e6},
    }
    out = {"measured_0p8b_laptop": meas, "achieved_laptop": ach, "assumptions": {
        "N_9b_text": P9["N"], "flop_per_token_stage1_9b": flops("stage1", P9),
        "flop_per_token_stage2_9b": flops("stage2", P9), "h100_peak": H100_PEAK, "rate_usd_per_h": RATE_BATCH,
        "eval_overhead_fraction": 0.15}, "plans": {}}
    for name, toks in plan.items():
        row = {}
        for mfu in (0.25, 0.35):
            sec = sum(toks[s] * flops(s, P9) / (H100_PEAK * mfu) for s in toks) * 1.15
            row[f"mfu{int(mfu * 100)}"] = {"H100_h": sec / 3600, "usd": sec / 3600 * RATE_BATCH}
        row["tokens"] = toks
        out["plans"][name] = row
    # memory plan, one 80 GB H100
    gdn_p = P9["n_attn"] * P9["gdn_new_params"]
    B, T = 8, 4096
    mem = {
        "weights_bf16_GB": 2 * 9.0e9 / 1e9,
        "trainable_gdn_params": gdn_p,
        "fp32_master+adam_GB": 16 * gdn_p / 1e9,
        "ckpt_layer_inputs_GB(B8xT4096)": P9["layers"] * B * T * P9["hidden"] * 2 / 1e9,
        "one_layer_recompute_working_set_GB(est)": 6.0,
        "kl_chunk_logits_GB(512 tok x 248k x fp32 x4 live)": 512 * 248320 * 4 * 4 / 1e9,
    }
    mem["total_GB_est"] = sum(v for k, v in mem.items() if k.endswith("GB") or "GB(" in k)
    out["memory_plan_80GB"] = mem
    # <= $20 qualification spec (not launched)
    budget_h = 20 / RATE_BATCH
    out["qualification_spec_le_20usd"] = {
        "hardware": "1x H100-80GB, SF Compute batch $2.97/h; hard runtime ceiling 6.0 h (= $17.8); stop node after",
        "budget_H100_h": budget_h,
        "steps": [
            "0.25 h setup: env (torch 2.8, transformers 5.17, fla 0.5.2), download Qwen3.5-9B, tokenise 60M tokens",
            "0.25 h: tests/test_gdn.py on H100 (fla vs pure-torch fwd/grad, module vs HF layer) -> PASS required",
            "0.5 h: stock probes (ppl, NIAH 512-16k, MQAR, MMLU-285) + SWA+sinks control",
            "0.5 h: stage-1 alignment of all 8 layers, 200 steps @ 8x4096 -> measured tok/s, peak memory",
            "0.5 h: importance (8 single-swap KLs) -> conversion order",
            "2.5 h: ONE-layer conversion (lowest-KL layer): stage-2 KL ~30M tokens, gate eval, probes",
            "0.5 h: throughput/memory (prefill 4k-32k stock vs converted), write receipt",
            "1.0 h: slack",
        ],
        "stop_conditions": ["any test_gdn FAIL", "stage-2 tok/s < 50% of cost-model mfu25 prediction after 50 steps",
                            "peak memory > 76 GB", "6.0 h wall clock"],
        "exit_proof": "receipt with measured 9B tok/s (stage1, stage2) and peak memory -> re-price the full m1 run",
    }
    json.dump(out, open(os.path.join(d, "cost_model.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
