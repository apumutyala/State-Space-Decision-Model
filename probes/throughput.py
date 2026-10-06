"""Prefill throughput / peak memory (measured) and decode-state bytes (analytic) for a given swap configuration."""
from __future__ import annotations

import statistics
import time

import torch


def state_bytes(model, modes, seq_len, kv_bytes=2, state_dtype_bytes=4):
    cfg = model.cfg
    n_attn = sum(1 for i in model.attn_layers if modes.get(i, "teacher") == "teacher")
    kv_per_tok = n_attn * 2 * cfg.num_key_value_heads * cfg.head_dim * kv_bytes
    n_gdn = cfg.num_hidden_layers - n_attn
    gdn_state = (cfg.linear_num_value_heads * cfg.linear_key_head_dim * cfg.linear_value_head_dim
                 + (2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim
                    + cfg.linear_num_value_heads * cfg.linear_value_head_dim) * (cfg.linear_conv_kernel_dim - 1))
    fixed = n_gdn * gdn_state * state_dtype_bytes
    return {"n_attention_layers": n_attn, "n_gdn_layers": n_gdn, "kv_bytes_per_token": kv_per_tok,
            "fixed_state_bytes": fixed, "total_bytes_at_len": fixed + kv_per_tok * seq_len}


@torch.no_grad()
def measure(model, modes, lengths, reps=3):
    out = {}
    with model.modes(modes):
        for L in lengths:
            x = torch.randint(100, 10000, (1, L), device="cuda")
            try:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model.hidden(x)  # warm-up / autotune
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    base = torch.cuda.memory_allocated()
                    ts = []
                    for _ in range(reps):
                        t = time.perf_counter()
                        model.hidden(x)
                        torch.cuda.synchronize()
                        ts.append(time.perf_counter() - t)
                med = statistics.median(ts)
                out[str(L)] = {"median_s": med, "spread_s": [min(ts), max(ts)], "tok_s": L / med,
                               "peak_activation_GB": (torch.cuda.max_memory_allocated() - base) / 1e9,
                               "state": state_bytes(model, modes, L)}
            except torch.OutOfMemoryError:
                out[str(L)] = {"oom": True, "state": state_bytes(model, modes, L)}
                torch.cuda.empty_cache()
    return out
