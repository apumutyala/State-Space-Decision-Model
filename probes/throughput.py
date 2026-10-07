"""Prefill throughput / peak memory (measured) and decode-state bytes (analytic) for a given swap configuration."""
from __future__ import annotations

import contextlib
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


@torch.no_grad()
def call_latency(model, modes, prefix_len=2048, tail=(64, 64), small_len=16, reps=6, autocast_dtype=torch.bfloat16):
    """Per-call cost of the prefix-fork readout on THIS model (jobs/common/readout_cost.py: wall = positions / tok_s +
    calls x call_s). Same shapes as jobs/rwkv7/probe_rwkv7.py: ``pack_call_s`` = one packed [64, 64] continuation from
    a 2k-token state through ``convert.model.ConvertedStateOps`` (the path both probe9b Index arms run), ``small_call_s``
    = a 16-token plain forward. Median of reps-1 after a warm-up. Before this, the probe9b readout preflight had to
    borrow the RWKV7 fitted 0.098 s/call (PERF_AUDIT)."""
    from convert.model import ConvertedStateOps

    dev = next(model.lm.parameters()).device
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    g = torch.Generator().manual_seed(0)
    vocab = model.cfg.vocab_size
    ops = ConvertedStateOps(model, autocast_dtype=autocast_dtype if dev.type == "cuda" else None)

    def med(fn):
        ts = []
        for _ in range(reps):
            sync()
            t = time.perf_counter()
            fn()
            sync()
            ts.append(time.perf_counter() - t)
        ts = sorted(ts[1:])
        return ts[len(ts) // 2]

    # no_grad: a readout never builds a graph. The small call runs under the same autocast as the pack call: job-5cmat's
    # converted arm (fp32 student GDN in a bf16 model) failed here with "mat1 and mat2 ... BFloat16 != float" outside it
    ac = (torch.autocast(device_type="cuda", dtype=autocast_dtype) if dev.type == "cuda" and autocast_dtype is not None
          else contextlib.nullcontext())
    with model.modes(modes), torch.no_grad():
        _, st = ops.prefill(torch.randint(1, vocab, (1, prefix_len), generator=g).to(dev))
        tl = torch.randint(1, vocab, tail, generator=g).to(dev)
        pack = med(lambda: ops.extend(st, tl, tail[1]))

        def small_call():
            with ac:
                h = model.hidden(tl[:1, :small_len])[:, -1:]
            return model.lm_head(h)

        small = med(small_call)
    return {"pack_call_s": pack, "small_call_s": small, "prefix_len": prefix_len, "tail": list(tail),
            "small_len": small_len, "reps": reps}
