"""RWKV7-G1k-13.3B memory / prefill-throughput probe (jobs/rwkv7 stage A, phase probe). GPU.

python -m jobs.rwkv7.probe_rwkv7 --model-dir <snapshot> --out probe.json [--lengths 2048,8192,16384] [--reps 2]

Measures what the m2 pilot and the m3 latency story depend on, on the real bundled code ("chunked" pure-PyTorch WKV):
weights resident GB, prefill tok/s and peak GB per length (logits for the last position only, no grad), and the
analytic constant decode-state bytes. Also re-checks the trained-context assumption used by the readout (25,600).
"""
from __future__ import annotations

import argparse
import json
import time


def state_bytes(cfg) -> int:
    """Per sequence: WKV [heads, hd, hd] in wkv_state_dtype + two token shifts [hidden] in bf16, per layer."""
    import torch

    wkv = cfg.num_heads * cfg.head_dim * cfg.head_dim * torch.finfo(getattr(torch, cfg.wkv_state_dtype)).bits // 8
    return cfg.num_hidden_layers * (wkv + 2 * cfg.hidden_size * 2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="2048,8192,16384")
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args(argv)
    import torch

    import third_party.rwkv7_hf as V
    from third_party.rwkv7_hf.modeling_rwkv7 import Rwkv7ForCausalLM

    code = V.verify_snapshot_code(a.model_dir)
    t = time.time()
    m = Rwkv7ForCausalLM.from_pretrained(a.model_dir, dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
    out = {"load_s": time.time() - t, "weights_gb": torch.cuda.memory_allocated() / 1e9, "code_sha256": code,
           "wkv_implementation": m.config.wkv_implementation, "state_bytes_per_seq": state_bytes(m.config),
           "params": sum(p.numel() for p in m.parameters()), "prefill": {}}
    g = torch.Generator(device="cpu").manual_seed(0)
    for L in [int(x) for x in a.lengths.split(",")]:
        ids = torch.randint(1, m.config.vocab_size, (1, L), generator=g).cuda()
        torch.cuda.reset_peak_memory_stats()
        times = []
        try:
            with torch.no_grad():
                for _ in range(a.reps + 1):  # first rep = warm-up
                    torch.cuda.synchronize()
                    t = time.time()
                    m(input_ids=ids, logits_to_keep=1, use_cache=False)
                    torch.cuda.synchronize()
                    times.append(time.time() - t)
            best = min(times[1:])
            out["prefill"][str(L)] = {"tok_s": L / best, "s": best, "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
        except torch.OutOfMemoryError as e:
            out["prefill"][str(L)] = {"oom": str(e)[:200], "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
        torch.cuda.empty_cache()
        json.dump(out, open(a.out, "w"), indent=1)
    # small-call latency (readout cost model, jobs/common/readout_cost.py): the prefix-fork readout makes ~2 calls per
    # row, one of them a packed tail batch; time a [64, 64] continuation from a 2k-token state, median of 5
    try:
        from decide.backends.state_fork import RWKV7StateOps

        ops = RWKV7StateOps(m)
        with torch.no_grad():
            _, st = ops.prefill(torch.randint(1, m.config.vocab_size, (1, 2048), generator=g).cuda())
            tail = torch.randint(1, m.config.vocab_size, (64, 64), generator=g).cuda()
            ts = []
            for _ in range(6):
                torch.cuda.synchronize()
                t = time.time()
                ops.extend(st, tail, 64)
                torch.cuda.synchronize()
                ts.append(time.time() - t)
        ts = sorted(ts[1:])
        out["pack_call_s"] = ts[len(ts) // 2]  # one packed call of 4,096 positions incl. state copies
        with torch.no_grad():
            ts = []
            for _ in range(6):
                torch.cuda.synchronize()
                t = time.time()
                m(input_ids=tail[:1, :16], logits_to_keep=1, use_cache=False)
                torch.cuda.synchronize()
                ts.append(time.time() - t)
        ts = sorted(ts[1:])
        out["small_call_s"] = ts[len(ts) // 2]  # fixed per-call overhead (16 tokens)
    except Exception as e:  # noqa: BLE001  measurement only; the preflight then falls back to the fitted value
        out["call_probe_error"] = repr(e)[:300]
    out["peak_gb"] = max(v["peak_gb"] for v in out["prefill"].values() if "peak_gb" in v)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
