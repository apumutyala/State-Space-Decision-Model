"""Hardware qualification (paid rung 2): numerics, throughput, memory on one GPU. No capability claims.

    python -m convert.qual --config configs/qual_h100.json --model 0p8b|9b --out <dir>

Per model: load -> stage 1 (alignment of every attention layer's GDN, N steps, Guard) -> stage 2 (end-to-end KL on ONE
layer, wall-clock capped, Guard) -> numerics sanity on gate windows (finite, KL >= 0, NLLs finite) -> prefill throughput
and peak memory, stock vs one-layer-converted. Results go to <out>/qual_<model>.json in the layout scripts/decide_9b.py
reads. A pre-registered stop condition (align.StopCondition) is recorded and re-raised as the process exit code (3/4).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

from convert import align, data
from convert.model import attach_gdn, init_gdn, load
from probes.throughput import measure as measure_throughput


def run(cfg, mkey, out_dir):
    mc = cfg["models"][mkey]
    res_path = os.path.join(out_dir, f"qual_{mkey}.json")
    R = {"model": mkey, "config": mc, "t_start": time.time()}

    def save():
        tmp = res_path + ".tmp"
        json.dump(R, open(tmp, "w"), indent=1)
        os.replace(tmp, res_path)

    def log(rec, phase=""):
        print(json.dumps({"t": time.strftime("%H:%M:%S"), "model": mkey, "phase": phase, **rec}), flush=True)

    try:
        torch.manual_seed(cfg["seed"])
        if cfg.get("vram_fraction"):
            torch.cuda.set_per_process_memory_fraction(cfg["vram_fraction"])
        t = time.time()
        model, tok, mdir = load(os.path.expandvars(mc["model_dir"]) if mc.get("model_dir") else None, kernel="fla")
        R["load_s"], R["model_dir"] = time.time() - t, mdir
        R["weights_gb"] = torch.cuda.memory_allocated() / 1e9
        p, meta = data.build(tok, "train", max_tokens=cfg["train_max_tokens"])
        R["data_train"] = meta
        train = data.load_tokens(p)
        gp, gmeta = data.build(tok, "gate")
        R["data_gate"] = gmeta
        L, B = mc["seq_len"], mc["batch"]
        gate_w = data.fixed_windows(data.load_tokens(gp), L, mc["gate_windows"])
        A = model.attn_layers
        for i in A:
            attach_gdn(model, i, init_gdn(model, i, mc["init"], cfg["seed"]))
        bt = data.Batcher(train, L, B, seed=cfg["seed"] + 2)
        align.calibrate_out_scale(model, A, next(bt))
        save()

        torch.cuda.reset_peak_memory_stats()
        g1 = align.Guard(mc["stage1"].get("guard"), f"{mkey}:stage1")
        hist, ntok, sec = align.stage1_align(model, A, bt, mc["stage1"]["steps"], mc["stage1"]["lr"],
                                             log=lambda r: log(r, "stage1"), guard=g1)
        R["stage1"] = {"tokens": ntok, "seconds": sec, "tok_s": ntok / sec, "steady_tok_s": g1.rate,
                       "peak_gb": g1.peak_gb(), "hist": hist}
        save()

        layer = A[-1] if mc.get("stage2_layer", "last") == "last" else int(mc["stage2_layer"])
        modes = {layer: "student"}
        torch.cuda.reset_peak_memory_stats()
        g2 = align.Guard(mc["stage2"].get("guard"), f"{mkey}:stage2")
        bt2 = data.Batcher(train, L, B, seed=cfg["seed"] + 100)
        hist, ntok, sec = align.stage2_kl(model, [layer], modes, bt2, mc["stage2"]["steps"], mc["stage2"]["lr"],
                                          log=lambda r: log(r, "stage2"), guard=g2)
        step = {"layer": layer, "tokens": ntok, "seconds": sec, "train_tok_s": ntok / sec, "steady_tok_s": g2.rate,
                "peak_gb": g2.peak_gb(), "hist": hist}
        R["progressive"] = {"steps": [step]}
        save()

        ge = align.evaluate(model, gate_w, modes)
        finite = all(math.isfinite(v) for v in ge.values())
        R["numerics"] = {"gate_eval_one_layer": ge, "finite": finite, "kl_nonneg": ge["kl"] >= -1e-6}
        save()
        if not (finite and ge["kl"] >= -1e-6):
            raise align.StopCondition(f"{mkey}: numerics sanity failed {ge}", 2)

        tc = mc["throughput"]
        R["throughput"] = {"stock": measure_throughput(model, {}, tc["lengths"], tc.get("reps", 2)),
                           "converted_1": measure_throughput(model, modes, tc["lengths"], tc.get("reps", 2))}
        R["seconds_total"] = time.time() - R["t_start"]
        save()
        return 0
    except align.StopCondition as e:
        R["stopped"] = {"reason": e.reason, "exit_code": e.code, **e.data}
        save()
        print(json.dumps({"STOP_CONDITION": e.reason, "exit_code": e.code}), flush=True)
        return e.code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    os.makedirs(a.out, exist_ok=True)
    sys.exit(run(cfg, a.model, a.out))


if __name__ == "__main__":
    main()
