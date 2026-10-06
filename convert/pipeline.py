"""m1-prep conversion pilot: attention -> GDN on Qwen3.5 (pipeline qualification on 0.8B, not a 9B claim).

    python -m convert.pipeline --config configs/m1_pilot_0p8b.json --out receipts/<dir>

Resumable: every finished phase is written to <out>/results.json and GDN weights to <ckpt_dir>; re-running skips
finished phases.

Phases
  controls     stock hybrid, SWA+sinks on all attention layers, attention deleted (all zero-training)
  init_cmp     stage-1 alignment from three inits (random / clone_neighbor / reuse_attn), short; pick lowest rel-MSE
  stage1       hidden-state alignment of every attention layer's GDN replacement (simultaneous, teacher-driven)
  importance   KL(teacher || model with only layer i swapped) on the gate split -> conversion order (ascending)
  progressive  for each layer in order: swap, end-to-end KL (all swapped GDNs train), gate (dNLL + fidelity),
               accept / rollback (or force, recording the verdict), probes -> one curve point
  throughput   prefill tok/s and peak memory, stock vs converted; analytic decode-state bytes
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import random
import subprocess
import time

import numpy as np
import torch

from convert import align, data
from convert.model import attach_gdn, init_gdn, load
from probes import core as probes
from probes.throughput import measure as measure_throughput


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


class Results:
    def __init__(self, path):
        self.path = path
        self.d = json.load(open(path)) if os.path.exists(path) else {}

    def __contains__(self, k):
        return k in self.d

    def __getitem__(self, k):
        return self.d[k]

    def set(self, k, v):
        self.d[k] = v
        tmp = self.path + ".tmp"
        json.dump(self.d, open(tmp, "w"), indent=1)
        os.replace(tmp, self.path)


def env_info():
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], text=True).strip()
        except Exception:
            return None

    import transformers

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0), "platform": platform.platform(),
            "fla": __import__("importlib.metadata").metadata.version("flash-linear-attention")
            if __import__("importlib.util").util.find_spec("fla") else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    os.makedirs(a.out, exist_ok=True)
    ckpt_dir = cfg.get("ckpt_dir", os.path.join(data.DATA_ROOT, "ckpt", os.path.basename(a.out.rstrip("/\\"))))
    os.makedirs(ckpt_dir, exist_ok=True)
    R = Results(os.path.join(a.out, "results.json"))
    logf = open(os.path.join(a.out, "train_log.jsonl"), "a")

    def log(rec, phase=""):
        rec = {"t": time.strftime("%H:%M:%S"), "phase": phase, **rec}
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    seed_all(cfg["seed"])
    if cfg.get("vram_fraction"):  # fail fast with OOM (driver retries) instead of spilling into shared memory
        torch.cuda.set_per_process_memory_fraction(cfg["vram_fraction"])
    model, tok, mdir = load(cfg.get("model_dir"), kernel=cfg.get("kernel", "fla"))
    if "env" not in R:
        R.set("env", {**env_info(), "model_dir": mdir})
    splits = {}
    for s in ("train", "gate", "eval"):
        p, meta = data.build(tok, s, max_tokens=cfg["data"].get("train_max_tokens") if s == "train" else None)
        splits[s] = data.load_tokens(p)
        R.set(f"data_{s}", meta) if f"data_{s}" not in R else None
    L, B = cfg["seq_len"], cfg["batch"]
    gate_w = data.fixed_windows(splits["gate"], L, cfg["gate"]["n_windows"])
    A = model.attn_layers
    pc = cfg

    def run_probes(tag, modes):
        key = f"probe_{tag}"
        if key in R:
            return R[key]
        t0 = time.time()
        with model.modes(modes):
            res = probes.run_all(model, tok, splits["eval"], pc)
        res["modes"] = {str(k): v for k, v in modes.items()}
        res["seconds"] = time.time() - t0
        R.set(key, res)
        torch.cuda.empty_cache()
        log({"tag": tag, "ppl1k": res["ppl"]["ctx1024"]["ppl"], "niah": res["niah"]["mean_acc"],
             "mqar": res["mqar"]["mean_acc"], "mmlu": res["mmlu"]["acc"]}, "probe")
        return res

    # ------------------------------------------------------------------ controls
    run_probes("stock", {})
    for w in cfg["controls"].get("swa_windows", [512]):
        for sw in model.swaps.values():
            sw.swa_window, sw.swa_sinks = w, cfg["controls"].get("swa_sinks", 4)
        run_probes(f"swa{w}_sinks{cfg['controls'].get('swa_sinks', 4)}", {i: "swa" for i in A})
    if cfg["controls"].get("zero", True):
        run_probes("attn_deleted", {i: "zero" for i in A})

    def save_gdns(name, layers):
        torch.save({i: model.swaps[i].gdn.state_dict() for i in layers}, os.path.join(ckpt_dir, name + ".pt"))

    def load_gdns(name):
        sd = torch.load(os.path.join(ckpt_dir, name + ".pt"))
        for i, s in sd.items():
            if model.swaps[i].gdn is None:
                attach_gdn(model, i, init_gdn(model, i, "random", cfg["seed"]))
            model.swaps[i].gdn.load_state_dict(s)

    # ------------------------------------------------------------------ init comparison (diagnostic)
    ic = cfg["init_cmp"]
    if "init_cmp" not in R and ic.get("steps", 0) > 0:
        out = {}
        for how in ic["inits"]:
            seed_all(cfg["seed"])
            for i in A:
                attach_gdn(model, i, init_gdn(model, i, how, cfg["seed"]))
            bt = data.Batcher(splits["train"], L, B, seed=cfg["seed"] + 1)
            if cfg.get("calibrate_out_scale", True):
                align.calibrate_out_scale(model, A, next(bt))
            hist, ntok, sec = align.stage1_align(model, A, bt, ic["steps"], ic["lr"],
                                                 log=lambda r, h=how: log(r, f"init_cmp:{h}"))
            out[how] = {"first": hist[0], "last": hist[-1], "tokens": ntok, "seconds": sec,
                        "mean_last_rel_mse": float(np.mean([v for k, v in hist[-1].items() if k.startswith("rel_mse")]))}
        best = min(out, key=lambda h: out[h]["mean_last_rel_mse"])
        R.set("init_cmp", {"results": out, "chosen": best})
    init_how = R["init_cmp"]["chosen"] if "init_cmp" in R else cfg["init"]

    # ------------------------------------------------------------------ stage 1
    s1 = cfg["stage1"]
    if "stage1" not in R:
        seed_all(cfg["seed"])
        for i in A:
            attach_gdn(model, i, init_gdn(model, i, init_how, cfg["seed"]))
        init_eval = {i: align.evaluate(model, gate_w[:8], {i: "student"}) for i in A} if s1.get("eval_init") else None
        bt = data.Batcher(splits["train"], L, B, seed=cfg["seed"] + 2)
        scales = align.calibrate_out_scale(model, A, next(bt)) if cfg.get("calibrate_out_scale", True) else None
        hist, ntok, sec = align.stage1_align(model, A, bt, s1["steps"], s1["lr"], log=lambda r: log(r, "stage1"))
        save_gdns("stage1", A)
        R.set("stage1", {"init": init_how, "hist": hist, "tokens": ntok, "seconds": sec, "tok_s": ntok / sec,
                         "init_eval_single_swap": init_eval,
                         "out_scale_calibration": {str(k): v for k, v in (scales or {}).items()}})
    else:
        load_gdns("stage1")

    # ------------------------------------------------------------------ importance (KL of single swaps)
    if "importance" not in R:
        imp = {}
        for i in A:
            imp[i] = align.evaluate(model, gate_w, {i: "student"})
            log({"layer": i, **imp[i]}, "importance")
        order = sorted(A, key=lambda i: imp[i]["kl"]) if cfg.get("order", "kl") == "kl" else list(cfg["order"])
        R.set("importance", {"per_layer": {str(k): v for k, v in imp.items()}, "order": order,
                             "criterion": "ascending KL(teacher||single-swap student) on gate split after stage1"})
    order = R["importance"]["order"]

    # ------------------------------------------------------------------ progressive conversion
    s2, gate = cfg["stage2"], cfg["gate"]
    prog = R["progressive"] if "progressive" in R else {"steps": [], "converted": []}
    stage1_state = {i: copy.deepcopy(model.swaps[i].gdn.state_dict()) for i in A}
    if prog["steps"]:
        load_gdns(f"prog_{len(prog['steps'])}")
    converted = list(prog["converted"])
    prev_nll = prog["steps"][-1]["gate_eval"]["nll_student"] if prog["steps"] else None
    if prev_nll is None:
        prev_nll = align.evaluate(model, gate_w, {})["nll_teacher"]
    for k, i in enumerate(order):
        if k < len(prog["steps"]):
            continue
        snap = {j: copy.deepcopy(model.swaps[j].gdn.state_dict()) for j in converted}
        trial = converted + [i]
        modes = {j: "student" for j in trial}
        seed_all(cfg["seed"] + 10 + k)
        bt = data.Batcher(splits["train"], L, B, seed=cfg["seed"] + 100 + k)
        hist, ntok, sec = align.stage2_kl(model, trial if s2.get("train_all_converted", True) else [i], modes, bt,
                                          s2["steps"], s2["lr"], log=lambda r: log(r, f"stage2:L{i}"))
        ge = align.evaluate(model, gate_w, modes)
        step_dnll = ge["nll_student"] - prev_nll
        passed = bool(step_dnll <= gate["max_step_delta_nll"] and ge["hidden_cos"] >= gate["min_hidden_cos"])
        policy = gate.get("policy", "force")
        accepted = passed or policy == "force"
        rec = {"k": k, "layer": i, "gate_eval": ge, "step_delta_nll": step_dnll, "gate_pass": passed,
               "policy": policy, "accepted": accepted, "train_tokens": ntok, "train_seconds": sec,
               "train_tok_s": ntok / sec, "hist": hist}
        if accepted:
            converted = trial
            prev_nll = ge["nll_student"]
        else:  # rollback: layer i returns to attention, other GDNs restored to their pre-step weights
            for j, sd in snap.items():
                model.swaps[j].gdn.load_state_dict(sd)
            model.swaps[i].gdn.load_state_dict(stage1_state[i])
        rec["converted_after"] = list(converted)
        log({kk: v for kk, v in rec.items() if kk != "hist"}, "gate")
        save_gdns(f"prog_{k + 1}", [j for j in A if model.swaps[j].gdn is not None])
        if accepted:
            tag = f"conv{len(converted)}_kept{len(A) - len(converted)}"
            rec["probe_tag"] = tag
            run_probes(tag, {j: "student" for j in converted})
        prog["steps"].append(rec)
        prog["converted"] = converted
        R.set("progressive", prog)

    # ------------------------------------------------------------------ throughput / memory
    if "throughput" not in R and cfg.get("throughput"):
        tcfg = cfg["throughput"]
        res = {"stock": measure_throughput(model, {}, tcfg["lengths"], tcfg.get("reps", 3)),
               f"converted_{len(converted)}": measure_throughput(model, {j: "student" for j in converted},
                                                                 tcfg["lengths"], tcfg.get("reps", 3))}
        R.set("throughput", res)
    log({"done": True, "converted": converted}, "end")


if __name__ == "__main__":
    main()
