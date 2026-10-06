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
import sys
import time

import numpy as np

# must be set before CUDA initialises (as convert/qual.py, job-c4zkm 9B OOM): the caching allocator grows segments
# instead of fragmenting. The probe9b stage shapes are the ones job-4j3vu measured with this setting (peak 31.9 GB).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from convert import align, data, memmodel  # noqa: E402
from convert.model import attach_gdn, init_gdn, load  # noqa: E402
from convert.progressive import run_progressive  # noqa: E402
from probes import core as probes  # noqa: E402
from probes.throughput import call_latency  # noqa: E402
from probes.throughput import measure as measure_throughput  # noqa: E402


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
        hook = os.environ.get("PHASE_HOOK")  # e.g. the paid runner's per-phase artifact export; never fatal
        if hook:
            try:
                subprocess.run(hook + " " + k, shell=True, timeout=600)
            except Exception as e:  # noqa: BLE001
                print(f"PHASE_HOOK failed for {k}: {e}", flush=True)


def env_info():
    def git(*a):
        try:
            return subprocess.check_output(["git", *a], text=True).strip()
        except Exception:
            return None

    import transformers

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "platform": platform.platform(),
            "fla": __import__("importlib.metadata").metadata.version("flash-linear-attention")
            if __import__("importlib.util").util.find_spec("fla") else None}


def main():
    try:
        _main()
    except align.StopCondition as e:
        print(json.dumps({"STOP_CONDITION": e.reason, "exit_code": e.code, **e.data}), flush=True)
        if _R is not None:
            _R.set("stopped", {"reason": e.reason, "exit_code": e.code, **e.data})
        sys.exit(e.code)


_R = None


def _main():
    global _R
    t_start = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    os.makedirs(a.out, exist_ok=True)
    ckpt_dir = cfg.get("ckpt_dir", os.path.join(data.DATA_ROOT, "ckpt", os.path.basename(a.out.rstrip("/\\"))))
    os.makedirs(ckpt_dir, exist_ok=True)
    R = _R = Results(os.path.join(a.out, "results.json"))
    if os.environ.get("DEADLINE_MINUTES"):  # the paid runner sets the remaining budget
        cfg["deadline_minutes"] = float(os.environ["DEADLINE_MINUTES"])
    deadline = 60 * cfg["deadline_minutes"] if cfg.get("deadline_minutes") else None
    past_deadline = lambda: bool(deadline and time.time() - t_start > deadline)  # noqa: E731
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
    model, tok, mdir = load(os.path.expandvars(cfg["model_dir"]) if cfg.get("model_dir") else None,
                            kernel=cfg.get("kernel", "fla"))
    if "env" not in R:
        R.set("env", {**env_info(), "model_dir": mdir})
    splits = {}
    for s in ("train", "gate", "eval"):
        p, meta = data.build(tok, s, max_tokens=cfg["data"].get("train_max_tokens") if s == "train" else None)
        splits[s] = data.load_tokens(p)
        R.set(f"data_{s}", meta) if f"data_{s}" not in R else None
    L, B = cfg["seq_len"], cfg["batch"]
    # micro-batch with gradient accumulation (exactly the full-batch gradient, tests/test_microbatch.py). The 9B at
    # 8 x 4096 without it OOMed at 68.9 GB (job-c4zkm); job-4j3vu measured micro_batch 4 at 31.9 GB peak.
    MB = cfg.get("micro_batch")
    dev = next(model.lm.parameters()).device
    gate_w = data.fixed_windows(splits["gate"], L, cfg["gate"]["n_windows"])
    A = model.attn_layers
    pc = cfg
    try:
        dims = memmodel.Dims.from_config(model.cfg)
    except Exception:  # noqa: BLE001  (FLOP accounting is a receipt field, never a reason to stop)
        dims = None
    timing = dict(R["timing"]) if "timing" in R else {}

    def timed(name, t0):
        timing[name] = round(time.time() - t0, 2)
        R.set("timing", timing)

    timed("load_and_data", t_start)
    _tcache = {}

    def gate_teacher():
        # the teacher is frozen for the whole run: its final hidden states on the gate windows are computed ONCE and
        # reused by the importance ranking (one evaluate per attention layer), the gate baseline and every gate
        if "h" not in _tcache:
            t0 = time.time()
            _tcache["h"] = align.teacher_hidden(model, gate_w)
            timed("gate_teacher_cache", t0)
        return _tcache["h"]

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
    t0 = time.time()
    run_probes("stock", {})
    for w in cfg["controls"].get("swa_windows", [512]):
        for sw in model.swaps.values():
            sw.swa_window, sw.swa_sinks = w, cfg["controls"].get("swa_sinks", 4)
        run_probes(f"swa{w}_sinks{cfg['controls'].get('swa_sinks', 4)}", {i: "swa" for i in A})
    if cfg["controls"].get("zero", True):
        run_probes("attn_deleted", {i: "zero" for i in A})
    timed("controls", t0)

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
            bt = data.Batcher(splits["train"], L, B, seed=cfg["seed"] + 1, device=dev)
            if cfg.get("calibrate_out_scale", True):
                align.calibrate_out_scale(model, A, next(bt))
            hist, ntok, sec = align.stage1_align(model, A, bt, ic["steps"], ic["lr"],
                                                 log=lambda r, h=how: log(r, f"init_cmp:{h}"), micro_batch=MB)
            out[how] = {"first": hist[0], "last": hist[-1], "tokens": ntok, "seconds": sec,
                        "mean_last_rel_mse": float(np.mean([v for k, v in hist[-1].items() if k.startswith("rel_mse")]))}
        best = min(out, key=lambda h: out[h]["mean_last_rel_mse"])
        R.set("init_cmp", {"results": out, "chosen": best})
    init_how = R["init_cmp"]["chosen"] if "init_cmp" in R else cfg["init"]

    # ------------------------------------------------------------------ stage 1
    s1 = cfg["stage1"]
    if "stage1" not in R:
        t0 = time.time()
        seed_all(cfg["seed"])
        for i in A:
            attach_gdn(model, i, init_gdn(model, i, init_how, cfg["seed"]))
        init_eval = {i: align.evaluate(model, gate_w[:8], {i: "student"}) for i in A} if s1.get("eval_init") else None
        bt = data.Batcher(splits["train"], L, B, seed=cfg["seed"] + 2, device=dev)
        scales = align.calibrate_out_scale(model, A, next(bt)) if cfg.get("calibrate_out_scale", True) else None
        f1 = memmodel.flops_stage1(dims, A) if dims else None
        g1 = align.Guard(s1.get("guard"), "stage1", flop_per_token=f1)
        hist, ntok, sec = align.stage1_align(model, A, bt, s1["steps"], s1["lr"], log=lambda r: log(r, "stage1"),
                                             guard=g1, micro_batch=MB)
        save_gdns("stage1", A)
        R.set("stage1", {"init": init_how, "hist": hist, "tokens": ntok, "seconds": sec, "tok_s": ntok / sec,
                         "steady_tok_s": g1.rate, "peak_gb": g1.peak_gb(), "micro_batch": MB or B,
                         "flop_per_token": f1, "mfu": (g1.rate or 0) * f1 / memmodel.H100_PEAK if f1 else None,
                         "reuse_info": {str(i): getattr(model.swaps[i].gdn, "reuse_info", None) for i in A},
                         "init_eval_single_swap": init_eval,
                         "out_scale_calibration": {str(k): v for k, v in (scales or {}).items()}})
        timed("stage1", t0)
    else:
        load_gdns("stage1")

    # ------------------------------------------------------------------ importance (KL of single swaps)
    if "importance" not in R:
        t0 = time.time()
        imp = {}
        for i in A:
            imp[i] = align.evaluate(model, gate_w, {i: "student"}, teacher=gate_teacher())
            log({"layer": i, **imp[i]}, "importance")
        order = sorted(A, key=lambda i: imp[i]["kl"]) if cfg.get("order", "kl") == "kl" else list(cfg["order"])
        R.set("importance", {"per_layer": {str(k): v for k, v in imp.items()}, "order": order,
                             "criterion": "ascending KL(teacher||single-swap student) on gate split after stage1"})
        timed("importance", t0)
    order = R["importance"]["order"]

    # ------------------------------------------------------------------ progressive conversion
    s2, gate = cfg["stage2"], cfg["gate"]
    prog = R["progressive"] if "progressive" in R else {"steps": [], "converted": []}
    stage1_state = {i: copy.deepcopy(model.swaps[i].gdn.state_dict()) for i in A}

    class Ops:
        def load(self, n):
            load_gdns(f"prog_{n}")

        def snapshot(self, converted):
            return {j: copy.deepcopy(model.swaps[j].gdn.state_dict()) for j in converted}

        def train(self, k, i, trial):
            seed_all(cfg["seed"] + 10 + k)
            bt = data.Batcher(splits["train"], L, B, seed=cfg["seed"] + 100 + k, device=dev)
            layers_train = trial if s2.get("train_all_converted", True) else [i]
            f2 = memmodel.flops_stage2(dims, trial, n_train=len(layers_train)) if dims else None
            g2 = align.Guard(s2.get("guard"), f"stage2:L{i}", flop_per_token=f2)
            hist, ntok, sec = align.stage2_kl(model, layers_train,
                                              {j: "student" for j in trial}, bt, s2["steps"], s2["lr"],
                                              log=lambda r: log(r, f"stage2:L{i}"), guard=g2, micro_batch=MB)
            return {"train_tokens": ntok, "train_seconds": sec, "train_tok_s": ntok / sec,
                    "steady_tok_s": g2.rate, "peak_gb": g2.peak_gb(), "micro_batch": MB or B, "flop_per_token": f2,
                    "mfu": (g2.rate or 0) * f2 / memmodel.H100_PEAK if f2 else None, "hist": hist}

        def evaluate(self, trial):
            return align.evaluate(model, gate_w, {j: "student" for j in trial}, teacher=gate_teacher())

        def restore(self, snap, i):
            for j, sd in snap.items():
                model.swaps[j].gdn.load_state_dict(sd)
            model.swaps[i].gdn.load_state_dict(stage1_state[i])

        def save(self, n):
            save_gdns(f"prog_{n}", [j for j in A if model.swaps[j].gdn is not None])

        def teacher_nll(self):
            return align.evaluate(model, gate_w, {}, teacher=gate_teacher())["nll_teacher"]

        def probe(self, converted):
            tag = f"conv{len(converted)}_kept{len(A) - len(converted)}"
            run_probes(tag, {j: "student" for j in converted})
            return tag

    t0 = time.time()
    converted = run_progressive(order, prog, gate, Ops(), cfg.get("progressive", {}).get("max_layers"),
                                past_deadline, lambda p: R.set("progressive", p), log)
    timed("progressive", t0)

    # ------------------------------------------------------------------ throughput / memory
    if "throughput" not in R and cfg.get("throughput"):
        t0 = time.time()
        tcfg = cfg["throughput"]
        res = {"stock": measure_throughput(model, {}, tcfg["lengths"], tcfg.get("reps", 3)),
               f"converted_{len(converted)}": measure_throughput(model, {j: "student" for j in converted},
                                                                 tcfg["lengths"], tcfg.get("reps", 3))}
        # readout per-call cost on THIS model (the Index preflight's call_s; measurement only, never fatal)
        if tcfg.get("call_latency", True):
            for arm, modes in (("stock", {}), (f"converted_{len(converted)}", {j: "student" for j in converted})):
                try:
                    res[arm]["call_latency"] = call_latency(model, modes)
                except Exception as e:  # noqa: BLE001
                    res[arm]["call_latency_error"] = repr(e)[:300]
        R.set("throughput", res)
        timed("throughput", t0)
    log({"done": True, "converted": converted}, "end")


if __name__ == "__main__":
    main()
