"""Q9-AF v2: ALL 8 attention layers of Qwen3.5-9B -> Gated DeltaNet at once, layer-wise then end-to-end KD on a generic
mixture, gated end to end (jobs/q9af_v2/JOB_SPEC.md). No per-layer gate, no rollback: the model is attention-free by
construction, and its quality is judged as a whole (held-out generic-text KL + the stock Index proxy600 screen).

    python -m convert.joint --config configs/q9af_v2_h100.json --mix <mixdata dir> --out <dir>

Phases (each finished phase is written to <out>/results.json; GDN weights to <ckpt_dir>):
  stock      probes of the stock model (wikitext ppl, MMLU-285, MQAR, S-NIAH), the reference for every drop
  init_cmp   diagnostic: 40 stage-1 steps from reuse_attn and from clone_neighbor; the pre-registered init is reuse_attn
             (q/k/v/gate/o from the attention projections) unless clone_neighbor's rel-MSE is < 0.8x reuse's
  stage1     MOHAWK-style hidden alignment of all 8 GDNs (teacher-driven, simultaneous), wall-clock capped
  stage2     end-to-end KL(teacher || student) with ALL 8 GDNs in student mode and training, wall-clock capped; every
             ``eval_every_min`` a held-out KL curve point and a checkpoint (``stage2_last.pt``)
  final      held-out generic KL per source (the binding KL), wikitext gate numbers (comparable to job-2xe5w),
             converted probes, full MMLU stock vs converted, throughput + call latency of both arms
Every training window is drawn once from one permutation (``convert.mixdata.WindowStream``): no text is re-read.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from convert import align, data, memmodel  # noqa: E402
from convert.mixdata import SOURCES, WindowStream  # noqa: E402
from convert.model import attach_gdn, init_gdn, load, load_tokenizer, tokenizer_dir  # noqa: E402
from convert.pipeline import Results, env_info, seed_all  # noqa: E402

INIT_DEFAULT = "reuse_attn"
INIT_ALT = "clone_neighbor"
INIT_SWITCH_RATIO = 0.8  # clone_neighbor replaces reuse_attn only if its init_cmp rel-MSE is < 0.8 x reuse's


def stage2_joint(model, layers, stream, lr, guard, micro_batch, log, eval_fn=None, eval_every_min=None, save_fn=None,
                 warmup=20, log_every=25):
    """align.stage2_kl with every layer in ``layers`` in student mode and training, plus a periodic held-out eval and
    checkpoint. LR: warmup then cosine to 0.1x on the guard's wall-clock fraction."""
    params = list(model.gdn_parameters(layers))
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    modes = {j: "student" for j in layers}
    model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.train()
    hist, curve, ntok, t0 = [], [], 0, time.time()
    next_eval = time.time() + 60 * eval_every_min if eval_every_min else None
    step = 0
    try:
        while True:
            for g in opt.param_groups:
                g["lr"] = align._cosine_lr(step, 10 ** 9, warmup, lr, guard)
            x = next(stream)
            opt.zero_grad(set_to_none=True)
            kl = align.stage2_accumulate(model, modes, x, micro_batch)
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ntok += x.numel()
            stop = guard.step(step, ntok)
            if step % log_every == 0 or stop:
                rec = {"step": step, "tokens": ntok, "tok_s": ntok / (time.time() - t0), "kl": round(kl, 5),
                       "grad_norm": round(float(gn), 4), "lr": opt.param_groups[0]["lr"], "peak_gb": align._peak_gb()}
                hist.append(rec)
                log(rec)
            if eval_fn and (stop or (next_eval and time.time() >= next_eval)):
                model.lm.eval()
                ev = {"step": step, "tokens": ntok, "minutes": round((time.time() - t0) / 60, 1), **eval_fn()}
                model.lm.train()
                curve.append(ev)
                log({"curve": ev})
                if save_fn:
                    save_fn("stage2_last", ntok)
                next_eval = time.time() + 60 * eval_every_min if eval_every_min else None
            step += 1
            if stop:
                break
    finally:
        model.lm.eval()
        model.lm.gradient_checkpointing_disable()
    return hist, curve, ntok, time.time() - t0, step


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
    ap = argparse.ArgumentParser(prog="convert.joint")
    ap.add_argument("--config", required=True)
    ap.add_argument("--mix", required=True, help="convert.mixdata build output dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    os.makedirs(a.out, exist_ok=True)
    ckpt_dir = os.path.expandvars(cfg["ckpt_dir"])
    if "$" in ckpt_dir:
        raise SystemExit(f"ckpt_dir {ckpt_dir!r}: unset environment variable")
    os.makedirs(ckpt_dir, exist_ok=True)
    R = _R = Results(os.path.join(a.out, "results.json"))
    logf = open(os.path.join(a.out, "train_log.jsonl"), "a")

    def log(rec, phase=""):
        rec = {"t": time.strftime("%H:%M:%S"), "phase": phase, **rec}
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)

    timing = dict(R["timing"]) if "timing" in R else {}

    def timed(name, t0):
        timing[name] = round(time.time() - t0, 2)
        R.set("timing", timing)

    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=1)
    seed_all(cfg["seed"])
    if cfg.get("vram_fraction"):
        torch.cuda.set_per_process_memory_fraction(cfg["vram_fraction"])
    model, tok, mdir = load(os.path.expandvars(cfg["model_dir"]), kernel=cfg.get("kernel", "fla"))
    if cfg.get("tokenizer_dir"):  # opt-in (jobs/q9af_v2 mode clefflash): tokenize with another pinned tokenizer
        tok = load_tokenizer(cfg)
    if "env" not in R:
        R.set("env", {**env_info(), "model_dir": mdir,
                      **({"tokenizer_dir": tokenizer_dir(cfg)} if cfg.get("tokenizer_dir") else {})})
    A = list(model.attn_layers)
    if A != cfg["attention_layers"]:
        raise SystemExit(f"attention layers {A} != config {cfg['attention_layers']}")
    L, B, MB = cfg["seq_len"], cfg["batch"], cfg.get("micro_batch")
    dev = next(model.lm.parameters()).device

    # ------------------------------------------------------------------ data
    t0 = time.time()
    mix = json.load(open(os.path.join(a.mix, "mix.json")))
    if "mix" not in R:
        R.set("mix", {k: v for k, v in mix.items()})
    train = data.load_tokens(os.path.join(a.mix, "train.u32"))
    stream = WindowStream(train, L, B, seed=cfg["seed"] + 7, device=dev)
    held = {s: data.fixed_windows(data.load_tokens(os.path.join(a.mix, f"heldout_{s}.u32")), L, cfg["heldout_windows"][s])
            for s in SOURCES}
    for s, w in held.items():
        if len(w) != cfg["heldout_windows"][s]:
            raise SystemExit(f"held-out {s}: {len(w)} windows < {cfg['heldout_windows'][s]}")
    p_eval, _ = data.build(tok, "eval")
    eval_toks = data.load_tokens(p_eval)
    p_gate, _ = data.build(tok, "gate")
    gate_w = data.fixed_windows(data.load_tokens(p_gate), L, cfg["wikitext_gate_windows"])
    timed("load_and_data", t_start)
    tcache = {}

    def teacher(name, windows):
        if name not in tcache:
            tcache[name] = align.teacher_hidden(model, windows)
        return tcache[name]

    def heldout_eval(n_per=None, modes=None):
        """Held-out KL / dNLL / hidden_cos per source and token-weighted over sources (all 8 layers as student)."""
        modes = modes if modes is not None else {j: "student" for j in A}
        out, tot, w = {}, {"kl": 0.0, "delta_nll": 0.0, "hidden_cos": 0.0, "nll_student": 0.0, "nll_teacher": 0.0}, 0
        for s in SOURCES:
            win = held[s] if n_per is None else held[s][:n_per]
            tc = teacher(f"held_{s}", held[s])
            tc = tc if n_per is None else tc[: -(-len(win) // 4)]
            e = align.evaluate(model, win, modes, teacher=tc)
            out[s] = e
            for k in tot:
                tot[k] += e[k] * len(win)
            w += len(win)
        out["mix"] = {k: v / w for k, v in tot.items()}
        return out

    def save_gdns(name, ntok=None):
        sd = {i: model.swaps[i].gdn.state_dict() for i in A}
        tmp = os.path.join(ckpt_dir, name + ".pt.tmp")
        torch.save(sd, tmp)
        os.replace(tmp, os.path.join(ckpt_dir, name + ".pt"))
        if ntok is not None:
            with open(os.path.join(ckpt_dir, name + ".json"), "w") as f:
                json.dump({"tokens": ntok, "t": time.strftime("%FT%TZ", time.gmtime())}, f)

    def load_gdns(name):
        sd = torch.load(os.path.join(ckpt_dir, name + ".pt"), map_location=dev)
        for i in A:
            if model.swaps[i].gdn is None:
                attach_gdn(model, i, init_gdn(model, i, "random", cfg["seed"]))
            model.swaps[i].gdn.load_state_dict(sd[i])

    def run_probes(tag, modes):
        if f"probe_{tag}" in R:
            return R[f"probe_{tag}"]
        from probes import core as probes

        t1 = time.time()
        with model.modes(modes):
            res = probes.run_all(model, tok, eval_toks, cfg)
        res["modes"] = {str(k): v for k, v in modes.items()}
        res["seconds"] = time.time() - t1
        R.set(f"probe_{tag}", res)
        torch.cuda.empty_cache()
        log({"tag": tag, "ppl1k": res["ppl"]["ctx1024"]["ppl"], "niah": res["niah"]["mean_acc"],
             "mqar": res["mqar"]["mean_acc"], "mmlu": res["mmlu"]["acc"]}, "probe")
        return res

    # ------------------------------------------------------------------ stock probes
    t0 = time.time()
    run_probes("stock", {})
    timed("stock_probes", t0)
    try:
        dims = memmodel.Dims.from_config(model.cfg)
    except Exception:  # noqa: BLE001  (FLOP accounting is a receipt field, never a reason to stop)
        dims = None

    # ------------------------------------------------------------------ init comparison (pre-registered choice rule)
    ic = cfg["init_cmp"]
    if "init_cmp" not in R:
        t0 = time.time()
        res = {}
        for how in (INIT_DEFAULT, INIT_ALT):
            seed_all(cfg["seed"])
            for i in A:
                attach_gdn(model, i, init_gdn(model, i, how, cfg["seed"]))
            align.calibrate_out_scale(model, A, next(stream))
            hist, ntok, sec = align.stage1_align(model, A, stream, ic["steps"], ic["lr"],
                                                 log=lambda r, h=how: log(r, f"init_cmp:{h}"), micro_batch=MB)
            res[how] = {"last": hist[-1], "tokens": ntok, "seconds": sec,
                        "mean_last_rel_mse": float(np.mean([v for k, v in hist[-1].items() if k.startswith("rel_mse")]))}
        chosen = INIT_ALT if res[INIT_ALT]["mean_last_rel_mse"] < INIT_SWITCH_RATIO * res[INIT_DEFAULT]["mean_last_rel_mse"] \
            else INIT_DEFAULT
        R.set("init_cmp", {"results": res, "chosen": chosen, "rule": f"{INIT_ALT} iff its rel-MSE < "
                           f"{INIT_SWITCH_RATIO} x {INIT_DEFAULT}'s after {ic['steps']} steps",
                           "windows_used_after": stream.windows_used})
        timed("init_cmp", t0)
    else:
        stream.skip(R["init_cmp"]["windows_used_after"] - stream.windows_used)
    init_how = R["init_cmp"]["chosen"]

    # ------------------------------------------------------------------ stage 1: layer-wise alignment, all 8
    s1 = cfg["stage1"]
    if "stage1" not in R:
        t0 = time.time()
        seed_all(cfg["seed"])
        for i in A:
            attach_gdn(model, i, init_gdn(model, i, init_how, cfg["seed"]))
        scales = align.calibrate_out_scale(model, A, next(stream))
        f1 = memmodel.flops_stage1(dims, A) if dims else None
        g1 = align.Guard(s1["guard"], "stage1", flop_per_token=f1)
        hist, ntok, sec = align.stage1_align(model, A, stream, 10 ** 9, s1["lr"], log=lambda r: log(r, "stage1"),
                                             guard=g1, micro_batch=MB)
        save_gdns("stage1", ntok)
        ev = heldout_eval(cfg["curve_windows_per_source"])
        R.set("stage1", {"init": init_how, "hist": hist, "tokens": ntok, "seconds": sec, "steady_tok_s": g1.rate,
                         "peak_gb": g1.peak_gb(), "flop_per_token": f1,
                         "mfu": (g1.rate or 0) * f1 / memmodel.H100_PEAK if f1 else None,
                         "reuse_info": {str(i): getattr(model.swaps[i].gdn, "reuse_info", None) for i in A},
                         "out_scale_calibration": {str(k): v for k, v in scales.items()},
                         "heldout_after": ev, "windows_used_after": stream.windows_used})
        log({"stage1_heldout_mix": ev["mix"]}, "stage1")
        timed("stage1", t0)
    else:
        load_gdns("stage1")
        stream.skip(R["stage1"]["windows_used_after"] - stream.windows_used)

    # ------------------------------------------------------------------ stage 2: end-to-end KL, all 8 GDNs jointly
    s2 = cfg["stage2"]
    if "stage2" not in R:
        t0 = time.time()
        seed_all(cfg["seed"] + 10)
        for i in A:
            attach_gdn(model, i, model.swaps[i].gdn)  # fp32 master weights, requires_grad
        f2 = memmodel.flops_stage2(dims, A, n_train=len(A)) if dims else None
        g2 = align.Guard(s2["guard"], "stage2", flop_per_token=f2)
        hist, curve, ntok, sec, steps = stage2_joint(
            model, A, stream, s2["lr"], g2, MB, lambda r: log(r, "stage2"),
            eval_fn=lambda: heldout_eval(cfg["curve_windows_per_source"]), eval_every_min=s2["eval_every_min"],
            save_fn=save_gdns)
        save_gdns("final", ntok)
        R.set("stage2", {"hist": hist, "curve": curve, "tokens": ntok, "seconds": sec, "steps": steps,
                         "steady_tok_s": g2.rate, "peak_gb": g2.peak_gb(), "flop_per_token": f2,
                         "mfu": (g2.rate or 0) * f2 / memmodel.H100_PEAK if f2 else None,
                         "windows_used_after": stream.windows_used,
                         "train_tokens_total": stream.windows_used * L, "train_stream_tokens": int(len(train)),
                         "stream_passes": stream.passes,
                         "lr": s2["lr"], "layers": A})
        timed("stage2", t0)
    else:
        load_gdns("final")

    # ------------------------------------------------------------------ final measurements (all 8 layers converted)
    conv = {j: "student" for j in A}
    if "final_heldout" not in R:
        t0 = time.time()
        R.set("final_heldout", heldout_eval(None))
        timed("final_heldout", t0)
    if "wikitext_gate" not in R:  # the job-2xe5w gate quantities on the same 32 wikitext-103 validation windows
        R.set("wikitext_gate", align.evaluate(model, gate_w, conv, teacher=teacher("gate", gate_w)))
    t0 = time.time()
    run_probes("conv8", conv)
    timed("conv_probes", t0)
    if cfg.get("mmlu_full") and "mmlu_full" not in R:
        from probes import core as probes

        t0 = time.time()
        mf = {}
        for arm, modes in (("stock", {}), ("final", conv)):
            with model.modes(modes):
                res = probes.mmlu_full(model, tok, batch_tokens=cfg["mmlu_full"]["batch_tokens"])
            res["modes"] = {str(k): v for k, v in modes.items()}
            mf[arm] = res
            log({"tag": f"mmlu_full:{arm}", "acc": res["acc"], "n": res["n"]}, "probe")
            torch.cuda.empty_cache()
        R.set("mmlu_full", mf)
        timed("mmlu_full", t0)
    if "throughput" not in R and cfg.get("throughput"):
        from probes.throughput import call_latency
        from probes.throughput import measure as measure_throughput

        t0 = time.time()
        tc = cfg["throughput"]
        res = {"stock": measure_throughput(model, {}, tc["lengths"], tc.get("reps", 2)),
               "converted_8": measure_throughput(model, conv, tc["lengths"], tc.get("reps", 2))}
        for arm, modes in (("stock", {}), ("converted_8", conv)):
            try:
                res[arm]["call_latency"] = call_latency(model, modes)
            except Exception as e:  # noqa: BLE001
                res[arm]["call_latency_error"] = repr(e)[:300]
        R.set("throughput", res)
        timed("throughput", t0)
    log({"done": True, "converted": A, "minutes": round((time.time() - t_start) / 60, 1)}, "end")


if __name__ == "__main__":
    main()
