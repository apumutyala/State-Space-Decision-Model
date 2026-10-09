"""Q9-AF v3: Qwen3.5-9B -> fully attention-free (all 8 attention layers -> Gated DeltaNet) with the 2025-26 recipe
upgrades over v2 (jobs/q9af_v3/README.md: the pre-registered plan and gate). Three subcommands, one per GPU phase:

    python -m convert.v3 gen    --config C --mix <mixdata dir> --out <gen dir>                 (job A, phase G)
    python -m convert.v3 stageA --config C --mix <mixdata dir> --gen <gen dir> --out <dir>     (job A, phase P)
    python -m convert.v3 stageB --config C --a <stage-A artifact dir> --out <dir>              (job B, phase P)

gen     the long-context block split (``lc.u32`` for job B), then TEACHER-GENERATED distillation data: prompts from the
        decontaminated mixture (chat docs up to the first assistant header; web / math doc prefixes), completions
        sampled from the stock open-weight Qwen3.5-9B (all attention, T = 1.0), packed into 4,096-token rows with a
        completion mask; wall-clock and token capped. Also the disjoint on-policy prompt set for job B.
stageA  stage 1 (MOHAWK alignment of all 8 GDNs from the reuse_attn init, teacher-driven) -> per-layer SENSITIVITY probe
        (held-out KL with only that layer converted) -> the pre-registered order + stage budgets
        (``convert.distill.conversion_plan``) -> per-group GATED conversion: each stage switches its layers to GDN and
        distils ALL converted layers jointly on a mixed stream (generic windows: full-sequence KL; teacher-generated
        rows: completion-only KL + CE), evaluates held-out KL every ``eval_every_min`` and keeps the BEST checkpoint
        of the stage (restored before the next stage). Ends attention-free (8/8).
stageB  from stage A's GDNs: a 32k-token LONG-CONTEXT stage (full-sequence KL on the reserved 32k blocks), then a short
        ON-POLICY stage (the student samples, the teacher scores every position: reverse KL), each keeping its best
        checkpoint; final held-out KL (4k mix, 32k, teacher-generated completions) + throughput of both arms.
Every stage is wall-clock capped (``align.Guard``: rate floor exit 3, memory stop exit 4).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from convert import align, data, gendata  # noqa: E402
from convert.distill import conversion_plan, distill_accumulate, generate  # noqa: E402
from convert.mixdata import SOURCES  # noqa: E402
from convert.model import attach_gdn, init_gdn, load  # noqa: E402
from convert.pipeline import Results, env_info, seed_all  # noqa: E402

INIT = "reuse_attn"  # pre-registered (v2's default; no init_cmp in v3)


class Ctx:
    """Shared plumbing of the three subcommands: model, results, log, timing, checkpoints, held-out evals."""

    def __init__(self, cfg, out, need_model=True):
        self.cfg, self.out = cfg, out
        os.makedirs(out, exist_ok=True)
        self.R = Results(os.path.join(out, "results.json"))
        self.logf = open(os.path.join(out, "train_log.jsonl"), "a")
        self.t_start = time.time()
        json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
        seed_all(cfg["seed"])
        if cfg.get("vram_fraction") and torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(cfg["vram_fraction"])
        self.model, self.tok, mdir = load(os.path.expandvars(cfg["model_dir"]), kernel=cfg.get("kernel", "fla"))
        if "env" not in self.R:
            self.R.set("env", {**env_info(), "model_dir": mdir})
        self.A = list(self.model.attn_layers)
        if self.A != cfg["attention_layers"]:
            raise SystemExit(f"attention layers {self.A} != config {cfg['attention_layers']}")
        self.dev = next(self.model.lm.parameters()).device
        self.ck = os.path.expandvars(cfg["ckpt_dir"])
        if "$" in self.ck:
            raise SystemExit(f"ckpt_dir {self.ck!r}: unset environment variable")
        os.makedirs(self.ck, exist_ok=True)
        self.tcache = {}
        self.timing = dict(self.R["timing"]) if "timing" in self.R else {}

    def log(self, rec, phase=""):
        rec = {"t": time.strftime("%H:%M:%S"), "phase": phase, **rec}
        self.logf.write(json.dumps(rec) + "\n")
        self.logf.flush()
        print(json.dumps(rec), flush=True)

    def timed(self, name, t0):
        self.timing[name] = round(time.time() - t0, 2)
        self.R.set("timing", self.timing)

    def ids(self, s):
        return self.tok.convert_tokens_to_ids(s)

    # ---------------------------------------------------------------- held-out
    def set_heldout(self, mixdir):
        cfg, L = self.cfg, self.cfg["seq_len"]
        self.held = {s: data.fixed_windows(data.load_tokens(os.path.join(mixdir, f"heldout_{s}.u32")), L,
                                           cfg["heldout_windows"][s]) for s in SOURCES}
        for s, w in self.held.items():
            if len(w) != cfg["heldout_windows"][s]:
                raise SystemExit(f"held-out {s}: {len(w)} windows < {cfg['heldout_windows'][s]}")
        lc = cfg.get("long_eval")
        self.long_held = None
        if lc:
            self.long_held = data.fixed_windows(data.load_tokens(os.path.join(mixdir, f"heldout_{lc['source']}.u32")),
                                                lc["seq_len"], lc["windows"])
            if len(self.long_held) != lc["windows"]:
                raise SystemExit(f"long held-out: {len(self.long_held)} windows < {lc['windows']}")

    def teacher(self, name, windows, bs=4):
        if name not in self.tcache:
            self.tcache[name] = align.teacher_hidden(self.model, windows, bs)
        return self.tcache[name]

    def heldout_eval(self, modes, n_per=None):
        """Held-out KL / dNLL / hidden_cos per source and token-weighted (the v2 quantity, same windows)."""
        out, tot, w = {}, {"kl": 0.0, "delta_nll": 0.0, "hidden_cos": 0.0, "nll_student": 0.0, "nll_teacher": 0.0}, 0
        for s in SOURCES:
            win = self.held[s] if n_per is None else self.held[s][:n_per]
            tc = self.teacher(f"held_{s}", self.held[s])
            tc = tc if n_per is None else tc[: -(-len(win) // 4)]
            e = align.evaluate(self.model, win, modes, teacher=tc)
            out[s] = e
            for k in tot:
                tot[k] += e[k] * len(win)
            w += len(win)
        out["mix"] = {k: v / w for k, v in tot.items()}
        return out

    def long_eval(self, modes):
        if self.long_held is None:
            return None
        return align.evaluate(self.model, self.long_held, modes, bs=1,
                              teacher=self.teacher("long", self.long_held, bs=1))

    @torch.no_grad()
    def masked_eval(self, rows, mask, modes, bs=4):
        """Completion-only KL(teacher || student) and student/teacher CE on teacher-generated rows."""
        if rows is None or not len(rows):
            return None
        w_all = gendata.pred_weights(mask)
        kl = ce_s = ce_t = wt = 0.0
        for s in range(0, len(rows), bs):
            x = torch.from_numpy(rows[s:s + bs].astype(np.int64)).to(self.dev)
            wp = torch.from_numpy(w_all[s:s + bs]).to(self.dev)
            y = torch.zeros_like(x)
            y[:, :-1] = x[:, 1:]
            ac = torch.autocast("cuda", dtype=torch.bfloat16) if self.dev.type == "cuda" else _null()
            with ac:
                with self.model.modes({}):
                    ht = self.model.hidden(x)
                with self.model.modes(modes):
                    hs = self.model.hidden(x)
            for b in range(x.shape[0]):
                for c in range(0, x.shape[1], 512):
                    wc = wp[b, c:c + 512]
                    if not bool((wc > 0).any()):
                        continue
                    ls = self.model.lm_head(hs[b, c:c + 512]).float().log_softmax(-1)
                    lt = self.model.lm_head(ht[b, c:c + 512]).float().log_softmax(-1)
                    yy = y[b, c:c + 512, None]
                    kl += float(((lt.exp() * (lt - ls)).sum(-1) * wc).sum())
                    ce_s -= float((ls.gather(-1, yy)[:, 0] * wc).sum())
                    ce_t -= float((lt.gather(-1, yy)[:, 0] * wc).sum())
                    wt += float(wc.sum())
        return {"kl": kl / wt, "ce_student": ce_s / wt, "ce_teacher": ce_t / wt, "positions": wt}

    # ---------------------------------------------------------------- GDN checkpoints
    def gdn_state(self, layers=None):
        return {i: {k: v.detach().to("cpu", copy=True) for k, v in self.model.swaps[i].gdn.state_dict().items()}
                for i in (layers or self.A) if self.model.swaps[i].gdn is not None}

    def load_state(self, sd):
        for i, s in sd.items():
            i = int(i)
            if self.model.swaps[i].gdn is None:
                attach_gdn(self.model, i, init_gdn(self.model, i, "random", self.cfg["seed"]))
            self.model.swaps[i].gdn.load_state_dict(s)

    def save(self, name, sd=None, meta=None):
        sd = sd if sd is not None else self.gdn_state()
        tmp = os.path.join(self.ck, name + ".pt.tmp")
        torch.save(sd, tmp)
        os.replace(tmp, os.path.join(self.ck, name + ".pt"))
        with open(os.path.join(self.ck, name + ".json"), "w") as f:
            json.dump({"t": time.strftime("%FT%TZ", time.gmtime()), **(meta or {})}, f)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ----------------------------------------------------------------------------- one gated distillation stage
def run_stage(C, name, train_layers, modes, next_batch, guard_cfg, lr, eval_fn, log_every=25, warmup=20,
              before=None):
    """One capped stage: AdamW over the GDNs of ``train_layers``; ``next_batch()`` -> (step_fn, ntok) where step_fn
    accumulates the batch's gradient and returns a log dict. Every ``eval_every_min`` and at the end: ``eval_fn()``
    -> {"crit": float, ...}; the stage KEEPS its best-criterion checkpoint (restored at the end). ``before`` = the
    criterion at entry (the stage's starting point counts as a candidate)."""
    params = list(C.model.gdn_parameters(train_layers))
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    g = align.Guard(guard_cfg, name)
    every = guard_cfg.get("eval_every_min")
    hist, curve, ntok, step, t0 = [], [], 0, 0, time.time()
    best = {"crit": before["crit"], "step": -1, "tokens": 0, "sd": C.gdn_state(train_layers)} if before else None
    next_eval = time.time() + 60 * every if every else None
    C.model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    C.model.lm.train()
    exhausted = False
    try:
        while True:
            nb = next_batch()
            if nb is None:  # the stage's data ran out: stop gracefully (final eval, keep-best still apply)
                exhausted = stop = True
            else:
                for gp in opt.param_groups:
                    gp["lr"] = align._cosine_lr(step, 10 ** 9, warmup, lr, g)
                opt.zero_grad(set_to_none=True)
                fn, n = nb
                rec = fn()
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                ntok += n
                stop = g.step(step, ntok)
                if step % log_every == 0 or stop:
                    r = {"step": step, "tokens": ntok, "tok_s": ntok / (time.time() - t0),
                         "grad_norm": round(float(gn), 4), "lr": opt.param_groups[0]["lr"], "peak_gb": align._peak_gb(),
                         **rec}
                    hist.append(r)
                    C.log(r, name)
            if stop or (next_eval and time.time() >= next_eval):
                C.model.lm.eval()
                C.model.lm.gradient_checkpointing_disable()
                ev = {"step": step, "tokens": ntok, "minutes": round((time.time() - t0) / 60, 1), **eval_fn()}
                C.model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                C.model.lm.train()
                curve.append(ev)
                C.log({"curve": ev}, name)
                if best is None or ev["crit"] < best["crit"]:
                    best = {"crit": ev["crit"], "step": step, "tokens": ntok, "sd": C.gdn_state(train_layers)}
                    C.save(f"{name}_best", best["sd"], {"step": step, "tokens": ntok, "crit": ev["crit"]})
                next_eval = time.time() + 60 * every if every else None
            if nb is not None:
                step += 1
            if stop:
                break
    finally:
        C.model.lm.eval()
        C.model.lm.gradient_checkpointing_disable()
    C.load_state(best["sd"])
    return {"hist": hist, "curve": curve, "tokens": ntok, "steps": step, "seconds": time.time() - t0,
            "steady_tok_s": g.rate, "peak_gb": g.peak_gb(), "best": {k: v for k, v in best.items() if k != "sd"},
            "kept_last": best["step"] == step - 1, "train_layers": [int(i) for i in train_layers],
            "data_exhausted": exhausted}


# ----------------------------------------------------------------------------- gen (job A, phase G)
def cmd_gen(cfg, mixdir, out):
    C = Ctx(cfg, out)
    gc = cfg["gen"]
    mix = json.load(open(os.path.join(mixdir, "mix.json")))
    train = data.load_tokens(os.path.join(mixdir, "train.u32"))
    eos, im_end = C.ids("<|endoftext|>"), C.ids("<|im_end|>")
    split = gendata.BlockSplit(len(train), cfg["seq_len"], cfg["lc"]["seq_len"], cfg["lc"]["blocks"], cfg["seed"])
    if not os.path.exists(os.path.join(out, "lc.u32")):
        lc = split.lc_tokens(train)
        lc.tofile(os.path.join(out, "lc.u32.tmp"))
        os.replace(os.path.join(out, "lc.u32.tmp"), os.path.join(out, "lc.u32"))
    json.dump({**split.identity(), "lc_sha256": _sha(os.path.join(out, "lc.u32")),
               "train_sha256": mix.get("train_sha256")}, open(os.path.join(out, "split.json"), "w"), indent=1)
    spans = gendata.doc_spans(train, eos)
    bounds = gendata.source_bounds(mix)
    header = C.tok(gendata.CHAT_ASSISTANT_HEADER, add_special_tokens=False)["input_ids"]
    rng_seed = cfg["seed"] + 41
    pc = {k: gc[k] for k in ("doc_prompt_tokens", "min_continuation", "max_prompt_tokens")}
    prompts, used, pstats = gendata.select_prompts(train, spans, bounds, split, header,
                                                   gendata.split_counts(gc["n_prompts"], gc["shares"]), pc, rng_seed)
    want = gendata.split_counts(gc["n_prompts"], gc["shares"])
    short = {s: (pstats[s]["taken"], n) for s, n in want.items() if n and pstats[s]["taken"] < 0.5 * n}
    if short:  # e.g. the chat header never matched: fail loudly instead of distilling on a silently skewed mix
        raise SystemExit(f"prompt selection short of half its quota (taken, wanted): {short}; stats {pstats}")
    oc = gc["onpolicy"]
    opc = {**pc, "max_prompt_tokens": oc["max_prompt_tokens"]}
    op, _, opstats = gendata.select_prompts(train, spans, bounds, split, header,
                                            gendata.split_counts(oc["n_prompts"], oc["shares"]), opc, rng_seed + 1,
                                            exclude=used)
    gendata.save_prompts(os.path.join(out, "onpolicy.npz"), op)
    # deterministic generation order: seeded shuffle, then length-sorted inside super-chunks (less padding without
    # biasing a time-capped prefix towards short prompts)
    order = list(np.random.default_rng(rng_seed + 2).permutation(len(prompts)))
    sc = gc["batch"] * gc.get("sort_chunk_batches", 8)
    order = [i for k in range(0, len(order), sc) for i in sorted(order[k:k + sc], key=lambda j: len(prompts[j]["ids"]))]
    g = torch.Generator(device=C.dev).manual_seed(cfg["seed"] + 43)
    samples, bstats, ntok, t0 = [], [], 0, time.time()
    stop_reason = "prompts_exhausted"
    for k in range(0, len(order), gc["batch"]):
        if (time.time() - t0) / 60 >= gc["max_minutes"]:
            stop_reason = "max_minutes"
            break
        if ntok >= gc["target_completion_tokens"]:
            stop_reason = "target_tokens"
            break
        batch = [prompts[i] for i in order[k:k + gc["batch"]]]
        comps, st = generate(C.model, {}, [p["ids"] for p in batch], gc["max_new"], [eos, im_end], eos,
                             gc["temperature"], g, C.dev)
        for p, c in zip(batch, comps):
            samples.append((p["ids"], c, p["src"]))
        ntok += st["completion_tokens"]
        st["tok_s"] = st["completion_tokens"] / max(st["seconds"], 1e-9)
        bstats.append(st)
        if len(bstats) % 10 == 1:
            C.log({"batch": len(bstats), "samples": len(samples), "completion_tokens": ntok,
                   "tok_s_cum": ntok / (time.time() - t0), **st}, "gen")
    sec = time.time() - t0
    rows, mask, kst = gendata.pack([(p, c) for p, c, _ in samples], cfg["seq_len"], eos)
    hr = min(gc["heldout_rows"], len(rows) // 4)
    np.save(os.path.join(out, "gen_rows.npy"), rows[: len(rows) - hr])
    np.save(os.path.join(out, "gen_mask.npy"), mask[: len(rows) - hr])
    np.savez(os.path.join(out, "gen_heldout.npz"), rows=rows[len(rows) - hr:], mask=mask[len(rows) - hr:])
    by_src = {s: sum(len(c) for _, c, ss in samples if ss == s) for s in SOURCES}
    res = {"teacher": "stock Qwen3.5-9B (all attention layers; open weights; config model_dir)",
           "sampling": {"temperature": gc["temperature"], "max_new": gc["max_new"], "stop": ["<|endoftext|>",
                                                                                            "<|im_end|>"]},
           "prompts": {"selected": len(prompts), "stats": pstats, "used_for_generation": len(samples)},
           "onpolicy_prompts": {"selected": len(op), "stats": opstats},
           "stop_reason": stop_reason, "seconds": sec, "completion_tokens": ntok,
           "completion_tokens_by_source": by_src, "tok_s": ntok / max(sec, 1e-9), "batches": len(bstats),
           "pack": kst, "train_rows": int(len(rows) - hr), "heldout_rows": int(hr),
           "split": split.identity(), "batch_stats_head": bstats[:3]}
    C.R.set("gen", res)
    C.timed("gen", C.t_start)
    C.log({"done": True, **{k: res[k] for k in ("completion_tokens", "tok_s", "stop_reason", "train_rows")}}, "gen")


def _sha(p):
    from convert.mixdata import sha256_file

    return sha256_file(p)


# ----------------------------------------------------------------------------- stage A (job A, phase P)
def cmd_stage_a(cfg, mixdir, gendir, out):
    C = Ctx(cfg, out)
    m, A, L, B, MB = C.model, C.A, cfg["seq_len"], cfg["batch"], cfg.get("micro_batch")
    mix = json.load(open(os.path.join(mixdir, "mix.json")))
    if "mix" not in C.R:
        C.R.set("mix", mix)
    train = data.load_tokens(os.path.join(mixdir, "train.u32"))
    split = gendata.BlockSplit(len(train), L, cfg["lc"]["seq_len"], cfg["lc"]["blocks"], cfg["seed"])
    sp = json.load(open(os.path.join(gendir, "split.json")))
    if {k: sp[k] for k in split.identity()} != split.identity():
        raise SystemExit("stage A block split != the gen phase's split.json")
    generic = gendata.IndexedWindowStream(train, L, B, split.short_windows(), cfg["seed"] + 7, C.dev)
    rows = np.load(os.path.join(gendir, "gen_rows.npy"), mmap_mode="r")
    mask = np.load(os.path.join(gendir, "gen_mask.npy"), mmap_mode="r")
    gh = np.load(os.path.join(gendir, "gen_heldout.npz"))
    gh_rows, gh_mask = gh["rows"], gh["mask"]
    dc = cfg["distill"]
    stream = gendata.MixedStream(generic, rows, mask, B, dc["gen_frac"], cfg["seed"], C.dev)
    C.set_heldout(mixdir)
    cw = cfg["curve_windows_per_source"]

    # ---------------- stage 1: init + MOHAWK alignment of all 8 (teacher-driven, simultaneous)
    s1 = cfg["stage1"]
    if "stage1" not in C.R:
        t0 = time.time()
        seed_all(cfg["seed"])
        for i in A:
            attach_gdn(m, i, init_gdn(m, i, INIT, cfg["seed"]))
        scales = align.calibrate_out_scale(m, A, next(generic))
        g1 = align.Guard(s1["guard"], "stage1")
        hist, ntok, sec = align.stage1_align(m, A, generic, 10 ** 9, s1["lr"], log=lambda r: C.log(r, "stage1"),
                                             guard=g1, micro_batch=MB)
        C.save("stage1", meta={"tokens": ntok})
        C.R.set("stage1", {"init": INIT, "hist": hist, "tokens": ntok, "seconds": sec, "steady_tok_s": g1.rate,
                           "peak_gb": g1.peak_gb(), "out_scale_calibration": {str(k): v for k, v in scales.items()},
                           "reuse_info": {str(i): getattr(m.swaps[i].gdn, "reuse_info", None) for i in A},
                           "windows_used_after": generic.windows_used})
        C.timed("stage1", t0)
    else:
        C.load_state(torch.load(os.path.join(C.ck, "stage1.pt"), map_location=C.dev))
        generic.used = 0
        for _ in range(C.R["stage1"]["windows_used_after"] // B):
            generic.take()

    # ---------------- sensitivity probe -> the pre-registered order and stage budgets
    if "plan" not in C.R:
        t0 = time.time()
        sens, detail = {}, {}
        for i in A:
            e = C.heldout_eval({i: "student"}, cw)
            sens[i], detail[str(i)] = e["mix"]["kl"], e["mix"]
            C.log({"layer": i, "kl": e["mix"]["kl"]}, "probe")
        plan = conversion_plan(sens, dc["group_size"], dc["kd_minutes"], dc["floor_minutes"])
        plan["detail"] = detail
        C.R.set("plan", plan)
        C.timed("probe", t0)
    plan = C.R["plan"]

    # ---------------- gated per-group conversion
    stages = dict(C.R["stages"]) if "stages" in C.R else {}
    for k, st in enumerate(plan["stages"]):
        key = f"stage{k + 1}"
        if key in stages:
            C.load_state(torch.load(os.path.join(C.ck, key + ".pt"), map_location=C.dev))
            continue
        conv = st["converted"]
        modes = {j: "student" for j in conv}

        def next_batch(modes=modes):
            x, w, kind = next(stream)
            if kind == "generic":
                return (lambda: {"kind": 0, "kl": round(align.stage2_accumulate(m, modes, x, MB), 5)}), x.numel()
            return (lambda: {"kind": 1, **{k_: round(v, 5) for k_, v in distill_accumulate(
                m, modes, x, w, MB, "fkl", dc["ce_weight"]).items() if k_ != "weight"}}), x.numel()

        def eval_fn(modes=modes):
            e = C.heldout_eval(modes, cw)
            return {"crit": e["mix"]["kl"], "heldout": e}

        guard = {**dc["guard"], "max_minutes": st["minutes"], "eval_every_min": dc["eval_every_min"]}
        before = eval_fn()
        res = run_stage(C, key, conv, modes, next_batch, guard, dc["lr"], eval_fn, before=before)
        res.update({"adds": st["adds"], "converted": conv, "minutes_budget": st["minutes"], "entry": before,
                    "stream": stream.state()})
        C.save(key, meta={"converted": conv, "best": res["best"]})
        stages[key] = res
        C.R.set("stages", stages)
    allm = {j: "student" for j in A}
    if "final" not in C.R:
        t0 = time.time()
        C.save("stageA", meta={"converted": A})
        C.R.set("final", {"heldout": C.heldout_eval(allm), "gen_heldout": C.masked_eval(gh_rows, gh_mask, allm),
                          "stream": stream.state(),
                          "train_tokens_total": stream.state()["generic_windows_used"] * L})
        C.timed("final", t0)
    C.log({"done": True, "converted": A, "minutes": round((time.time() - C.t_start) / 60, 1)}, "end")


# ----------------------------------------------------------------------------- stage B (job B, phase P)
def cmd_stage_b(cfg, adir, out):
    C = Ctx(cfg, out)
    m, A, MB = C.model, C.A, cfg.get("micro_batch")
    sd = torch.load(os.path.join(adir, "gdn_converted.pt"), map_location=C.dev, weights_only=True)
    if sorted(int(i) for i in sd) != A:
        raise SystemExit(f"stage-A export layers {sorted(sd)} != {A}")
    for i in A:
        attach_gdn(m, i, init_gdn(m, i, "random", cfg["seed"]))
        m.swaps[i].gdn.load_state_dict(sd[int(i)] if int(i) in sd else sd[str(i)])
    if os.path.exists(os.path.join(C.ck, "stageB_last.pt")):
        C.load_state(torch.load(os.path.join(C.ck, "stageB_last.pt"), map_location=C.dev))
    C.set_heldout(adir)
    gh = np.load(os.path.join(adir, "gen_heldout.npz"))
    allm = {j: "student" for j in A}
    cw = cfg["curve_windows_per_source"]

    def eval_fn():
        e = C.heldout_eval(allm, cw)
        return {"crit": e["mix"]["kl"], "heldout": e}

    if "entry" not in C.R:
        C.R.set("entry", {"heldout": C.heldout_eval(allm), "long": C.long_eval(allm),
                          "gen_heldout": C.masked_eval(gh["rows"], gh["mask"], allm)})
    # ---------------- long context (32k)
    lcc = cfg["long_context"]
    if "long_context" not in C.R:
        lc = data.load_tokens(os.path.join(adir, "lc.u32"))
        n = len(lc) // lcc["seq_len"]
        stream = gendata.IndexedWindowStream(lc, lcc["seq_len"], lcc["batch"], np.arange(n), cfg["seed"] + 61, C.dev)

        def nb_lc():
            if stream.pos + lcc["batch"] > len(stream.order):  # every reserved 32k block read once: stop
                return None
            x = next(stream)
            return (lambda: {"kl": round(align.stage2_accumulate(m, allm, x, lcc["micro_batch"]), 5)}), x.numel()

        res = run_stage(C, "long_context", A, allm, nb_lc, {**lcc["guard"], "eval_every_min": lcc["eval_every_min"]},
                        lcc["lr"], eval_fn, before=eval_fn())
        res.update({"windows_used": stream.windows_used, "windows_total": int(n), "passes": stream.passes,
                    "long_after": C.long_eval(allm)})
        C.save("stageB_last", meta={"after": "long_context"})
        C.R.set("long_context", res)
    # ---------------- on-policy (student samples, teacher scores: reverse KL)
    op = cfg["onpolicy"]
    if "onpolicy" not in C.R:
        prompts = gendata.load_prompts(os.path.join(adir, "onpolicy.npz"))
        order = np.random.default_rng(cfg["seed"] + 71).permutation(len(prompts))
        eos, im_end = C.ids("<|endoftext|>"), C.ids("<|im_end|>")
        g = torch.Generator(device=C.dev).manual_seed(cfg["seed"] + 73)
        state = {"pos": 0, "rollout_s": 0.0, "completion_tokens": 0}

        def nb_op():
            if state["pos"] + op["rollout_batch"] > len(order):
                return None
            batch = [prompts[i]["ids"] for i in order[state["pos"]:state["pos"] + op["rollout_batch"]]]
            state["pos"] += op["rollout_batch"]
            was = m.lm.training
            m.lm.eval()
            m.lm.gradient_checkpointing_disable()
            t1 = time.time()
            comps, st = generate(m, allm, batch, op["max_new"], [eos, im_end], eos, op["temperature"], g, C.dev)
            state["rollout_s"] += time.time() - t1
            state["completion_tokens"] += st["completion_tokens"]
            m.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if was:
                m.lm.train()
            P = max(len(p) for p in batch)
            T = P + max(1, max(len(c) for c in comps))
            x = torch.full((len(batch), T), eos, dtype=torch.long)
            w = torch.zeros((len(batch), T))
            for r, (p, c) in enumerate(zip(batch, comps)):
                x[r, P - len(p):P] = torch.as_tensor([int(t) for t in p])
                if c:
                    x[r, P:P + len(c)] = torch.as_tensor(c)
                    w[r, P - 1:P - 1 + len(c)] = 1.0  # position t predicts token t + 1: the completion tokens
            x, w = x.to(C.dev), w.to(C.dev)
            return (lambda: {k_: round(v, 5) for k_, v in distill_accumulate(m, allm, x, w, op["micro_batch"],
                                                                             "rkl").items() if k_ != "weight"}), \
                int(w.sum())

        res = run_stage(C, "onpolicy", A, allm, nb_op, {**op["guard"], "eval_every_min": op["eval_every_min"]},
                        op["lr"], eval_fn, before=eval_fn())
        res.update({"prompts_used": state["pos"], "prompts_total": len(prompts), "rollout_seconds": state["rollout_s"],
                    "rollout_completion_tokens": state["completion_tokens"],
                    "tokens_note": "tokens = completion positions trained (the rollouts' scored tokens)"})
        C.save("stageB_last", meta={"after": "onpolicy"})
        C.R.set("onpolicy", res)
    # ---------------- final measurements (all 8 layers converted)
    if "final" not in C.R:
        t0 = time.time()
        C.save("final", meta={"converted": A})
        C.R.set("final", {"heldout": C.heldout_eval(allm), "long": C.long_eval(allm),
                          "gen_heldout": C.masked_eval(gh["rows"], gh["mask"], allm)})
        C.timed("final", t0)
    if "throughput" not in C.R and cfg.get("throughput"):
        from probes.throughput import call_latency
        from probes.throughput import measure as measure_throughput

        tc = cfg["throughput"]
        res = {"stock": measure_throughput(m, {}, tc["lengths"], tc.get("reps", 2)),
               "converted_8": measure_throughput(m, allm, tc["lengths"], tc.get("reps", 2))}
        for arm, modes in (("stock", {}), ("converted_8", allm)):
            try:
                res[arm]["call_latency"] = call_latency(m, modes)
            except Exception as e:  # noqa: BLE001
                res[arm]["call_latency_error"] = repr(e)[:300]
        C.R.set("throughput", res)
    C.log({"done": True, "converted": A, "minutes": round((time.time() - C.t_start) / 60, 1)}, "end")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="convert.v3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    a = sub.add_parser("stageA")
    b = sub.add_parser("stageB")
    for p in (g, a, b):
        p.add_argument("--config", required=True)
        p.add_argument("--out", required=True)
    for p in (g, a):
        p.add_argument("--mix", required=True)
    a.add_argument("--gen", required=True)
    b.add_argument("--a", required=True, help="stage-A artifact dir (gdn_converted.pt, lc.u32, onpolicy.npz, held-out)")
    ar = ap.parse_args(argv)
    cfg = json.load(open(ar.config))
    try:
        if ar.cmd == "gen":
            cmd_gen(cfg, ar.mix, ar.out)
        elif ar.cmd == "stageA":
            cmd_stage_a(cfg, ar.mix, ar.gen, ar.out)
        else:
            cmd_stage_b(cfg, ar.a, ar.out)
    except align.StopCondition as e:
        print(json.dumps({"STOP_CONDITION": e.reason, "exit_code": e.code, **e.data}), flush=True)
        sys.exit(e.code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
