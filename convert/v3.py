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
from convert.model import attach_from_state, attach_gdn, init_gdn, load  # noqa: E402
from convert.recall import MultiStream, RecallSynth  # noqa: E402
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
            g = self.model.swaps[i].gdn
            if g is None or g.in_proj_qkv.weight.shape != s["in_proj_qkv.weight"].shape:
                attach_from_state(self.model, i, s, seed=self.cfg["seed"])
            else:
                g.load_state_dict(s)

    def new_gdn(self, i, how=None, dk="cfg"):
        """Attach a fresh converted GDN for layer i: the config's ``gdn.dk`` (v3: widened key dim) unless overridden."""
        dk = (self.cfg.get("gdn") or {}).get("dk") if dk == "cfg" else dk
        return attach_gdn(self.model, i, init_gdn(self.model, i, how or self.cfg.get("init", "reuse_attn"),
                                                  self.cfg["seed"], dk=dk))

    # ---------------------------------------------------------------- recall probe (the v2 probe settings)
    def recall_eval(self, modes):
        """MQAR + NIAH (probes.core, the exact probes v2's MQAR / NIAH drops were measured with) in ``modes``."""
        from probes import core as probes

        rp = self.cfg["recall_probe"]
        if getattr(self, "_eval_toks", None) is None:
            p_eval, _ = data.build(self.tok, "eval")
            self._eval_toks = data.load_tokens(p_eval)
        t0 = time.time()
        with self.model.modes(modes):
            mq = probes.mqar(self.model, self.tok, n_pairs=rp["mqar_pairs"], n_seq=rp["mqar_n_seq"])
            nh = probes.niah(self.model, self.tok, self._eval_toks, lengths=rp["niah_lengths"], n_per=rp["niah_n_per"])
        return {"mqar": mq, "niah": nh, "mqar_mean": mq["mean_acc"], "niah_mean": nh["mean_acc"],
                "seconds": time.time() - t0}

    def recall_drop(self, rec, ref):
        return {"mqar_drop": ref["mqar_mean"] - rec["mqar_mean"], "niah_drop": ref["niah_mean"] - rec["niah_mean"]}

    def recall_synth(self, train, window_ids):
        from probes.core import _key_words

        return RecallSynth(self.tok, train, window_ids, self.cfg["seq_len"], self.cfg["seed"] + 101,
                           self.cfg["recall_data"], exclude_words=_key_words(self.tok))

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
def cmd_gen(cfg, mixdir, out, eval_index_dir=None):
    C = Ctx(cfg, out)
    gc = cfg["gen"]
    if gc.get("check_templates", True):  # the instruction templates must pass the full-eval-v1 filter
        from convert.mixdata import full_filter, load_eval_index_dir

        if not eval_index_dir:
            raise SystemExit("gen.check_templates needs --eval-index-dir (the shipped full-eval-v1 index)")
        chk = gendata.check_templates(full_filter(load_eval_index_dir(eval_index_dir))[0])
        C.R.set("template_check", chk)
        if chk["flagged"]:
            raise SystemExit(f"instruction templates flagged by full-eval-v1: {chk['flagged']}")
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
    kinds = [*SOURCES, *gendata.INSTRUCTION_KINDS]
    want = {k: v for k, v in zip(kinds, _counts(gc["n_prompts"], {k: gc["shares"].get(k, 0.0) for k in kinds}))}
    prompts, used, pstats = gendata.select_prompts(train, spans, bounds, split, header,
                                                   {s: want[s] for s in SOURCES}, pc, rng_seed)
    ic = {**gc["instruct"], "max_prompt_tokens": gc["max_prompt_tokens"]}
    ip, iused, istats = gendata.instruction_prompts(C.tok, train, spans, bounds, split,
                                                    {k: want[k] for k in gendata.INSTRUCTION_KINDS}, ic, rng_seed + 3,
                                                    exclude=used)
    prompts, used = prompts + ip, used | iused
    pstats.update(istats)
    short = {k: (pstats[k]["taken"], n) for k, n in want.items() if n and pstats[k]["taken"] < 0.5 * n}
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
    by_src = {s: sum(len(c) for _, c, ss in samples if ss == s) for s in kinds}
    res = {"teacher": "stock Qwen3.5-9B (all attention layers; open weights; config model_dir)",
           "sampling": {"temperature": gc["temperature"], "max_new": gc["max_new"], "stop": ["<|endoftext|>",
                                                                                            "<|im_end|>"]},
           "prompts": {"selected": len(prompts), "wanted": want, "stats": pstats, "used_for_generation": len(samples)},
           "onpolicy_prompts": {"selected": len(op), "stats": opstats},
           "stop_reason": stop_reason, "seconds": sec, "completion_tokens": ntok,
           "completion_tokens_by_source": by_src, "tok_s": ntok / max(sec, 1e-9), "batches": len(bstats),
           "pack": kst, "train_rows": int(len(rows) - hr), "heldout_rows": int(hr),
           "split": split.identity(), "batch_stats_head": bstats[:3]}
    C.R.set("gen", res)
    C.timed("gen", C.t_start)
    C.log({"done": True, **{k: res[k] for k in ("completion_tokens", "tok_s", "stop_reason", "train_rows")}}, "gen")


def _counts(n, shares):
    """Largest-remainder split of n over the (ordered) share keys."""
    keys = list(shares)
    raw = {k: n * shares[k] for k in keys}
    out = {k: int(np.floor(v)) for k, v in raw.items()}
    for k in sorted(keys, key=lambda k: -(raw[k] - out[k]))[: n - sum(out.values())]:
        out[k] += 1
    return [out[k] for k in keys]


def _sha(p):
    from convert.mixdata import sha256_file

    return sha256_file(p)


def _stream(C, train, split, gendir=None):
    """The pre-registered recall-heavy training stream (config ``mix_fracs``): generic windows, teacher-generated
    rows, synthetic recall rows."""
    cfg = C.cfg
    generic = gendata.IndexedWindowStream(train, cfg["seq_len"], cfg["batch"], split.short_windows(), cfg["seed"] + 7,
                                          C.dev)
    fr = dict(cfg["mix_fracs"])
    rows = mask = None
    if gendir and fr.get("gen"):
        rows = np.load(os.path.join(gendir, "gen_rows.npy"), mmap_mode="r")
        mask = np.load(os.path.join(gendir, "gen_mask.npy"), mmap_mode="r")
    elif fr.get("gen"):
        fr["generic"] += fr.pop("gen")
    rec = C.recall_synth(train, split.short_windows()) if fr.get("recall") else None
    return generic, MultiStream(generic, fr, cfg["batch"], cfg["seed"], rows, mask, rec, C.dev)


def _batch_fn(C, stream, modes, ce_weight):
    m, MB = C.model, C.cfg.get("micro_batch")

    def next_batch():
        x, w, kind = next(stream)
        if kind == "generic":
            return (lambda: {"kind": 0, "kl": round(align.stage2_accumulate(m, modes, x, MB), 5)}), x.numel()
        code = 1 if kind == "gen" else 2
        return (lambda: {"kind": code, **{k_: round(v, 5) for k_, v in distill_accumulate(
            m, modes, x, w, MB, "fkl", ce_weight).items() if k_ != "weight"}}), x.numel()

    return next_batch


# ----------------------------------------------------------------------------- stage A (job A, phase P)
def cmd_stage_a(cfg, mixdir, gendir, out, until_stage=None):
    """Resumable: every finished stage is in results.json + <ck>/stage<k>.pt (with the stream state), so the runner
    can run one stage per invocation (``--until-stage k``) and gate each on recall + proxy600 in between."""
    C = Ctx(cfg, out)
    m, A, L, MB = C.model, C.A, cfg["seq_len"], cfg.get("micro_batch")
    mix = json.load(open(os.path.join(mixdir, "mix.json")))
    if "mix" not in C.R:
        C.R.set("mix", mix)
    train = data.load_tokens(os.path.join(mixdir, "train.u32"))
    split = gendata.BlockSplit(len(train), L, cfg["lc"]["seq_len"], cfg["lc"]["blocks"], cfg["seed"])
    sp = json.load(open(os.path.join(gendir, "split.json")))
    if {k: sp[k] for k in split.identity()} != split.identity():
        raise SystemExit("stage A block split != the gen phase's split.json")
    generic, stream = _stream(C, train, split, gendir)
    gh = np.load(os.path.join(gendir, "gen_heldout.npz"))
    gh_rows, gh_mask = gh["rows"], gh["mask"]
    dc = cfg["distill"]
    C.set_heldout(mixdir)
    cw = cfg["curve_windows_per_source"]
    if "recall_stock" not in C.R:
        C.R.set("recall_stock", C.recall_eval({}))
    rstock = C.R["recall_stock"]

    # ---------------- stage 1: init (widened key dim) + MOHAWK alignment of all 8 (teacher-driven, simultaneous)
    s1 = cfg["stage1"]
    if "stage1" not in C.R:
        t0 = time.time()
        seed_all(cfg["seed"])
        for i in A:
            C.new_gdn(i)
        scales = align.calibrate_out_scale(m, A, next(generic))
        g1 = align.Guard(s1["guard"], "stage1")
        hist, ntok, sec = align.stage1_align(m, A, generic, 10 ** 9, s1["lr"], log=lambda r: C.log(r, "stage1"),
                                             guard=g1, micro_batch=MB)
        C.save("stage1", meta={"tokens": ntok})
        C.R.set("stage1", {"init": cfg.get("init", "reuse_attn"), "gdn": cfg.get("gdn"), "hist": hist, "tokens": ntok,
                           "seconds": sec, "steady_tok_s": g1.rate, "peak_gb": g1.peak_gb(),
                           "out_scale_calibration": {str(k): v for k, v in scales.items()},
                           "reuse_info": {str(i): getattr(m.swaps[i].gdn, "reuse_info", None) for i in A},
                           "stream": stream.state()})
        C.timed("stage1", t0)

    # ---------------- per-layer probe: single-layer RECALL damage (+ KL) -> the pre-registered order and budgets
    if "plan" not in C.R:
        C.load_state(torch.load(os.path.join(C.ck, "stage1.pt"), map_location=C.dev))
        t0 = time.time()
        dmg, kl, detail = {}, {}, {}
        for i in A:
            e = C.heldout_eval({i: "student"}, cw)
            r = C.recall_eval({i: "student"})
            d = C.recall_drop(r, rstock)
            dmg[i], kl[i] = d["mqar_drop"] + d["niah_drop"], e["mix"]["kl"]
            detail[str(i)] = {"kl": e["mix"]["kl"], **d, "mqar": r["mqar_mean"], "niah": r["niah_mean"]}
            C.log({"layer": i, **detail[str(i)]}, "probe")
        plan = conversion_plan(dmg, dc["group_size"], dc["kd_minutes"], dc["floor_minutes"],
                               descending=dc.get("order", "ascending_recall") != "ascending_recall", tiebreak=kl)
        plan["detail"] = detail
        C.R.set("plan", plan)
        C.timed("probe", t0)
    plan = C.R["plan"]

    # ---------------- recall-gated per-group conversion (one stage per invocation when --until-stage is given)
    stages = dict(C.R["stages"]) if "stages" in C.R else {}
    last = "stage1"
    for k, st in enumerate(plan["stages"]):
        key = f"stage{k + 1}"
        if key in stages:
            last = key
            continue
        if until_stage is not None and k + 1 > until_stage:
            break
        C.load_state(torch.load(os.path.join(C.ck, last + ".pt"), map_location=C.dev))
        stream.restore(stages[last]["stream"] if last in stages else C.R["stage1"]["stream"])
        conv = st["converted"]
        modes = {j: "student" for j in conv}

        def eval_fn(modes=modes):
            e = C.heldout_eval(modes, cw)
            return {"crit": e["mix"]["kl"], "heldout": e}

        guard = {**dc["guard"], "max_minutes": st["minutes"], "eval_every_min": dc["eval_every_min"]}
        before = eval_fn()
        res = run_stage(C, key, conv, modes, _batch_fn(C, stream, modes, dc["ce_weight"]), guard, dc["lr"], eval_fn,
                        before=before)
        rec = C.recall_eval(modes)
        res.update({"adds": st["adds"], "converted": conv, "minutes_budget": st["minutes"], "entry": before,
                    "stream": stream.state(), "recall": {k_: rec[k_] for k_ in ("mqar_mean", "niah_mean")},
                    "recall_detail": rec, "recall_drop_vs_stock": C.recall_drop(rec, rstock)})
        C.save(key, meta={"converted": conv, "best": res["best"]})
        stages[key] = res
        C.R.set("stages", stages)
        last = key
    allm = {j: "student" for j in A}
    if len(stages) == len(plan["stages"]) and "final" not in C.R:
        t0 = time.time()
        C.save("stageA", meta={"converted": A})
        C.R.set("final", {"heldout": C.heldout_eval(allm), "gen_heldout": C.masked_eval(gh_rows, gh_mask, allm),
                          "recall": stages[last]["recall"], "stream": stages[last]["stream"]})
        C.timed("final", t0)
    C.log({"done": len(stages) == len(plan["stages"]), "stages_done": sorted(stages),
           "minutes": round((time.time() - C.t_start) / 60, 1)}, "end")


# ----------------------------------------------------------------------------- sens (the cheap first job, phase P)
def cmd_sens(cfg, mixdir, out):
    """Per-layer sensitivity table under the recall-heavy mix + the widened GDN: can ANY single attention layer be
    converted with a small recall loss? Stage 1 (all 8, widened) once; then for each layer alone: KD for
    ``sens.minutes_per_layer`` on generic + synthetic-recall rows (only that layer converted and trained), then the
    recall probe + held-out KL. Capacity control: the ``sens.control_layers`` worst layers again at the config's
    key dim (base width), with their own short stage 1. The table is judged by jobs/q9af_v3/rule.py ``sensitivity``."""
    C = Ctx(cfg, out)
    m, A, L, MB, sc = C.model, C.A, cfg["seq_len"], cfg.get("micro_batch"), cfg["sens"]
    train = data.load_tokens(os.path.join(mixdir, "train.u32"))
    split = gendata.BlockSplit(len(train), L, cfg["lc"]["seq_len"], cfg["lc"]["blocks"], cfg["seed"])
    generic, stream = _stream(C, train, split)
    C.set_heldout(mixdir)
    cw = cfg["curve_windows_per_source"]
    if "recall_stock" not in C.R:
        C.R.set("recall_stock", C.recall_eval({}))
    rstock = C.R["recall_stock"]
    if "stage1" not in C.R:
        for i in A:
            C.new_gdn(i)
        align.calibrate_out_scale(m, A, next(generic))
        g1 = align.Guard(cfg["stage1"]["guard"], "stage1")
        hist, ntok, sec = align.stage1_align(m, A, generic, 10 ** 9, cfg["stage1"]["lr"],
                                             log=lambda r: C.log(r, "stage1"), guard=g1, micro_batch=MB)
        C.save("stage1", meta={"tokens": ntok})
        C.R.set("stage1", {"tokens": ntok, "seconds": sec, "steady_tok_s": g1.rate, "gdn": cfg.get("gdn"),
                           "hist": hist[-3:]})
    s1 = torch.load(os.path.join(C.ck, "stage1.pt"), map_location=C.dev)
    table = dict(C.R["table"]) if "table" in C.R else {}

    def one(i, tag):
        modes = {i: "student"}

        def eval_fn():
            return {"crit": C.heldout_eval(modes, cw)["mix"]["kl"]}

        res = run_stage(C, f"{tag}_L{i}", [i], modes, _batch_fn(C, stream, modes, cfg["distill"]["ce_weight"]),
                        {**cfg["distill"]["guard"], "max_minutes": sc["minutes_per_layer"]}, cfg["distill"]["lr"],
                        eval_fn)
        rec = C.recall_eval(modes)
        return {"layer": i, "width": tag, "dk": int(m.swaps[i].gdn.dk), "kl": res["best"]["crit"],
                "tokens": res["tokens"], "steady_tok_s": res["steady_tok_s"], "mqar": rec["mqar_mean"],
                "niah": rec["niah_mean"], **C.recall_drop(rec, rstock), "recall_detail": rec}

    for i in A:
        if f"wide_L{i}" in table:
            continue
        C.load_state({i: s1[i]})
        table[f"wide_L{i}"] = one(i, "wide")
        C.R.set("table", table)
        C.log({k: v for k, v in table[f"wide_L{i}"].items() if k != "recall_detail"}, "sens")
    worst = sorted(A, key=lambda i: -(table[f"wide_L{i}"]["mqar_drop"] + table[f"wide_L{i}"]["niah_drop"]))
    for i in worst[: sc["control_layers"]]:
        if f"base_L{i}" in table:
            continue
        C.new_gdn(i, dk=None)  # the config's key dim: v2's capacity
        align.calibrate_out_scale(m, [i], next(generic))
        align.stage1_align(m, [i], generic, 10 ** 9, cfg["stage1"]["lr"], log=lambda r: C.log(r, "stage1_ctrl"),
                           guard=align.Guard({**cfg["stage1"]["guard"], "max_minutes": sc["control_stage1_minutes"]},
                                             "stage1_ctrl"), micro_batch=MB)
        table[f"base_L{i}"] = one(i, "base")
        C.R.set("table", table)
    C.R.set("stream", stream.state())
    C.log({"done": True, "rows": len(table)}, "end")


# ----------------------------------------------------------------------------- stage B (job B, phase P)
def cmd_stage_b(cfg, adir, out):
    C = Ctx(cfg, out)
    m, A = C.model, C.A
    sd = torch.load(os.path.join(adir, "gdn_converted.pt"), map_location=C.dev, weights_only=True)
    if sorted(int(i) for i in sd) != A:
        raise SystemExit(f"stage-A export layers {sorted(sd)} != {A}")
    for i in A:
        attach_from_state(m, i, sd[int(i)] if int(i) in sd else sd[str(i)], seed=cfg["seed"])
    if os.path.exists(os.path.join(C.ck, "stageB_last.pt")):
        C.load_state(torch.load(os.path.join(C.ck, "stageB_last.pt"), map_location=C.dev))
    C.set_heldout(adir)
    gh = np.load(os.path.join(adir, "gen_heldout.npz"))
    allm = {j: "student" for j in A}
    cw = cfg["curve_windows_per_source"]

    def eval_fn():
        e = C.heldout_eval(allm, cw)
        return {"crit": e["mix"]["kl"], "heldout": e}

    if "recall_stock" not in C.R:
        C.R.set("recall_stock", C.recall_eval({}))
    if "entry" not in C.R:
        C.R.set("entry", {"heldout": C.heldout_eval(allm), "long": C.long_eval(allm),
                          "gen_heldout": C.masked_eval(gh["rows"], gh["mask"], allm), "recall": C.recall_eval(allm)})
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
        rec = C.recall_eval(allm)
        C.R.set("final", {"heldout": C.heldout_eval(allm), "long": C.long_eval(allm),
                          "gen_heldout": C.masked_eval(gh["rows"], gh["mask"], allm), "recall": rec,
                          "recall_drop_vs_stock": C.recall_drop(rec, C.R["recall_stock"])})
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
    s_ = sub.add_parser("sens")
    for p in (g, a, b, s_):
        p.add_argument("--config", required=True)
        p.add_argument("--out", required=True)
    for p in (g, a, s_):
        p.add_argument("--mix", required=True)
    g.add_argument("--eval-index-dir", default=None, help="the shipped full-eval-v1 index (template check)")
    a.add_argument("--gen", required=True)
    a.add_argument("--until-stage", type=int, default=None, help="stop after this conversion stage (runner gating)")
    b.add_argument("--a", required=True, help="stage-A artifact dir (gdn_converted.pt, lc.u32, onpolicy.npz, held-out)")
    ar = ap.parse_args(argv)
    cfg = json.load(open(ar.config))
    try:
        if ar.cmd == "gen":
            cmd_gen(cfg, ar.mix, ar.out, ar.eval_index_dir)
        elif ar.cmd == "stageA":
            cmd_stage_a(cfg, ar.mix, ar.gen, ar.out, ar.until_stage)
        elif ar.cmd == "sens":
            cmd_sens(cfg, ar.mix, ar.out)
        else:
            cmd_stage_b(cfg, ar.a, ar.out)
    except align.StopCondition as e:
        print(json.dumps({"STOP_CONDITION": e.reason, "exit_code": e.code, **e.data}), flush=True)
        sys.exit(e.code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
