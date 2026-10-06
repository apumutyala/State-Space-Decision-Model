"""Training stages for attention -> GDN conversion.

stage1_align  MOHAWK-style hidden-state alignment: each new GDN is trained to reproduce its attention layer's mixer
              output given the *teacher's* input to that layer (rest of the network frozen and teacher-driven).
              All requested layers are aligned from one teacher forward (each layer back-propagated inside its hook,
              so one student graph is alive at a time; micro-batched with gradient accumulation). Diagnostic only
              (CODEX_CRITICAL_COMPARISON: layerwise matching is not evidence of end-to-end recovery).
stage2_kl     end-to-end distillation: KL(teacher || student) on next-token distributions, student = model with the
              converted set in GDN mode; only GDN parameters train.
evaluate      NLL (student and teacher), KL, final-hidden cosine fidelity on fixed windows.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

AC = dict(device_type="cuda", dtype=torch.bfloat16)


def _cosine_lr(step, total, warmup, base, guard=None):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    if guard is not None:
        p = max(p, guard.time_fraction())  # wall-clock-capped stages anneal on time
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


class StopCondition(Exception):
    """A pre-registered stop condition fired; the pipeline records it and exits with ``code``."""

    def __init__(self, reason, code, data=None):
        super().__init__(reason)
        self.reason, self.code, self.data = reason, code, data or {}


class Guard:
    """Wall-clock cap + pre-registered stop conditions for an unattended (paid) run.

    cfg keys: max_minutes (graceful: the stage ends early), min_tok_s or min_mfu + check_after_steps (exit 3),
    max_peak_gb (exit 4). Throughput is measured from step 10 (after kernel autotune) to check_after_steps.
    """

    def __init__(self, cfg=None, name="", flop_per_token=None, peak_flops=989e12):
        cfg = cfg or {}
        self.name = name
        self.max_sec = 60 * cfg["max_minutes"] if cfg.get("max_minutes") else None
        self.min_tok_s = cfg.get("min_tok_s")
        # min_mfu (preferred): the floor is re-derived from the stage's real FLOP/token, so changing the micro-batch,
        # the converted layer or the prefix sharing cannot silently loosen or tighten it
        if cfg.get("min_mfu") is not None:
            if not flop_per_token:
                raise ValueError(f"{name}: min_mfu needs flop_per_token")
            self.min_tok_s = cfg["min_mfu"] * peak_flops / flop_per_token
        self.check_after = cfg.get("check_after_steps", 50)
        self.max_peak_gb = cfg.get("max_peak_gb")
        self.t0 = time.time()
        self.mark = None
        self.rate = None

    def time_fraction(self):
        return (time.time() - self.t0) / self.max_sec if self.max_sec else 0.0

    def peak_gb(self):
        return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    def step(self, step, ntok):
        """Call after each optimiser step. Returns True when the wall-clock cap says stop (graceful)."""
        if step == 10:
            self.mark = (time.time(), ntok)
        if self.mark and step == max(self.check_after, 11):
            self.rate = (ntok - self.mark[1]) / max(time.time() - self.mark[0], 1e-9)
            if self.min_tok_s and self.rate < self.min_tok_s:
                raise StopCondition(f"{self.name}: tok/s {self.rate:.0f} < {self.min_tok_s} after {step} steps", 3,
                                    {"tok_s": self.rate})
        if self.max_peak_gb and self.peak_gb() > self.max_peak_gb:
            raise StopCondition(f"{self.name}: peak {self.peak_gb():.1f} GB > {self.max_peak_gb} GB", 4,
                                {"peak_gb": self.peak_gb()})
        return bool(self.max_sec and time.time() - self.t0 > self.max_sec)


def _micro(x, micro_batch):
    mb = micro_batch or x.shape[0]
    if x.shape[0] % mb:
        raise ValueError(f"batch {x.shape[0]} not divisible by micro_batch {mb}")
    return [x[s:s + mb] for s in range(0, x.shape[0], mb)]


def stage1_accumulate(model, layers, x, micro_batch=None):
    """Accumulate stage-1 gradients for batch x into the GDN .grad buffers (caller zeroes/steps).

    Loss per aligned layer = mean over sequences of ||GDN(x_t) - attn(x_t)||^2 / ||attn(x_t)||^2 (per-sequence relative
    MSE), so micro-batch accumulation equals the full-batch gradient exactly. Memory: the teacher runs under no_grad, each
    layer's student graph is back-propagated INSIDE its hook (one GDN graph alive at a time, not all of them), and the
    teacher stops after the last aligned layer.
    """
    B = x.shape[0]
    swaps = [model.swaps[i] for i in layers]
    losses = {i: 0.0 for i in layers}

    def hook(mod, args, kwargs, out):
        xi = kwargs.get("hidden_states", args[0] if args else None).detach()
        y = out[0].detach().float()
        with torch.enable_grad():
            ys = mod.gdn(xi).float()
            per_seq = ((ys - y) ** 2).sum((1, 2)) / (y ** 2).sum((1, 2))
            loss = per_seq.sum() / B
            loss.backward()
        losses[mod.layer_idx] += loss.item()

    hs = [sw.register_forward_hook(hook, with_kwargs=True) for sw in swaps]
    try:
        with model.modes({}), torch.no_grad(), torch.autocast(**AC):
            for xb in _micro(x, micro_batch):
                h, ctx = model.embed(xb)
                model.run_layers(h, ctx, 0, max(layers) + 1)
    finally:
        for h in hs:
            h.remove()
    return losses


def stage1_align(model, layers, batcher, steps, lr, warmup=20, log_every=25, log=print, guard=None, micro_batch=None):
    params = [p for i in layers for p in model.swaps[i].gdn.parameters()]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    hist, ntok, t0 = [], 0, time.time()
    for step in range(steps):
        for gpar in opt.param_groups:
            gpar["lr"] = _cosine_lr(step, steps, warmup, lr, guard)
        x = next(batcher)
        opt.zero_grad(set_to_none=True)
        losses = stage1_accumulate(model, layers, x, micro_batch)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        ntok += x.numel()
        stop = guard.step(step, ntok) if guard else False
        if step % log_every == 0 or step == steps - 1 or stop:
            rec = {"step": step, "tokens": ntok, "tok_s": ntok / (time.time() - t0), "peak_gb": _peak_gb(),
                   **{f"rel_mse_L{k}": round(v, 5) for k, v in losses.items()}}
            hist.append(rec)
            log(rec)
        if stop:
            break
    return hist, ntok, time.time() - t0


def _peak_gb():
    return round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0.0


@torch.no_grad()
def calibrate_out_scale(model, layers, x):
    """Rescale each new GDN's out_proj so its output norm matches the attention output norm on one batch."""
    scales = {}
    store = {}

    def hook(mod, args, kwargs, out):
        h = kwargs.get("hidden_states", args[0] if args else None)
        store[mod.layer_idx] = (out[0].float().norm() / mod.gdn(h).float().norm()).item()

    hs = [model.swaps[i].register_forward_hook(hook, with_kwargs=True) for i in layers]
    try:
        with model.modes({}), torch.autocast(**AC):
            model.hidden(x)
    finally:
        for h in hs:
            h.remove()
    for i in layers:
        model.swaps[i].gdn.out_proj.weight.mul_(store[i])
        scales[i] = store[i]
    return scales


def _kl_chunk(hs, ht, w):
    ls = F.linear(hs, w).float().log_softmax(-1)
    lt = F.linear(ht, w).float().log_softmax(-1)
    return (lt.exp() * (lt - ls)).sum()


def kl_from_hidden(hs, ht, lm_head, chunk=512):
    """Mean per-token KL(teacher||student) computed in checkpointed token chunks (bounded logits memory)."""
    hs = hs.reshape(-1, hs.shape[-1])
    ht = ht.reshape(-1, ht.shape[-1])
    tot = 0.0
    for s in range(0, hs.shape[0], chunk):
        tot = tot + checkpoint(_kl_chunk, hs[s:s + chunk], ht[s:s + chunk], lm_head.weight, use_reentrant=False)
    return tot / hs.shape[0]


def stage2_accumulate(model, student_modes, x, micro_batch=None, chunk=512):
    """Accumulate stage-2 (end-to-end KL) gradients for batch x; returns the batch-mean per-token KL.

    The prefix before the first student layer is identical for teacher and student, so it runs ONCE under no_grad and
    is shared; the teacher suffix runs under no_grad; only the student suffix builds a graph (activation-checkpointed
    when the model is in gradient-checkpointing mode). Loss = sum of per-token KL / tokens in the FULL batch, so
    micro-batch accumulation equals the full-batch gradient.
    """
    first = min(i for i, m in student_modes.items() if m == "student")
    n = len(model.lm.layers)
    total = x.shape[0] * x.shape[1]
    kl_total = 0.0
    for xb in _micro(x, micro_batch):
        with torch.no_grad(), torch.autocast(**AC), model.modes({}):
            h0, ctx = model.embed(xb)
            h0 = model.run_layers(h0, ctx, 0, first)
            ht = model.run_layers(h0, ctx, first, n, final_norm=True)
        # backward must run inside the modes context: checkpointed layers re-run their forward during backward
        with model.modes(student_modes):
            with torch.autocast(**AC):
                hs = model.run_layers(h0, ctx, first, n, final_norm=True)
                loss = kl_from_hidden(hs, ht, model.lm_head, chunk) * (xb.numel() / total)
            loss.backward()
        kl_total += loss.item()
    return kl_total


def stage2_kl(model, layers_train, student_modes, batcher, steps, lr, warmup=20, log_every=25, log=print,
              grad_ckpt=True, guard=None, micro_batch=None):
    params = list(model.gdn_parameters(layers_train))
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    if grad_ckpt:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.lm.train()
    hist, ntok, t0 = [], 0, time.time()
    try:
        for step in range(steps):
            for gpar in opt.param_groups:
                gpar["lr"] = _cosine_lr(step, steps, warmup, lr, guard)
            x = next(batcher)
            opt.zero_grad(set_to_none=True)
            kl = stage2_accumulate(model, student_modes, x, micro_batch)
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ntok += x.numel()
            stop = guard.step(step, ntok) if guard else False
            if step % log_every == 0 or step == steps - 1 or stop:
                rec = {"step": step, "tokens": ntok, "tok_s": ntok / (time.time() - t0), "kl": round(kl, 5),
                       "grad_norm": round(float(gn), 4), "peak_gb": _peak_gb()}
                hist.append(rec)
                log(rec)
            if stop:
                break
    finally:
        model.lm.eval()
        if grad_ckpt:
            model.lm.gradient_checkpointing_disable()
    return hist, ntok, time.time() - t0


@torch.no_grad()
def evaluate(model, windows, student_modes, bs=4):
    """NLL of teacher and student, KL(t||s), and mean cosine of final hidden states."""
    nll_s = nll_t = kl = cos = 0.0
    n = 0
    for s in range(0, len(windows), bs):
        x = windows[s:s + bs].cuda()
        with torch.autocast(**AC):
            with model.modes({}):
                ht = model.hidden(x)
            with model.modes(student_modes):
                hs = model.hidden(x)
        cos += F.cosine_similarity(hs.float(), ht.float(), dim=-1).sum().item()
        for b in range(x.shape[0]):
            for c in range(0, x.shape[1] - 1, 512):
                e = min(c + 512, x.shape[1] - 1)
                ls = model.lm_head(hs[b, c:e]).float().log_softmax(-1)
                lt = model.lm_head(ht[b, c:e]).float().log_softmax(-1)
                y = x[b, c + 1:e + 1]
                nll_s -= ls.gather(-1, y[:, None]).sum().item()
                nll_t -= lt.gather(-1, y[:, None]).sum().item()
                kl += (lt.exp() * (lt - ls)).sum().item()
        n += x.shape[0] * (x.shape[1] - 1)
    tot_pos = len(windows) * windows.shape[1]
    return {"nll_student": nll_s / n, "nll_teacher": nll_t / n, "delta_nll": (nll_s - nll_t) / n, "kl": kl / n,
            "hidden_cos": cos / tot_pos}
