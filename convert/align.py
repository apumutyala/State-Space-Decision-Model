"""Training stages for attention -> GDN conversion.

stage1_align  MOHAWK-style hidden-state alignment: each new GDN is trained to reproduce its attention layer's mixer
              output given the *teacher's* input to that layer (rest of the network frozen and teacher-driven).
              All requested layers are aligned simultaneously from one teacher forward. Diagnostic only
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

    cfg keys: max_minutes (graceful: the stage ends early), min_tok_s + check_after_steps (exit 3),
    max_peak_gb (exit 4). Throughput is measured from step 10 (after kernel autotune) to check_after_steps.
    """

    def __init__(self, cfg=None, name=""):
        cfg = cfg or {}
        self.name = name
        self.max_sec = 60 * cfg["max_minutes"] if cfg.get("max_minutes") else None
        self.min_tok_s = cfg.get("min_tok_s")
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


def stage1_align(model, layers, batcher, steps, lr, warmup=20, log_every=25, log=print, guard=None):
    swaps = [model.swaps[i] for i in layers]
    params = [p for sw in swaps for p in sw.gdn.parameters()]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    losses = {}

    def hook(mod, args, kwargs, out):
        x = kwargs.get("hidden_states", args[0] if args else None).detach()
        y = out[0].detach().float()
        with torch.enable_grad():
            ys = mod.gdn(x).float()
            losses[mod.layer_idx] = ((ys - y) ** 2).sum() / (y ** 2).sum()

    hs = [sw.register_forward_hook(hook, with_kwargs=True) for sw in swaps]
    hist, ntok, t0 = [], 0, time.time()
    try:
        with model.modes({}):  # all teacher
            for step in range(steps):
                for gpar in opt.param_groups:
                    gpar["lr"] = _cosine_lr(step, steps, warmup, lr, guard)
                x = next(batcher)
                losses.clear()
                with torch.no_grad(), torch.autocast(**AC):
                    # run only up to the last aligned layer would be cheaper; full pass keeps the code simple
                    model.hidden(x)
                loss = sum(losses.values())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                ntok += x.numel()
                stop = guard.step(step, ntok) if guard else False
                if step % log_every == 0 or step == steps - 1 or stop:
                    rec = {"step": step, "tokens": ntok, "tok_s": ntok / (time.time() - t0),
                           "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                           **{f"rel_mse_L{k}": round(v.item(), 5) for k, v in losses.items()}}
                    hist.append(rec)
                    log(rec)
                if stop:
                    break
    finally:
        for h in hs:
            h.remove()
    return hist, ntok, time.time() - t0


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


def stage2_kl(model, layers_train, student_modes, batcher, steps, lr, warmup=20, log_every=25, log=print,
              grad_ckpt=True, guard=None):
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
            with torch.no_grad(), torch.autocast(**AC), model.modes({}):
                ht = model.hidden(x)
            opt.zero_grad(set_to_none=True)
            # backward must run inside the modes context: checkpointed layers re-run their forward during backward
            with model.modes(student_modes):
                with torch.autocast(**AC):
                    hs = model.hidden(x)
                    loss = kl_from_hidden(hs, ht, model.lm_head)
                loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ntok += x.numel()
            stop = guard.step(step, ntok) if guard else False
            if step % log_every == 0 or step == steps - 1 or stop:
                rec = {"step": step, "tokens": ntok, "tok_s": ntok / (time.time() - t0), "kl": round(loss.item(), 5),
                       "grad_norm": round(float(gn), 4), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
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
