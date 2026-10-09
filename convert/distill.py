"""Q9-AF v3 training pieces (jobs/q9af_v3): batched sampling, completion-only / on-policy distillation losses, and the
pre-registered per-layer conversion order.

generate          batched ancestral sampling from the model in any mix of modes (teacher = stock attention, student =
                  the converted GDNs): left-padded prompts, one prefill, then one-token steps through the cached path
                  (attention KV in an HF DynamicCache, GDN states in ``cache._ssdm_state``, as ConvertedStateOps). Decode
                  steps use the fp32 recurrent GDN kernel (one token: no chunk padding).
distill_accumulate  gradients of a weighted per-position loss with the student = ``modes`` and the teacher = stock
                  attention, on one batch (micro-batched; the loss is normalised by the FULL batch's weight, so
                  accumulation equals the full-batch gradient). Losses:
                    ``fkl``  KL(teacher || student) per position (+ ``ce_weight`` x CE of the student on the batch's own
                             next tokens): GenDistill-style completion-only distillation on teacher-generated rows;
                    ``rkl``  KL(student || teacher) per position: on-policy distillation on the student's own samples
                             (GKD-style: the teacher scores every position of the student's rollout; no gradient
                             through the sampling).
                  Weights are per PREDICTION position (``convert.gendata.pred_weights``); 0 = not in the loss.
Order rule        ``conversion_plan``: the pre-registered DtR-style order (see jobs/q9af_v3/README.md): descending
                  single-layer sensitivity, groups of ``group_size``, stage minutes floor + sensitivity share.
"""
from __future__ import annotations

import contextlib
import time

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _autocast(dev):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if dev.type == "cuda" else contextlib.nullcontext()


def _gdns(model):
    for layer in model.lm.layers:
        la = getattr(layer, "linear_attn", None)
        if la is not None and hasattr(la, "gdn"):
            yield la.gdn
    for sw in model.swaps.values():
        if sw.gdn is not None:
            yield sw.gdn


@contextlib.contextmanager
def recurrent_decode(model):
    """Every GDN (native + converted) on the fp32 recurrent kernel for one-token decode steps; restored after."""
    old = [(g, g.kernel) for g in _gdns(model)]
    try:
        for g, _ in old:
            g.kernel = "recurrent"
        yield
    finally:
        for g, k in old:
            g.kernel = k


@torch.no_grad()
def generate(model, modes, prompts, max_new, stop_ids, pad_id, temperature=1.0, generator=None, device=None):
    """Sample one completion per prompt. Returns (completions: list of lists incl. the stop token when one was drawn,
    stats). Rows that stopped keep stepping on ``pad_id`` (their tokens are discarded)."""
    from transformers import DynamicCache

    dev = device or next(model.lm.parameters()).device
    B = len(prompts)
    P = max(len(p) for p in prompts)
    ids = torch.full((B, P), pad_id, dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, P - len(p):] = torch.as_tensor([int(t) for t in p], dtype=torch.long)
    ids = ids.to(dev)
    stop = torch.as_tensor(sorted(set(stop_ids)), device=dev)
    cache = DynamicCache(config=model.cfg)
    cache._ssdm_state = {}
    t0 = time.time()
    out = torch.full((B, max_new), pad_id, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    n = 0
    with model.modes(modes), _autocast(dev):
        h = model.lm(input_ids=ids, past_key_values=cache, use_cache=True).last_hidden_state[:, -1]
        t_prefill = time.time() - t0
        for t in range(max_new):
            logits = model.lm_head(h).float()
            if temperature > 0:
                nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=generator)[:, 0]
            else:
                nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
            out[:, t] = nxt
            n = t + 1
            done |= torch.isin(nxt, stop)
            if bool(done.all()) or t == max_new - 1:
                break
            with recurrent_decode(model):
                h = model.lm(input_ids=nxt[:, None], past_key_values=cache, use_cache=True).last_hidden_state[:, -1]
    out = out[:, :n].tolist()
    comps = []
    stops = set(int(s) for s in stop_ids)
    for row in out:
        c = []
        for tok in row:
            c.append(tok)
            if tok in stops:
                break
        comps.append(c)
    sec = time.time() - t0
    return comps, {"rows": B, "prompt_len_max": P, "steps": n, "seconds": sec, "prefill_s": t_prefill,
                   "completion_tokens": sum(len(c) for c in comps), "stopped": int(done.sum())}


# ----------------------------------------------------------------------------- losses
def _chunk_terms(hs, ht, w_head, y, wt, mode, ce_weight):
    """Weighted sums over one token chunk: (divergence, ce)."""
    ls = F.linear(hs, w_head).float().log_softmax(-1)
    lt = F.linear(ht, w_head).float().log_softmax(-1)
    if mode == "fkl":
        div = (lt.exp() * (lt - ls)).sum(-1)
    elif mode == "rkl":
        div = (ls.exp() * (ls - lt)).sum(-1)
    else:
        raise ValueError(mode)
    loss = (div * wt).sum()
    ce = -(ls.gather(-1, y[:, None])[:, 0] * wt).sum() if ce_weight else loss.new_zeros(())
    return loss + ce_weight * ce, ce.detach(), loss.detach()


def weighted_loss(hs, ht, lm_head, y, wt, mode="fkl", ce_weight=0.0, chunk=512):
    """Sum over positions of wt * (divergence + ce_weight * CE), checkpointed in token chunks (bounded logits memory).
    Returns (total, ce_sum, div_sum)."""
    hs, ht = hs.reshape(-1, hs.shape[-1]), ht.reshape(-1, ht.shape[-1])
    y, wt = y.reshape(-1), wt.reshape(-1).float()
    tot = ce = div = 0.0
    for s in range(0, hs.shape[0], chunk):
        if not bool((wt[s:s + chunk] > 0).any()):
            continue
        a, b, c = checkpoint(_chunk_terms, hs[s:s + chunk], ht[s:s + chunk], lm_head.weight, y[s:s + chunk],
                             wt[s:s + chunk], mode, ce_weight, use_reentrant=False)
        tot, ce, div = tot + a, ce + b, div + c
    return tot, ce, div


def distill_accumulate(model, student_modes, x, wpos, micro_batch=None, mode="fkl", ce_weight=0.0, chunk=512):
    """Accumulate the gradient of sum(w * (div + ce_weight * CE)) / sum(w) over the batch into the GDN grads.
    x [B, T] token ids; wpos [B, T] prediction-position weights (position t predicts x[:, t + 1]; the last column
    is ignored). Returns per-weighted-position means {"div", "ce", "weight"}."""
    first = min(i for i, m in student_modes.items() if m == "student")
    n = len(model.lm.layers)
    wpos = wpos.to(x.device).float().clone()
    wpos[:, -1] = 0.0
    total = float(wpos.sum())
    if total <= 0:
        return {"div": 0.0, "ce": 0.0, "weight": 0.0}
    y = torch.zeros_like(x)
    y[:, :-1] = x[:, 1:]
    dev = x.device
    div_s = ce_s = 0.0
    mb = micro_batch or x.shape[0]
    for s in range(0, x.shape[0], mb):
        xb, yb, wb = x[s:s + mb], y[s:s + mb], wpos[s:s + mb]
        if float(wb.sum()) <= 0:
            continue
        with torch.no_grad(), _autocast(dev), model.modes({}):
            h0, ctx = model.embed(xb)
            h0 = model.run_layers(h0, ctx, 0, first)
            ht = model.run_layers(h0, ctx, first, n, final_norm=True)
        with model.modes(student_modes):  # backward inside: checkpointed layers re-run their forward
            with _autocast(dev):
                hs = model.run_layers(h0, ctx, first, n, final_norm=True)
                loss, ce, div = weighted_loss(hs, ht, model.lm_head, yb, wb, mode, ce_weight, chunk)
                loss = loss / total
            loss.backward()
        div_s += float(div)
        ce_s += float(ce)
    return {"div": div_s / total, "ce": ce_s / total, "weight": total}


# ----------------------------------------------------------------------------- the pre-registered order rule
def conversion_plan(sens: dict, group_size: int, kd_minutes: float, floor_minutes: float) -> dict:
    """sens: {layer: held-out KL with ONLY that layer converted (its stage-1-aligned GDN)}.

    Order: descending sensitivity (most sensitive first; ties -> lower layer index first). Groups: consecutive
    ``group_size`` layers of that order (the last group may be smaller). Stage k minutes:
    floor + (kd_minutes - n_groups * floor) * S_k / sum(S), S_k = summed sensitivity of the layers stage k ADDS
    (every stage trains all layers converted so far). Pure function of the probe: pre-registered, deterministic."""
    if group_size < 1 or not sens:
        raise ValueError("need group_size >= 1 and a sensitivity per layer")
    order = sorted(sens, key=lambda i: (-float(sens[i]), int(i)))
    groups = [order[k:k + group_size] for k in range(0, len(order), group_size)]
    free = kd_minutes - len(groups) * floor_minutes
    if free < 0:
        raise ValueError(f"kd_minutes {kd_minutes} < {len(groups)} stages x floor {floor_minutes}")
    S = [sum(max(float(sens[i]), 0.0) for i in g) for g in groups]
    tot = sum(S)
    share = [s / tot if tot > 0 else 1.0 / len(groups) for s in S]
    minutes = [floor_minutes + free * f for f in share]
    stages, conv = [], []
    for g, m, f in zip(groups, minutes, share):
        conv = conv + list(g)
        stages.append({"adds": [int(i) for i in g], "converted": sorted(int(i) for i in conv), "minutes": round(m, 3),
                       "share": round(f, 6)})
    return {"order": [int(i) for i in order], "rule": "descending single-layer held-out KL (ties: lower index first); "
            f"groups of {group_size}; stage minutes = {floor_minutes} + {free:g} x share of the added layers' KL",
            "sensitivity": {str(int(i)): float(sens[i]) for i in order}, "stages": stages}
