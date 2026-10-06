"""Prefix-fork option scoring for the HF readout: read the shared prompt once, then score every (question, option)
continuation from copies of that state, instead of re-reading the whole prompt for every (question, option) row.

Why. The JRT prompt of every question of a request is ``[schema(all fields) + state] + [re-read of question q]`` and
each option is scored as ``prompt_q + continuation_o``. The recompute path forwards ``prompt_q + cont_o`` for every
(q, o): sum over questions of K_q x |prompt| tokens. On a 17-question, 3-option ContractNLI row with a 4.5k-token
state that is 229k tokens for ~46 s (measured, RWKV7-13.3B, H100, job-mhc3h); the 2,008-row DEV subset is 236.7M
tokens (``jobs/common/readout_volume_rwkv7.json``), projected 763 min. The fork path is ~5M tokens.

How. Segments are encoded one by one (no BPE merge across segments), so the token prefix P shared by every question's
prompt (template head, schema of all fields, state) is exact. The fork path makes
  1. one call over P (batch 1);
  2. packed calls over every (q, o) tail ``S_q + cont_o[:-1]`` from copies of P's state, right-padded to the longest
     tail of the pack and packed up to ``token_budget`` positions per call.
``log p(cont_o | P + S_q)`` is read from the tail logits at positions |S_q|-1 .. |S_q|+len(cont_o)-2. Right padding is
exact here: every model in use is causal (recurrent state or causal attention), the padded positions come after every
position that is read, and the state after a pack is discarded. Same tokens, same math; only the recurrence is split
into chunks and batched differently, so scores equal the recompute path up to float rounding (fp32 CPU test: 1e-5).

Calls per request: 1 + ceil(tail positions / token_budget), typically 2. (A first version with separate per-question
suffix and per-option calls made 18,440 small calls on the subset, ~30 of a projected 45 min at the fitted ~0.1 s/call.)

A plugin opts in by providing ``state_ops`` with
  prefill(ids[1, T]) -> (logits[1, 1, V], state)                     state after T tokens
  extend(state, ids[B, t], n_last) -> (logits[B, n_last, V], state)  never mutates ``state``; a batch-1 state is
                                                                     expanded to B copies
  select(state, i) -> state                                          batch-1 copy of row i
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

TOKEN_BUDGET = 4096  # tail positions per packed call: logits <= 4096 x V floats (1 GB fp32 at V = 65,536)
PAD_ID = 0  # any valid id: padded positions are never read


def common_prefix_len(seqs: list[list[int]]) -> int:
    """Longest common token prefix, capped so every sequence keeps at least one own token (the logit that scores
    the first continuation token must come from a forward over that token)."""
    if not seqs:
        return 0
    m = min(len(s) for s in seqs)
    n = 0
    first = seqs[0]
    while n < m and all(s[n] == first[n] for s in seqs):
        n += 1
    return min(n, m - 1)


@dataclass
class ForkCall:
    kind: str  # "prefix" | "pack"
    tails: list[dict[str, Any]] = field(default_factory=list)  # {qid, option, ids, first, cont}
    width: int = 0
    tokens: int = 0


def fork_plan(rows: list[dict[str, Any]], max_batch_rows: int, token_budget: int = TOKEN_BUDGET
              ) -> tuple[int, list[ForkCall]]:
    """rows: [{qid, option, prompt, cont}] (one per option, the recompute layout). -> (prefix_len, calls).
    ``tokens`` counts real (unpadded) tokens."""
    prompts: dict[str, list[int]] = {}
    for r in rows:
        prompts.setdefault(r["qid"], r["prompt"])
    L = common_prefix_len(list(prompts.values()))
    calls = [ForkCall("prefix", tokens=L)] if L > 0 else []
    tails = []
    for r in rows:
        suf = prompts[r["qid"]][L:]
        tails.append({"qid": r["qid"], "option": r["option"], "ids": suf + r["cont"][:-1], "first": len(suf) - 1,
                      "cont": r["cont"]})
    tails.sort(key=lambda t: (-len(t["ids"]), t["qid"], t["option"]))  # deterministic packing
    cur: list[dict[str, Any]] = []
    for t in tails:
        if cur and (len(cur) >= max_batch_rows or len(cur[0]["ids"]) * (len(cur) + 1) > token_budget):
            calls.append(ForkCall("pack", cur, len(cur[0]["ids"]), sum(len(x["ids"]) for x in cur)))
            cur = []
        cur.append(t)
    if cur:
        calls.append(ForkCall("pack", cur, len(cur[0]["ids"]), sum(len(x["ids"]) for x in cur)))
    return L, calls


def fork_score(rows, ops, max_batch_rows: int, device: str, check_deadline: Callable[[], None],
               token_budget: int = TOKEN_BUDGET):
    """-> (logits {qid: {option: sum logprob}}, calls, total_tokens, call_ms). Needs a shared prefix (L >= 1); the
    backend uses the recompute path otherwise (every JRT prompt shares at least the template head)."""
    import time

    import torch

    L, calls = fork_plan(rows, max_batch_rows, token_budget)
    first_prompt = rows[0]["prompt"]
    logits: dict[str, dict[str, float]] = {}
    for r in rows:
        logits.setdefault(r["qid"], {})
    p_state = None
    call_ms: list[float] = []
    total = 0
    for c in calls:
        check_deadline()
        t0 = time.perf_counter()
        with torch.inference_mode():
            if c.kind == "prefix":
                _, p_state = ops.prefill(torch.tensor([first_prompt[:L]], dtype=torch.long, device=device))
            else:
                B, W = len(c.tails), c.width
                nmax = max(len(t["cont"]) for t in c.tails)
                ids = torch.full((B, W), PAD_ID, dtype=torch.long)
                pos = torch.zeros((B, nmax), dtype=torch.long)
                tgt = torch.zeros((B, nmax), dtype=torch.long)
                live = torch.zeros((B, nmax), dtype=torch.bool)
                for b, t in enumerate(c.tails):
                    ids[b, : len(t["ids"])] = torch.tensor(t["ids"], dtype=torch.long)
                    n = len(t["cont"])
                    pos[b, :n] = torch.arange(t["first"], t["first"] + n)
                    tgt[b, :n] = torch.tensor(t["cont"], dtype=torch.long)
                    live[b, :n] = True
                ids, pos, tgt, live = (x.to(device) for x in (ids, pos, tgt, live))
                lg, _ = ops.extend(p_state, ids, W)  # [B, W, V]
                sel = lg.gather(1, pos.unsqueeze(-1).expand(-1, -1, lg.shape[-1]))  # [B, nmax, V]
                lp = torch.log_softmax(sel.float(), dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                seq = (lp * live).sum(-1).tolist()
                for t, v in zip(c.tails, seq):
                    logits[t["qid"]][t["option"]] = float(v)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        call_ms.append(round((time.perf_counter() - t0) * 1000.0, 3))
        total += c.tokens
    return logits, len(calls), total, call_ms


# ----------------------------------------------------------------------------- per-family state ops
def cache_batch_map(cache, fn) -> None:
    """Apply ``fn`` (a batch-axis op returning a NEW tensor) to every per-sequence tensor of an HF-style cache:
    attention keys/values, linear-attention conv/recurrent states (list or dict slots), and convert.model's
    ``_ssdm_state``. transformers' own ``batch_repeat_interleave`` is missing on ``LinearAttentionLayer`` (5.14), and
    RWKV-7's in-place ``reorder_cache`` would write into buffers a copy may share; assignment avoids both."""
    import torch

    for layer in getattr(cache, "layers", []) or []:
        for name in ("keys", "values"):
            v = getattr(layer, name, None)
            if isinstance(v, torch.Tensor) and v.ndim > 0:
                setattr(layer, name, fn(v))
        for name in ("conv_states", "recurrent_states"):
            slots = getattr(layer, name, None)
            items = slots.items() if isinstance(slots, dict) else enumerate(slots) if isinstance(slots, list) else ()
            for k, v in list(items):
                if isinstance(v, torch.Tensor) and v.ndim > 0:
                    slots[k] = fn(v)
    st = getattr(cache, "_ssdm_state", None)
    if st:
        cache._ssdm_state = {k: tuple(fn(x) for x in v) for k, v in st.items()}


def expand(cache, B: int) -> None:
    cache_batch_map(cache, lambda t: t.repeat_interleave(B, 0))


def select_row(cache, i: int) -> None:
    cache_batch_map(cache, lambda t: t[i: i + 1].clone())


def _deepcopy_cache(cache, keep=()):
    memo = {id(x): x for x in keep if x is not None}
    return copy.deepcopy(cache, memo)


class RWKV7StateOps:
    """Bundled HF RWKV-7 (``state=`` / ``Rwkv7Cache``)."""

    def __init__(self, model):
        self.model = model

    def _clone(self, st):
        return _deepcopy_cache(st, [getattr(st, "config", None)])

    def prefill(self, ids):
        out = self.model(input_ids=ids, use_cache=True, logits_to_keep=1)
        return out.logits, out.state

    def extend(self, state, ids, n_last):
        st = self._clone(state)
        if ids.shape[0] > 1 and _batch(st) == 1:
            expand(st, ids.shape[0])
        out = self.model(input_ids=ids, state=st, use_cache=True, logits_to_keep=n_last)
        return out.logits, out.state

    def select(self, state, i):
        st = self._clone(state)
        select_row(st, i)
        return st


class HFCacheStateOps:
    """transformers models with ``past_key_values`` (Qwen3.5 hybrid: attention KV + GDN conv/recurrent states)."""

    def __init__(self, model):
        self.model = model

    def _clone(self, c):
        return _deepcopy_cache(c, [getattr(c, "config", None)])

    def prefill(self, ids):
        out = self.model(input_ids=ids, use_cache=True, logits_to_keep=1)
        return out.logits, out.past_key_values

    def extend(self, cache, ids, n_last):
        c = self._clone(cache)
        if ids.shape[0] > 1 and _batch(c) == 1:
            expand(c, ids.shape[0])
        out = self.model(input_ids=ids, past_key_values=c, use_cache=True, logits_to_keep=n_last)
        return out.logits, out.past_key_values

    def select(self, cache, i):
        c = self._clone(cache)
        select_row(c, i)
        return c


def _tensors(obj):
    import torch

    seen = []
    for layer in getattr(obj, "layers", []) or []:
        for name in ("keys", "values", "conv_states", "recurrent_states"):
            v = getattr(layer, name, None)
            vals = v if isinstance(v, (list, tuple)) else ([v] if v is not None else [])
            if isinstance(v, dict):
                vals = list(v.values())
            seen += [x for x in vals if isinstance(x, torch.Tensor) and x.ndim > 0]
    for v in (getattr(obj, "_ssdm_state", None) or {}).values():
        seen += [x for x in v if isinstance(x, torch.Tensor)]
    return seen


def _batch(obj) -> int:
    ts = _tensors(obj)
    return ts[0].shape[0] if ts else 1


def _device(obj):
    ts = _tensors(obj)
    return ts[0].device if ts else "cpu"
