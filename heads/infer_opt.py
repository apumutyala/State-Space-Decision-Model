"""Inference-only speed options for the trained decision model (EVALUATION / serving; never used by training).

Profiled on D1 (job-c7pq2 art-bm3e6, 2,000 ok Index DEV subset rows, analysis/COST_LATENCY_2026-10-08.md): the child
scores at 2.78x its stock parent's wall time, and a per-row least-squares fit puts ~69 % of the child's time on a
per-QUESTION cost of ~307 ms (parent: ~36 ms). Three options remove it, each opt-in (``DecisionModelBackend(...,
inference=...)``, default off = byte-identical to before):

``memo_encode``  the backend tokenises one full prompt per question in BOTH ``plan()`` and ``score()`` (2 x Q passes
                 over the same state text; 107k tok/s on CPU = ~126 ms per subset row per pass, up to 5.9 s for a
                 64-question row). A per-request segment memo (the stock readout's ``_enc_memo``) encodes each distinct
                 segment once. Exact: the same encoder on the same text gives the same ids.
``packed_fork``  ``heads.prefix_features`` ran one backbone call per distinct suffix length plus one per QUESTION for
                 the cache fork (1 + G + Q calls of a 61-layer model). ``packed`` runs the shared prefix once, then
                 every (question, option) tail ``suffix_q + continuation_k`` from copies of that state in packed,
                 right-padded calls (the stock readout's ``decide.backends.state_fork`` layout): typically 2 calls per
                 request. Exact up to float rounding (causal; padded positions are after every position read).
``merge_lora``   folds every LoRA update into its base weight (W + s * B @ A, computed in fp32, stored in the base
                 dtype) and drops the side matmuls, the per-call fp32 -> bf16 casts of A and B and ~6 kernel launches
                 per wrapped Linear (366 on RWKV7-13.3B). Exact in exact arithmetic (fp32 test: 1e-6); in bf16 the
                 merged weight is rounded once, so logits move at bf16 resolution: GPU-checked by the A/B
                 (jobs/m2_first_cell mode ``latency_ab``), never assumed.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn

OPTS = ("memo_encode", "packed_fork", "merge_lora")
PRESETS = {"base": (), "exact": ("memo_encode", "packed_fork"), "fast": ("memo_encode", "packed_fork", "merge_lora")}


def parse(spec: str | dict | None) -> dict[str, bool]:
    """``None`` / "base" -> {} (off); a preset name, "a+b" list, or a dict -> {opt: True}."""
    if not spec:
        return {}
    if isinstance(spec, dict):
        names = [k for k, v in spec.items() if v]
    elif spec in PRESETS:
        names = list(PRESETS[spec])
    else:
        names = [x for x in spec.split("+") if x]
    bad = [x for x in names if x not in OPTS]
    if bad:
        raise ValueError(f"inference options {bad}: known {OPTS} or presets {sorted(PRESETS)}")
    return {k: True for k in OPTS if k in names}


class MemoEncoder:
    """Segment memo around an ``EncodeFn`` (text, trusted) -> ids. Exceptions are not cached (re-raised each call).
    Bounded: cleared past ``max_entries`` (a request's segments; plan + score of the same request hit)."""

    def __init__(self, encode: Callable[..., list[int]], max_entries: int = 4096) -> None:
        self.encode = encode
        self.max_entries = max_entries
        self._memo: dict[tuple[str, bool], tuple[int, ...]] = {}
        self.hits = 0
        self.misses = 0
        ident = getattr(encode, "identity", None)
        if callable(ident):
            self.identity = ident  # the served revision names the underlying encoder, unchanged

    def __call__(self, text: str, trusted: bool = True) -> list[int]:
        key = (text, bool(trusted))
        ids = self._memo.get(key)
        if ids is None:
            self.misses += 1
            ids = tuple(self.encode(text, trusted))
            if len(self._memo) >= self.max_entries:
                self._memo.clear()
            self._memo[key] = ids
        else:
            self.hits += 1
        return list(ids)


def _is_lora(m: nn.Module) -> bool:  # duck-typed: heads never imports train (train.lora.LoRALinear)
    return all(hasattr(m, a) for a in ("base", "lora_A", "lora_B", "scaling")) and isinstance(m.base, nn.Linear)


@torch.no_grad()
def merge_lora(model: nn.Module) -> int:
    """Replace every LoRA-wrapped Linear by its base Linear with W <- W + scaling * B @ A (fp32 math, cast back to the
    base dtype). In place, for inference only (the LoRA parameters are gone afterwards). Returns the count."""
    names = [n for n, m in model.named_modules() if _is_lora(m)]
    for name in names:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        lm = getattr(parent, attr)
        if not isinstance(lm.dropout, nn.Identity):
            raise ValueError(f"{name}: LoRA dropout is active; merge is only defined for inference (dropout off)")
        base = lm.base
        w = base.weight
        delta = (lm.lora_B.detach().float() @ lm.lora_A.detach().float()) * float(lm.scaling)
        base.weight.data = (w.detach().float() + delta.to(w.device)).to(w.dtype)
        base.weight.requires_grad_(False)
        setattr(parent, attr, base)
    return len(names)


def describe(opts: dict[str, Any]) -> dict[str, Any]:
    return {k: bool(opts.get(k)) for k in OPTS}
