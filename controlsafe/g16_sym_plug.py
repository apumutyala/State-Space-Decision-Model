"""``score_fn(state, options)`` for the evidence lane's invariance harness (tests/guarantees/test_reorder_invariance.py):

    SSDM_INV_HEAD=controlsafe.g16_sym_plug:score_fn python -m pytest tests/guarantees/test_reorder_invariance.py

A TINY random trust-gated backbone (``tiny_hf_backbone``, hidden 32, CPU) + a randomised ``G16SymHead`` read the
sym-fork row of a request whose untrusted state is ``state`` and whose options are ``options`` (id, description); the
host declares the instructions and every description instruction-trusted. Returns {id: probability}. Test-only.
"""
from __future__ import annotations

from typing import Any, Sequence

_CACHE: dict[str, Any] = {}
INSTR = "Pick the action that fits the record."


def request_for(state: Any, options: Sequence[tuple[str, str]], declare_criteria: bool = True) -> dict:
    from decide.provenance import criteria_path, instructions_path

    crit = {oid: desc for oid, desc in options}
    segs = {instructions_path("q"): [{"start": 0, "end": len(INSTR), "trust": "instruction",
                                      "origin": "host_template"}]}
    if declare_criteria:
        for oid, desc in options:
            if isinstance(desc, str) and desc:
                segs[criteria_path("q", oid)] = [{"start": 0, "end": len(desc), "trust": "instruction",
                                                  "origin": "host_template"}]
    return {"request_id": "r", "model": "t", "state": {"note": state},
            "questions": {"q": {"type": "choice", "instructions": INSTR, "criteria": crit}},
            "ssdm": {"provenance": {"segments": segs}}}


def tiny_head(family: str = "q9af", r: int = 4, c: float = 1.0, seed: int = 0, stream_mode: str = "recompute",
              option_trust: str = "declared"):
    import torch

    from controlsafe import monitor as M
    from controlsafe import trust_gate as TG
    from controlsafe.g16_sym import G16SymHead
    from heads.backbone import tiny_hf_backbone

    bb = tiny_hf_backbone(family, seed=seed, fork_mode="cache" if stream_mode == "cache" else "recompute")
    M.long_memory_(bb.model, seed=seed)
    g = TG.TrustGatedBackbone(bb, bridge_rank=4, stream_fork="cache" if stream_mode == "cache" else "recompute")
    h = G16SymHead(g, r=r, c=c, seed=seed, stream_mode=stream_mode, option_trust=option_trust)
    gen = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for p in h.own_parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.5)
        for br in g.bridges:  # non-zero bridges: D reads P (as after training)
            br.lora_B.copy_(torch.randn(br.lora_B.shape, generator=gen) * 0.1)
    return h.float().eval()


def row_for(req: dict):
    from heads.layout import ByteEncoder, build_row
    from train.data import TOY_TEMPLATE

    return build_row(req, "q", ByteEncoder(), template=TOY_TEMPLATE, layout="sym-fork")


def score_fn(state: Any, options: Sequence[tuple[str, str]]) -> dict[str, float]:
    import torch

    if "head" not in _CACHE:
        _CACHE["head"] = tiny_head()
    with torch.no_grad():
        out = _CACHE["head"](row_for(request_for(state, options)))
    return out.probs()
