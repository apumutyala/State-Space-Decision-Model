"""``score_fn(state, options) -> {id: prob}`` adapters of REAL heads on a tiny random RWKV-7 for the evidence lane's
reorder / relabel harness (tests/guarantees/test_reorder_invariance.py, ``SSDM_INV_HEAD=heads.inv_adapters:<fn>``).

* ``rwkv7_sym_fork``: the D21 (A) path: layout ``sym-fork`` (G1 vendor format, ``jrt-rwkv-g1-v2`` head), cache fork,
  head ``joint_record_sym`` with the lexical prior reading descriptions only. Claimed: invariant at tolerance 0.
* ``rwkv7_g1v2_v3w``: D19's own path (template ``jrt-rwkv-g1-v3w``, layout ``g1v2``, head ``joint_record``): the ids are
  in the prompt, so relabeling must move the probabilities (the harness's teeth on the real model).

The heads' zero-init knobs (``w``, ``lex_scale``) are set to non-zero seeded values so the record mixer and the lexical
prior actually contribute (at init the joint heads equal the fork head, which would make the check partly vacuous).
CPU, fp32, built once per process.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

INSTR = "Which option fits the record?"


def _request(state: str, options: Sequence[tuple[str, str]]) -> dict[str, Any]:
    return {"model": "toy", "state": {"msg": state},
            "questions": {"q": {"type": "choice", "instructions": INSTR, "criteria": dict(options)}}}


@lru_cache(maxsize=None)
def _model(head: str, seed: int = 0):
    import torch

    from heads.backbone import tiny_hf_backbone
    from heads.model import DecisionModel

    torch.manual_seed(seed)
    bb = tiny_hf_backbone("rwkv7", seed=seed, fork_mode="cache")
    m = DecisionModel(bb, head, seed=seed).float().eval()
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        m.head.w.copy_(torch.randn(m.head.w.shape, generator=g))
        m.head.lex_scale.fill_(1.5)
    return m


def _score(head: str, template: str, layout: str, state: str, options: Sequence[tuple[str, str]]) -> dict[str, float]:
    import torch

    from heads.layout import ByteEncoder, build_row

    row = build_row(_request(state, options), "q", ByteEncoder(), template=template, layout=layout)
    with torch.no_grad():
        lp = _model(head)([row]).log_probs[0, : row.n_options].double().exp()
    return {o: float(p) for o, p in zip(row.option_ids, lp)}


def rwkv7_sym_fork(state: str, options: Sequence[tuple[str, str]]) -> dict[str, float]:
    return _score("joint_record_sym", "jrt-rwkv-g1-v2", "sym-fork", state, options)


def rwkv7_g1v2_v3w(state: str, options: Sequence[tuple[str, str]]) -> dict[str, float]:
    from decide.prompt_g1v3 import G1V3W_ID

    return _score("joint_record", G1V3W_ID, "g1v2", state, options)
