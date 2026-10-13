"""``score_fn(state, options) -> {id: p}`` adapters for the evidence lane's independent invariance harness
(claude/ev-guarantee-proofs ``tests/guarantees/test_reorder_invariance.py``, ``SSDM_INV_HEAD=module:function``).

Each call goes through the WHOLE served path of this branch: ``heads.layout.build_row`` -> tiny random Qwen3.5 hybrid
(CPU, fp32, cache fork) + fork head -> ``decide.Decider`` (adapter softmax, calibration identity) and returns the
full-precision served probabilities (``ssdm.served_probabilities``), keyed by the caller's ids.

    SSDM_INV_HEAD=jobs.sym_eval.inv_adapter:sym_fork_score   # must pass at tol 0 (bit-identical)
    SSDM_INV_HEAD=jobs.sym_eval.inv_adapter:ordered_score    # the old ordered layout: must FAIL relabel (teeth)
    SSDM_INV_HEAD=jobs.sym_eval.inv_adapter:sym_fast_fork2_score    # heads.sym_fast fork2 readout: tol 0
    SSDM_INV_HEAD=jobs.sym_eval.inv_adapter:sym_fast_packed1_score  # heads.sym_fast packed1 readout: tol 0
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence


@lru_cache(maxsize=8)
def _decider(layout: str, sym_exec: str | None = None):
    import torch

    from decide import Decider
    from heads.backbone import tiny_hf_backbone
    from heads.decide_backend import DecisionModelBackend
    from heads.layout import ByteEncoder
    from heads.model import DecisionModel
    from train.data import TOY_TEMPLATE

    torch.manual_seed(0)
    model = DecisionModel(tiny_hf_backbone("qwen3.5", seed=0, fork_mode="cache"), "fork", seed=0).float().eval()
    return Decider(DecisionModelBackend(model, ByteEncoder(), template=TOY_TEMPLATE, layout=layout, sym_exec=sym_exec))


def _score(layout: str, state: Any, options: Sequence[tuple[str, str]], sym_exec: str | None = None) -> dict[str, float]:
    req = {"model": "inv-harness", "state": state,
           "questions": {"q": {"type": "choice", "instructions": "Which option fits the state?",
                               "criteria": {str(i): c for i, c in options}}}}
    resp = _decider(layout, sym_exec).decide(req)
    ss = resp.get("ssdm") or {}
    if ss.get("status", "ok") != "ok":
        raise RuntimeError(f"decider refused: {ss.get('error')}")
    return {str(k): float(v) for k, v in ss["served_probabilities"]["q"].items()}


def sym_fork_score(state: Any, options: Sequence[tuple[str, str]]) -> dict[str, float]:
    return _score("sym-fork", state, options)


def ordered_score(state: Any, options: Sequence[tuple[str, str]]) -> dict[str, float]:
    return _score("ordered", state, options)


def sym_fast_fork2_score(state: Any, options: Sequence[tuple[str, str]]) -> dict[str, float]:
    return _score("sym-fork", state, options, sym_exec="fork2")


def sym_fast_packed1_score(state: Any, options: Sequence[tuple[str, str]]) -> dict[str, float]:
    return _score("sym-fork", state, options, sym_exec="packed1")
