"""DecisionModelBackend row cache (JEV2-4B speed fix): decide.adapter's plan() + score() on one request build each
question's row ONCE, and every served output is bit-identical to rebuilding the rows (the pre-fix path). CPU, tiny
models (torch; runs in the remote test job)."""
from __future__ import annotations

import copy
import json

import pytest

torch = pytest.importorskip("torch")

from heads import ByteEncoder, DecisionModel  # noqa: E402
from heads.backbone import tiny_hf_backbone  # noqa: E402

ENC = ByteEncoder()
REQ = {"model": "t", "state": "ticket: checkout down since 9am; vendor Acme; invoice INV-7 overdue",
       "questions": {"dept": {"type": "choice", "instructions": "team?",
                              "criteria": {"billing": "payments", "tech": "outages", "sales": None}},
                     "down": {"type": "noul", "instructions": "is a service down?"},
                     "urg": {"type": "score", "criteria": ["later", "week", "today"]}}}


def _backend(cls=None, layout="ordered"):
    from heads.decide_backend import DecisionModelBackend

    bb = tiny_hf_backbone("qwen3.5", fork_mode="cache")
    m = DecisionModel(bb, "fork", None, seed=0).eval()
    return (cls or DecisionModelBackend)(m, ENC, template="jrt-plain-v1", layout=layout)


def _no_cache_cls():
    from heads.decide_backend import DecisionModelBackend

    class NoCache(DecisionModelBackend):  # the pre-fix behaviour: rows rebuilt on every call
        def _rows(self, request):
            self._row_cache = None
            return super()._rows(request)
    return NoCache


@pytest.mark.parametrize("layout", ["ordered", "sym-fork"])  # g1v2 needs the shim template (tokenizer), not the toy one
def test_plan_then_score_builds_rows_once_and_outputs_identical(monkeypatch, layout):
    import heads.decide_backend as DB
    from decide import Decider

    calls = []
    real = DB.build_row
    monkeypatch.setattr(DB, "build_row", lambda *a, **k: calls.append(a[1]) or real(*a, **k))
    torch.manual_seed(0)
    cached = _backend(layout=layout)
    torch.manual_seed(0)
    plain = _backend(_no_cache_cls(), layout=layout)
    plain.model.load_state_dict(cached.model.state_dict())
    r1 = Decider(cached).decide(copy.deepcopy(REQ))
    n_cached = len(calls)
    calls.clear()
    r2 = Decider(plain).decide(copy.deepcopy(REQ))
    assert n_cached == len(REQ["questions"])  # once per question (plan + score)
    assert len(calls) == 2 * len(REQ["questions"])  # the old path: twice
    strip = lambda r: json.dumps({k: v for k, v in r.items() if k not in ("ssdm",)}, sort_keys=True)  # noqa: E731
    assert strip(r1) == strip(r2)
    assert r1["answers"] == r2["answers"]


def test_changed_request_rebuilds():
    be = _backend()
    a = be._rows(REQ)
    assert be._rows(copy.deepcopy(REQ)) is a  # same content: cache hit
    req2 = copy.deepcopy(REQ)
    req2["state"] += " (updated)"
    b = be._rows(req2)
    assert b is not a and list(b[0].ids) != list(a[0].ids)
