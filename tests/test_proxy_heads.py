"""Proxy heads + loop on SYNTHETIC cached features (train/proxy_heads.py, train/proxy_loop.py). CPU, $0."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from train import proxy_loop as PL  # noqa: E402
from train.proxy_heads import AFMixer, AttnRef, ForkProxy, ProxyQuestion, ProxyRequest, make_proxy_head  # noqa: E402
from train.proxy_rule import CANDIDATE_FAMILIES, RECIPES  # noqa: E402

D = 16


def _q(g, qid, qtype, k, label=-1, bank="state"):
    r = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    return ProxyQuestion(qid, qtype, [str(i) for i in range(k)] if qtype != "noul" else ["true", "false"],
                         r(k, D).bfloat16(), r(k, D).bfloat16(), r(k, D).bfloat16(), r(D).bfloat16(), r(D).bfloat16(),
                         r(k) - 3, torch.full((k,), 2.0), "digits" if qtype == "score" else "option_n", bank, label)


def synthetic(n: int, seed: int, learnable: bool = True) -> list[ProxyRequest]:
    """The gold option's fork feature carries a fixed direction, so a head can learn it."""
    g = torch.Generator().manual_seed(seed)
    sig = torch.randn(D, generator=torch.Generator().manual_seed(99))
    out = []
    for i in range(n):
        qs = []
        for j, (qt, k) in enumerate((("choice", 4), ("noul", 2), ("score", 3))[: 1 + i % 3]):
            y = int(torch.randint(k, (1,), generator=g))
            q = _q(g, f"q{j}", qt, k, y)
            if learnable:
                q.fork = (q.fork.float() + torch.zeros(k, D).index_fill_(0, torch.tensor([y]), 1) * sig).bfloat16()
            qs.append(q)
        out.append(ProxyRequest(f"r{i}", "train", qs, {"state": torch.randn(5, D, generator=g).bfloat16()}))
    return out


@pytest.mark.parametrize("rid", [r["id"] for r in RECIPES if r["arm"] == "stock"])
def test_every_recipe_builds_trains_and_is_deterministic(rid):
    rec = next(r for r in RECIPES if r["id"] == rid)
    data = synthetic(24, 0)
    budget = {**PL.TRAIN, "epochs": 1, "grad_accum": 4, "warmup_steps": 2}
    h1, i1 = PL.train_head(rec, data, seed=3, budget=budget)
    h2, i2 = PL.train_head(rec, data, seed=3, budget=budget)
    assert i1["steps"] == 6 and i1["loss_last"] == i2["loss_last"]
    for (n, a), (_, b) in zip(h1.state_dict().items(), h2.state_dict().items()):
        assert torch.equal(a, b), n
    assert h1.attention_free == (rec["family"] in CANDIDATE_FAMILIES)
    for r in data[:3]:
        for lp, q in zip(h1(r), r.questions):
            assert lp.shape == (q.k,) and abs(float(lp.exp().sum()) - 1) < 1e-4


def test_mixer_learns_the_signal_and_beats_chance():
    data, dev = synthetic(120, 0), synthetic(60, 1)
    head, _ = PL.train_head(RECIPES[1], data, seed=0, budget={**PL.TRAIN, "epochs": 3, "grad_accum": 4})
    assert PL.nll(head, dev)["acc"] > 0.8


@pytest.mark.parametrize("cls", [AFMixer, AttnRef, ForkProxy])
def test_option_permutation_equivariance(cls):
    torch.manual_seed(0)
    head = cls(D).eval()
    r = synthetic(1, 5, learnable=False)[0]
    q = r.questions[0]  # choice, 4 options (score levels are ordinal and never permuted)
    perm = torch.tensor([2, 0, 3, 1])
    q2 = ProxyQuestion(q.qid, q.qtype, [q.option_ids[i] for i in perm], q.fork[perm], q.e[perm], q.lex[perm], q.q_pre,
                       q.q_end, q.lm_lp[perm], q.lm_ntok[perm], q.key_class, q.bank_key)
    a = head(ProxyRequest("a", "x", [q], r.banks))[0]
    b = head(ProxyRequest("b", "x", [q2], r.banks))[0]
    assert torch.allclose(a[perm], b, atol=1e-5)


def test_mixer_field_order_invariance_and_field_mixing_matters():
    torch.manual_seed(0)
    head = AFMixer(D).eval()
    r = synthetic(3, 7, learnable=False)[2]  # three questions
    a = {q.qid: lp for q, lp in zip(r.questions, head(r))}
    rr = ProxyRequest("x", "x", list(reversed(r.questions)), r.banks)
    b = {q.qid: lp for q, lp in zip(rr.questions, head(rr))}
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-5)
    alone = head(ProxyRequest("y", "x", r.questions[:1], r.banks))[0]
    assert not torch.allclose(alone, a[r.questions[0].qid], atol=1e-4)  # other fields change the decision


def test_routing_is_a_recurrence_not_attention():
    """No normalisation over the evidence: doubling every chunk vector changes the routed state (an attention-weighted
    average of v would only change through v itself); and the mixer's ablation flag removes the bank's influence."""
    torch.manual_seed(0)
    head = AFMixer(D).eval()
    r = synthetic(1, 3, learnable=False)[0]
    base = head(r)[0]
    r2 = ProxyRequest("z", "x", r.questions, {"state": torch.randn(9, D).bfloat16()})
    assert not torch.allclose(base, head(r2)[0])
    off = make_proxy_head({"family": "mixer", "no_route": True}, D, 0).eval()
    assert torch.allclose(off(r)[0], off(r2)[0])


def test_fork_proxy_is_the_m2_fork_head():
    from heads.features import HeadInput
    from heads.heads import FinalStateFork

    head = make_proxy_head({"family": "fork"}, D, 4).eval()
    assert isinstance(head.head, FinalStateFork)
    q = synthetic(1, 2, learnable=False)[0].questions[0]
    f = q.fork.float()[None]
    x = HeadInput(h=f.new_zeros((1, 1, D)), token_mask=torch.ones(1, 1, dtype=torch.bool),
                  evidence_mask=torch.ones(1, 1, dtype=torch.bool), q_pre=q.q_pre.float()[None],
                  q_end=q.q_end.float()[None], e=q.e.float()[None], option_mask=torch.ones(1, q.k, dtype=torch.bool),
                  fork=f, ks=[q.k])
    assert torch.allclose(head.logits(ProxyRequest("a", "x", [q]))[0], head.head(x).logits[0])


def test_write_results_kit_rows_and_unanswerable_rows_are_errors(tmp_path):
    reqs = synthetic(2, 0, learnable=False)
    rows = {}
    for r in reqs:
        qs = {}
        for q in r.questions:
            if q.qtype == "noul":
                qs[q.qid] = {"type": "noul", "instructions": "?"}
            elif q.qtype == "score":
                qs[q.qid] = {"type": "score", "instructions": "?", "criteria": ["a", "b", "c"]}
            else:
                qs[q.qid] = {"type": "choice", "instructions": "?", "criteria": {o: None for o in q.option_ids}}
        rows[r.rid] = {"questions": qs, "_evaluation": {"run_id": r.rid, "catalog_id": 1}}
    rows["missing"] = {"questions": {}, "_evaluation": {"run_id": "missing", "catalog_id": 1}}
    reqs[1].skipped = {"q9": "T > dev_max_t"}
    preds = PL.stock_predictions(reqs)
    c = PL.write_results(tmp_path / "r.jsonl", rows, reqs, preds, "t")
    assert c == {"ok": 1, "error": 2}
    lines = [json.loads(x) for x in (tmp_path / "r.jsonl").read_text(encoding="utf-8").splitlines()]
    ok = lines[0]
    assert ok["status"] == "ok" and ok["response"]["answers"]["q0"]["type"] == "choice"
    p = ok["response"]["answers"]["q0"]["probabilities"]
    assert abs(sum(p.values()) - 1) < 1e-3
    assert {x["status"] for x in lines[1:]} == {"error"}
