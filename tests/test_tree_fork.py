"""Tree fork (prefix -> question -> option, each read once), CPU, tiny fp32 models:

* inference (``heads.infer_opt`` ``tree_fork``; ``heads.prefix_features.tree_prefix_features``): features and head
  log-probs equal the packed / grouped / plain paths within 1e-5 for every head; pack limits that split suffixes and
  continuations; non-RWKV backbones fall back to the packed path; the token plan (``tree_plan``) is what the executor
  actually reads, and it never reads more than ``packed_plan``; the backend presets ``tree`` / ``exact_tree`` equal
  the base backend;
* training (``TrainConfig.share_prefix`` -> ``HiddenStateBackbone.share_prefix`` in ``fork_features_rows``): features,
  fork features, loss and every LoRA / head gradient equal the original step (1e-5; cosine >= 0.99999), with and
  without activation checkpointing and ``cont_plan="single"``; execution-only (the config hash is unchanged).

    python -X utf8 -m pytest -q tests/test_tree_fork.py
"""

from __future__ import annotations

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from heads import HEADS, ByteEncoder, DecisionModel, build_row  # noqa: E402
from heads import prefix_features as PF  # noqa: E402
from heads.backbone import tiny_hf_backbone  # noqa: E402

ENC = ByteEncoder()
SMALL = {"clef_control": {"width": 16, "heads": 2, "feedforward": 32, "layers": 1, "routing_layers": 1}}
TOL = 1e-5
TPL = "jrt-rwkv-g1-v1"  # D1's template: multi-token continuations " <id>\n\n"

REQ_A = {"model": "t", "state": "ticket: checkout down since 9am; vendor Acme; invoice INV-7 overdue",
         "questions": {"dept": {"type": "choice", "instructions": "team?",
                                "criteria": {"billing": "payments", "tech": "outages", "sales": None}},
                       "down": {"type": "noul", "instructions": "is a service down?"},
                       "urg": {"type": "score", "criteria": ["later", "week", "today"]},
                       "inv": {"type": "choice", "instructions": "invoice? answer from the text above, carefully",
                               "criteria": {"paid": None, "late": None, "disputed_by_customer": None}}}}
REQ_B = {"model": "t", "state": "Weather report: heavy rain expected; roads closed near the river.",
         "questions": {"risk": {"type": "choice", "instructions": "flood risk?",
                                "criteria": {"low": None, "medium": None, "high": None}},
                       "go": {"type": "noul", "instructions": "safe to drive?"}}}
REQ_ONE = {"model": "t", "state": "a single question row", "questions": {"q": {"type": "noul", "instructions": "ok?"}}}


def rows(req, template=TPL):
    return [build_row(req, q, ENC, template=template, layout="ordered") for q in req["questions"]]


def _model(family, head, lora=False):
    bb = tiny_hf_backbone(family, fork_mode="cache")
    if lora:
        from train.lora import apply_lora

        apply_lora(bb.model, bb.spec.lora_targets, rank=4, check_counts=False, generator=torch.Generator().manual_seed(1))
        g = torch.Generator().manual_seed(2)
        with torch.no_grad():
            for n, p in bb.model.named_parameters():
                if n.endswith("lora_B"):
                    p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return DecisionModel(bb, head, SMALL.get(head), seed=0).eval()


def _close(a, b, tol=TOL):
    fin = torch.isfinite(b)
    assert torch.equal(fin, torch.isfinite(a))
    assert torch.allclose(a[fin], b[fin], atol=tol, rtol=tol), (a - b).abs()[fin].max()


# ----------------------------------------------------------------------------------------------------- inference
@pytest.mark.parametrize("head", sorted(HEADS))
def test_tree_equals_packed_grouped_and_plain(head):
    m = _model("rwkv7", head)
    assert PF.tree_supported(m.backbone)
    rs = rows(REQ_A) + rows(REQ_B) + rows(REQ_ONE)
    with torch.no_grad():
        t = m(rs, shared_prefix=True, prefix_mode="tree").log_probs
        p = m(rs, shared_prefix=True, prefix_mode="packed").log_probs
        g = m(rs, shared_prefix=True, prefix_mode="grouped").log_probs
        plain = m(rs, shared_prefix=False).log_probs
    for ref in (p, g, plain):
        _close(t, ref)
    assert torch.equal(t.argmax(-1), plain.argmax(-1))


@pytest.mark.parametrize("need_fork", [True, False])
def test_tree_features_under_pack_limits(need_fork):
    m = _model("rwkv7", "fork")
    rs = rows(REQ_A) + rows(REQ_B)
    with torch.no_grad():
        h0, f0 = PF.shared_prefix_features(m.backbone, rs, need_fork)
        for max_rows, max_tokens in ((1, 10_000), (2, 10_000), (64, 12), (3, 40)):
            h1, f1 = PF.tree_prefix_features(m.backbone, rs, need_fork, max_rows=max_rows, max_tokens=max_tokens)
            assert h1.shape == h0.shape
            _close(h1, h0)
            if need_fork:
                _close(f1, f0)
            else:
                assert f1 is None and f0 is None


def test_non_rwkv_backbone_falls_back_to_packed():
    m = _model("q9af", "fork")
    assert not PF.tree_supported(m.backbone)
    rs = rows(REQ_A, "jrt-plain-v1")
    with torch.no_grad():
        _close(m(rs, shared_prefix=True, prefix_mode="tree").log_probs,
               m(rs, shared_prefix=True, prefix_mode="packed").log_probs)


def test_tree_reads_the_planned_tokens_and_fewer_than_packed():
    """The executor's real tokens equal ``tree_plan`` (counted on the model's input ids, mask-aware), and the
    question suffix is no longer re-read per option."""
    m = _model("rwkv7", "fork")
    seen = {"tok": 0, "pos": 0, "calls": 0}

    def hook(mod, args, kw):
        ids, mask = kw["input_ids"], kw.get("attention_mask")
        seen["calls"] += 1
        seen["pos"] += ids.numel()
        seen["tok"] += int(mask.sum()) if mask is not None else ids.numel()

    for req in (REQ_A, REQ_B, REQ_ONE):
        rs = rows(req)
        ids, conts = [r.ids for r in rs], [r.continuations for r in rs]
        for limits in ({}, {"max_rows": 2, "max_tokens": 10_000}):
            seen.update(tok=0, pos=0, calls=0)
            h = m.backbone.model.register_forward_pre_hook(hook, with_kwargs=True)
            try:
                with torch.no_grad():
                    PF.tree_prefix_features(m.backbone, rs, True, **limits)
            finally:
                h.remove()
            plan = PF.tree_plan(ids, conts, **limits)
            # continuation calls are unmasked: their padding counts as tokens in the hook, as positions in the plan
            assert seen["calls"] == plan["calls"] and seen["pos"] == plan["positions"]
            assert plan["tokens"] <= seen["tok"] <= plan["positions"]
        tp, pp = PF.tree_plan(ids, conts), PF.packed_plan(ids, conts)
        L = tp["prefix"]
        assert tp["tokens"] == L + sum(len(x) - L for x in ids) + sum(len(c) for cs in conts for c in cs)
        assert pp["tokens"] == L + sum((len(x) - L) * len(cs) + sum(len(c) for c in cs) for x, cs in zip(ids, conts))
        assert tp["tokens"] < pp["tokens"]


@pytest.mark.parametrize("preset", ["exact_tree", "tree"])
def test_backend_tree_presets_equal_base(preset):
    from heads.decide_backend import DecisionModelBackend

    base = DecisionModelBackend(_model("rwkv7", "fork", lora=True), ENC, template=TPL)
    tree = DecisionModelBackend(_model("rwkv7", "fork", lora=True), ENC, template=TPL, inference=preset)
    assert tree.inference["tree_fork"] and tree.revision()["inference"] == sorted(tree.inference)
    assert tree.revision()["backbone"] == base.revision()["backbone"]
    for req in (REQ_A, REQ_B, REQ_ONE):
        ra, rb = base.score(req, lambda: None), tree.score(req, lambda: None)
        for q in ra.logits:
            for o, v in ra.logits[q].items():
                assert abs(rb.logits[q][o] - v) <= TOL * max(1.0, abs(v)), (q, o, v, rb.logits[q][o])
            assert max(ra.logits[q], key=ra.logits[q].get) == max(rb.logits[q], key=rb.logits[q].get)


def test_parse_tree_presets():
    from heads.infer_opt import parse

    assert parse("tree") == {"memo_encode": True, "merge_lora": True, "tree_fork": True}
    assert parse("exact_tree") == {"memo_encode": True, "tree_fork": True}
    assert parse("fast") == {"memo_encode": True, "packed_fork": True, "merge_lora": True}  # unchanged


# ------------------------------------------------------------------------------------------------------ training
def _items():
    from train.data import Item

    rs = rows(REQ_A) + rows(REQ_B) + rows(REQ_ONE) + rows(REQ_B)  # 9 rows: shared template head, some same-request
    return [Item(f"{i}", r, i % r.n_options) for i, r in enumerate(rs[:8])]


def _trainer(items, **flags):
    from train.guard import SplitGuard
    from train.losses import LossConfig
    from train.trainer import TrainConfig, Trainer

    bb = tiny_hf_backbone("rwkv7", seed=0, fork_mode="cache")
    model = DecisionModel(bb, "fork", {}, pad_id=0, seed=0)
    cfg = TrainConfig(head="fork", fork_mode="cache", lora_rank=4, loss=LossConfig(arm="clef"), batch_size=1,
                      grad_accum=8, max_steps=1, seed=0, **flags)
    tr = Trainer(model, cfg, items, [], guard=SplitGuard.synthetic_only("tree_fork equivalence"))
    tr._order = lambda epoch: list(range(len(items)))
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n.endswith("lora_B"):
                p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(len(n))) * 0.05)
    return tr


def _step_grads(tr):
    grads = {}
    real = tr.opt.step

    def capture(*a, **k):
        for n, p in tr.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grads[n] = p.grad.detach().clone()
        return real(*a, **k)

    tr.opt.step = capture
    loss = tr.train_step()
    tr.opt.step = real
    return loss, grads


def _grad_check(a, b):
    (la, ga), (lb, gb) = a, b
    assert set(ga) == set(gb) and any(n.endswith("lora_A") for n in ga)
    assert abs(la - lb) <= TOL * abs(lb), (la, lb)
    va = torch.cat([g.flatten() for _, g in sorted(ga.items())])
    vb = torch.cat([g.flatten() for _, g in sorted(gb.items())])
    cos = float(torch.nn.functional.cosine_similarity(va, vb, dim=0))
    worst = max(float((ga[n] - gb[n]).abs().max()) for n in ga)
    assert worst <= TOL and cos >= 0.99999, (worst, cos)
    return worst, cos


def test_share_prefix_fork_features_rows_equal():
    m = _model("rwkv7", "fork", lora=True).train()
    bb = m.backbone
    rs = [it.row for it in _items()]
    args = ([r.ids for r in rs], [r.continuations for r in rs])
    h, f = bb.fork_features_rows(*args)
    bb.share_prefix = True
    calls = []
    hk = bb.model.register_forward_pre_hook(lambda mod, a, k: calls.append(k["input_ids"].shape), with_kwargs=True)
    try:
        hs, fs = bb.fork_features_rows(*args)
    finally:
        hk.remove()
    assert calls[0][0] == 1 and calls[0][1] > 30  # the shared template head ran once, B=1
    assert hs.shape == h.shape
    _close(hs, h)
    for a, b in zip(fs, f):
        _close(a, b)
    bb.prefix_groups = [[0, 3, 5], [1], [2, 4, 6, 7]]  # sub-groups (cont_plan single), one with a single row
    hg, fg = bb.fork_features_rows(*args)
    bb.prefix_groups = None
    _close(hg, h)
    for a, b in zip(fg, f):
        _close(a, b)


@pytest.mark.parametrize("flags", [dict(row_batch=8), dict(row_batch=3), dict(row_batch=8, act_ckpt=True),
                                   dict(row_batch=8, row_tokens=600, act_ckpt=True, row_plan="min_cost",
                                        cont_plan="single")])
def test_share_prefix_step_equals_original_step(flags):
    items = _items()
    base = _step_grads(_trainer(items))
    plain = _step_grads(_trainer(items, **flags))
    tr = _trainer(items, share_prefix=True, **flags)
    assert tr.model.backbone.share_prefix
    b1 = []  # not vacuous: the shared head really ran as its own B=1 call
    hk = tr.model.backbone.model.register_forward_pre_hook(
        lambda mod, a, k: b1.append(k["input_ids"].shape[0] == 1 and k.get("state") is None), with_kwargs=True)
    try:
        _grad_check(_step_grads(tr), base)
    finally:
        hk.remove()
    assert any(b1)
    _grad_check(_step_grads(_trainer(items, share_prefix=True, **flags)), plain)


def test_share_prefix_is_execution_only():
    from train.trainer import TrainConfig, with_execution

    cfg = TrainConfig(head="fork", fork_mode="cache")
    assert "share_prefix" not in cfg.to_json()
    assert dataclasses.replace(cfg, share_prefix=True, row_batch=8).hash() == cfg.hash()
    assert with_execution(cfg, {"share_prefix": True}).share_prefix and not with_execution(cfg, {}).share_prefix
