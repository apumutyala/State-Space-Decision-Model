"""heads.infer_opt (inference-only speed options for the trained decision model), CPU, tiny models, fp32:
packed fork == grouped shared prefix == plain path; merged LoRA == unmerged LoRA; the segment memo changes no ids and
encodes each segment once; options off = the previous revision id."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from heads import HEADS, ByteEncoder, DecisionModel, build_row  # noqa: E402
from heads.backbone import tiny_hf_backbone  # noqa: E402

ENC = ByteEncoder()
SMALL = {"clef_control": {"width": 16, "heads": 2, "feedforward": 32, "layers": 1, "routing_layers": 1}}
TOL = 1e-5

REQ_A = {"model": "t", "state": "ticket: checkout down since 9am; vendor Acme; invoice INV-7 overdue",
         "questions": {"dept": {"type": "choice", "instructions": "team?",
                                "criteria": {"billing": "payments", "tech": "outages", "sales": None}},
                       "down": {"type": "noul", "instructions": "is a service down?"},
                       "urg": {"type": "score", "criteria": ["later", "week", "today"]},
                       "inv": {"type": "choice", "instructions": "invoice?", "criteria": {"paid": None, "late": None}}}}
REQ_B = {"model": "t", "state": "Weather report: heavy rain expected; roads closed near the river.",
         "questions": {"risk": {"type": "choice", "instructions": "flood risk?",
                                "criteria": {"low": None, "medium": None, "high": None}},
                       "go": {"type": "noul", "instructions": "safe to drive?"}}}
REQ_ONE = {"model": "t", "state": "a single question row", "questions": {"q": {"type": "noul", "instructions": "ok?"}}}


def rows(req):
    return [build_row(req, q, ENC, template="jrt-plain-v1", layout="ordered") for q in req["questions"]]


def _model(family, head, lora: bool = False):
    bb = tiny_hf_backbone(family, fork_mode="cache")
    if lora:
        from train.lora import apply_lora

        apply_lora(bb.model, bb.spec.lora_targets, rank=4, check_counts=False, generator=torch.Generator().manual_seed(1))
        g = torch.Generator().manual_seed(2)
        with torch.no_grad():  # B is zero-initialised: make the update non-trivial
            for n, p in bb.model.named_parameters():
                if n.endswith("lora_B"):
                    p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return DecisionModel(bb, head, SMALL.get(head), seed=0).eval()


def _close(a, b):
    fin = torch.isfinite(b)
    assert torch.equal(fin, torch.isfinite(a))
    assert torch.allclose(a[fin], b[fin], atol=TOL, rtol=TOL), (a - b).abs()[fin].max()


@pytest.mark.parametrize("family", ["rwkv7", "q9af"])
@pytest.mark.parametrize("head", sorted(HEADS))
def test_packed_equals_grouped_and_plain(family, head):
    m = _model(family, head)
    rs = rows(REQ_A) + rows(REQ_B) + rows(REQ_ONE)  # three prefix groups, one with a single question
    with torch.no_grad():
        p = m(rs, shared_prefix=True, prefix_mode="packed").log_probs
        g = m(rs, shared_prefix=True, prefix_mode="grouped").log_probs
        plain = m(rs, shared_prefix=False).log_probs
    _close(p, g)
    _close(p, plain)


@pytest.mark.parametrize("need_fork", [True, False])
def test_packed_features_with_many_small_packs(need_fork):
    """Pack limits that force one tail per call and mixed widths: same h and fork as the grouped path."""
    from heads import prefix_features as PF

    m = _model("rwkv7", "fork")
    rs = rows(REQ_A) + rows(REQ_B)
    with torch.no_grad():
        h0, f0 = PF.shared_prefix_features(m.backbone, rs, need_fork)
        for max_rows, max_tokens in ((1, 10_000), (2, 10_000), (64, 12)):
            h1, f1 = PF.packed_prefix_features(m.backbone, rs, need_fork, max_rows=max_rows, max_tokens=max_tokens)
            assert h1.shape == h0.shape
            assert torch.allclose(h1, h0, atol=TOL, rtol=TOL), (h1 - h0).abs().max()
            if need_fork:
                assert torch.allclose(f1, f0, atol=TOL, rtol=TOL), (f1 - f0).abs().max()
            else:
                assert f1 is None and f0 is None


def test_packed_uses_fewer_backbone_calls():
    from heads import prefix_features as PF

    m = _model("rwkv7", "fork")
    rs = rows(REQ_A)
    calls = []
    orig = PF._run
    PF._run = lambda *a, **k: calls.append(1) or orig(*a, **k)
    try:
        with torch.no_grad():
            PF.shared_prefix_features(m.backbone, rs, True)
            n_grouped = len(calls)
            calls.clear()
            PF.packed_prefix_features(m.backbone, rs, True)
            n_packed = len(calls)
    finally:
        PF._run = orig
    assert n_packed == 2  # prefix + one pack
    assert n_grouped >= 1 + len(rs)  # prefix + >= 1 suffix call + one fork call per question


@pytest.mark.parametrize("family", ["rwkv7", "q9af"])
def test_merge_lora_equals_unmerged(family):
    from heads import infer_opt

    m = _model(family, "fork", lora=True)
    rs = rows(REQ_A) + rows(REQ_B)
    with torch.no_grad():
        before = m(rs, shared_prefix=True).log_probs
        n = infer_opt.merge_lora(m.backbone.model)
        after = m(rs, shared_prefix=True).log_probs
    assert n > 0
    assert not any(n.endswith(("lora_A", "lora_B")) for n, _ in m.named_parameters())
    _close(after, before)


def test_memo_encoder_same_ids_one_encode_per_segment():
    from heads.infer_opt import MemoEncoder

    seen = []

    def enc(text, trusted=True):
        seen.append(text)
        return ENC(text, trusted)

    memo = MemoEncoder(enc)
    a = [build_row(REQ_A, q, ENC, template="jrt-plain-v1") for q in REQ_A["questions"]]
    b = [build_row(REQ_A, q, memo, template="jrt-plain-v1") for q in REQ_A["questions"]]
    b2 = [build_row(REQ_A, q, memo, template="jrt-plain-v1") for q in REQ_A["questions"]]  # score() after plan()
    assert [r.ids for r in a] == [r.ids for r in b] == [r.ids for r in b2]
    assert [r.state_span for r in a] == [r.state_span for r in b]
    assert len(seen) == len(set((t) for t in seen)) == memo.misses
    assert seen.count(REQ_A["state"]) <= 1


@pytest.mark.parametrize("preset", ["exact", "fast"])
def test_backend_options_equal_base(preset):
    from heads.decide_backend import DecisionModelBackend

    base_m = _model("rwkv7", "fork", lora=True)
    fast_m = _model("rwkv7", "fork", lora=True)  # same seeds: identical weights
    base = DecisionModelBackend(base_m, ENC, template="jrt-plain-v1")
    fast = DecisionModelBackend(fast_m, ENC, template="jrt-plain-v1", inference=preset)
    assert "inference" not in base.revision()
    assert fast.revision()["inference"] == sorted(fast.inference)
    assert fast.revision()["backbone"] == base.revision()["backbone"]  # hashed before the merge
    for req in (REQ_A, REQ_B, REQ_ONE):
        assert fast.plan(req) == base.plan(req)
        ra, rb = base.score(req, lambda: None), fast.score(req, lambda: None)
        assert ra.logits.keys() == rb.logits.keys()
        for q in ra.logits:
            for o, v in ra.logits[q].items():
                assert abs(rb.logits[q][o] - v) <= TOL * max(1.0, abs(v)), (q, o, v, rb.logits[q][o])
        assert rb.features == ra.features


def test_parse_rejects_unknown_options():
    from heads.infer_opt import parse

    assert parse(None) == {} and parse("base") == {}
    assert parse("fast") == {"memo_encode": True, "packed_fork": True, "merge_lora": True}
    assert parse("packed_fork") == {"packed_fork": True}
    with pytest.raises(ValueError):
        parse("speculative")
