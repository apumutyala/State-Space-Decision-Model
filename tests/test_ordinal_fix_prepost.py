"""The ordinal fix (6be1c8f, heads/features.py) pre vs post, on one code base: the pre-fix branch is exactly the post-fix
one with G1V2_LAYOUT unmatched (``r.layout in ("ordered", G1V2_LAYOUT)`` -> ``r.layout == "ordered"``), so monkeypatching
that module global reproduces the pre-fix code path byte for byte.

* under g1v2 (v3w), an ordered-trained head (D1's: fork head, trained under layout ordered) gets e != 0 post-fix (e = 0
  pre-fix), and its score questions move;
* under v1 (layout ordered, the readout_ab_d1pf control) every head input and output is byte-identical pre/post;
* under g1v2, choice / noul questions through the fork head are byte-identical pre/post: FinalStateFork never reads e
  and TypedOutput reads it only for score questions (the 2,008-row subset has none: postfix_landing counts them)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import heads.features as HF  # noqa: E402
from heads.backbone import tiny_hf_backbone  # noqa: E402
from heads.layout import ByteEncoder, build_row  # noqa: E402
from heads.model import DecisionModel  # noqa: E402

PRE_FIX = "<pre-fix: g1v2 not matched>"
REQ = {"model": "t", "state": {"note": "color is blue, size 2, it is urgent"},
       "questions": {"s": {"type": "score", "instructions": "size?", "criteria": ["0", "1", "2"]},
                     "c": {"type": "choice", "instructions": "color?", "criteria": {"red": "r", "blue": None}},
                     "u": {"type": "noul", "instructions": "Is it urgent?"}}}


def _rows(layout, template, qids=("s", "c", "u")):
    return [build_row(REQ, q, ByteEncoder(), template=template, layout=layout) for q in qids]


def _forward(rows, monkeypatch, pre):
    with monkeypatch.context() as m:
        if pre:
            m.setattr(HF, "G1V2_LAYOUT", PRE_FIX)
        torch.manual_seed(0)
        model = DecisionModel(tiny_hf_backbone("rwkv7", seed=0, fork_mode="cache"), "fork")  # D1's head kind
        model.eval()
        with torch.no_grad():
            fw = model(rows)
    return fw


def _b(t):
    return t.detach().cpu().contiguous().numpy().tobytes()


def test_g1v2_e_nonzero_postfix_for_ordered_trained_head(monkeypatch):
    rows = _rows("g1v2", "jrt-rwkv-g1-v3w")
    pre, post = _forward(rows, monkeypatch, True), _forward(rows, monkeypatch, False)
    assert float(pre.inputs.e.abs().sum()) == 0.0  # the bug: every g1v2 row had e = 0
    for i, r in enumerate(rows):
        assert float(post.inputs.e[i, : r.n_options].abs().sum(-1).min()) > 0  # every real option has e != 0
    # the fork features / head logits do not read e, the score question's ordinal output does
    assert _b(pre.inputs.fork) == _b(post.inputs.fork)
    i_s = [r.qid for r in rows].index("s")
    assert not torch.equal(pre.log_probs[i_s], post.log_probs[i_s])


def test_v1_ordered_byte_identical_pre_post(monkeypatch):
    rows = _rows("ordered", "jrt-rwkv-g1-v1")
    pre, post = _forward(rows, monkeypatch, True), _forward(rows, monkeypatch, False)
    assert float(post.inputs.e.abs().sum()) > 0
    for f in ("e", "fork", "q_pre", "q_end", "option_mask"):
        assert _b(getattr(pre.inputs, f)) == _b(getattr(post.inputs, f)), f
    assert _b(pre.log_probs) == _b(post.log_probs)


def test_g1v2_choice_noul_unchanged_through_the_fork_head(monkeypatch):
    rows = _rows("g1v2", "jrt-rwkv-g1-v3w", qids=("c", "u"))
    pre, post = _forward(rows, monkeypatch, True), _forward(rows, monkeypatch, False)
    assert not torch.equal(pre.inputs.e, post.inputs.e)  # e did change ...
    assert _b(pre.log_probs) == _b(post.log_probs)  # ... and nothing a choice / noul answer reads moved
