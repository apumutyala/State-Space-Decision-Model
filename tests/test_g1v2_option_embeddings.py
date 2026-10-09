"""g1v2 rows (templates v2 / v2f / v3w) feed real option embeddings e to the heads: before the fix e was all zeros, so
typed.ordinal.gap.0.weight never got a gradient (research-notes/D10_DIVERGENCE.md, side finding 8)."""

import pytest

torch = pytest.importorskip("torch")

from heads.backbone import tiny_hf_backbone  # noqa: E402
from heads.layout import ByteEncoder, build_row  # noqa: E402
from heads.model import DecisionModel  # noqa: E402

REQ = {"model": "t", "state": {"note": "color is blue, size 2"},
       "questions": {"s": {"type": "score", "instructions": "size?", "criteria": ["0", "1", "2"]},
                     "c": {"type": "choice", "instructions": "color?", "criteria": {"red": "r", "blue": None}}}}


@pytest.mark.parametrize("layout,template", [("ordered", "jrt-rwkv-g1-v1"), ("g1v2", "jrt-rwkv-g1-v2f"),
                                             ("g1v2", "jrt-rwkv-g1-v3w")])
def test_option_embeddings_and_ordinal_gap_gradient(layout, template):
    pytest.importorskip("transformers")
    rows = [build_row(REQ, q, ByteEncoder(), template=template, layout=layout) for q in REQ["questions"]]
    m = DecisionModel(tiny_hf_backbone("rwkv7", seed=0, fork_mode="cache"), "fork")
    fw = m(rows)
    (-fw.log_probs[0, 1] - fw.log_probs[1, 0]).backward()
    assert float(fw.inputs.e.abs().sum()) > 0
    assert float(m.typed.ordinal.gap[0].weight.grad.abs().sum()) > 0
