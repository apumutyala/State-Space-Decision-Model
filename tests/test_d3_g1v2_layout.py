"""D3: the g1v2 training layout (heads.layout) reads exactly the R1 / C1 v2 prompt (decide.prompt.render_g1v2)."""
import json
from pathlib import Path

import pytest

from decide.prompt import G1V2_ID, render_g1v2
from heads.layout import ByteEncoder, _g1v2_code_table, build_row

REQ = {"model": "ssdm-m2", "state": {"note": "The customer\n\n\nlost the card yesterday.\n"},
       "questions": {"intent": {"type": "choice", "instructions": "Pick the intent.\n\nOne only.\n",
                                "criteria": {"lost_card": "lost or stolen card", "balance": "check\n\nbalance",
                                             "zz_other": "something else"}},
                     "urgent": {"type": "noul"}}}
TAIL = "\n\nAssistant: <think></think>\nANSWER:"


@pytest.mark.parametrize("qid", ["intent", "urgent"])
def test_g1v2_row_is_the_r1_prompt(qid):
    enc = ByteEncoder()
    row = build_row(REQ, qid, enc, template=G1V2_ID, layout="g1v2")
    text = render_g1v2(REQ, qid, _g1v2_code_table())
    assert row.ids == [0] + enc(text)  # F2 token 0 + F5 whole-string tokenization
    body = text[text.index("User: ") + 6: -len(TAIL)]
    assert text.endswith(TAIL) and "\n\n" not in body  # F1 + F3
    n = row.n_options
    assert row.continuations == [enc(" " + c) for c in _g1v2_code_table()[:n]]  # F4: " <CODE>"
    dec = lambda a, b: bytes(i - 4 for i in row.ids[a:b]).decode()  # noqa: E731
    assert "lost the card" in dec(*row.state_span)
    assert dec(*row.field_span).startswith("FIELD")
    if qid == "intent":
        assert [dec(*s) for s in row.option_spans] == ["balance: check\nbalance", "lost_card: lost or stolen card",
                                                       "zz_other: something else"]
    assert row.meta["template"] == G1V2_ID and row.layout == "g1v2"


def test_g1v2_needs_its_template():
    with pytest.raises(ValueError):
        build_row(REQ, "intent", ByteEncoder(), template="jrt-rwkv-g1-v1", layout="g1v2")


def test_g1v2_cache_fork_equals_recompute():
    torch = pytest.importorskip("torch")
    from heads.backbone import tiny_hf_backbone
    from heads.model import DecisionModel

    rows = [build_row(REQ, q, ByteEncoder(), template=G1V2_ID, layout="g1v2") for q in REQ["questions"]]
    out = []
    for mode in ("cache", "recompute"):
        torch.manual_seed(0)
        out.append(DecisionModel(tiny_hf_backbone("rwkv7", seed=0, fork_mode=mode), "fork")(rows).log_probs.detach())
    assert torch.allclose(out[0], out[1], atol=1e-4)


def test_d3_cell_config_uses_g1v2():
    d = Path(__file__).resolve().parents[1] / "jobs/m2_first_cell"
    for f in d.glob("m2_first_cell_rwkv7_d3_*.json"):
        cfg = json.loads(f.read_text(encoding="utf-8"))
        assert cfg["template"] == G1V2_ID and cfg["layout"] == "g1v2", f.name
