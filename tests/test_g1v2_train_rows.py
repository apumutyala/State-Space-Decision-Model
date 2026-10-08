"""g1v2 training rows (heads.layout layout "g1v2"): option_order permutes the option axis, never the prompt; the
segments are the v2 readout's own (decide.prompt.build_prompt_g1v2). Real-tokenizer check:
scripts/check_g1v2_train_equivalence.py."""
from __future__ import annotations

from decide.prompt import G1V2_ID, build_prompt_g1v2
from heads.layout import ByteEncoder, _g1v2_code_table, _g1v2_segments_paths, build_row

REQ = {"model": "t", "state": {"note": "color is blue\n\n\nsize 2"},
       "questions": {"c": {"type": "choice", "instructions": "color?", "criteria": {"red": "r", "blue": None, "green": "g"}},
                     "s": {"type": "score", "instructions": "size?", "criteria": ["0", "1", "2"]}}}


def test_segments_are_the_readout_segments():
    table = _g1v2_code_table()
    for qid in REQ["questions"]:
        mine = _g1v2_segments_paths(REQ, qid, table)
        ro = build_prompt_g1v2(REQ, qid, table)
        assert [(t, tr) for t, tr, _, _ in mine] == [(t, lab == "template") for t, lab in ro]


def test_option_order_permutes_axis_not_prompt():
    enc = ByteEncoder()
    base = build_row(REQ, "c", enc, template=G1V2_ID, layout="g1v2")
    perm = build_row(REQ, "c", enc, template=G1V2_ID, layout="g1v2", option_order=[2, 0, 1])
    assert perm.ids == base.ids
    assert perm.option_ids == [base.option_ids[k] for k in (2, 0, 1)]
    assert perm.continuations == [base.continuations[k] for k in (2, 0, 1)]
    assert perm.option_spans == [base.option_spans[k] for k in (2, 0, 1)]
    assert base.option_ids == ["blue", "green", "red"]  # canonical (sorted id) order, as the readout codes them
