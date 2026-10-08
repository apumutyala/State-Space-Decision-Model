"""Template jrt-rwkv-g1-v3w under layout g1v2: the row is the v3w readout's prompt (decide.prompt_g1v3.render_g1v3 with
the readout's word-id set and code-key label width) and the target is the token the readout scores. Real-tokenizer
check: scripts/check_g1v2_train_equivalence.py --template jrt-rwkv-g1-v3w."""
from __future__ import annotations

import re

import pytest

from decide.prompt_g1v3 import G1V3W_ID, render_g1v3
from heads.layout import ByteEncoder, _v3_code_table, build_row, v3w_context, v3w_first_token


class WordEncoder:
    """Toy whole-string tokenizer: " word" / "word" / one other char per token (so " yes" is ONE token, as in G1k)."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.inv: dict[int, str] = {}

    def __call__(self, text: str, trusted: bool = True) -> list[int]:
        out = []
        for t in re.findall(r" ?[A-Za-z_]+| ?\d+|[\s\S]", text):
            if t not in self.vocab:
                self.vocab[t] = len(self.vocab) + 1  # 0 is the control id
                self.inv[self.vocab[t]] = t
            out.append(self.vocab[t])
        return out

    def token_bytes(self, i: int) -> bytes:
        return self.inv[i].encode("utf-8")


WORDS = {"type": "choice", "instructions": "Is it relevant?", "criteria": {"yes": None, "no": "not relevant"}}
OPAQUE = {"type": "choice", "instructions": "Pick one.", "criteria": {f"option_{i}": f"desc {i}" for i in range(12)}}
REQ = {"model": "t", "state": "some evidence\n\n\nhere", "questions": {"rel": WORDS, "pick": OPAQUE}}


def test_word_ids_scored_by_first_token():
    enc = WordEncoder()
    row = build_row(REQ, "rel", enc, template=G1V3W_ID, layout="g1v2")
    ctx = v3w_context(REQ, "rel", enc, _v3_code_table())
    assert ctx["words"] == frozenset({"rel"})  # option_0.. are opaque
    text = render_g1v3(REQ, "rel", _v3_code_table(), None, False, True, ctx["words"])
    assert row.ids == [0] + enc(text) and "\n- no: not relevant" in text and "A. " not in text.split("QUESTION (")[1]
    assert row.continuations == [[v3w_first_token(enc, "no")], [v3w_first_token(enc, "yes")]]
    assert row.meta["words"] and row.answer_pos == len(row.ids) - 1


def test_coded_catalog_question_gets_code_key_and_v3_codes():
    enc = WordEncoder()
    row = build_row(REQ, "pick", enc, template=G1V3W_ID, layout="g1v2")
    assert row.meta["catalog_chars"] == 64 and not row.meta["words"]
    text = render_g1v3(REQ, "pick", _v3_code_table(), 64, False, True, frozenset({"rel"}))
    assert row.ids == [0] + enc(text) and "CODE KEY" in text
    assert row.continuations == [enc(" " + c) for c in _v3_code_table()[:12]]


def test_budget_ladder_steps_down():
    enc = WordEncoder()
    full = v3w_context(REQ, "pick", enc, _v3_code_table())
    n64 = 1 + len(enc(render_g1v3(REQ, "pick", _v3_code_table(), 64, False, True, full["words"])))
    assert v3w_context(REQ, "pick", enc, _v3_code_table(), budget=n64)["catalog_chars"] == 64
    assert v3w_context(REQ, "pick", enc, _v3_code_table(), budget=n64 - 1)["catalog_chars"] in (24, None)
    assert v3w_context(REQ, "pick", enc, _v3_code_table(), budget=5)["catalog_chars"] is None


def test_byte_encoder_never_has_word_ids():
    row = build_row(REQ, "rel", ByteEncoder(), template=G1V3W_ID, layout="g1v2")  # every " <id>" starts with b" "
    assert not row.meta["words"] and row.continuations == [ByteEncoder()(" " + c) for c in _v3_code_table()[:2]]


def test_v3w_template_identity_is_the_words_readout():
    pytest.importorskip("torch")
    from decide.backends.rwkv_g1_v3 import renderer_identity
    from heads.decide_backend import _template_identity
    from heads.layout import resolve_template

    assert _template_identity(resolve_template(G1V3W_ID)) == renderer_identity("words")
