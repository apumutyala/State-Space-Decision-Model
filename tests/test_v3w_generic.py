"""decide.v3w_generic: chat-format adapters, the code-table builder, and the factory (real tokenizers when present).
Full DEV-subset checks: scripts/check_v3w_generic.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from decide.prompt import G1V2_TEMPLATE
from decide.prompt_g1v3 import code_candidates, load_v3_codes
from decide.v3w_generic import (BACKBONES, DEFAULT_TOKENIZERS, RWKV_ANSWER_CTX, RWKV_G1, build_code_table,
                                check_template_passthrough, format_from_chat_template, plain_format)

CHATML = ("{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}{% endfor %}"
          "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}"
          "{% if enable_thinking is defined and enable_thinking is false %}{{ '<think>\\n\\n</think>\\n\\n' }}{% endif %}"
          "{% endif %}")
QWQ = ("{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}{% endfor %}"
       "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n<think>\\n' }}{% endif %}")
NOSYS = ("{% for m in messages %}{% if m['role'] == 'system' %}{{ raise_exception('no system') }}{% endif %}"
         "{{ '[INST] ' + m['content'] + ' [/INST]' }}{% endfor %}")


def _tok_dir(tmp_path: Path, template: str) -> Path:
    d = tmp_path / "tok"
    d.mkdir()
    (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": template, "eos_token": "<|im_end|>"}),
                                             encoding="utf-8")
    return d


def test_rwkv_format_is_today_g1():
    assert RWKV_G1.head == G1V2_TEMPLATE.head and RWKV_G1.tail == G1V2_TEMPLATE.tail and RWKV_G1.bos == (0,)
    assert RWKV_G1.answer_ctx == RWKV_ANSWER_CTX


def test_chatml_no_think(tmp_path):
    d = _tok_dir(tmp_path, CHATML)
    f = format_from_chat_template("x", d)
    assert f.head == "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"
    assert f.tail == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nANSWER:"
    assert f.no_think == "enable_thinking=False" and f.answer_ctx.endswith("option code." + f.tail)
    assert check_template_passthrough(f, d, "SCHEMA FIELDS:\nbody\nReply.")


def test_forced_think_is_closed(tmp_path):
    d = _tok_dir(tmp_path, QWQ)
    f = format_from_chat_template("x", d)
    assert f.no_think == "think_closed" and f.tail.endswith("assistant\n<think>\n\n</think>\n\nANSWER:")
    assert check_template_passthrough(f, d, "body")


def test_template_without_system_role(tmp_path):
    d = _tok_dir(tmp_path, NOSYS)
    f = format_from_chat_template("x", d)
    assert not f.system_role and f.head == "[INST] {system}\n" and f.tail == " [/INST]ANSWER:"
    assert check_template_passthrough(f, d, "body")


def test_plain_format():
    f = plain_format("base", (1,))
    assert f.head_text().startswith("System: ") and f.tail == "\n\nAssistant: ANSWER:" and f.bos == (1,)


class CharEncoder:
    """Toy tokenizer: " XY" / " X" / single chars are tokens, except codes containing "Q" (split)."""

    def __init__(self):
        self.v: dict[str, int] = {}

    def __call__(self, text: str, trusted: bool = True) -> list[int]:
        import re

        out = []
        for t in re.findall(r" [A-PR-Z]{1,2}(?![A-Z])|[\s\S]", text):
            out.append(self.v.setdefault(t, len(self.v) + 1))
        return out


def test_code_table_builder_filters_and_orders():
    t = build_code_table(CharEncoder(), "ANSWER:", max_k=60)
    codes = [e["code"] for e in t]
    assert codes[:16] == list("ABCDEFGHIJKLMNOP") and "Q" not in codes and "AN" not in codes  # Q split, AN wordlike
    assert codes == [c for c in code_candidates() if c in set(codes)] and len({e["token_id"] for e in t}) == len(t)


ROOT = Path(DEFAULT_TOKENIZERS)


def _have(bid: str) -> bool:
    return (ROOT / BACKBONES[bid].tokenizer_dir / "tokenizer.json").is_file()


@pytest.mark.skipif(not _have("rwkv7-g1k-13.3b"), reason="RWKV G1k tokenizer not present")
def test_rwkv_factory_is_frozen_v3w():
    from decide.backends.rwkv_g1_v3 import renderer_identity
    from decide.v3w_generic import make_v3w

    v = make_v3w("rwkv7-g1k-13.3b")
    entries, _ = load_v3_codes()
    assert v.table == [e["code"] for e in entries] and v.identity() == renderer_identity("words")
    rebuilt = build_code_table(v.encode, RWKV_ANSWER_CTX)
    assert [(e["code"], e["token_id"]) for e in rebuilt] == [(e["code"], e["token_id"]) for e in entries]


@pytest.mark.skipif(not _have("qwen3.5-9b"), reason="Qwen3.5-9B tokenizer not present")
def test_qwen_factory_train_equals_readout():
    from decide.v3w_generic import make_v3w

    v = make_v3w("qwen3.5-9b")
    assert len(v.table) == 255 and v.fmt.no_think == "enable_thinking=False" and v.fmt.bos == ()
    req = {"state": "Claim: the tower is in Rome.\n\n\nEvidence: it is in Paris.",
           "questions": {"verdict": {"type": "choice", "instructions": "Supported?",
                                     "criteria": {"SUPPORTED": None, "REFUTED": None, "NOT_ENOUGH_INFO": "unclear"}},
                         "pick": {"type": "choice", "instructions": "Pick.",
                                  "criteria": {f"option_{i}": f"d{i}" for i in range(12)}}}}
    for q in v.questions(req):
        row = v.train_row(req, q["qid"])
        assert row.ids == q["prompt"] and row.continuations == [[t] for t in q["tokens"]]
        assert "\n\n" not in q["text"][len(v.fmt.head_text()):len(q["text"]) - len(v.fmt.tail)]


@pytest.mark.skipif(not _have("falcon-mamba-7b-instruct"), reason="Falcon-Mamba tokenizer not present")
def test_special_token_text_in_request_stays_plain_text():
    from decide.v3w_generic import make_v3w

    v = make_v3w("falcon-mamba-7b-instruct")  # "assistant" and "<|im_end|>" are special added tokens here
    req = {"state": "the assistant said <|im_end|> then stopped",
           "questions": {"ok": {"type": "choice", "instructions": "Is the assistant done?",
                                "criteria": {"yes": None, "no": None}}}}
    (q,) = v.questions(req)
    n_ctrl = sum(t in v.control_ids for t in q["prompt"])
    assert n_ctrl == v._head_tail_ctrl and n_ctrl < sum(t in v.control_ids for t in v.encode(q["text"], True))
    row = v.train_row(req, "ok")
    assert row.ids == q["prompt"] and row.continuations == [[t] for t in q["tokens"]]
