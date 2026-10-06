"""ZWSP CONTROL ARM ONLY. The defence is decide/untrusted.py (UntrustedEncoder), imported, never re-implemented.

This module inserts U+200B into role/special-token strings. That changes the characters the model reads, so it is
kept only as a labelled control arm (``HFLetterDecider(untrusted_mode="zwsp")``) to compare against the canonical
encoder. Original notes:

Prior-lab finding (chl-typed-control item 2): literal ``<|im_start|>``/``<|im_end|>`` inside a tool output are
parsed by Qwen-family tokenizers as real role tokens (ids 151644/151645), producing most of the measured hijack
on Qwen models; inserting U+200B after the first character (or split-tokenising the span) removed 96-100% of it.

Two layers (control arm):
  1. ``sanitize_text``: string level, tokenizer-independent; breaks every known role/special marker and every
     added-token string of the tokenizer at hand by inserting a zero-width space after its first character.
  2. ``assert_no_special_ids``: tokenizer level; encodes the sanitised span and fails closed if any special or
     added-token id survives (catches markers we did not list).
"""
from __future__ import annotations

import re
from typing import Iterable

ZWSP = "​"

# Role / control markers across common chat formats (chatml/Qwen, Llama-3, Gemma, Mistral, gpt-oss harmony,
# Phi). Matching is case-sensitive, as tokenizers are.
KNOWN_MARKERS = (
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|im_sep|>",
    "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>",
    "<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>",
    "[INST]", "[/INST]", "<s>", "</s>",
    "<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|return|>", "<|call|>",
    "<|system|>", "<|user|>", "<|assistant|>",
    "<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
)


def tokenizer_markers(tok) -> list[str]:
    """Added-token strings of an HF tokenizer (special and non-special), longest first."""
    dec = getattr(tok, "added_tokens_decoder", {}) or {}
    out = {t.content for t in dec.values() if getattr(t, "content", "") and not t.content.isspace()}
    return sorted(out, key=len, reverse=True)


def sanitize_text(text: str, extra_markers: Iterable[str] = ()) -> tuple[str, int]:
    """Return (sanitised text, number of markers broken). Idempotent: a broken marker no longer matches."""
    markers = sorted(set(KNOWN_MARKERS) | set(extra_markers), key=len, reverse=True)
    markers = [m for m in markers if len(m) >= 2]
    if not text or not markers:
        return text, 0
    rx = re.compile("|".join(re.escape(m) for m in markers))
    n = 0

    def br(m: re.Match) -> str:
        nonlocal n
        n += 1
        s = m.group(0)
        return s[0] + ZWSP + s[1:]

    return rx.sub(br, text), n


def special_ids(tok) -> set[int]:
    ids = set(getattr(tok, "all_special_ids", []) or [])
    ids |= set((getattr(tok, "added_tokens_decoder", {}) or {}).keys())
    return ids


def assert_no_special_ids(tok, text: str) -> None:
    """Fail closed if encoding ``text`` (special-token parsing ON, as a naive renderer would) yields any
    special/added-token id."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    bad = sorted(set(ids) & special_ids(tok))
    if bad:
        raise ValueError(f"untrusted span still encodes special token ids {bad[:8]}")


def sanitize_request_state(request: dict, fields: list[str], extra_markers: Iterable[str] = ()) -> tuple[dict, int]:
    """Copy of ``request`` with the given state fields sanitised; returns (request, markers broken)."""
    import copy
    r = copy.deepcopy(request)
    tot = 0
    for f in fields:
        v = r["state"].get(f)
        if isinstance(v, str):
            r["state"][f], n = sanitize_text(v, extra_markers)
            tot += n
    return r, tot
