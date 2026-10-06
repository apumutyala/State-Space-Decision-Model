"""The one canonical encoder for untrusted text. Import this; do not re-implement it.

Untrusted text means anything that comes from a request: the state, question and
option ids, instructions, option descriptions, and attack strings in safety_eval.
Such text must never encode to a control id. A control id is any special token
or any *added* token. Added tokens include non-special ones such as ``<think>``,
``</think>``, ``<tool_call>``, ``<tool_response>`` and ``<|fim_*|>`` on Qwen
tokenizers.

Method (``UNTRUSTED_ENCODER_ID``):

1. Find the added tokens that still parse as single ids under
   ``split_special_tokens=True`` (on Qwen3.5: 12 of 33). Cut the text into
   pieces so that no piece contains one of those strings, cutting each
   occurrence at its midpoint, repeatedly. Special tokens are not cut, because
   the flag already splits them, so text without leaking tokens encodes exactly
   as with the flag alone.
2. Encode each piece separately with ``add_special_tokens=False,
   split_special_tokens=True`` and concatenate the ids.
3. Fail closed. If any control id still appears (for example a 1-character added
   token that cannot be cut, or a tokenizer that ignores the flags), raise
   ``ControlTokenError``. Callers map it to ``status = unsupported``,
   ``code = control_tokens_in_input``.

The model sees exactly the same characters. Only BPE merges at the cut points
differ, so ``decode(ids) == text``. Inserting U+200B (ZWSP) instead changes the
text the model reads, so ZWSP is only a labelled control arm in safety_eval,
not the defence.

Limit: no text template can stop plain-text role spoofing (a state containing
``END STATE`` or ``Assistant:``). Only control-token spoofing is prevented.
"""

from __future__ import annotations

from typing import Any

UNTRUSTED_ENCODER_ID = "untrusted-split-special+cut-leaking-added-v1"
CONTROL_TOKENS_CODE = "control_tokens_in_input"


class ControlTokenError(ValueError):
    """Untrusted text could not be encoded without control ids (fail-closed)."""

    status = "unsupported"
    code = CONTROL_TOKENS_CODE


def control_ids_and_contents(tokenizer: Any) -> tuple[frozenset[int], tuple[str, ...]]:
    ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    contents = set(getattr(tokenizer, "all_special_tokens", []) or [])
    for tid, tok in (getattr(tokenizer, "added_tokens_decoder", None) or {}).items():
        ids.add(int(tid))
        contents.add(getattr(tok, "content", str(tok)))
    return frozenset(int(i) for i in ids), tuple(sorted((c for c in contents if c), key=len, reverse=True))


class UntrustedEncoder:
    """``UntrustedEncoder(hf_tokenizer).encode(text) -> list[int]``; raises ControlTokenError."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.control_ids, contents = control_ids_and_contents(tokenizer)
        # Cut only tokens that survive split_special_tokens (non-special added tokens such as
        # <think>); special tokens are already split by the flag, so their encoding is unchanged.
        self._contents = tuple(
            c for c in contents
            if set(tokenizer(c, add_special_tokens=False, split_special_tokens=True).input_ids) & self.control_ids
        )

    def identity(self) -> dict[str, Any]:
        return {"id": UNTRUSTED_ENCODER_ID, "n_control_ids": len(self.control_ids)}

    def _pieces(self, text: str) -> list[str]:
        pieces, stack = [], [text]
        while stack:
            piece = stack.pop()
            hit = None
            for c in self._contents:
                at = piece.find(c)
                if at >= 0 and len(c) >= 2:
                    hit = (at, c)
                    break
            if hit is None:
                pieces.append(piece)
                continue
            at, c = hit
            cut = at + len(c) // 2
            stack.append(piece[cut:])  # processed after the left part (stack order)
            stack.append(piece[:cut])
        return pieces

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in self._pieces(text):
            if piece:
                ids += self.tokenizer(piece, add_special_tokens=False, split_special_tokens=True).input_ids
        bad = sorted({i for i in ids if i in self.control_ids})
        if bad:
            raise ControlTokenError(f"untrusted text encodes to control ids {bad[:8]}; refused")
        return ids


def encode_untrusted(tokenizer: Any, text: str) -> list[int]:
    """Convenience wrapper; builds the control-id table on each call."""
    return UntrustedEncoder(tokenizer).encode(text)
