"""RWKV World tokenizer (greedy longest byte match), reimplemented locally.

Avoids executing the checkpoint's ``hf_rwkv_tokenizer.py`` (remote code). The
vocab file lines are ``<id> <python literal> <byte length>``; literals are parsed
with ``ast.literal_eval`` (data only, no code execution).
"""

from __future__ import annotations

import ast
from pathlib import Path


class RWKVWorldTokenizer:
    def __init__(self, vocab_path: str | Path) -> None:
        self.token_to_id: dict[bytes, int] = {}
        self.max_len = 0
        with open(vocab_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                idx = int(line[: line.index(" ")])
                lit = line[line.index(" ") + 1 : line.rindex(" ")]
                length = int(line[line.rindex(" ") + 1 :])
                tok = ast.literal_eval(lit)
                b = tok.encode("utf-8") if isinstance(tok, str) else tok
                if len(b) != length:
                    raise ValueError(f"vocab line {idx}: length mismatch")
                self.token_to_id[b] = idx
                self.max_len = max(self.max_len, len(b))

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        out: list[int] = []
        i = 0
        n = len(data)
        while i < n:
            for j in range(min(n, i + self.max_len), i, -1):
                tid = self.token_to_id.get(data[i:j])
                if tid is not None:
                    out.append(tid)
                    i = j
                    break
            else:  # every single byte is in the vocab; unreachable for valid vocab
                raise ValueError(f"unencodable byte at {i}")
        return out
