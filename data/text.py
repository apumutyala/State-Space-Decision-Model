"""Text normalisation, segment extraction and stable hashing shared by the converter and the dedup guard.

Everything here is deterministic across processes (no use of Python's salted ``hash``).
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Iterator

import numpy as np

_NON_ALNUM = re.compile(r"[\W_]+")  # Unicode-aware: letters/digits of every script are kept
# "Premise: ...", "Topic: ...", "Candidate: ..." -> also index the text without the short label prefix.
_LABEL_PREFIX = re.compile(r"^\s*[A-Za-z][A-Za-z0-9 _/()#-]{0,30}:\s+")

MIN_SEGMENT_CHARS = 12  # normalised segments shorter than this are not item-identifying ("Yes", "4", "Sarah")


def normalize(text: str) -> str:
    """NFKC, casefold, every run of non-word characters (Unicode-aware; underscore included) -> one space."""
    t = unicodedata.normalize("NFKC", text).casefold()
    return _NON_ALNUM.sub(" ", t).strip()


def words(text: str) -> list[str]:
    n = normalize(text)
    return n.split() if n else []


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def h64(s: str) -> int:
    """Stable unsigned 64-bit hash of a string."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def strings_in(obj: Any) -> Iterator[str]:
    """Every string value (not dict keys) inside a JSON-like object, depth first."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from strings_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from strings_in(v)
    elif obj is not None and not isinstance(obj, bool):
        # numbers inside structured states (prices, ids) are kept as text for exact matching only
        yield str(obj)


def raw_segments(state: Any, questions: dict) -> list[str]:
    """Item text an engine sees: every string in state, every instruction, every option description."""
    out = list(strings_in(state))
    for q in (questions or {}).values():
        out.extend(strings_in(q.get("instructions")))
        out.extend(strings_in(q.get("criteria")))
    return out


def segments(state: Any, questions: dict) -> set[str]:
    """Normalised segments: whole strings, their lines, and lines without a short 'Label:' prefix."""
    segs: set[str] = set()
    for s in raw_segments(state, questions):
        variants = [s]
        if "\n" in s:
            variants.extend(s.split("\n"))
        for v in list(variants):
            m = _LABEL_PREFIX.match(v)
            if m:
                variants.append(v[m.end():])
        for v in variants:
            n = normalize(v)
            if len(n) >= MIN_SEGMENT_CHARS:
                segs.add(n)
    # the whole state as one unit (structured states: canonical JSON of the normalised strings)
    if state not in (None, "", {}, []):
        whole = normalize(" ".join(strings_in(state)))
        if len(whole) >= MIN_SEGMENT_CHARS:
            segs.add(whole)
    return segs


def payload_texts(state: Any, questions: dict) -> list[str]:
    """Full strings (normalised) used for 13-gram shingling; lines are substrings so are covered."""
    out = []
    for s in raw_segments(state, questions):
        n = normalize(s)
        if n:
            out.append(n)
    return out


_P = np.uint64(1099511628211)  # FNV prime; arithmetic wraps mod 2**64


def word_ids(ws: list[str]) -> np.ndarray:
    return np.fromiter((h64(w) for w in ws), dtype=np.uint64, count=len(ws))


def ngram_hashes(ids: np.ndarray, n: int) -> np.ndarray:
    """Polynomial rolling hash of every word n-gram (uint64, wraps). Empty if fewer than n words."""
    m = ids.shape[0] - n + 1
    if m <= 0:
        return np.empty(0, dtype=np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for i in range(n):
            h = h * _P + ids[i:i + m]
    return h


def char_shingles(norm: str, k: int = 5) -> np.ndarray:
    """Hashes of character k-grams of a normalised string (uint32, stable)."""
    if len(norm) < k:
        grams = {norm}
    else:
        grams = {norm[i:i + k] for i in range(len(norm) - k + 1)}
    return np.fromiter((h64(g) & 0xFFFFFFFF for g in grams), dtype=np.uint64, count=len(grams))


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
