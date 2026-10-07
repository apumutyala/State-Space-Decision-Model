"""Option-key classes and the E5 short-code rendering (techniques/10 §4: the stock readout loses on opaque keys).

``key_class(option_ids)`` classifies a question's option ids into one of ``KEY_CLASSES``:
  letter    single / double upper-case letters (A, B, ..., AA)
  option_n  an index behind a word: ``option_3``, ``chord_10``, ``headline_1``
  digits    plain integers
  hash      opaque hex-like tokens (``k_5cf8035f``, 8+ hex chars)
  semantic  anything else (words, labels)
It is a property of the REQUEST (its option ids), never of the benchmark, so a head feature or rendering that depends
on it is one fixed rendering in the Index's sense.

``e5_recode(request)``: every CHOICE question whose keys are ``option_n`` or ``hash`` gets short codes A, B, ..., Z, AA,
... assigned in the canonical (sorted) order of its original ids; descriptions are kept. Returns the recoded request and,
per question, ``{code: original_id}`` so answers map back exactly. Noul / score questions are never recoded (their
option ids carry meaning: true/false, ordinal levels).
"""

from __future__ import annotations

import copy
import re
from typing import Any

KEY_CLASSES = ("letter", "option_n", "digits", "hash", "semantic")
OPAQUE = ("option_n", "hash")
_LETTER = re.compile(r"[A-Z]{1,2}")
_OPTN = re.compile(r"[A-Za-z]+[ _\-]?\d+")  # option_3, chord_10, headline_1: an index behind a word
_DIGITS = re.compile(r"\d+")
_HASH = re.compile(r"(?i)(?P<pre>[a-z]{1,3}_)?[0-9a-f]{8,}")


def _one(oid: str) -> str:
    s = str(oid)
    if _LETTER.fullmatch(s):
        return "letter"
    if _DIGITS.fullmatch(s):
        return "digits"
    m = _HASH.fullmatch(s)  # before option_n: k_08527554 is a hash that happens to be all digits
    if m and (m.group("pre") or any(c.isdigit() for c in s)):
        return "hash"
    if _OPTN.fullmatch(s):
        return "option_n"
    return "semantic"


def key_class(option_ids) -> str:
    """The class shared by ALL ids of one question; mixed classes -> 'semantic'."""
    cls = {_one(o) for o in option_ids}
    return cls.pop() if len(cls) == 1 else "semantic"


def short_code(i: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA, ... (bijective base 26)."""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def e5_recode(request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    from decide.schema import canonical_options

    req = copy.deepcopy(request)
    back: dict[str, dict[str, str]] = {}
    for qid, q in req["questions"].items():
        if q.get("type") != "choice":
            continue
        opts = canonical_options(q)
        ids = [o for o, _ in opts]
        if key_class(ids) not in OPAQUE:
            continue
        codes = [short_code(i) for i in range(len(ids))]
        crit = {str(k): v for k, v in q["criteria"].items()}  # choice criteria are a dict (decide.schema)
        q["criteria"] = {c: crit[o] for c, o in zip(codes, ids)}
        back[qid] = dict(zip(codes, ids))
    return req, back


def e5_labels(labels: dict[str, str], back: dict[str, dict[str, str]]) -> dict[str, str]:
    """Training labels under the E5 rendering (original id -> code)."""
    out = dict(labels)
    for qid, m in back.items():
        if qid in out:
            inv = {v: k for k, v in m.items()}
            out[qid] = inv[str(out[qid])]
    return out
