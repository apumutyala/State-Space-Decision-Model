"""Canonicalised H2O shim questions (layout ``g1v2-canon``; A-PREREG-JEV3-LM).

The served system reads every option in a CANONICAL order that is a function of the option contents only, under the
shim's own letters (A, B, C, ... in that order), so neither the request's option ids nor its display order ever reach
the model; the answer is mapped back from canonical position to the request's ids OUTSIDE the model. This gives exact
reorder + relabel invariance of the SERVED SYSTEM by canonicalisation (docs/guarantees/REORDER_INVARIANCE.md 2T, C1-C5;
the model never sees display order or ids).

Per type (``canonical_question``):
  choice  content c_i = N(description) (N = the 2T C5 normaliser: Unicode NFC, whitespace runs collapsed to one space,
          stripped); an option without a description is rendered by N(its id) (the id IS its content; listed in
          ``id_as_content``: relabel invariance is not claimed for it). Canonical order = sorted by (c_i, sha256(c_i)).
          Exact duplicates (equal c_i) form a ``tie_set``: they render identically, the served distribution gives every
          member the same probability (``equalize_ties``) and the decision is the set (decide.schema.wire_answer).
  noul    (true, false), each N(description) (defaults filled as decide.schema.NOUL_DEFAULT_CRITERIA).
  score   content = "level v of K: <rubric>" (byte-identical to the evidence op's render_level, claude/ev-score-inv
          @c47c0a46) for wire items (criteria list: v = the list index); an ev_score_perturb request
          (question.options = {handle: content}, question.scale) is read as given. Canonical order = the level value v
          PARSED FROM CONTENT (never the handle, never display order); unparseable content, a K that disagrees with the
          option count, values other than 0..K-1, duplicate handles or a scale that is not the contents in level order
          FAIL CLOSED (ValueError).
The canonical question handed to the shim renderer has option names "001".."K" (zero-padded, so canonical_options'
string sort keeps the canonical order) and descriptions = the canonical contents; the shim labels them A, B, C, ...
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any

CANON_LAYOUT = "g1v2-canon"
#: TRAINING only (lead 2026-10-12, evidence claude/ev-canonical-prior @1ccdc676: the canonical order puts the train gold
#: on A 35% vs 28% chance, a letter prior the public items lack): choice options in a seeded random order (seed = the
#: row's content hash + a salt; one shuffle per row; the cell trains <= 1 epoch), lettered in that order. Inference is
#: always CANON_LAYOUT. Score rows stay in level order and noul in (true, false) in both.
CANON_SHUFFLE_LAYOUT = "g1v2-canon-shuffle"
CANON_LAYOUTS = (CANON_LAYOUT, CANON_SHUFFLE_LAYOUT)
SHUFFLE_SALT = "jev3-lm-train-shuffle/1"
CANON_VERSION = "g1v2-canon/2"
TYPED_SCORE_RENDERING = ("ev-score-level-as-content/1: content = 'level v of K: <rubric>' (v = 0..K-1); scale canonical; "
                         "no handles in scale")
_LEVEL_RE = re.compile(r"level (0|[1-9][0-9]{0,3}) of ([1-9][0-9]{0,3}): ", re.S)
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """2T C5: Unicode NFC, whitespace runs -> one space, stripped."""
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def desc_text(v: Any) -> str:
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def level_content(v: int, k: int, rubric: Any) -> str:
    """== scripts/ev_score_perturb.render_level (the evidence lane's score rendering), byte for byte."""
    return f"level {int(v)} of {int(k)}: {rubric if isinstance(rubric, str) else json.dumps(rubric, ensure_ascii=False)}"


def parse_level(content: Any) -> tuple[int, int]:
    m = _LEVEL_RE.match(content if isinstance(content, str) else "")
    if m is None:
        raise ValueError(f"canon score: no level value in the option content {str(content)[:40]!r}")
    return int(m.group(1)), int(m.group(2))


def content_key(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def sort_key(content: str) -> tuple[str, str]:
    """The canonical order key (evidence lane): NFC + whitespace-collapse + casefold, then the sha256 of the rendered
    content as the tie-break (distinct contents that casefold alike are still ordered by content alone)."""
    return normalize(content).casefold(), content_key(content)


def canonical_question(q: dict[str, Any], shuffle_seed: str | None = None) -> dict[str, Any]:
    """{"question": the canonical question for the shim renderer, "ids": the request's option ids in canonical order,
    "contents": canonical contents (same order), "tie_sets": groups of request ids with equal content (len > 1),
    "id_as_content": ids rendered by their own id}."""
    t = q["type"]
    if t == "noul":
        from .schema import NOUL_DEFAULT_CRITERIA

        crit = dict(NOUL_DEFAULT_CRITERIA)
        crit.update(q.get("criteria") or {})
        if set(crit) != {"true", "false"}:
            raise ValueError("canon noul: criteria keys must be true / false")
        ids = ["true", "false"]
        cont = [normalize(desc_text(crit[i])) for i in ids]
        return {"question": {**{k: v for k, v in q.items() if k not in ("criteria", "options", "scale")},
                             "criteria": dict(zip(ids, cont))},
                "ids": ids, "contents": cont, "tie_sets": [], "id_as_content": []}
    if t == "choice":
        crit = q["criteria"]
        if not isinstance(crit, dict) or not crit:
            raise ValueError("canon choice: criteria must be a non-empty dict")
        items, as_id = [], []
        for oid, d in crit.items():
            c = normalize(desc_text(d)) if d is not None else ""
            if not c:
                c = normalize(str(oid))
                as_id.append(str(oid))
            items.append((c, content_key(c), str(oid)))
        # canonical order: content, then its hash; exact duplicates keep their (irrelevant) relative order -- they
        # render identically and get equal served probability, so which duplicate sits where never changes an output
        items.sort(key=lambda x: sort_key(x[0]))
        if shuffle_seed is not None:  # TRAINING rows only (CANON_SHUFFLE_LAYOUT): a seeded random presented order
            import random

            rng = random.Random(int(hashlib.sha256(shuffle_seed.encode("utf-8")).hexdigest()[:16], 16))
            rng.shuffle(items)
        groups: dict[str, list[str]] = {}
        for c, _, oid in items:
            groups.setdefault(c, []).append(oid)
        ids, cont = [oid for _, _, oid in items], [c for c, _, _ in items]
        names = [f"{k + 1:03d}" for k in range(len(ids))]
        return {"question": {**{k: v for k, v in q.items() if k not in ("criteria", "options", "scale")},
                             "criteria": dict(zip(names, cont))},
                "ids": ids, "contents": cont, "tie_sets": sorted(sorted(g) for g in groups.values() if len(g) > 1),
                "id_as_content": sorted(as_id)}
    if t == "score":
        if isinstance(q.get("options"), dict):  # an ev_score_perturb request: {handle: content} in display order
            disp = [(str(h), str(c)) for h, c in q["options"].items()]
        else:
            crit = q["criteria"]
            if not isinstance(crit, list) or len(crit) < 2:
                raise ValueError("canon score: criteria must be a list of >= 2 levels")
            disp = [(str(i), level_content(i, len(crit), r)) for i, r in enumerate(crit)]
        if len({h for h, _ in disp}) != len(disp):
            raise ValueError("canon score: duplicate handles")
        vk = [parse_level(c) for _, c in disp]
        k = len(disp)
        if any(kk != k for _, kk in vk) or sorted(v for v, _ in vk) != list(range(k)):
            raise ValueError(f"canon score: level values {sorted(vk)} are not exactly 0..{k - 1} of {k}")
        axis = [it for _, it in sorted(zip([v for v, _ in vk], disp), key=lambda x: x[0])]
        if q.get("scale") is not None and list(q["scale"]) != [c for _, c in axis]:
            raise ValueError("canon score: question.scale is not the option contents in level order")
        cont = [normalize(c) for _, c in axis]
        return {"question": {**{kk: v for kk, v in q.items() if kk not in ("criteria", "options", "scale")},
                             "criteria": cont},
                "ids": [h for h, _ in axis], "contents": cont, "tie_sets": [], "id_as_content": []}
    raise ValueError(f"canon: unknown question type {t!r}")


def shuffle_seed_of(request: dict[str, Any], qid: str) -> str:
    """The training shuffle's seed: the salt + the row's content (state + the question), so one row always gets the
    same order and different rows get independent orders; never the request's ids or display order alone."""
    q = request["questions"][qid]
    body = {"state": request.get("state"), "instructions": q.get("instructions"), "type": q.get("type"),
            "criteria": sorted([normalize(desc_text(v)) for v in (q.get("criteria") or {}).values()])
            if isinstance(q.get("criteria"), dict) else q.get("criteria")}
    return SHUFFLE_SALT + ":" + hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                                           default=str).encode("utf-8")).hexdigest()


def canonical_request(request: dict[str, Any], qid: str, shuffle: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """(the request with question ``qid`` canonicalised -- or, ``shuffle`` (training), in its seeded random order --
    and the canonicalisation record)."""
    rec = canonical_question(request["questions"][qid], shuffle_seed_of(request, qid) if shuffle else None)
    req2 = {**request, "questions": {**request["questions"], qid: rec["question"]}}
    return req2, rec


def equalize_ties(logits: dict[str, float], tie_sets: list[list[str]]) -> dict[str, float]:
    """Every member of a tie set gets log(mean exp) of the members' log-probabilities (total mass kept), computed in a
    fixed (sorted-value) order, so the served distribution is the same whichever duplicate sat where."""
    import math

    out = dict(logits)
    for g in tie_sets:
        vals = sorted(float(logits[o]) for o in g)
        m = vals[-1]
        v = m + math.log(sum(math.exp(x - m) for x in vals) / len(vals))
        for o in g:
            out[o] = v
    return out
