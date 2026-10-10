"""RAGTruth statement rephrase (B-PREREG-RAGTRUTH-REPHRASE, state PREREG.md): a READOUT rewrite of the Index's
catalog-59 question, opt-in only.

Why: the Index asks RAGTruth as a declarative statement with a negation ("The response contains content that is not
supported by the context in the prompt."), the ``true`` option restates it and ``false`` is the affirmative. D13 ranks
the 300 RAGTruth TRAIN items INVERTED under it (job-3zmj2: AUROC 0.418 [0.353, 0.481]; Summary track 0.33).

RT1 (the one pre-registered candidate) keeps the question type ``noul``, the option ids and the polarity (``true`` =
hallucinated), so the decider's p(true) is still p(hallucinated) and the scorer sees the same answer shape; only the
instruction and the two option texts change. A question is rewritten iff its type is ``noul`` and its instructions +
criteria equal ``ORIGINAL`` exactly (content match only: never the question id, row id, metadata, _evaluation or
gold). Every other question and the state pass through unchanged.

    from decide.ragtruth_rephrase import rephrased
    decide = rephrased(decider.decide, "rt1")   # same call signature; "original" = identity
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Callable

#: the Index's catalog-59 question block (every DEV RAGTruth row; jobs.m2_first_cell.ragtruth_polarity.QUESTION["q"])
ORIGINAL: dict[str, Any] = {
    "type": "noul",
    "instructions": "The response contains content that is not supported by the context in the prompt.",
    "criteria": {"true": "The response contains content that is not supported by the context in the prompt.",
                 "false": "All content of the response is supported by the context in the prompt."},
}

#: the pre-registered wording (fixed in PREREG before any prediction under it existed); no other wording is allowed
WORDINGS: dict[str, dict[str, Any]] = {
    "rt1": {
        "type": "noul",
        "instructions": ("Hallucination check. The prompt contains the source material (the passage, data or article) "
                         "that the response must be based on. Does the response state any fact, detail or claim that "
                         "is absent from that source material or contradicts it?"),
        "criteria": {"true": ("Yes: the response states at least one fact, detail or claim that is absent from the "
                              "source material in the prompt or contradicts it (hallucinated)."),
                     "false": ("No: every fact, detail and claim in the response appears in the source material in "
                               "the prompt (faithful).")},
    },
}
NAMES = ("original",) + tuple(WORDINGS)


def wording_sha256(name: str) -> str:
    """Identity of a wording (canonical JSON of its question block)."""
    q = ORIGINAL if name == "original" else WORDINGS[name]
    return hashlib.sha256(json.dumps(q, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def matches(q: Any) -> bool:
    """True iff ``q`` is exactly the Index catalog-59 question (type, instructions, criteria; key order ignored)."""
    return (isinstance(q, dict) and q.get("type") == ORIGINAL["type"]
            and q.get("instructions") == ORIGINAL["instructions"] and q.get("criteria") == ORIGINAL["criteria"])


def rephrase_request(request: dict[str, Any], name: str) -> tuple[dict[str, Any], list[str]]:
    """(request with every matching question replaced by wording ``name``, ids of the replaced questions).
    ``name == "original"`` returns the request unchanged (same object) and []."""
    if name == "original":
        return request, []
    if name not in WORDINGS:
        raise ValueError(f"ragtruth rephrase {name!r}: one of {NAMES}")
    qs = request.get("questions") or {}
    hit = [k for k, q in qs.items() if matches(q)]
    if not hit:
        return request, []
    new_qs = {k: (copy.deepcopy(WORDINGS[name]) if k in hit else q) for k, q in qs.items()}
    # any extra keys a question carries (none on catalog 59) are kept
    for k in hit:
        for kk, vv in qs[k].items():
            if kk not in ("type", "instructions", "criteria"):
                new_qs[k][kk] = copy.deepcopy(vv)
    return {**request, "questions": new_qs}, hit


def rephrased(decide: Callable[[dict], dict], name: str) -> Callable[[dict], dict]:
    """``decide`` with the request rewritten first. The response is returned as the decider produced it (RT1 keeps the
    option ids and polarity, so answers stay keyed and typed exactly as for the original question); the rewritten
    question ids are recorded in ``response["ssdm"]["ragtruth_rephrase"]``."""
    if name == "original":
        return decide
    if name not in WORDINGS:
        raise ValueError(f"ragtruth rephrase {name!r}: one of {NAMES}")
    sha = wording_sha256(name)

    def call(req: dict) -> dict:
        new, hit = rephrase_request(req, name)
        resp = decide(new)
        if hit and isinstance(resp, dict) and isinstance(resp.get("ssdm"), dict):
            resp["ssdm"]["ragtruth_rephrase"] = {"wording": name, "wording_sha256": sha, "questions": hit}
        return resp

    return call
