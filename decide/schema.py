"""Request validation, canonical option order and SystemOne wire conversion.

Wire semantics follow Cloudflare's ``joint_schema_model.py`` (clef-flash, sha256
0e304cf7...a3, read as text only, never executed):

* ``noul``  -> options ``true``/``false`` (default descriptions, overridable via
  ``criteria``); answer ``{"type": "noul", "noul": p_true}``.
* ``choice`` -> options are the ``criteria`` keys; answer has ``choice`` (argmax),
  ``confidence`` (= probability of the selected option) and ``probabilities``.
* ``score`` -> options are list indices ``"0".."k-1"`` of ``criteria``; answer has
  ``score`` (= raw expected ordinal index, sum_i i * p_i), ``confidence`` (= max p),
  ``legend`` and ``probabilities``.

Deliberate differences from Clef (each required by the canonical interface):

* Clef silently truncates the state to fit ``max_length``. We never truncate:
  an over-long input returns ``status = exhausted``.
* Clef breaks argmax ties by request insertion order. We break ties by sorted
  option id so that the answer is invariant to option order in the request.
* Validation is stricter than Clef: a 1-level ``score`` is ``invalid_request``
  (Clef accepts it and returns score 0); noul ``criteria`` keys other than
  true/false are ``invalid_request`` (Clef ignores them); option ids and
  question ids containing control characters or special-token markers
  (``<|``, ``|>``) are ``invalid_option_id`` / ``invalid_request``.
* ``usage.input_tokens`` is the total number of tokens forwarded (all option
  rows of all questions); Clef reports the length of its single joint sequence.
  The longest single model input is ``diagnostics.input_counts.max_call_input_tokens``.
* Non-ok responses carry a top-level ``error`` object (``{status, code,
  message}``) besides the ``ssdm`` envelope, with ``answers == {}``. Clef raises
  ``ValueError`` instead. Transports must map ``status != ok`` to an HTTP error;
  consumers must branch on ``ssdm.status`` (or ``error``), never on ``answers``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

QUESTION_TYPES = ("noul", "choice", "score")
NOUL_DEFAULT_CRITERIA = {
    "true": "The proposition is true or the answer is yes.",
    "false": "The proposition is false or the answer is no.",
}
#: SystemOne option cardinality limit per question.
MAX_OPTIONS_WIRE = 255
WIRE_ROUND = 4
MAX_ID_CHARS = 128
_FORBIDDEN_ID_MARKERS = ("<|", "|>")


def _bad_id(value: str) -> bool:
    return (
        not value
        or len(value) > MAX_ID_CHARS
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in value)
        or any(m in value for m in _FORBIDDEN_ID_MARKERS)
    )


class RequestRejected(Exception):
    """Raised during validation; carries the status and a bounded reason code."""

    def __init__(self, status: str, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def render(value: Any) -> str:
    """Same rendering as Clef: strings verbatim, everything else canonical JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_options(question: dict[str, Any]) -> list[tuple[str, Any]]:
    """Options in canonical order: noul (true, false); choice sorted by id; score by level.

    Canonical order makes every order-dependent readout (causal prompts, batching)
    invariant to the order in which the caller listed choice options.
    """
    qtype = question["type"]
    if qtype == "noul":
        criteria = dict(NOUL_DEFAULT_CRITERIA)
        criteria.update(question.get("criteria") or {})
        return [(key, criteria[key]) for key in ("true", "false")]
    if qtype == "choice":
        return sorted((str(k), v) for k, v in question["criteria"].items())
    return [(str(i), v) for i, v in enumerate(question["criteria"])]


def canonical_question_ids(questions: dict[str, Any]) -> list[str]:
    return sorted(str(q) for q in questions)


def validate_legacy(request: Any) -> None:
    """Validate the SystemOne-compatible part. Raises RequestRejected."""
    if not isinstance(request, dict):
        raise RequestRejected("error", "invalid_request", "request must be a JSON object")
    if not isinstance(request.get("model"), str):
        raise RequestRejected("error", "invalid_request", "model (string) is required")
    if "state" not in request:
        raise RequestRejected("error", "invalid_request", "state is required")
    if request.get("images") or request.get("videos"):
        raise RequestRejected(
            "unsupported", "media_unsupported", "images/videos are not supported by text-only backends"
        )
    questions = request.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise RequestRejected("error", "invalid_request", "at least one question is required")
    for qid, q in questions.items():
        if _bad_id(str(qid)):
            raise RequestRejected("error", "invalid_request", f"question id {qid!r} has control characters or markers")
        if not isinstance(q, dict):
            raise RequestRejected("error", "invalid_request", f"{qid}: question must be an object")
        qtype = q.get("type")
        if qtype not in QUESTION_TYPES:
            raise RequestRejected(
                "error", "unknown_question_type", f"{qid}: type must be noul, choice, or score"
            )
        crit = q.get("criteria")
        if qtype == "choice":
            if not isinstance(crit, dict) or not crit:
                raise RequestRejected("error", "empty_criteria", f"{qid}: criteria must be a non-empty object")
            for oid in crit:
                if _bad_id(str(oid)):
                    raise RequestRejected(
                        "error", "invalid_option_id",
                        f"{qid}: option id {oid!r} is empty, too long, or has control characters/special-token markers",
                    )
            n = len(crit)
        elif qtype == "score":
            if not isinstance(crit, list) or not crit:
                raise RequestRejected("error", "empty_criteria", f"{qid}: criteria must be a non-empty list")
            if len(crit) < 2:
                raise RequestRejected("error", "invalid_request", f"{qid}: score needs at least 2 levels")
            n = len(crit)
        else:
            if crit is not None and (
                not isinstance(crit, dict) or not set(crit) <= {"true", "false"}
            ):
                raise RequestRejected(
                    "error", "invalid_request", f"{qid}: noul criteria may only describe true/false"
                )
            n = 2
        if n > MAX_OPTIONS_WIRE:
            raise RequestRejected(
                "unsupported",
                "too_many_options",
                f"{qid}: {n} options exceeds the cardinality limit {MAX_OPTIONS_WIRE}",
            )


def wire_answer(question: dict[str, Any], probs: dict[str, float], tie: dict[str, Any] | None = None) -> dict[str, Any]:
    """SystemOne answer from full-precision per-option probabilities.

    ``tie`` (sym-fork rows, ``heads.symmetric.sym_tie_info``): content-based decisions. Exact-probability ties between
    DIFFERENT options are broken by content hash (never position or id); exact duplicates (equal content) form the
    ``tie_set`` -- the decision, invariant as a set (a single id among duplicates cannot be relabel-invariant; "choice"
    then names the set's smallest id and ``tie_set`` says it is one of several equivalent ids). Probabilities are
    written with ``tie["digits"]`` digits so wire rounding cannot create ties a reader breaks by id. None: unchanged."""
    qtype = question["type"]
    r = WIRE_ROUND if tie is None else int(tie.get("digits", WIRE_ROUND))
    if qtype == "noul":
        return {"type": "noul", "noul": round(probs["true"], r)}
    if qtype == "choice":
        options = [str(o) for o in question["criteria"]]  # caller's order, as Clef
        if tie is None:
            best = max(sorted(options), key=lambda o: probs[o])  # tie -> smallest id
            extra: dict[str, Any] = {}
        else:
            pmax = max(probs[o] for o in options)
            top = [o for o in options if probs[o] == pmax]
            key = tie["content_key"]
            k0 = min(key[o] for o in top)  # content-hash order among different near-tied options
            tie_set = sorted(o for o in top if key[o] == k0)  # exact duplicates of the winning content
            best, extra = tie_set[0], {"tie_set": tie_set}
        return {
            "type": "choice",
            "choice": best,
            "confidence": round(probs[best], r),
            "probabilities": {o: round(probs[o], r) for o in options},
            **extra,
        }
    levels = [str(i) for i in range(len(question["criteria"]))]
    return {
        "type": "score",
        "score": round(sum(i * probs[lv] for i, lv in enumerate(levels)), r),
        "confidence": round(max(probs[lv] for lv in levels), r),
        "legend": dict(zip(levels, question["criteria"])),
        "probabilities": {lv: round(probs[lv], r) for lv in levels},
    }


def request_fingerprint(request: dict[str, Any]) -> str:
    """Content identity of the decision-relevant part of a request (order-insensitive)."""
    questions = {}
    for qid, q in request["questions"].items():
        questions[str(qid)] = {
            "type": q["type"],
            "instructions": q.get("instructions"),
            "options": canonical_options(q),
        }
    payload = {"state": request["state"], "questions": questions}
    env = request.get("ssdm")
    if isinstance(env, dict) and env.get("provenance") is not None:
        payload["provenance"] = env["provenance"]
    return sha256_text(canonical_json(payload))
