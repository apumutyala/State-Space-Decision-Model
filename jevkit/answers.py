"""Answer validity and conversion of a SystemOne answer to a distribution over an item's exact label set.

Rules (public jevbench/scoring.py + the v1.6 method):

* The distribution must be over exactly the label set (no missing, no extra keys), finite, each value in [0, 1].
* It must sum to 1 within 1e-3 (strict); inside 2e-2 it is treated as rounded and renormalised; outside it is
  INVALID. An invalid answer counts wrong for Intelligence and is excluded from Calibration. Nothing is repaired.
* noul ``{"noul": P(yes)}`` -> {"yes": p, "no": 1 - p}; the decision is Yes at p >= 0.8, No at p <= 0.2, else an
  abstention (wrong for Intelligence).
* choice: predicted = argmax of the distribution, ties to the lexicographically smallest label (the ``choice`` field
  is not trusted; a disagreement is recorded). A label-only answer (``choice`` with no probabilities) is valid for
  Intelligence and carries no calibration.
* score: probabilities keyed "0".."K-1"; the Intelligence prediction is the expected level sum_k k p_k; a label-only
  answer uses its numeric ``score`` (must lie in [0, K-1]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .items import Item
from .spec import NOUL_NO_MAX, NOUL_YES_MIN, RENORM_SUM_TOL, STRICT_SUM_TOL


@dataclass
class Parsed:
    valid: bool
    probs: dict[str, float] | None = None  # over item.labels, renormalised; None when label-only / invalid
    predicted: str | None = None  # label the answer decides (None: invalid, or a noul abstention)
    expected_level: float | None = None  # score: E[level]
    strict_valid: bool = False
    renormalized: bool = False
    label_only: bool = False
    choice_field_mismatch: bool = False
    error: str | None = None

    @property
    def has_probs(self) -> bool:
        return self.valid and self.probs is not None


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def argmax_label(probs: dict[str, float]) -> str:
    best, best_p = None, -1.0
    for k in sorted(probs):
        if probs[k] > best_p:
            best, best_p = k, probs[k]
    return best  # type: ignore[return-value]


def validate_distribution(probs: Any, labels: list[str]) -> tuple[dict[str, float], bool, bool]:
    """-> (clean distribution, strict_valid, renormalized); raises ValueError when invalid."""
    if not isinstance(probs, dict):
        raise ValueError("probabilities is not an object")
    got, want = set(map(str, probs)), set(labels)
    if got != want or len(probs) != len(labels):
        raise ValueError(f"label set mismatch: missing={sorted(want - got)} extra={sorted(got - want)}")
    clean = {}
    for k, v in probs.items():
        f = _num(v)
        if f is None or f < 0.0 or f > 1.0:
            raise ValueError(f"probability for {k!r} is not a number in [0, 1]")
        clean[str(k)] = f
    total = sum(clean.values())
    if abs(total - 1.0) <= STRICT_SUM_TOL:
        return clean, True, False
    if abs(total - 1.0) <= RENORM_SUM_TOL and total > 0:
        return {k: v / total for k, v in clean.items()}, False, True
    raise ValueError(f"probabilities sum to {total:.6f}, outside the {RENORM_SUM_TOL} band")


def noul_decision(p_yes: float) -> str | None:
    if p_yes >= NOUL_YES_MIN:
        return "yes"
    if p_yes <= NOUL_NO_MAX:
        return "no"
    return None


def parse_answer(item: Item, answer: Any) -> Parsed:
    """Never raises: an unusable answer is ``Parsed(valid=False, error=...)``."""
    if answer is None:
        return Parsed(False, error="no answer")
    if not isinstance(answer, dict):
        return Parsed(False, error="answer is not an object")
    if answer.get("type") not in (None, item.type):
        return Parsed(False, error=f"answer type {answer.get('type')!r} != question type {item.type!r}")
    try:
        if item.type == "noul":
            p = _num(answer.get("noul"))
            if p is None or not 0.0 <= p <= 1.0:
                raise ValueError("noul is not a probability in [0, 1]")
            return Parsed(True, {"yes": p, "no": 1.0 - p}, noul_decision(p), strict_valid=True)
        if item.type == "choice":
            if answer.get("probabilities") is None:
                c = answer.get("choice")
                if c is None or str(c) not in item.labels:
                    raise ValueError("label-only choice outside the label set")
                return Parsed(True, None, str(c), strict_valid=True, label_only=True)
            probs, strict, renorm = validate_distribution(answer["probabilities"], item.labels)
            pred = argmax_label(probs)
            mism = answer.get("choice") is not None and str(answer["choice"]) != pred
            return Parsed(True, probs, pred, strict_valid=strict, renormalized=renorm, choice_field_mismatch=mism)
        # score
        if answer.get("probabilities") is None:
            s = _num(answer.get("score"))
            if s is None or not 0.0 <= s <= item.k - 1:
                raise ValueError("label-only score outside [0, K-1]")
            return Parsed(True, None, str(int(round(s))), expected_level=s, strict_valid=True, label_only=True)
        probs, strict, renorm = validate_distribution(answer["probabilities"], item.labels)
        ev = sum(int(k) * v for k, v in probs.items())
        return Parsed(True, probs, argmax_label(probs), expected_level=ev, strict_valid=strict, renormalized=renorm)
    except ValueError as e:
        return Parsed(False, error=str(e))


def answer_from_probs(item: Item, probs: dict[str, float]) -> dict:
    """Build a SystemOne answer from a distribution over item.labels (used by adapters and test fixtures)."""
    if item.type == "noul":
        return {"type": "noul", "noul": float(probs["yes"])}
    pmax = max(probs.values())
    if item.type == "choice":
        k = len(probs)
        conf = (k * pmax - 1) / (k - 1) if k > 1 else 1.0
        return {"type": "choice", "choice": argmax_label(probs), "probabilities": dict(probs), "confidence": conf}
    levels = [str(i) for i in range(item.k)]
    crit = item.question.get("criteria") or levels
    return {"type": "score", "score": sum(i * probs[lv] for i, lv in enumerate(levels)),
            "probabilities": {lv: float(probs[lv]) for lv in levels},
            "legend": {lv: (c if isinstance(c, str) else str(c)) for lv, c in zip(levels, crit)}, "confidence": pmax}
