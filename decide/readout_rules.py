"""Opt-in, parameter-free readout rules ("r1") applied to served probabilities after scoring.

Both rules change only the ``choice`` of yes/no choice questions (and its ``confidence``); every probability stage
(raw, served, policy-conditioned) is left as the model produced it. Both read ONLY the request (``state`` +
``questions``, what the kit hands an engine: ``engine(state, questions)``) and the model's own served probabilities.
No gold, no gold count, no row metadata (``id`` / ``metadata.category`` / ``_evaluation``), nothing fitted.

``bfcl_exactly_one``
    Trigger: ``state["task"] == "BFCL tool-name selection"`` and every question is a yes/no choice. The kit builds this
    benchmark only from BFCL's single-call categories (``decision_index/suite/build/adapters_mechanical.py``
    ``build_bfcl``: ``categories = ["simple", "live_simple", "live_multiple"]``; ``docs/suite.md`` row 1). In BFCL's
    own category definitions simple / multiple / live_simple / live_multiple each expect exactly ONE function call
    (parallel / parallel_multiple are the multi-call categories, and the kit does not include them). Rule: yes for
    the single tool with the highest p(yes), no for every other tool (ties: smallest question id).

``sata_top_m``
    Trigger: ``state`` has exactly the keys ``paragraph`` / ``question`` / ``options`` (the kit's SATA-Bench
    normalisation, ``decision_index/suite/build/normalize_text.py`` ``sata``), the questions are exactly
    ``option_0 .. option_{n-1}`` with n = len(state["options"]), each a yes/no choice. Select-all-that-apply has at
    least one correct option. Rule: m = max(1, round(sum_i p_i(yes))) from the model's served probabilities; yes for
    the m options with the highest p(yes) (ties: smallest question id), no for the rest. ``round`` is Python's
    (half to even), matching ``analysis/readout_levers_d13.py``.

``READOUT_RULES_CAP`` is the input cap the rules readout serves under for every backend (32,768: the cap Clef's full-DEV
reference was measured at), replacing the adapter default 16,384 (Clef's ``encode_record`` default; no Index/kit
protocol requires it).
"""

from __future__ import annotations

from typing import Any, Mapping

RULES_ID = "ssdm.readout_rules.r1"
READOUT_RULES_CAP = 32_768
BFCL_TASK = "BFCL tool-name selection"
SATA_STATE_KEYS = frozenset({"paragraph", "question", "options"})
RULES = ("bfcl_exactly_one", "sata_top_m")


def _is_yes_no(q: Any) -> bool:
    return (isinstance(q, Mapping) and q.get("type") == "choice" and isinstance(q.get("criteria"), Mapping)
            and set(map(str, q["criteria"])) == {"yes", "no"})


def detect(request: Mapping[str, Any]) -> str | None:
    """Which rule applies to this request (from the request alone), or None."""
    state, questions = request.get("state"), request.get("questions")
    if not isinstance(state, Mapping) or not isinstance(questions, Mapping) or not questions:
        return None
    if not all(_is_yes_no(q) for q in questions.values()):
        return None
    if state.get("task") == BFCL_TASK:
        return "bfcl_exactly_one"
    opts = state.get("options")
    if (set(state) == SATA_STATE_KEYS and isinstance(opts, list) and len(opts) == len(questions)
            and set(questions) == {f"option_{j}" for j in range(len(opts))}):
        return "sata_top_m"
    return None


def selected_yes(rule: str, p_yes: Mapping[str, float]) -> set[str]:
    """Question ids answered yes under ``rule`` given the model's p(yes) per question (nothing else)."""
    ranked = sorted(p_yes, key=lambda k: (-float(p_yes[k]), k))
    if rule == "bfcl_exactly_one":
        m = 1
    elif rule == "sata_top_m":
        m = max(1, round(sum(float(v) for v in p_yes.values())))
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return set(ranked[:m])


def apply_rules(request: Mapping[str, Any], answers: dict[str, Any],
                probabilities: Mapping[str, Mapping[str, float]] | None = None) -> dict[str, Any]:
    """Rewrite ``answers`` (wire answers, in place) under the rule that applies; returns a diagnostics record.

    ``probabilities``: full-precision served probabilities per question (the adapter passes them); default = the wire
    answers' own (rounded) ``probabilities`` (offline re-scoring of saved results)."""
    rule = detect(request)
    if rule is None:
        return {"id": RULES_ID, "rule": None, "changed": []}
    probs = probabilities if probabilities is not None else {k: a["probabilities"] for k, a in answers.items()}
    p_yes = {k: float(probs[k]["yes"]) for k in request["questions"]}
    yes = selected_yes(rule, p_yes)
    changed = []
    for k in sorted(request["questions"]):
        new = "yes" if k in yes else "no"
        a = answers[k]
        if a["choice"] != new:
            changed.append(k)
            a["choice"] = new
            if "confidence" in a:
                a["confidence"] = a["probabilities"][new]
    return {"id": RULES_ID, "rule": rule, "changed": changed}
