"""Local, label-free act/defer decisions from uncertainty and manipulation signals."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from keel.protocol import Answer, DecisionRequest, DecisionResult, decision_of, text_of

GATE_VERSION = "keel-gate/1"


def _unit_interval(value: float, name: str) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")


def _freeze_probabilities(values: Mapping[str, Mapping[str, float]]) -> Mapping[str, Mapping[str, float]]:
    if not values:
        raise ValueError("judge outputs need at least one question")
    frozen = {}
    for qid, probs in values.items():
        if not isinstance(qid, str) or not qid or not probs:
            raise ValueError("question ids and distributions must be nonempty")
        for option, probability in probs.items():
            if not isinstance(option, str) or not option:
                raise ValueError("option ids must be nonempty strings")
            _unit_interval(probability, f"{qid}/{option} probability")
        if not math.isclose(math.fsum(probs.values()), 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError(f"{qid}: probabilities must sum to one")
        frozen[qid] = MappingProxyType(dict(probs))
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class JudgeOutputs:
    """Already-computed distributions, keyed by question id and then original option id.

    ``trust`` supplies the decisions; ``base`` is only a detector input. Construct both
    from the same pinned judge and request under the two protocol rendering modes.
    This class copies its inputs; it neither scores a model nor accepts gold labels.
    """

    trust: Mapping[str, Mapping[str, float]]
    base: Mapping[str, Mapping[str, float]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "trust", _freeze_probabilities(self.trust))
        if self.base is not None:
            object.__setattr__(self, "base", _freeze_probabilities(self.base))
            if self.trust.keys() != self.base.keys() or any(
                self.trust[qid].keys() != self.base[qid].keys() for qid in self.trust
            ):
                raise ValueError("trust/base question and option ids must agree")


class ManipulationSignal(Protocol):
    """A frozen, versioned detector. Custom implementations must not access labels.

    Include all tuned configuration in the version and freeze it on the fit split.
    Return a finite score in [0, 1]; larger means more suspected manipulation.
    """

    name: str
    version: str

    def __call__(self, request: DecisionRequest, outputs: JudgeOutputs) -> float: ...


@dataclass(frozen=True)
class RenderDivergence:
    """Maximum per-question Jensen-Shannon divergence in bits (range [0, 1]).

    Large divergence is sensitivity to the rendering, not proof of an attack.
    Both distributions are supplied by the caller; there is no remote computation.
    """

    name: str = "render_divergence"
    version: str = "render-divergence/1"

    def __call__(self, request: DecisionRequest, outputs: JudgeOutputs) -> float:
        if outputs.base is None:
            raise ValueError("RenderDivergence requires both trust and base distributions")
        divergences = []
        for question in request.questions:
            p, q = outputs.trust[question.qid], outputs.base[question.qid]
            terms = []
            for option in p:
                midpoint = (p[option] + q[option]) / 2.0
                if p[option] > 0:
                    terms.append(0.5 * p[option] * math.log2(p[option] / midpoint))
                if q[option] > 0:
                    terms.append(0.5 * q[option] * math.log2(q[option] / midpoint))
            divergences.append(min(1.0, max(0.0, math.fsum(terms))))
        return max(divergences)


# Each rule contributes at most once across all untrusted segments. These literal
# patterns, flags, weights and aggregation define instruction-lexicon/1.
LEXICON_PATTERNS: tuple[tuple[str, str, float], ...] = (
    ("override", r"\b(?:ignore|disregard|override|forget)\s+(?:(?:all|the|any|your|these)\s+)*"
     r"(?:(?:previous|prior|above|earlier|system|developer|safety)\s+)*"
     r"(?:instructions?|rules?|polic(?:y|ies)|messages?|constraints?)\b", 0.65),
    ("authority", r"\b(?:system|developer|administrator|admin|supervisor|operator)\s*"
     r"(?:message|instruction|notice|directive|says|orders?|commands?)\b", 0.50),
    ("role_marker", r"(?:<\|(?:im_start|start_header_id)\|>\s*(?:system|developer)|"
     r"\[\s*(?:system|developer)\s*\]|</?\s*(?:system|developer)\s*>)", 0.65),
    ("direct_answer", r"\b(?:you\s+must\s+(?:answer|output|return|select|choose)|"
     r"(?:answer|output|return|select|choose)\s+(?:only|exactly|instead))\b", 0.40),
    ("priority", r"\b(?:(?:highest|higher|top)[ -]priority|"
     r"(?:supersedes?|overrides?)\s+(?:all|any|previous|prior|system|developer))\b", 0.50),
)
_COMPILED_LEXICON = tuple((name, re.compile(pattern, re.IGNORECASE), weight)
                          for name, pattern, weight in LEXICON_PATTERNS)


@dataclass(frozen=True)
class InstructionLexicon:
    """Transparent weak baseline, not a semantic instruction/attack detector.

    Scans only host-labelled untrusted content (strings or protocol JSON text).
    Score = min(1, sum of weights of distinct matched rules). Repetition cannot
    inflate one rule; quotes can trigger false positives; obfuscation can evade it.
    """

    name: str = "instruction_lexicon"
    version: str = "instruction-lexicon/1"

    def matches(self, request: DecisionRequest) -> tuple[str, ...]:
        texts = [text_of(segment.content) for segment in request.segments if segment.trust == "untrusted"]
        return tuple(name for name, pattern, _ in _COMPILED_LEXICON
                     if any(pattern.search(text) for text in texts))

    def __call__(self, request: DecisionRequest, outputs: JudgeOutputs) -> float:
        matched = set(self.matches(request))
        return min(1.0, math.fsum(weight for name, _, weight in LEXICON_PATTERNS if name in matched))


@dataclass(frozen=True)
class GateThresholds:
    uncertainty: float
    manipulation: float

    def __post_init__(self) -> None:
        _unit_interval(self.uncertainty, "uncertainty threshold")
        _unit_interval(self.manipulation, "manipulation threshold")

    def accepts(self, uncertainty: float, manipulation: float) -> bool:
        """Strict inequalities: equality defers, including a zero threshold."""
        return uncertainty < self.uncertainty and manipulation < self.manipulation


@dataclass(frozen=True)
class GateAssessment:
    uncertainty: float
    one_minus_max: float
    margin: float
    manipulation: float
    signal_scores: Mapping[str, float]
    predictions: Mapping[str, str]


def _check_question_options(request: DecisionRequest, outputs: JudgeOutputs) -> None:
    if set(outputs.trust) != {question.qid for question in request.questions}:
        raise ValueError("judge outputs must cover exactly the request's questions")
    for question in request.questions:
        if question.type == "noul":
            options = {"true", "false"}
        elif question.type == "score":
            options = {str(i) for i in range(len(question.criteria))}
        elif question.type == "choice":
            options = {str(option) for option in question.criteria}
        else:
            raise ValueError(f"unsupported question type: {question.type}")
        if set(outputs.trust[question.qid]) != options:
            raise ValueError(f"{question.qid}: output option ids differ from the question")


@dataclass(frozen=True)
class ActDeferGate:
    """Act on the entire request only when BOTH worst-case scores are below threshold.

    For multiple questions, uncertainty is the maximum 1-p_max (or maximum
    1-margin); manipulation is the maximum across configured signals. Every
    answer is cleared on deferral. This is a decision, not host authorization.
    """

    thresholds: GateThresholds
    signals: tuple[ManipulationSignal, ...] = (InstructionLexicon(),)
    uncertainty_metric: str = "one_minus_max"

    def __post_init__(self) -> None:
        object.__setattr__(self, "signals", tuple(self.signals))
        if self.uncertainty_metric not in ("one_minus_max", "one_minus_margin"):
            raise ValueError("uncertainty_metric must be one_minus_max or one_minus_margin")
        if not self.signals:
            raise ValueError("a manipulation-aware gate requires at least one signal")
        names = [signal.name for signal in self.signals]
        if len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError("signal names must be nonempty and unique")
        if any(not signal.version or not callable(signal) for signal in self.signals):
            raise ValueError("signals must be versioned callables")

    def specification(self) -> dict:
        """The threshold-independent family; pipeline identity is recorded by certification."""
        return {"version": GATE_VERSION, "uncertainty_metric": self.uncertainty_metric,
                "signals": [{"name": signal.name, "version": signal.version} for signal in self.signals],
                "aggregation": "max across questions/signals", "comparison": "strictly below"}

    def assess(self, request: DecisionRequest, outputs: JudgeOutputs) -> GateAssessment:
        _check_question_options(request, outputs)
        uncertainty, margin = 0.0, 1.0
        predictions = {}
        for question in request.questions:
            probabilities = outputs.trust[question.qid]
            top = sorted(probabilities.values(), reverse=True)
            uncertainty = max(uncertainty, 1.0 - top[0])
            margin = min(margin, top[0] - (top[1] if len(top) > 1 else 0.0))
            predictions[question.qid] = decision_of(dict(probabilities), question.type)
        scores = {}
        for signal in self.signals:
            score = signal(request, outputs)
            _unit_interval(score, f"{signal.name} score")
            scores[signal.name] = score
        return GateAssessment(uncertainty if self.uncertainty_metric == "one_minus_max" else 1.0 - margin,
                              uncertainty, margin, max(scores.values()), MappingProxyType(scores),
                              MappingProxyType(predictions))

    def evaluate(self, request: DecisionRequest, outputs: JudgeOutputs) -> DecisionResult:
        assessment = self.assess(request, outputs)
        reasons = []
        if assessment.uncertainty >= self.thresholds.uncertainty:
            reasons.append("uncertainty_at_or_above_threshold")
        if assessment.manipulation >= self.thresholds.manipulation:
            reasons.append("manipulation_at_or_above_threshold")
        act = not reasons
        answers = {question.qid: Answer(question.qid, dict(outputs.trust[question.qid]),
                                       assessment.predictions[question.qid] if act else None)
                   for question in request.questions}
        return DecisionResult(request.request_id, answers, act, reasons,
                              {"uncertainty": assessment.uncertainty, "one_minus_max": assessment.one_minus_max,
                               "margin": assessment.margin, "manipulation": assessment.manipulation,
                               **{f"manipulation/{name}": value
                                  for name, value in assessment.signal_scores.items()}})
