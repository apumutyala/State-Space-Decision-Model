"""Typed, trust-labelled decision requests and results (Keel protocol v1).

Authority comes only from the host: a ``Segment`` is ``trusted`` because the host says so (operator policy, operator
notes, authenticated metadata), never because of anything written inside its content. Untrusted segments carry
evidence; they can never carry authority. Every type round-trips through plain JSON (``to_json`` / ``from_json``).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

PROTOCOL_VERSION = "keel-protocol/1"

Trust = Literal["trusted", "untrusted"]
SegmentKind = Literal["record", "tool_output", "user_message", "retrieved", "operator_note", "metadata"]
QuestionType = Literal["choice", "noul", "score"]

TRUSTED_KINDS: frozenset[str] = frozenset({"operator_note", "metadata"})


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Segment:
    """One piece of the input. ``content`` is text or JSON-able data; ``parent_ids`` records derivation (a summary of
    a tool output keeps the tool output as parent and inherits its trust: see ``derive``)."""

    segment_id: str
    content: Any
    trust: Trust
    kind: SegmentKind
    source: str = ""
    parent_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.trust not in ("trusted", "untrusted"):
            raise ValueError(f"segment {self.segment_id}: trust must be trusted or untrusted")
        if self.trust == "untrusted" and self.kind in TRUSTED_KINDS:
            raise ValueError(f"segment {self.segment_id}: kind {self.kind!r} is host-authored and must be trusted")
        if self.trust == "trusted" and self.kind not in TRUSTED_KINDS:
            raise ValueError(f"segment {self.segment_id}: only operator_note / metadata segments may be trusted")


def derive(segment_id: str, content: Any, parents: list[Segment], kind: SegmentKind = "retrieved",
           source: str = "derived") -> Segment:
    """A segment computed from others is untrusted if any parent is untrusted (trust never increases by derivation)."""
    trust: Trust = "trusted" if parents and all(p.trust == "trusted" for p in parents) else "untrusted"
    if trust == "untrusted" and kind in TRUSTED_KINDS:
        kind = "retrieved"
    return Segment(segment_id, content, trust, kind, source, tuple(p.segment_id for p in parents))


@dataclass(frozen=True)
class Question:
    qid: str
    type: QuestionType
    instructions: str
    criteria: Any = None  # choice: {option_id: description}; noul: {"true": ..., "false": ...} or None; score: [level rubrics]

    def as_wire(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            d["criteria"] = self.criteria
        return d


@dataclass(frozen=True)
class OperatorPolicy:
    """The deployment's trusted policy: free-text rules plus, for agent gates, the authorized action set."""

    policy_id: str
    text: str = ""
    version: str = "1"
    authorized_actions: tuple[str, ...] = ()


@dataclass(frozen=True)
class DecisionRequest:
    request_id: str
    policy: OperatorPolicy
    segments: tuple[Segment, ...]
    questions: tuple[Question, ...]

    def __post_init__(self) -> None:
        ids = [s.segment_id for s in self.segments]
        if len(set(ids)) != len(ids):
            raise ValueError(f"request {self.request_id}: duplicate segment ids")
        qids = [q.qid for q in self.questions]
        if len(set(qids)) != len(qids) or not qids:
            raise ValueError(f"request {self.request_id}: question ids must be non-empty and unique")

    def question(self, qid: str) -> Question:
        for q in self.questions:
            if q.qid == qid:
                return q
        raise KeyError(qid)

    def trusted(self) -> list[Segment]:
        return [s for s in self.segments if s.trust == "trusted"]

    def untrusted(self) -> list[Segment]:
        return [s for s in self.segments if s.trust == "untrusted"]

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["protocol"] = PROTOCOL_VERSION
        return d

    @staticmethod
    def from_json(d: dict[str, Any]) -> "DecisionRequest":
        if d.get("protocol", PROTOCOL_VERSION) != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol {d.get('protocol')!r}")
        pol = d["policy"]
        return DecisionRequest(
            request_id=d["request_id"],
            policy=OperatorPolicy(pol["policy_id"], pol.get("text", ""), pol.get("version", "1"),
                                  tuple(pol.get("authorized_actions", ()))),
            segments=tuple(Segment(s["segment_id"], s["content"], s["trust"], s["kind"], s.get("source", ""),
                                   tuple(s.get("parent_ids", ()))) for s in d["segments"]),
            questions=tuple(Question(q["qid"], q["type"], q["instructions"], q.get("criteria"))
                            for q in d["questions"]),
        )

    def fingerprint(self) -> str:
        """sha256 of the canonical JSON: the input identity a receipt records."""
        return sha256_json(self.to_json())


@dataclass
class Answer:
    qid: str
    probs: dict[str, float]  # option id -> probability (sums to 1)
    decision: str | None  # argmax option id, or None when the gate defers this question


@dataclass
class DecisionResult:
    request_id: str
    answers: dict[str, Answer]
    act: bool
    defer_reasons: list[str] = field(default_factory=list)
    signals: dict[str, float] = field(default_factory=dict)
    receipt_id: str = ""


def from_jevbench_item(item: dict[str, Any], policy: OperatorPolicy | None = None,
                       extra: tuple[Segment, ...] = ()) -> DecisionRequest:
    """A JevBench-shaped item (state + one question) as a Keel request: the state is one untrusted record segment."""
    q = item["question"]
    return DecisionRequest(
        request_id=str(item["id"]),
        policy=policy or OperatorPolicy("default"),
        segments=(Segment("record", item["state"], "untrusted", "record", "item"),) + tuple(extra),
        questions=(Question("q", q["type"], q.get("instructions", ""), q.get("criteria")),),
    )
