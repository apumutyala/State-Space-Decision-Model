"""Immutable checked meanings. These objects carry no model-authored proof authority."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, tuple):
        return [_json_value(part) for part in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


@dataclass(frozen=True, slots=True)
class Fact:
    subject: str
    predicate: str
    object: str | int | bool | Decimal
    unit: str | None
    interval: str
    source_ref: str
    record_ref: str
    record_version: int
    field: str
    quantity_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {name: _json_value(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ConstantValue:
    value: str | int | bool | Decimal
    value_type: str
    unit: str | None
    interval: str
    quantity_kind: str
    constant_ref: str

    def to_dict(self) -> dict[str, Any]:
        return {name: _json_value(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class AssumptionValue:
    """A host-declared alternate value, never an authoritative Fact."""

    assumption_ref: str
    premise: Fact
    value: str | int | bool | Decimal

    @property
    def interval(self) -> str:
        return self.premise.interval

    @property
    def unit(self) -> str | None:
        return self.premise.unit

    @property
    def quantity_kind(self) -> str:
        return self.premise.quantity_kind

    def to_dict(self) -> dict[str, Any]:
        return {"epistemic_class": "hypothesis", "assumption_ref": self.assumption_ref,
                "premise": self.premise.to_dict(), "value": _json_value(self.value)}


@dataclass(frozen=True, slots=True)
class Comparison:
    left: Fact | ConstantValue | AssumptionValue
    right: Fact | ConstantValue | AssumptionValue
    relation: str
    interval: str

    def to_dict(self) -> dict[str, Any]:
        return {"left": self.left.to_dict(), "right": self.right.to_dict(), "relation": self.relation, "interval": self.interval}


@dataclass(frozen=True, slots=True)
class ConditionalContent:
    assumption: AssumptionValue
    conclusion: Comparison

    def to_dict(self) -> dict[str, Any]:
        return {"assumption": self.assumption.to_dict(), "conclusion": self.conclusion.to_dict()}


@dataclass(frozen=True, slots=True)
class AskContent:
    category: str
    obligation_ref: str
    references: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"category": self.category, "obligation_ref": self.obligation_ref, "references": list(self.references)}


@dataclass(frozen=True, slots=True)
class SpeechAct:
    kind: str
    content: Fact | Comparison | ConditionalContent | AskContent | tuple[SpeechAct, ...] | SpeechAct
    dependencies: tuple[str, ...]
    derivation: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "content": _json_value(self.content), "dependencies": list(self.dependencies), "derivation": list(self.derivation)}


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class CheckResult:
    status: str
    reason: str | None
    outputs: tuple[SpeechAct, ...] = ()
    dependencies: tuple[str, ...] = ()
    context_ref: str | None = None
    policy_ref: str | None = None
    registry_ref: str | None = None
    audience: str | None = None
    style: str | None = None
    _program_json: str | None = field(default=None, repr=False, compare=False)
    _binding_digest: str | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "outputs": [output.to_dict() for output in self.outputs],
            "dependencies": list(self.dependencies),
            "context_ref": self.context_ref,
            "policy_ref": self.policy_ref,
            "registry_ref": self.registry_ref,
            "audience": self.audience,
            "style": self.style,
        }
