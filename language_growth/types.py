"""Immutable checked meanings. These objects carry no model-authored proof authority."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, is_dataclass
from decimal import Decimal
from typing import Any


MAX_OUTPUT_BYTES = 262144


class OutputLimitError(ValueError):
    """A complete released representation exceeds the profile's byte budget."""


class _OutputBudget:
    def __init__(self) -> None:
        self.remaining = MAX_OUTPUT_BYTES

    def consume(self, count: int) -> None:
        self.remaining -= count
        if self.remaining < 0:
            raise OutputLimitError("exceeded output limit")


def _json_value(value: Any, budget: _OutputBudget | None = None) -> Any:
    """Build a bounded JSON tree, counting default ensure_ascii JSON bytes.

    Count before expanding repeated speech-act subgraphs so a small DAG cannot
    materialize an exponentially large result. Only ExactSum results acquire
    fixed-point formatting; source Facts/constants retain legacy formatting.
    """
    if budget is None:
        budget = _OutputBudget()
    if isinstance(value, Decimal):
        value = str(value)
    if isinstance(value, ExactSum):
        value = {name: getattr(value, name) for name in value.__dataclass_fields__}
        value["value"] = format(value["value"], "f")
    if is_dataclass(value):
        value = {name: getattr(value, name) for name in value.__dataclass_fields__}
    if isinstance(value, dict):
        budget.consume(2 + max(0, len(value) - 1) * 2)
        converted = {}
        for key, part in value.items():
            _json_value(key, budget)
            budget.consume(2)
            converted[key] = _json_value(part, budget)
        return converted
    if isinstance(value, (tuple, list)):
        budget.consume(2 + max(0, len(value) - 1) * 2)
        return [_json_value(part, budget) for part in value]
    if isinstance(value, str) and len(value) > MAX_OUTPUT_BYTES:
        raise OutputLimitError("exceeded output limit")
    budget.consume(len(json.dumps(value, ensure_ascii=True).encode("ascii")))
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
        return _json_value(self)


@dataclass(frozen=True, slots=True)
class ConstantValue:
    value: str | int | bool | Decimal
    value_type: str
    unit: str | None
    interval: str
    quantity_kind: str
    constant_ref: str

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclass(frozen=True, slots=True)
class Comparison:
    left: Fact | ConstantValue
    right: Fact | ConstantValue
    relation: str
    interval: str

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclass(frozen=True, slots=True)
class ExactSum:
    operands: tuple[Fact, ...]
    rule_ref: str
    value: Decimal
    unit: str | None
    interval: str
    quantity_kind: str

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclass(frozen=True, slots=True)
class AskContent:
    category: str
    obligation_ref: str
    references: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclass(frozen=True, slots=True)
class SpeechAct:
    kind: str
    content: Fact | Comparison | ExactSum | AskContent | tuple[SpeechAct, ...] | SpeechAct
    dependencies: tuple[str, ...]
    derivation: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


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
        return _json_value({
            "status": self.status,
            "reason": self.reason,
            "outputs": self.outputs,
            "dependencies": list(self.dependencies),
            "context_ref": self.context_ref,
            "policy_ref": self.policy_ref,
            "registry_ref": self.registry_ref,
            "audience": self.audience,
            "style": self.style,
        })
