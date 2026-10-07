"""Plain-text linearization of kernel-minted, release-validated speech acts.

This is a small controlled language, not a general text generation path. Every
string copied from a checked object is a JSON string literal in the output.
"""

from __future__ import annotations

import json
from decimal import Decimal

from .kernel import validate_release
from .types import AskContent, CheckResult, Comparison, ConstantValue, ExactSum, Fact, MAX_OUTPUT_BYTES, OutputLimitError, SpeechAct


def _quoted(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid checked value")
    if len(value) > MAX_OUTPUT_BYTES:
        raise OutputLimitError("exceeded output limit")
    return json.dumps(value, ensure_ascii=True)


def _value(value: str | int | bool | Decimal) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("invalid checked value")
        return str(value)
    if isinstance(value, str):
        return _quoted(value)
    raise ValueError("invalid checked value")


def _unit(unit: str | None) -> str:
    return "" if unit is None else " " + _quoted(unit)


def _subject(item: Fact | ConstantValue) -> str:
    if isinstance(item, Fact):
        return f"{_quoted(item.predicate)} for {_quoted(item.subject)}"
    if isinstance(item, ConstantValue):
        return f"constant {_quoted(item.constant_ref)}"
    raise ValueError("invalid checked operand")


def _operand_value(item: Fact | ConstantValue) -> str:
    if isinstance(item, Fact):
        return f"{_value(item.object)}{_unit(item.unit)}"
    if isinstance(item, ConstantValue):
        return f"{_value(item.value)}{_unit(item.unit)}"
    raise ValueError("invalid checked operand")


def _assertion(content: Fact | Comparison | ExactSum) -> str:
    if isinstance(content, Fact):
        return f"At {_quoted(content.interval)}, {_subject(content)} has value {_operand_value(content)}."
    if isinstance(content, Comparison):
        relations = {"greater_than": "greater than", "less_than": "less than", "equal": "equal to"}
        if content.relation not in relations:
            raise ValueError("invalid checked relation")
        return (
            f"At {_quoted(content.interval)}, the value of {_subject(content.left)} "
            f"({_operand_value(content.left)}) is {relations[content.relation]} "
            f"the value of {_subject(content.right)} ({_operand_value(content.right)})."
        )
    if isinstance(content, ExactSum):
        terms = ", ".join(f"{_subject(operand)} ({_operand_value(operand)})"
                          for operand in content.operands)
        return (
            f"At {_quoted(content.interval)}, the exact arithmetic {_quoted(content.quantity_kind)} "
            f"sum of these explicitly listed premises [{terms}] is "
            f"{format(content.value, 'f')}{_unit(content.unit)}."
        )
    raise ValueError("invalid checked assertion")


def _speech_parts(act: SpeechAct):
    if not isinstance(act, SpeechAct):
        raise ValueError("invalid checked speech act")
    if act.kind == "assert":
        yield _assertion(act.content)
        return
    if act.kind == "ask":
        if not isinstance(act.content, AskContent):
            raise ValueError("invalid checked ask")
        refs = ", ".join(_quoted(ref) for ref in act.content.references)
        yield (
            f"Please provide {_quoted(act.content.category)} for obligation "
            f"{_quoted(act.content.obligation_ref)}; references: [{refs}]."
        )
        return
    if act.kind == "conjoin":
        if not isinstance(act.content, tuple) or not act.content:
            raise ValueError("invalid checked conjunction")
        for index, item in enumerate(act.content):
            if index:
                yield " "
            yield from _speech_parts(item)
        return
    if act.kind == "explain":
        if not isinstance(act.content, SpeechAct):
            raise ValueError("invalid checked explanation")
        if (not isinstance(act.derivation, tuple) or not act.derivation
                or any(type(step) is not int or step <= 0 for step in act.derivation)):
            raise ValueError("invalid checked derivation")
        steps = ", ".join(str(step) for step in act.derivation)
        yield f"Checked derivation through program steps [{steps}]: "
        yield from _speech_parts(act.content)
        return
    raise ValueError("invalid checked speech act")


def _render_outputs(outputs: tuple[SpeechAct, ...]) -> str:
    parts = []
    remaining = MAX_OUTPUT_BYTES
    for index, act in enumerate(outputs):
        if index:
            remaining -= 1
            parts.append("\n")
        for part in _speech_parts(act):
            remaining -= len(part.encode("utf-8"))
            if remaining < 0:
                raise OutputLimitError("exceeded output limit")
            parts.append(part)
    return "".join(parts)


def _speech(act: SpeechAct) -> str:
    """Bounded single-act linearization retained for the renderer fixture API."""
    return _render_outputs((act,))


def render_checked(result: CheckResult, host: dict, catalog: dict) -> str:
    """Render only a currently authorized accepted kernel result.

    ``validate_release`` rejects forged, stale, revoked and rejected objects.
    Nothing in this function reads the proposal, model state, or raw evidence.
    The output medium is plain text; downstream HTML/Markdown must escape it.
    """
    checked = validate_release(result, host, catalog)
    if not isinstance(checked, CheckResult) or checked.status != "accepted" or checked.style != "plain":
        raise ValueError("release denied")
    if not checked.outputs:
        raise ValueError("release denied")
    return _render_outputs(checked.outputs)
