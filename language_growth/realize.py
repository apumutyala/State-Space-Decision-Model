"""Plain-text linearization of kernel-minted, release-validated speech acts.

This is a small controlled language, not a general text generation path. Every
string copied from a checked object is a JSON string literal in the output.
"""

from __future__ import annotations

import json
from decimal import Decimal

from .kernel import validate_release
from .types import AskContent, CheckResult, Comparison, ConstantValue, Fact, SpeechAct


PLAIN_PRODUCTION = "plain"
L3_FACT_PAIR_COORDINATION = "l3_fact_pair_coordination_v1"
LICENSED_PRODUCTIONS = frozenset({PLAIN_PRODUCTION, L3_FACT_PAIR_COORDINATION})


def is_licensed_production(production: object) -> bool:
    """Whole-production IDs only; caller text never becomes a connective."""
    return type(production) is str and production in LICENSED_PRODUCTIONS


def _quoted(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("invalid checked value")
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


def _assertion(content: Fact | Comparison) -> str:
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
    raise ValueError("invalid checked assertion")


def _speech(act: SpeechAct) -> str:
    if not isinstance(act, SpeechAct):
        raise ValueError("invalid checked speech act")
    if act.kind == "assert":
        return _assertion(act.content)
    if act.kind == "ask":
        if not isinstance(act.content, AskContent):
            raise ValueError("invalid checked ask")
        refs = ", ".join(_quoted(ref) for ref in act.content.references)
        return (
            f"Please provide {_quoted(act.content.category)} for obligation "
            f"{_quoted(act.content.obligation_ref)}; references: [{refs}]."
        )
    if act.kind == "conjoin":
        if not isinstance(act.content, tuple) or not act.content:
            raise ValueError("invalid checked conjunction")
        return " ".join(_speech(item) for item in act.content)
    if act.kind == "explain":
        if not isinstance(act.content, SpeechAct):
            raise ValueError("invalid checked explanation")
        if (not isinstance(act.derivation, tuple) or not act.derivation
                or any(type(step) is not int or step <= 0 for step in act.derivation)):
            raise ValueError("invalid checked derivation")
        steps = ", ".join(str(step) for step in act.derivation)
        return f"Checked derivation through program steps [{steps}]: {_speech(act.content)}"
    raise ValueError("invalid checked speech act")


def _coordinated_fact_pair(outputs: tuple[SpeechAct, ...]) -> str:
    """One L3 primitive: share an identical scope across two factual clauses."""
    if len(outputs) != 1 or outputs[0].kind != "conjoin":
        raise ValueError("unsupported production shape")
    acts = outputs[0].content
    if (not isinstance(acts, tuple) or len(acts) != 2
            or any(not isinstance(act, SpeechAct) or act.kind != "assert"
                   or not isinstance(act.content, Fact) for act in acts)):
        raise ValueError("unsupported production shape")
    left, right = (act.content for act in acts)
    if left.interval != right.interval:
        raise ValueError("unsupported production scope")
    return (
        f"At {_quoted(left.interval)}, {_subject(left)} has value {_operand_value(left)}, "
        f"and {_subject(right)} has value {_operand_value(right)}."
    )


def render_checked(result: CheckResult, host: dict, catalog: dict, *,
                   production: str = PLAIN_PRODUCTION) -> str:
    """Render only a currently authorized accepted kernel result.

    ``validate_release`` rejects forged, stale, revoked and rejected objects.
    Nothing in this function reads the proposal, model state, or raw evidence.
    The output medium is plain text; downstream HTML/Markdown must escape it.
    The optional finite production changes wording only, never the checked
    program, proposition order, source dependencies or authority. An unsupported
    production or speech-act shape has no fallback. This is an L3 primitive,
    not learned fluent language capability.
    """
    checked = validate_release(result, host, catalog)
    if not isinstance(checked, CheckResult) or checked.status != "accepted" or checked.style != "plain":
        raise ValueError("release denied")
    if not checked.outputs:
        raise ValueError("release denied")
    if not is_licensed_production(production):
        raise ValueError("unsupported production")
    if production == L3_FACT_PAIR_COORDINATION:
        return _coordinated_fact_pair(checked.outputs)
    return "\n".join(_speech(act) for act in checked.outputs)
