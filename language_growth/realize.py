"""Plain-text linearization of kernel-minted, release-validated speech acts.

This is a small controlled language, not a general text generation path. Every
string copied from a checked object is a JSON string literal in the output.
"""

from __future__ import annotations

import json
from decimal import Decimal

from .kernel import validate_release
from .types import AssumptionValue, AskContent, CheckResult, Comparison, ConditionalContent, ConstantValue, Fact, PlanContent, SpeechAct


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


def _subject(item: Fact | ConstantValue | AssumptionValue, *, recorded_baseline: bool = False) -> str:
    if isinstance(item, AssumptionValue):
        return "hypothetical " + _subject(item.premise)
    if isinstance(item, Fact):
        prefix = "recorded baseline " if recorded_baseline else ""
        return f"{prefix}{_quoted(item.predicate)} for {_quoted(item.subject)}"
    if isinstance(item, ConstantValue):
        return f"constant {_quoted(item.constant_ref)}"
    raise ValueError("invalid checked operand")


def _operand_value(item: Fact | ConstantValue | AssumptionValue) -> str:
    if isinstance(item, AssumptionValue):
        return f"{_value(item.value)}{_unit(item.unit)}"
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
        conditional = any(isinstance(item, AssumptionValue) for item in (content.left, content.right))
        return (
            f"At {_quoted(content.interval)}, the value of {_subject(content.left, recorded_baseline=conditional)} "
            f"({_operand_value(content.left)}) is {relations[content.relation]} "
            f"the value of {_subject(content.right, recorded_baseline=conditional)} ({_operand_value(content.right)})."
        )
    raise ValueError("invalid checked assertion")


def _speech(act: SpeechAct) -> str:
    if not isinstance(act, SpeechAct):
        raise ValueError("invalid checked speech act")
    if act.kind == "assert":
        return _assertion(act.content)
    if act.kind == "hypothesize":
        if not isinstance(act.content, ConditionalContent):
            raise ValueError("invalid checked hypothesis")
        assumption = act.content.assumption
        conclusion = _assertion(act.content.conclusion)
        return (f"If at {_quoted(assumption.interval)}, {_subject(assumption)} had value "
                f"{_operand_value(assumption)}, then {conclusion[0].lower()}{conclusion[1:]}")
    if act.kind == "ask":
        if not isinstance(act.content, AskContent):
            raise ValueError("invalid checked ask")
        refs = ", ".join(_quoted(ref) for ref in act.content.references)
        return (
            f"Please provide {_quoted(act.content.category)} for obligation "
            f"{_quoted(act.content.obligation_ref)}; references: [{refs}]."
        )
    if act.kind == "plan":
        if not isinstance(act.content, PlanContent) or not act.content.steps:
            raise ValueError("invalid checked plan")
        lines = [f"Proposed selected steps for workflow {_quoted(act.content.workflow_ref)}; selection may be incomplete; no actions executed."]
        for index, step in enumerate(act.content.steps, 1):
            conditions = " and ".join(
                f"at {_quoted(fact.interval)}, recorded {_subject(fact)} has value {_operand_value(fact)}"
                for fact in step.preconditions)
            ordering = (" after proposed steps [" + ", ".join(_quoted(ref) for ref in step.after) + "]"
                        if step.after else "")
            lines.append(f"Proposed step {index} ({_quoted(step.step_ref)}){ordering}: "
                         f"if {conditions}, propose action {_quoted(step.action_ref)}; execution requires separate authorization and precondition recheck.")
        return "\n".join(lines)
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
    return "\n".join(_speech(act) for act in checked.outputs)
