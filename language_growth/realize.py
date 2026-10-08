"""Plain-text linearization of kernel-minted, release-validated speech acts.

This is a small controlled language, not a general text generation path. Every
string copied from a checked object is a JSON string literal in the output.
"""

from __future__ import annotations

import json
from decimal import Decimal

from .kernel import validate_release
from .types import AssumptionValue, AskContent, CheckResult, Comparison, ConditionalContent, ConstantValue, ExactDifference, Fact, PlanContent, SpeechAct, decimal_text


PLAIN_PRODUCTION = "plain"
L3_FACT_PAIR_COORDINATION = "l3_fact_pair_coordination_v1"
L3_FACT_COMPARISON_CONTRAST = "l3_fact_comparison_contrast_v1"
LICENSED_PRODUCTIONS = frozenset({
    PLAIN_PRODUCTION, L3_FACT_PAIR_COORDINATION, L3_FACT_COMPARISON_CONTRAST,
})


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
        return decimal_text(value)
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


def _assertion(content: Fact | Comparison | ExactDifference) -> str:
    if isinstance(content, Fact):
        return f"At {_quoted(content.interval)}, {_subject(content)} has value {_operand_value(content)}."
    if isinstance(content, ExactDifference):
        return (
            f"At {_quoted(content.interval)}, the value of {_subject(content.left)} "
            f"({_operand_value(content.left)}) minus the value of {_subject(content.right)} "
            f"({_operand_value(content.right)}) is {_value(content.value)}{_unit(content.unit)}."
        )
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


def _contrasted_fact_comparison(outputs: tuple[SpeechAct, ...]) -> str:
    """Expand a checked numeric comparison into a two-sentence discourse.

    The second sentence refers back to the first value. The connective adds
    only the quantitative relation licensed by Compare, never cause or chronology.
    """
    if (len(outputs) != 1 or outputs[0].kind != "assert"
            or not isinstance(outputs[0].content, Comparison)):
        raise ValueError("unsupported production shape")
    comparison = outputs[0].content
    left, right = comparison.left, comparison.right
    if not isinstance(left, Fact) or not isinstance(right, Fact):
        raise ValueError("unsupported production operands")
    if (left.subject == right.subject or left.predicate != right.predicate
            or left.field != right.field or left.unit != right.unit
            or left.quantity_kind != right.quantity_kind
            or left.interval != right.interval or left.interval != comparison.interval):
        raise ValueError("unsupported production scope")
    if (type(left.object) not in (int, Decimal)
            or type(right.object) not in (int, Decimal)):
        raise ValueError("unsupported production operands")
    # Compare relates left to right; this clause describes right versus left.
    inverse = {"greater_than": "less than", "less_than": "greater than", "equal": "equal to"}
    if comparison.relation not in inverse:
        raise ValueError("unsupported production relation")
    connective = "Likewise" if comparison.relation == "equal" else "In contrast"
    return (
        f"{_assertion(left)} {connective}, at {_quoted(right.interval)}, "
        f"{_subject(right)} has value {_operand_value(right)}, "
        f"which is {inverse[comparison.relation]} the first value."
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
    if production == L3_FACT_COMPARISON_CONTRAST:
        return _contrasted_fact_comparison(checked.outputs)
    return "\n".join(_speech(act) for act in checked.outputs)
