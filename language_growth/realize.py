"""Plain-text linearization of kernel-minted, release-validated speech acts.

This is a small controlled language, not a general text generation path. Every
string copied from a checked object is a JSON string literal in the output.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import NamedTuple

from .kernel import validate_release
from .types import AskContent, CheckResult, Comparison, ConstantValue, Fact, SpeechAct


class EvidenceUse(NamedTuple):
    """One checked fact used by an emitted speech act.

    ``speech_path`` starts with the output index and then records child indexes
    through conjunctions/explanations. It is deliberately structural: callers
    can attach citations or inspect lineage without parsing rendered prose.
    """

    speech_path: tuple[int, ...]
    role: str
    source_ref: str
    record_ref: str
    record_version: int
    field: str
    interval: str


def checked_evidence_trace(result: CheckResult, host: dict, catalog: dict) -> tuple[EvidenceUse, ...]:
    """Return the evidence lineage for current, authorized checked output.

    The trace is derived only from kernel-minted speech acts after revalidating
    release authority. It contains references and versions, never evidence
    values or model-proposed strings. This is an audit hook for consumers that
    need citations; it does not itself release text or grant authority.
    """
    checked = validate_release(result, host, catalog)
    if not isinstance(checked, CheckResult) or checked.status != "accepted":
        raise ValueError("release denied")

    uses: list[EvidenceUse] = []

    def add(item: Fact | ConstantValue, path: tuple[int, ...], role: str) -> None:
        if isinstance(item, Fact):
            uses.append(EvidenceUse(path, role, item.source_ref, item.record_ref,
                                    item.record_version, item.field, item.interval))

    def visit(act: SpeechAct, path: tuple[int, ...]) -> None:
        if not isinstance(act, SpeechAct):
            raise ValueError("invalid checked speech act")
        if act.kind == "assert" and isinstance(act.content, Fact):
            add(act.content, path, "asserted")
        elif act.kind == "assert" and isinstance(act.content, Comparison):
            add(act.content.left, path, "comparison_left")
            add(act.content.right, path, "comparison_right")
        elif act.kind == "conjoin" and isinstance(act.content, tuple):
            for index, child in enumerate(act.content):
                visit(child, path + (index,))
        elif act.kind == "explain" and isinstance(act.content, SpeechAct):
            visit(act.content, path + (0,))

    for output_index, act in enumerate(checked.outputs):
        visit(act, (output_index,))
    return tuple(uses)


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
