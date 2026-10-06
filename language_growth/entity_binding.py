"""Conservative exact-mention resolution for explicit positive request forms.

Literal references bind only within the caller-supplied trusted allowed records.
Unsupported or ambiguous text is unresolved. This is a small controlled grammar,
not general natural-language interpretation or an authority checker.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
import re
from typing import Literal


@dataclass(frozen=True, slots=True)
class AllowedRecord:
    record_ref: str
    name: str
    fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Mention:
    record_ref: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Resolution:
    status: Literal["resolved", "ambiguous", "unsupported"]
    mentions: tuple[Mention, ...] = ()
    reason: str | None = None


def resolve_mentions(task_input: str, records: tuple[AllowedRecord, ...]) -> Resolution:
    """Resolve literal ASCII names with original half-open character spans.

    Positive-form admission is a full match, not a negation/injection blacklist.
    Any overlapping name, repeated mention or duplicate name/ref is ambiguous.
    Unknown syntax (including negation and appended instructions) fails closed.
    """
    if not isinstance(task_input, str) or not task_input.isascii() or not task_input or len(task_input) > 2048 or any(ord(c) < 32 for c in task_input):
        return Resolution("unsupported", reason="unsupported_text")
    if not records or len(records) > 64:
        return Resolution("unsupported", reason="invalid_records")
    names, refs, fields = set(), set(), set()
    for record in records:
        if (not record.name.isascii() or not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*(?:[ -][A-Za-z][A-Za-z0-9]*)*", record.name)
                or not record.record_ref or not record.fields
                or any(not re.fullmatch(r"[a-z][a-z0-9_]*", field) for field in record.fields)):
            return Resolution("unsupported", reason="unsupported_record_shape")
        if record.name.casefold() in names or record.record_ref in refs:
            return Resolution("ambiguous", reason="duplicate_name_or_ref")
        names.add(record.name.casefold())
        refs.add(record.record_ref)
        fields.update(record.fields)
    mentions = sorted((Mention(record.record_ref, match.start(), match.end())
        for record in records
        for match in re.finditer(r"(?<![A-Za-z0-9_])" + re.escape(record.name) + r"(?![A-Za-z0-9_])",
                                 task_input, flags=re.IGNORECASE | re.ASCII)), key=lambda item: (item.start, item.end))
    if not mentions:
        return Resolution("unsupported", reason="no_allowed_mention")
    if any(a.end > b.start for a,b in zip(mentions, mentions[1:])):
        return Resolution("ambiguous", reason="overlapping_names")
    if len({item.record_ref for item in mentions}) != len(mentions):
        return Resolution("ambiguous", reason="repeated_mention")
    # The NUL marker cannot appear in admitted input, preventing marker spoofing.
    scaffold = task_input
    for item in reversed(mentions):
        scaffold = scaffold[:item.start] + "\x00" + scaffold[item.end:]
    entity = re.escape("\x00")
    field = "(?:" + "|".join(re.escape(value.replace("_", " ")) for value in sorted(fields)) + ")"
    date = r"(?:January|February|March|April|May|June|July|August|September|October|November|December) [0-9]{4}"
    patterns = (
        rf"State {entity}'s {field} for {date}\.",
        rf"Provide a checked derivation of {entity}'s {field}\.",
        rf"For {date}, compare {entity}'s and {entity}'s {field}\.",
        rf"Tell me the {field} belonging to {entity} as of {date}\.",
        rf"Describe the checked reasoning behind the {field} belonging to {entity}\.",
        rf"With {entity} first and {entity} second, compare their {date} {field}\.",
    )
    if not any(re.fullmatch(pattern, scaffold, flags=re.IGNORECASE | re.ASCII) for pattern in patterns):
        return Resolution("unsupported", reason="unsupported_positive_form")
    return Resolution("resolved", tuple(mentions))


def overlay_request(request: dict, predicted: dict) -> dict:
    """No evaluator input accepted. Unresolved is a clarification disposition only."""
    state = json.loads(request["state"])
    records = tuple(AllowedRecord(record["record_ref"], record["entity_ref"], tuple(record["fields"]))
                    for record in state["records"])
    resolution = resolve_mentions(state["task_input"], records)
    anchor = {key: predicted[key] for key in ("kind", "field")}
    output = {"disposition": "unresolved", "operation_field": anchor,
              "resolution": asdict(resolution), "predicted": None}
    if resolution.status != "resolved":
        return output
    kind = predicted["kind"]
    expected = 2 if kind == "compare" else 1 if kind in ("read", "explain") else 0
    if expected == 0 or len(resolution.mentions) != expected:
        output["resolution"] = asdict(Resolution("unsupported", reason="operation_mention_count"))
        return output
    if kind != "compare" and "right" in predicted:
        output["resolution"] = asdict(Resolution("unsupported", reason="malformed_prediction"))
        return output
    target = copy.deepcopy(predicted)
    target["left"] = resolution.mentions[0].record_ref
    if kind == "compare":
        target["right"] = resolution.mentions[1].record_ref
    assert json.dumps(anchor, sort_keys=True).encode() == json.dumps(
        {key: target[key] for key in ("kind", "field")}, sort_keys=True).encode()
    output.update(disposition="resolved", predicted=target)
    return output
