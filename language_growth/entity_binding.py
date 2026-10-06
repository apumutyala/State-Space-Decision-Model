"""Conservative exact-mention resolution for explicit positive request forms.

Literal references bind only within the caller-supplied trusted allowed records.
The matched form retains requested operation/field and explicit month/year scope;
model disagreement or mismatched selected-field scope rejects.
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
    requested_kind: Literal["read", "explain", "compare"] | None = None
    requested_field: str | None = None
    requested_scope: str | None = None


def resolve_mentions(task_input: str, records: tuple[AllowedRecord, ...]) -> Resolution:
    """Resolve ASCII mentions and requested operation/field from supported forms.

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
                or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", field) for field in record.fields)):
            return Resolution("unsupported", reason="unsupported_record_shape")
        if record.name.casefold() in names or record.record_ref in refs:
            return Resolution("ambiguous", reason="duplicate_name_or_ref")
        names.add(record.name.casefold())
        refs.add(record.record_ref)
        fields.update(record.fields)
    field_surfaces = {}
    for field_name in sorted(fields):
        surface = field_name.replace("_", " ").casefold()
        if surface in field_surfaces and field_surfaces[surface] != field_name:
            return Resolution("ambiguous", reason="duplicate_field_surface")
        field_surfaces[surface] = field_name
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
    field = "(?P<field>" + "|".join(re.escape(value) for value in sorted(field_surfaces)) + ")"
    months = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
    date = "(?P<month>" + "|".join(months) + r") (?P<year>[0-9]{4})"
    patterns = (
        ("read", rf"State {entity}'s {field} for {date}\."),
        ("explain", rf"Provide a checked derivation of {entity}'s {field}\."),
        ("compare", rf"For {date}, compare {entity}'s and {entity}'s {field}\."),
        ("read", rf"Tell me the {field} belonging to {entity} as of {date}\."),
        ("explain", rf"Describe the checked reasoning behind the {field} belonging to {entity}\."),
        ("compare", rf"With {entity} first and {entity} second, compare their {date} {field}\."),
    )
    matches = [(kind, match) for kind, pattern in patterns
               if (match := re.fullmatch(pattern, scaffold, flags=re.IGNORECASE | re.ASCII)) is not None]
    if not matches:
        return Resolution("unsupported", reason="unsupported_positive_form")
    if len(matches) != 1:
        return Resolution("ambiguous", reason="ambiguous_positive_form")
    kind, match = matches[0]
    year = match.groupdict().get("year")
    scope = None if year is None else f"{year}-{next(i for i, month in enumerate(months, 1) if month.casefold() == match.group('month').casefold()):02d}"
    return Resolution("resolved", tuple(mentions), requested_kind=kind,
                      requested_field=field_surfaces[match.group("field").casefold()], requested_scope=scope)


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
    def reject(reason):
        output["resolution"] = asdict(Resolution("unsupported", reason=reason,
            requested_kind=resolution.requested_kind, requested_field=resolution.requested_field,
            requested_scope=resolution.requested_scope))
        return output
    if predicted["kind"] != resolution.requested_kind or predicted["field"] != resolution.requested_field:
        return reject("requested_selection_mismatch")
    kind = predicted["kind"]
    expected = 2 if kind == "compare" else 1 if kind in ("read", "explain") else 0
    if expected == 0 or len(resolution.mentions) != expected:
        return reject("operation_mention_count")
    if kind != "compare" and "right" in predicted:
        return reject("malformed_prediction")
    by_ref = {record["record_ref"]: record for record in state["records"]}
    for mention in resolution.mentions:
        descriptor = by_ref[mention.record_ref]["fields"].get(resolution.requested_field)
        if not isinstance(descriptor, dict):
            return reject("selected_field_missing")
        if resolution.requested_scope is not None and descriptor.get("temporal_scope") != resolution.requested_scope:
            return reject("requested_scope_mismatch")
    target = copy.deepcopy(predicted)
    target["left"] = resolution.mentions[0].record_ref
    if kind == "compare":
        target["right"] = resolution.mentions[1].record_ref
    assert json.dumps(anchor, sort_keys=True).encode() == json.dumps(
        {key: target[key] for key in ("kind", "field")}, sort_keys=True).encode()
    output.update(disposition="resolved", predicted=target)
    return output
