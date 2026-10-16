"""Deterministic, query-only grant baselines, not a production permission compiler.

Inputs are the authenticated task text and trusted tool schemas, never task IDs,
ground truth, environment state, or tool outputs. Unrecognized values stay free
in ``literals``; ``literal_required`` instead withholds incompletely scoped writes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from keel.host import ArgumentClause, Authority
from keel.protocol import OperatorPolicy

VERSION = "query-keywords-literals/1"
READ_PREFIXES = ("get_", "read_", "search_", "list_", "check_")
# These rules come from tool semantics, not benchmark prompts or answer keys.
WRITE_RULES = {
    "send_money": (r"\b(pay|send|transfer|reimburse|refund)\b",),
    "schedule_transaction": (r"\b(schedule|recurring|monthly|standing)\b|every month",),
    "update_scheduled_transaction": (r"\b(update|change|adjust|increase|decrease|reduce)\b", r"\b(transaction|payment|rent|transfer)\b"),
    "update_password": (r"\bpassword\b", r"\b(update|change|set|reset)\b"),
    "update_user_info": (r"\b(update|change|set|move|moved)\b", r"\b(address|name|street|city|information|info)\b"),
    "send_email": (r"\b(email|e-mail|mail)\b", r"\b(send|forward|share|reply|email|e-mail)\b"),
    "delete_email": (r"\b(delete|remove|trash)\b", r"\b(email|e-mail|mail|inbox)\b"),
    "create_calendar_event": (r"\b(create|add|schedule|book|organize|arrange)\b", r"\b(calendar|event|meeting|appointment)\b"),
    "cancel_calendar_event": (r"\b(cancel|delete|remove)\b", r"\b(calendar|event|meeting|appointment)\b"),
    "reschedule_calendar_event": (r"\b(reschedule|move|change|postpone)\b", r"\b(calendar|event|meeting|appointment)\b"),
    "add_calendar_event_participants": (r"\b(add|invite|include)\b", r"\b(calendar|event|meeting|appointment)\b"),
    "append_to_file": (r"\b(append|add|update|write)\b", r"\b(file|document|notes)\b"),
    "create_file": (r"\b(create|save|write)\b", r"\b(file|document)\b"),
    "delete_file": (r"\b(delete|remove|trash)\b", r"\b(file|document)\b"),
    "share_file": (r"\bshare\b", r"\b(file|document)\b"),
    "reserve_hotel": (r"\b(book|reserve|reservation)\b", r"\b(hotel|room|accommodation)\b"),
    "reserve_restaurant": (r"\b(book|reserve|reservation)\b", r"\b(restaurant|table|dinner|lunch)\b"),
    "reserve_car_rental": (r"\b(book|reserve|reservation|rent)\b", r"\b(car|vehicle)\b"),
    "send_direct_message": (r"\b(send|tell|message|inform|notify|dm)\b",),
    "send_channel_message": (r"\b(send|post|message|announce|write|reply)\b", r"\bchannel\b|#[\w-]+"),
    "add_user_to_channel": (r"\b(add|invite|include)\b", r"\bchannel\b|#[\w-]+"),
    "invite_user_to_slack": (r"\binvite\b",),
    "remove_user_from_slack": (r"\b(remove|delete|kick)\b",),
    "post_webpage": (r"\b(post|publish|upload|update)\b", r"\b(webpage|website|page|url)\b|https?://"),
}
# This arm tests routing/identity/value completeness, NOT content or flow safety.
SCOPE_KEYS = {
    "send_money": ("recipient", "amount"),
    "schedule_transaction": ("recipient", "amount", "date"),
    "update_scheduled_transaction": ("id",),
    "update_password": ("password",),
    "update_user_info": ("street", "city"),
    "send_email": ("recipients",), "delete_email": ("email_id",),
    "create_calendar_event": ("start_time", "end_time", "participants"),
    "cancel_calendar_event": ("event_id",),
    "reschedule_calendar_event": ("event_id", "new_start_time"),
    "add_calendar_event_participants": ("event_id", "participants"),
    "append_to_file": ("file_id",), "create_file": ("filename",),
    "delete_file": ("file_id",), "share_file": ("file_id", "email"),
    "reserve_hotel": ("hotel", "start_day", "end_day"),
    "reserve_restaurant": ("restaurant", "start_time"),
    "reserve_car_rental": ("company", "start_time"),
    "send_direct_message": ("recipient",), "send_channel_message": ("channel",),
    "add_user_to_channel": ("user", "channel"),
    "invite_user_to_slack": ("user", "user_email"),
    "remove_user_from_slack": ("user",), "post_webpage": ("url",),
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    properties: Mapping[str, Mapping[str, Any]]

    @classmethod
    def from_tool(cls, tool) -> "ToolSpec":
        return cls(tool.name, tool.description, tool.parameters.model_json_schema()["properties"])

    @property
    def defaults(self) -> dict:
        return {k: v["default"] for k, v in self.properties.items() if "default" in v}


@dataclass(frozen=True)
class QueryGrant:
    tools: tuple[str, ...]
    clauses: tuple[ArgumentClause, ...] = ()

    def authority(self, specs: Sequence[ToolSpec]) -> Authority:
        return Authority(OperatorPolicy(VERSION, "Authenticated query-derived grants", authorized_actions=self.tools),
                         {"agent": self.tools}, {"agent": ("agent",)}, argument_clauses=self.clauses,
                         argument_defaults={s.name: s.defaults for s in specs})

    def manifest(self) -> dict:
        return {"tools": list(self.tools), "clauses": [c.manifest() for c in self.clauses]}


def _unique(values: Sequence[Any]) -> list[Any]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def extract_literals(prompt: str) -> dict[str, list]:
    """Conservative lexical candidates; role resolution is intentionally incomplete."""
    emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", prompt)
    ibans = re.findall(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b", prompt)
    urls = [u.rstrip(".,;:!?)") for u in re.findall(r"https?://[^\s<>\"']+", prompt)]
    files = re.findall(r"(?<![\w@])(?:[\w./-]+\.(?:txt|pdf|csv|docx?|xlsx?|md|json))(?!\w)", prompt, re.I)
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", prompt)
    times = re.findall(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}\b", prompt)
    amounts = [float(x.replace(",", "")) for x in re.findall(
        r"(?:[$€£]\s*|\b(?:USD|EUR|GBP|CHF)\s+)(\d[\d,]*(?:\.\d+)?)", prompt)]
    amounts += [float(x.replace(",", "")) for x in re.findall(
        r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:dollars?|euros?|pounds?|USD|EUR|GBP|CHF)\b", prompt, re.I)]
    channels = re.findall(r"(?<!\w)#([\w-]+)", prompt)
    channels += re.findall(r"\bchannel\s+[\"'`]([^\"'`]+)[\"'`]", prompt, re.I)
    channels += re.findall(r"[\"'`]([^\"'`]+)[\"'`]\s+channel\b", prompt, re.I)
    names = re.findall(
        r"\b(?!(?:Tell|Send|Invite|Ask|Notify|Please|Email|Pay|Find|Add|Remove)\b)"
        r"[A-Z][a-z]+(?:[ -][A-Z][a-z]+)+\b", prompt)
    names += re.findall(r"\b(?:to|tell|notify|invite|user|from)\s+([A-Z][a-z]+)\b", prompt)
    return {key: _unique(values) for key, values in {
        "email": emails, "iban": ibans, "url": urls, "file": files, "date": dates,
        "time": [t.replace("T", " ") for t in times], "amount": amounts,
        "channel": channels, "name": names,
    }.items()}


def _array_property(prop: Mapping) -> bool:
    return prop.get("type") == "array" or any(p.get("type") == "array" for p in prop.get("anyOf", ()))


def _clause(prompt: str, spec: ToolSpec, literals: Mapping[str, list]) -> ArgumentClause:
    fixed, one_of, at_most = {}, {}, {}
    for key, prop in spec.properties.items():
        values = []
        if key in ("email", "user_email", "recipients", "participants", "sender"):
            values = literals["email"]
        elif key == "recipient":
            values = literals["iban"] if spec.name in ("send_money", "schedule_transaction", "update_scheduled_transaction") else literals["name"]
        elif key in ("user", "hotel", "restaurant", "company"):
            values = literals["name"]
        elif key == "channel":
            values = literals["channel"]
        elif key == "url":
            values = literals["url"]
        elif key in ("file_path", "filename"):
            values = literals["file"]
        elif key in ("date", "day", "start_day", "end_day"):
            values = literals["date"]
        elif key in ("start_time", "end_time", "new_start_time", "new_end_time"):
            values = literals["time"]
        elif key == "amount":
            values = literals["amount"]
            if len(values) == 1 and re.search(r"\b(at most|up to|no more than|maximum|limit)\b", prompt, re.I):
                at_most[key] = values[0]
                continue
        if not values:
            continue
        if _array_property(prop):
            # Host clauses match whole JSON arrays, not each member independently.
            fixed[key] = list(values)
        elif len(values) == 1:
            fixed[key] = values[0]
        else:
            one_of[key] = values
    return ArgumentClause(spec.name, fixed=fixed, one_of=one_of, at_most=at_most)


def plan_query(prompt: str, specs: Sequence[ToolSpec], *, mode: str = "literals") -> QueryGrant:
    """Grant discovery tools plus keyword-selected writes; optional literal clauses.

    Three frozen modes: tools, literals, literal_required. No environment-bearing
    object is accepted. Tool descriptions/schemas must themselves be host trusted.
    """
    if mode not in ("tools", "literals", "literal_required"):
        raise ValueError(f"unknown planner mode: {mode}")
    literals = extract_literals(prompt)
    tools, clauses = [], []
    for spec in specs:
        read = spec.name.startswith(READ_PREFIXES)
        rules = WRITE_RULES.get(spec.name)
        if not read and (rules is None or not all(re.search(r, prompt, re.I) for r in rules)):
            continue
        clause = _clause(prompt, spec, literals)
        constrained = set(clause.fixed) | set(clause.one_of) | set(clause.at_most)
        if mode == "literal_required" and not read and not set(SCOPE_KEYS[spec.name]) <= constrained:
            continue
        tools.append(spec.name)
        if mode != "tools" and constrained:
            clauses.append(clause)
    return QueryGrant(tuple(sorted(tools)), tuple(clauses))
