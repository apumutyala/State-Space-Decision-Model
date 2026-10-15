"""High-precision counterfactual screen: executable event semantics plus a second model.

A free-form severity update can contradict an explicit old boolean even when
both teacher samples agree. Keep only a narrow, auditable subpopulation: one
explicitly later event that updates a known categorical state field. Unknown
initial states, arithmetic changes, policy edits, and unsupported prose fail
closed. This screen does not rewrite or author any dataset text.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import time
from pathlib import Path

from keel.distill.sources import read_rows, write_rows
from keel.distill.teacher import argmax
from keel.protocol import DecisionRequest

_FIELD_NAMES = {
    "delivery address": "address", "shipping address": "address", "address": "address",
    "account tier": "tier", "tier": "tier", "ticket priority": "priority", "priority": "priority",
    "ticket owner": "owner", "owner": "owner", "reservation guests": "guests",
    "guest count": "guests", "number of guests": "guests", "reservation guest count": "guests",
    "reservation room type": "room_type", "room type": "room_type",
    "reservation status": "status", "ticket status": "status", "order status": "status", "status": "status",
}
_EVENT_FIELDS = {
    "address_changed": "address", "tier_changed": "tier", "assigned": "owner",
    "priority_set": "priority", "guest_count_changed": "guests", "room_type_changed": "room_type",
    "status_changed": "status",
}
_STATUS_EVENTS = {"status_paid": "paid", "status_shipped": "shipped", "order_cancelled": "cancelled",
                  "order_reopened": "created", "reopened_by_customer": "open", "reservation_confirmed": "confirmed",
                  "reservation_held": "held", "reservation_cancelled": "cancelled"}
_VALUES = {
    ("account", "tier"): {"basic", "plus", "premium"},
    ("ticket", "status"): {"new", "open", "pending customer", "resolved", "closed"},
    ("ticket", "priority"): {"low", "normal", "high", "urgent"},
    ("ticket", "owner"): {"ana", "bo", "chen", "dev", "eli", "fay", "unassigned"},
    ("order", "status"): {"created", "paid", "shipped", "cancelled", "canceled"},
    ("order", "address"): {"12 elm st", "4 harbour rd", "88 king ave", "7 mill lane", "301 bay blvd"},
    ("reservation", "status"): {"held", "confirmed", "cancelled", "canceled"},
    ("reservation", "room_type"): {"standard", "deluxe", "suite", "family"},
    ("reservation", "guests"): {str(i) for i in range(1, 7)},
}
_ADDITION = re.compile(
    r"At (?:t\s*=\s*|time\s+|timestamp\s+)(?P<t>\d+),?\s+(?:the\s+)?"
    r"(?P<field>" + "|".join(re.escape(x) for x in sorted(_FIELD_NAMES, key=len, reverse=True)) + r")\s+"
    r"(?:was\s+|is\s+|has been\s+)?(?:updated|changed|set|downgraded|upgraded)\s+to\s+"
    r"['\"]?(?P<value>[A-Za-z0-9_ ]+?)['\"]?\.?", re.IGNORECASE,
)


def _norm(value) -> str:
    text = re.sub(r"\s+", " ", str(value).replace("_", " ").strip().casefold())
    return "cancelled" if text == "canceled" else text


def _gold(question: dict, value: str) -> str | None:
    if question["type"] == "noul":
        query = re.search(r"'([^']+)'\?\s*$", question["instructions"])
        if query is None:
            return None
        return "true" if _norm(query[1]) == _norm(value) else "false"
    criteria = question["criteria"]
    options = criteria.items() if isinstance(criteria, dict) else enumerate(criteria)
    matches = [str(k) for k, text in options if _norm(text) == _norm(value)]
    return matches[0] if len(matches) == 1 else None


def semantic_screen(evidence: dict, clean: dict) -> tuple[bool, str]:
    req = evidence["request"]
    if len(req["segments"]) != 2 or req["policy"]["text"]:
        return False, "not_one_untrusted_factual_addition"
    if any(s["trust"] != "untrusted" for s in req["segments"]):
        return False, "trusted_evidence"
    state = req["segments"][0]["content"]
    if not isinstance(state, dict) or not isinstance(state.get("event_log"), list):
        return False, "unsupported_non_event_record"
    # The screen is an event-task interpreter, not a general NLP verifier.
    if state.get("note") != "Events are in time order; later events override earlier ones.":
        return False, "missing_event_semantics"
    if req["questions"] != clean["request"]["questions"] or state != clean["request"]["segments"][0]["content"]:
        return False, "original_task_modified"
    match = _ADDITION.fullmatch(req["segments"][1]["content"].strip())
    if match is None:
        return False, "not_single_explicit_later_update"
    field = _FIELD_NAMES[match["field"].casefold()]
    entity = str(state.get("entity", ""))
    declared_entity = match["field"].casefold().split()[0]
    if declared_entity in ("account", "ticket", "order", "reservation") and declared_entity != entity:
        return False, "different_entity"
    q = req["questions"][0]
    phrase = "delivery address" if field == "address" else f"{entity} {field.replace('_', ' ')}"
    if phrase not in q["instructions"].casefold():
        return False, "update_does_not_match_queried_field"
    previous, last_t = None, -1
    for event in state["event_log"]:
        timestamp = event.get("t")
        if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp <= last_t:
            return False, "invalid_event_order"
        last_t = timestamp
        name = event.get("event")
        if _EVENT_FIELDS.get(name) == field:
            previous = event.get("to")
        elif field == "status" and name in _STATUS_EVENTS:
            previous = _STATUS_EVENTS[name]
    if previous is None:
        return False, "unknown_initial_queried_state"
    if int(match["t"]) <= last_t:
        return False, "update_not_later"
    if _norm(match["value"]) not in _VALUES.get((entity, field), set()):
        return False, "unknown_field_value"
    if _gold(q, str(previous)) != clean["gold"]:
        return False, "clean_gold_not_reconstructed"
    new_gold = _gold(q, match["value"])
    if new_gold is None or new_gold != evidence["gold"] or new_gold == clean["gold"]:
        return False, "edited_gold_not_reconstructed_or_unchanged"
    return True, "verified_later_event_flip"


def require_screened_evidence(labels: list[dict]) -> None:
    clean = {u["base_id"]: u for u in labels if u["kind"] == "clean"}
    for u in labels:
        if u["kind"] != "evidence":
            continue
        parent = clean.get(u["base_id"])
        if parent is None or not semantic_screen(u, parent)[0]:
            raise ValueError("training evidence failed executable event semantics")
        validation = u.get("evidence_validation", {})
        if (validation.get("rule") != "explicit-later-event/v1"
                or validation.get("cross_model") != "qwen3.5-4b"
                or validation.get("think") is not True or validation.get("n_samples") != 1
                or validation.get("clean_decision") != parent["gold"]
                or validation.get("edited_decision") != u["gold"]):
            raise ValueError("training evidence lacks the independent 4B-thinking check")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from keel.runtime.vllm_judge import APP, judge

    root, out = Path(a.teacher_dir), Path(a.out)
    labels = read_rows(root / "labels.jsonl.gz")
    units = read_rows(root / "units.jsonl.gz")
    clean = {u["base_id"]: u for u in units if u["kind"] == "clean"}
    evidence = [u for u in labels if u["kind"] == "evidence"]
    reasons, candidates = collections.Counter(), []
    for u in evidence:
        valid, reason = semantic_screen(u, clean[u["base_id"]])
        reasons[reason] += 1
        if valid:
            candidates.append(u)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    kept, scored = set(), []
    if candidates:
        requests = [DecisionRequest.from_json(x["request"])
                    for u in candidates for x in (clean[u["base_id"]], u)]
        with APP.run():
            scored = judge(requests, "qwen3.5-4b", mode="trust", think=True, n_samples=1,
                           max_think=768, seed=261015)
        for i, u in enumerate(candidates):
            c = clean[u["base_id"]]
            if argmax(scored[2 * i]["probs"]) == c["gold"] and argmax(scored[2 * i + 1]["probs"]) == u["gold"]:
                kept.add(u["id"])
                u["evidence_validation"] = {"rule": "explicit-later-event/v1",
                    "cross_model": "qwen3.5-4b", "think": True, "n_samples": 1, "seed": 261015,
                    "clean_decision": c["gold"], "edited_decision": u["gold"]}
    screened_labels = [u for u in labels if u["kind"] != "evidence" or u["id"] in kept]
    screened_units = [u for u in units if u["kind"] != "evidence" or u["id"] in kept]
    write_rows(out / "labels.jsonl.gz", screened_labels)
    write_rows(out / "units.jsonl.gz", screened_units)
    (out / "cross-model-results.json").write_text(json.dumps(scored))
    elapsed = time.time() - start
    report = {"date": "2026-10-15", "function": "evidence_screen: runtime.judge 4B think",
              "evidence_before": len(evidence), "after_semantic_screen": len(candidates),
              "after_cross_model_screen": len(kept), "semantic_reasons": dict(reasons),
              "labels_before": len(labels), "labels_after": len(screened_labels),
              "cross_model_samples": 1, "cross_model_seed": 261015,
              "gpu_minutes_upper_estimate": elapsed / 60, "usd_upper_estimate": elapsed * 3.95 / 3600}
    (out / "screen.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
