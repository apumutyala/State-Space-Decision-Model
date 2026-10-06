"""Training items: one (request, question, gold option) per row, plus a toy task for tests.

Record format (JSONL): ``{"id": str, "request": <SystemOne request>, "labels": {qid: option_id}}``.
Gold option ids are mapped to the canonical option index of the row. Score labels may be the
level index as a string ("0".."k-1"); noul labels are "true"/"false".

The toy task exists only to exercise the trainer on CPU: a short state names one value of a
field and a choice question asks for it. It is synthetic and has no relation to Index items.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Iterable

from decide.prompt import Template
from decide.schema import validate_legacy
from heads.layout import EncodeFn, LayoutRow, build_row


@dataclass
class Item:
    id: str
    row: LayoutRow
    label: int


def items_from_records(records: Iterable[dict[str, Any]], encode: EncodeFn, layout: str = "ordered",
                       template: "str | Template" = "jrt-plain-v1") -> list[Item]:
    out = []
    for rec in records:
        req = rec["request"]
        validate_legacy(req)
        for qid, gold in sorted(rec["labels"].items()):
            row = build_row(req, qid, encode, template=template, layout=layout)
            gold = str(gold).lower() if row.qtype == "noul" else str(gold)
            if gold not in row.option_ids:
                raise ValueError(f"{rec['id']}/{qid}: gold {gold!r} not among options {row.option_ids}")
            out.append(Item(f"{rec['id']}/{qid}", row, row.option_ids.index(gold)))
    return out


def split_hash(items: list[Item]) -> str:
    return hashlib.sha256("\n".join(sorted(i.id for i in items)).encode()).hexdigest()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


COLORS = ["red", "green", "blue", "amber"]


def toy_records(n: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        color = rng.choice(COLORS)
        filler = rng.choice(["ok", "note", "id 7", "x"])
        size = rng.randrange(3)
        recs.append({
            "id": f"toy{seed}-{i}",
            "request": {
                "model": "toy",
                "state": f"{filler}; color={color}; size={size}",
                "questions": {
                    "c": {"type": "choice", "instructions": "color?", "criteria": {c: None for c in COLORS}},
                    "s": {"type": "score", "instructions": "size?", "criteria": ["0", "1", "2"]},
                },
            },
            "labels": {"c": color, "s": str(size)},
        })
    return recs


#: Compact template for CPU toys (no system prompt), so tiny recurrent models see short rows.
TOY_TEMPLATE = Template(id="toy-v1", head="", tail="\nA:", terminator="\n")
