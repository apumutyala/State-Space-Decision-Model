"""Regression: ``items.write_items`` must keep option (criteria) order as authored.

Same bug class as PR #245: ``write_items`` used ``json.dumps(..., sort_keys=True)``, which re-sorted
``question.criteria`` (and ``state`` dicts), so any item file written through it lost its option order. These tests
write items whose criteria are deliberately NOT sorted, read the file back with plain ``json.loads`` (dicts keep file
order) and through ``load_items``, and check the order survives; the last test patches the pre-fix ``sort_keys=True``
behaviour back in and asserts the check catches it.
"""

from __future__ import annotations

import json
import types

from jevkit import items as IT
from jevkit.items import Item

from ._fixtures import noul

KEYS = ["zeta", "alpha", "mid", "beta", "kappa"]


def _unsorted_items() -> list[Item]:
    ch = Item(id="cu", family="fam_choice", state={"zz": "last key first", "aa": "first key last"},
              question={"type": "choice", "instructions": "Which option applies?",
                        "criteria": {k: f"Option {k}." for k in KEYS}},
              labels=list(KEYS), expected="mid", split="public", tier="standard").validate()
    # noul criteria authored as true, false ("false" < "true" when sorted)
    return [ch, noul("n1")]


def _order_violations(path, items: list[Item]) -> list[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    bad = []
    for it, r in zip(items, rows, strict=True):
        if list(r["question"]["criteria"]) != list(it.question["criteria"]):
            bad.append(f"{it.id}: criteria {list(r['question']['criteria'])} != {list(it.question['criteria'])}")
        if isinstance(it.state, dict) and list(r["state"]) != list(it.state):
            bad.append(f"{it.id}: state keys {list(r['state'])} != {list(it.state)}")
    return bad


def test_write_items_preserves_criteria_order(tmp_path):
    items = _unsorted_items()
    assert list(items[0].question["criteria"]) != sorted(items[0].question["criteria"])
    assert list(items[1].question["criteria"]) != sorted(items[1].question["criteria"])
    p = tmp_path / "items.jsonl"
    IT.write_items(items, p)
    assert _order_violations(p, items) == []
    back = IT.load_items(p)
    assert [list(it.question["criteria"]) for it in back] == [list(it.question["criteria"]) for it in items]
    assert [it.to_dict() for it in back] == [it.to_dict() for it in items]


def test_order_check_catches_sort_keys(tmp_path, monkeypatch):
    real = json.dumps
    monkeypatch.setattr(IT, "json", types.SimpleNamespace(
        dumps=lambda obj, **kw: real(obj, **{**kw, "sort_keys": True}), loads=json.loads))
    items = _unsorted_items()
    p = tmp_path / "items.jsonl"
    IT.write_items(items, p)
    bad = _order_violations(p, items)
    assert any(b.startswith("cu: criteria") for b in bad), bad
    assert any(b.startswith("n1: criteria") for b in bad), bad
