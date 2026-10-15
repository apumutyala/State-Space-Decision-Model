"""Regression: ``perturb make`` must write option order to disk exactly as the operator produced it.

PR #245: the CLI wrote rows with ``json.dumps(..., sort_keys=True)``, which re-sorted ``question.criteria`` and so put
option_reverse / option_shuffle / key_relabel rows back into (sorted) base order on disk. Models never saw a reorder
and reorder flip rates were vacuously 0. These tests read the CLI's JSONL output with plain ``json.loads`` (dicts keep
file order) and check the criteria key order per operator; the last test re-runs the CLI with the pre-fix
``sort_keys=True`` behaviour patched in and asserts the checks catch it.
"""

from __future__ import annotations

import json
import types

from jevkit import cli
from jevkit import items as IT
from jevkit import perturb
from jevkit.cli import main
from jevkit.items import Item

from ._fixtures import choice, noul

OPS = "option_reverse,option_shuffle,key_relabel"
SEEDS = "0,1,2"


def _unsorted_choice(i: str) -> Item:
    """Base keys deliberately NOT in sorted order, so sorted != base != reversed."""
    keys = ["zeta", "alpha", "mid", "beta", "kappa"]
    return Item(id=i, family="fam_choice", state=f"state for {i}.",
                question={"type": "choice", "instructions": "Which option applies?",
                          "criteria": {k: f"Option {k}." for k in keys}},
                labels=list(keys), expected="mid", split="public", tier="standard").validate()


def _base_items() -> list[Item]:
    return [choice("c3", k=3), choice("c4", k=4, gold=2), choice("c6", k=6, gold=5), _unsorted_choice("cu"), noul("n1")]


def _make(tmp_path, items: list[Item]) -> list[dict]:
    ip = tmp_path / "items.jsonl"
    # written without sort_keys (IT.write_items sorts keys, which would itself erase the authored base order)
    ip.write_text("".join(json.dumps(it.to_dict(), ensure_ascii=False) + "\n" for it in items), encoding="utf-8")
    assert [list(it.question["criteria"]) for it in IT.load_items(str(ip))] == \
        [list(it.question["criteria"]) for it in items]
    out = tmp_path / "pert.jsonl"
    assert main(["perturb", "make", "--items", str(ip), "--out", str(out), "--ops", OPS, "--seeds", SEEDS]) == 0
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]


def _violations(rows: list[dict], items: list[Item]) -> list[str]:
    """Every way the on-disk criteria order disagrees with what the operator produced in memory."""
    base = {it.id: list(it.question["criteria"]) for it in items}
    expected = {p.item.id: list(p.item.question["criteria"])
                for p in perturb.make_perturbations(items, OPS.split(","), [int(s) for s in SEEDS.split(",")])}
    bad: list[str] = []
    seen_ops: set[str] = set()
    for r in rows:
        meta = r["perturbation"]
        op, keys, bk = meta["op"], list(r["question"]["criteria"]), base[meta["base_id"]]
        seen_ops.add(op)
        if keys != expected[r["id"]]:
            bad.append(f"{r['id']}: on-disk order {keys} != operator order {expected[r['id']]}")
        if op == "option_reverse" and keys != bk[::-1]:
            bad.append(f"{r['id']}: option_reverse order {keys} is not reversed base {bk[::-1]}")
        if op == "option_shuffle" and len(bk) >= 3 and keys == bk:
            bad.append(f"{r['id']}: option_shuffle left base order {bk}")
        if op == "key_relabel":
            lm = meta["label_map"]
            if set(keys) & set(bk) or sorted(keys) != sorted(f"opt_{i + 1}" for i in range(len(bk))):
                bad.append(f"{r['id']}: key_relabel ids {keys} not renamed to opt_1..opt_{len(bk)}")
            elif keys != [lm[k] for k in bk]:
                bad.append(f"{r['id']}: key_relabel order {keys} != relabelled base order {[lm[k] for k in bk]}")
    if not {"option_reverse", "option_shuffle", "key_relabel"} <= seen_ops:
        bad.append(f"missing operators in output: {seen_ops}")
    return bad


def test_perturb_make_preserves_option_order(tmp_path):
    items = _base_items()
    rows = _make(tmp_path, items)
    # 4 choice items x 3 ops x 3 seeds + noul x option_reverse/option_shuffle x 3 seeds (key_relabel is n/a for noul)
    assert len(rows) == 4 * 3 * 3 + 2 * 3
    assert _violations(rows, items) == []
    # spot-check the headline cases directly
    rev = next(r for r in rows if r["perturbation"]["op"] == "option_reverse" and r["perturbation"]["base_id"] == "c4")
    assert list(rev["question"]["criteria"]) == ["o3", "o2", "o1", "o0"]
    rel = next(r for r in rows if r["perturbation"]["op"] == "key_relabel" and r["perturbation"]["base_id"] == "cu")
    assert rel["id"] != "cu" and not set(rel["question"]["criteria"]) & {"zeta", "alpha", "mid", "beta", "kappa"}
    assert rel["expected"] == rel["perturbation"]["label_map"]["mid"]


def test_checks_fail_on_pre_fix_sort_keys(tmp_path, monkeypatch):
    """Re-create the pre-#245 writer (sort_keys=True) inside cli and confirm the order checks catch it."""
    real = json.dumps

    def sorted_dumps(obj, *args, **kw):
        kw["sort_keys"] = True
        return real(obj, *args, **kw)

    monkeypatch.setattr(cli, "json", types.SimpleNamespace(dumps=sorted_dumps, loads=json.loads))
    items = _base_items()
    rows = _make(tmp_path, items)
    bad = _violations(rows, items)
    assert any("option_reverse" in b and "c4" in b for b in bad), bad
    assert any("option_shuffle left base order" in b for b in bad), bad
    assert any("key_relabel" in b for b in bad), bad
