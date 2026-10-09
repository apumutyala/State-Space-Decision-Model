"""Robustness battery: 8 deterministic perturbation operators, flip rate and probability shift.

MD1's announcement discloses its perturbation FAMILIES (state paraphrase, instruction paraphrase, option description
paraphrase, option reorder / reverse / shuffle, key relabel, formatting noise) and a 1.3 % average flip rate with 0
reorder flips, but not its exact operators. These 8 are ours, chosen to cover the same families, and they are pure
functions of (item, seed): no model, no network, byte-identical on every machine.

==  =====================  ===========================================================================================
#   operator               what changes (meaning-preserving by construction)
==  =====================  ===========================================================================================
1   state_paraphrase       contraction / connective rewrite table over every string leaf of the state
2   instruction_paraphrase same table over the instructions + one of three neutral task frames ("Task: ...")
3   option_paraphrase      same table over option / level / noul descriptions + sentence-final period toggled
4   option_reverse         choice options listed in reverse order (noul: true/false order swapped; score: n/a)
5   option_shuffle         choice options listed in a seeded random order (score: n/a)
6   key_relabel            choice option keys renamed to neutral ids opt_1..opt_k in a seeded order (score/noul: n/a)
7   whitespace_noise       runs of spaces doubled, a leading and a trailing newline added to the state
8   markup_noise           string state quoted as a Markdown block ("> " per line); dict state with keys reversed
==  =====================  ===========================================================================================

An operator that does not apply to an item (reordering a 1-option set, reordering ordinal levels) returns None and
the item is reported as ``n/a`` for that operator, never as a non-flip.

Flip: the decision changes (choice / score argmax, noul argmax of {no, yes} - i.e. P(yes) crossing 0.5), after the
perturbed answer is mapped back through the operator's ``label_map``. Shift: total-variation distance and max |dp|
between the base and perturbed distributions. Invalid answers on either side are counted separately.
"""

from __future__ import annotations

import copy
import hashlib
import random
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .answers import argmax_label, parse_answer
from .items import Item

#: meaning-preserving rewrites, applied left-to-right with word boundaries, case of the first letter kept
_PARA = [
    ("do not", "don't"), ("does not", "doesn't"), ("did not", "didn't"), ("is not", "isn't"), ("are not", "aren't"),
    ("was not", "wasn't"), ("were not", "weren't"), ("cannot", "can't"), ("will not", "won't"),
    ("should not", "shouldn't"), ("would not", "wouldn't"), ("has not", "hasn't"), ("have not", "haven't"),
    ("it is", "it's"), ("that is", "that's"), ("there is", "there's"), ("we are", "we're"), ("they are", "they're"),
    ("e.g.", "for example"), ("i.e.", "that is,"), ("approximately", "about"), ("in order to", "to"),
]
_FRAMES = ["Task: {}", "Decide the following. {}", "{} Answer from the options given."]


def _rewrite(text: str, direction: int) -> str:
    """direction 0: expanded -> contracted (+ connectives); 1: contracted -> expanded."""
    pairs = _PARA if direction == 0 else [(b, a) for a, b in _PARA if b not in ("for example", "that is,", "about", "to")]
    out = text
    for a, b in pairs:
        pat = re.compile(r"(?<![\w'])" + re.escape(a) + r"(?![\w'])", re.IGNORECASE)

        def rep(m: re.Match, b=b) -> str:
            s = m.group(0)
            return (b[0].upper() + b[1:]) if s[:1].isupper() else b

        out = pat.sub(rep, out)
    return out


def _paraphrase(text: str, seed: int) -> str:
    """Try the seed's direction first; fall back to the other if it changes nothing."""
    d = seed % 2
    out = _rewrite(text, d)
    return out if out != text else _rewrite(text, 1 - d)


def _map_leaves(obj: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(obj, str):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _map_leaves(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_map_leaves(v, fn) for v in obj]
    return obj


def _rng(item: Item, op: str, seed: int) -> random.Random:
    h = hashlib.sha256(f"{item.id}\x1f{op}\x1f{seed}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


@dataclass
class Perturbed:
    item: Item
    op: str
    seed: int
    base_id: str
    label_map: dict[str, str] = field(default_factory=dict)  # base label -> perturbed label

    def to_dict(self) -> dict:
        d = self.item.to_dict()
        d["perturbation"] = {"op": self.op, "seed": self.seed, "base_id": self.base_id, "label_map": self.label_map}
        return d


def _clone(item: Item, op: str, seed: int, **changes: Any) -> Item:
    it = copy.deepcopy(item)
    for k, v in changes.items():
        setattr(it, k, v)
    it.id = f"{item.id}~{op}~{seed}"
    return it.validate()


def _identity_map(item: Item) -> dict[str, str]:
    return {lab: lab for lab in item.labels}


def op_state_paraphrase(item: Item, seed: int) -> Perturbed | None:
    st = _map_leaves(item.state, lambda s: _paraphrase(s, seed))
    if st == item.state:
        return None
    return Perturbed(_clone(item, "state_paraphrase", seed, state=st), "state_paraphrase", seed, item.id, _identity_map(item))


def op_instruction_paraphrase(item: Item, seed: int) -> Perturbed | None:
    frame = _FRAMES[_rng(item, "instruction_paraphrase", seed).randrange(len(_FRAMES))]
    q = copy.deepcopy(item.question)
    ins = q.get("instructions")
    if isinstance(ins, str):
        q["instructions"] = frame.format(_paraphrase(ins, seed))
    else:
        q["instructions"] = _map_leaves(ins, lambda s: _paraphrase(s, seed))
    if q == item.question:
        return None
    return Perturbed(_clone(item, "instruction_paraphrase", seed, question=q), "instruction_paraphrase", seed, item.id,
                     _identity_map(item))


def _toggle_period(s: str) -> str:
    s2 = s.rstrip()
    return s2[:-1] if s2.endswith(".") else s2 + "."


def op_option_paraphrase(item: Item, seed: int) -> Perturbed | None:
    q = copy.deepcopy(item.question)
    crit = q.get("criteria")
    fn = lambda s: _toggle_period(_paraphrase(s, seed))
    if isinstance(crit, dict):
        q["criteria"] = {k: (_map_leaves(v, fn) if v is not None else v) for k, v in crit.items()}
    elif isinstance(crit, list):
        q["criteria"] = [_map_leaves(v, fn) for v in crit]
    else:
        return None
    if q == item.question:
        return None
    return Perturbed(_clone(item, "option_paraphrase", seed, question=q), "option_paraphrase", seed, item.id,
                     _identity_map(item))


def _reorder(item: Item, op: str, seed: int, order_fn: Callable[[list], list]) -> Perturbed | None:
    q = copy.deepcopy(item.question)
    crit = q.get("criteria")
    if item.type == "score" or not isinstance(crit, dict) or len(crit) < 2:
        return None
    keys = list(crit)
    new = order_fn(keys)
    if new == keys:
        new = keys[1:] + keys[:1]  # a seeded shuffle that drew the identity: rotate instead (always a real reorder)
    q["criteria"] = {k: crit[k] for k in new}
    return Perturbed(_clone(item, op, seed, question=q, labels=list(item.labels)), op, seed, item.id, _identity_map(item))


def op_option_reverse(item: Item, seed: int) -> Perturbed | None:
    return _reorder(item, "option_reverse", seed, lambda ks: ks[::-1])


def op_option_shuffle(item: Item, seed: int) -> Perturbed | None:
    def sh(ks: list) -> list:
        ks = list(ks)
        _rng(item, "option_shuffle", seed).shuffle(ks)
        return ks

    return _reorder(item, "option_shuffle", seed, sh)


def op_key_relabel(item: Item, seed: int) -> Perturbed | None:
    if item.type != "choice":
        return None
    crit = item.question["criteria"]
    keys = list(crit)
    ids = [f"opt_{i + 1}" for i in range(len(keys))]
    _rng(item, "key_relabel", seed).shuffle(ids)
    lm = dict(zip(keys, ids))
    q = copy.deepcopy(item.question)
    q["criteria"] = {lm[k]: crit[k] for k in keys}
    exp = lm[item.expected] if item.expected is not None else None
    it = _clone(item, "key_relabel", seed, question=q, labels=[lm[k] for k in item.labels], expected=exp)
    return Perturbed(it, "key_relabel", seed, item.id, lm)


def op_whitespace_noise(item: Item, seed: int) -> Perturbed | None:
    fn = lambda s: "\n" + re.sub(r" +", lambda m: m.group(0) * 2, s) + "\n"
    st = _map_leaves(item.state, fn)
    if st == item.state:
        return None
    return Perturbed(_clone(item, "whitespace_noise", seed, state=st), "whitespace_noise", seed, item.id, _identity_map(item))


def op_markup_noise(item: Item, seed: int) -> Perturbed | None:
    if isinstance(item.state, str):
        st: Any = "\n".join("> " + ln for ln in item.state.split("\n"))
    elif isinstance(item.state, dict) and len(item.state) > 1:
        st = {k: item.state[k] for k in reversed(list(item.state))}
    else:
        return None
    return Perturbed(_clone(item, "markup_noise", seed, state=st), "markup_noise", seed, item.id, _identity_map(item))


OPERATORS: dict[str, Callable[[Item, int], Perturbed | None]] = {
    "state_paraphrase": op_state_paraphrase,
    "instruction_paraphrase": op_instruction_paraphrase,
    "option_paraphrase": op_option_paraphrase,
    "option_reverse": op_option_reverse,
    "option_shuffle": op_option_shuffle,
    "key_relabel": op_key_relabel,
    "whitespace_noise": op_whitespace_noise,
    "markup_noise": op_markup_noise,
}
REORDER_OPS = ("option_reverse", "option_shuffle", "key_relabel")


def make_perturbations(items: Iterable[Item], ops: Iterable[str] | None = None, seeds: Iterable[int] = (0,)) -> list[Perturbed]:
    out = []
    for it in items:
        for op in (ops or OPERATORS):
            for sd in seeds:
                p = OPERATORS[op](it, sd)
                if p is not None:
                    out.append(p)
    return out


# ------------------------------------------------------------------------------------------- scoring
def _dist(item: Item, answer: Any) -> dict[str, float] | None:
    p = parse_answer(item, answer)
    if not p.valid:
        return None
    if p.probs is not None:
        return p.probs
    return {lab: (1.0 if lab == p.predicted else 0.0) for lab in item.labels}  # label-only: a point mass


def _decision(item: Item, d: dict[str, float]) -> str:
    return argmax_label(d)


def flip_report(base_items: Iterable[Item], perturbed: Iterable[Perturbed], base_answers: dict[str, Any],
                pert_answers: dict[str, Any]) -> dict:
    """Per operator and per (operator, type): n, n/a, invalid, flips, flip rate, mean / max TV shift, max |dp|."""
    base = {it.id: it for it in base_items}
    rows: dict[tuple[str, str], dict] = {}
    for p in perturbed:
        b = base[p.base_id]
        key = (p.op, b.type)
        r = rows.setdefault(key, {"n": 0, "invalid": 0, "flips": 0, "tv": [], "maxdp": 0.0})
        db, dp_ = _dist(b, base_answers.get(b.id)), _dist(p.item, pert_answers.get(p.item.id))
        if db is None or dp_ is None:
            r["invalid"] += 1
            continue
        back = {lab: dp_[p.label_map[lab]] for lab in b.labels}  # perturbed distribution in base label space
        r["n"] += 1
        r["flips"] += int(_decision(b, db) != _decision(b, back))
        diffs = [abs(db[lab] - back[lab]) for lab in b.labels]
        r["tv"].append(0.5 * sum(diffs))
        r["maxdp"] = max(r["maxdp"], max(diffs))

    def fin(rs: list[dict]) -> dict:
        n = sum(r["n"] for r in rs)
        tv = [x for r in rs for x in r["tv"]]
        return {"n": n, "invalid": sum(r["invalid"] for r in rs), "flips": sum(r["flips"] for r in rs),
                "flip_rate": (sum(r["flips"] for r in rs) / n) if n else None,
                "mean_tv_shift": (sum(tv) / len(tv)) if tv else None, "max_tv_shift": max(tv) if tv else None,
                "max_abs_dp": max((r["maxdp"] for r in rs), default=None) if n else None}

    per_op = {op: fin([r for (o, _), r in rows.items() if o == op]) for op in sorted({o for o, _ in rows})}
    per_op_type = {f"{o}|{t}": fin([r]) for (o, t), r in sorted(rows.items())}
    rates = [v["flip_rate"] for v in per_op.values() if v["flip_rate"] is not None]
    reorder = [v for o, v in per_op.items() if o in REORDER_OPS]
    return {"per_operator": per_op, "per_operator_type": per_op_type,
            "average_flip_rate": (sum(rates) / len(rates)) if rates else None,
            "pooled": fin(list(rows.values())),
            "reorder_flips": sum(v["flips"] for v in reorder),
            "note": "operators are jevkit's own; MD1's exact 8 operators are unpublished"}


def run_battery(items: list[Item], answer_fn: Callable[[Item], Any], ops: Iterable[str] | None = None,
                seeds: Iterable[int] = (0,)) -> dict:
    """In-process battery: answer_fn(item) -> SystemOne answer for that item's single question."""
    perturbed = make_perturbations(items, ops, seeds)
    base_ans = {it.id: answer_fn(it) for it in items}
    pert_ans = {p.item.id: answer_fn(p.item) for p in perturbed}
    return flip_report(items, perturbed, base_ans, pert_ans)
