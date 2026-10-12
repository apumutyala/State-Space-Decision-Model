"""Property-test harness for reorder / relabel invariance (docs/guarantees/REORDER_INVARIANCE.md).

Plug a real head in as ``score_fn(state, options) -> probs`` where ``options`` is a list of ``(id, content)`` and
``probs`` is a sequence aligned to ``options`` or a dict keyed by id. ``check_reorder`` / ``check_relabel`` return
measured ``max_abs_dp`` and flips; ``assert_invariant`` turns them into a pass/fail at tolerance ``tol``.

Optional: ``SSDM_INV_HEAD=package.module:function`` makes ``test_plugged_head`` import that score_fn and check it.
Pure Python, no torch, runs in well under a second.
"""
from __future__ import annotations

import hashlib
import importlib
import itertools
import math
import os
import random
import re
from typing import Any, Callable, Sequence

import pytest

Option = tuple[str, str]
ScoreFn = Callable[[Any, Sequence[Option]], Any]


# ------------------------------------------------------------------------------------------------ harness
def _per_id(options: Sequence[Option], probs: Any) -> dict[str, float]:
    if isinstance(probs, dict):
        out = {str(k): float(v) for k, v in probs.items()}
        if set(out) != {o[0] for o in options}:
            raise ValueError("score_fn dict keys must be exactly the option ids")
        return out
    probs = list(probs)
    if len(probs) != len(options):
        raise ValueError("score_fn must return one probability per option")
    return {o[0]: float(p) for o, p in zip(options, probs)}


def _argmax_set(d: dict[str, float], content: dict[str, str], margin: float) -> frozenset[str]:
    """Contents (not ids, not positions) whose probability is within ``margin`` of the max: the order-free decision."""
    m = max(d.values())
    return frozenset(content[k] for k, v in d.items() if v >= m - margin)


def _orders(k: int, n_perms: int, rng: random.Random) -> list[list[int]]:
    ident = list(range(k))
    if math.factorial(k) - 1 <= n_perms:
        return [list(p) for p in itertools.permutations(ident) if list(p) != ident]
    out, seen = [ident[::-1]], {tuple(ident), tuple(ident[::-1])}  # always include the reverse
    while len(out) < n_perms:
        p = ident[:]
        rng.shuffle(p)
        if tuple(p) not in seen:
            seen.add(tuple(p))
            out.append(p)
    return out


def check_reorder(score_fn: ScoreFn, state: Any, options: Sequence[Option], n_perms: int = 24, seed: int = 0,
                  tie_margin: float = 0.0) -> dict:
    """All K! orders when that is <= n_perms + 1, else the reverse plus seeded shuffles. Compares per-id probs."""
    options = list(options)
    content = dict(options)
    base = _per_id(options, score_fn(state, options))
    dec = _argmax_set(base, content, tie_margin)
    maxdp, flips, orders = 0.0, 0, _orders(len(options), n_perms, random.Random(f"reorder|{seed}"))
    for perm in orders:
        o2 = [options[i] for i in perm]
        d = _per_id(o2, score_fn(state, o2))
        maxdp = max(maxdp, max(abs(d[k] - base[k]) for k in base))
        flips += _argmax_set(d, content, tie_margin) != dec
    return {"kind": "reorder", "n_checked": len(orders), "max_abs_dp": maxdp, "flips": flips,
            "argmax_equal": flips == 0}


def check_relabel(score_fn: ScoreFn, state: Any, options: Sequence[Option], n_relabels: int = 8, seed: int = 0,
                  tie_margin: float = 0.0, also_reorder: bool = True) -> dict:
    """Fresh neutral ids (a bijection sigma), optionally composed with a shuffle; maps back through sigma."""
    options = list(options)
    content = dict(options)
    base = _per_id(options, score_fn(state, options))
    dec = _argmax_set(base, content, tie_margin)
    rng = random.Random(f"relabel|{seed}")
    maxdp, flips = 0.0, 0
    for r in range(n_relabels):
        fresh = [f"x{rng.randrange(10 ** 6):06d}_{i}" for i in range(len(options))]
        rng.shuffle(fresh)
        sigma = {o[0]: fresh[i] for i, o in enumerate(options)}
        o2 = [(sigma[i], c) for i, c in options]
        if also_reorder:
            rng.shuffle(o2)
        d2 = _per_id(o2, score_fn(state, o2))
        back = {k: d2[sigma[k]] for k in base}
        maxdp = max(maxdp, max(abs(back[k] - base[k]) for k in base))
        flips += _argmax_set(back, content, tie_margin) != dec
    return {"kind": "relabel", "n_checked": n_relabels, "max_abs_dp": maxdp, "flips": flips,
            "argmax_equal": flips == 0}


def fp_tolerance(k: int, delta_logit: float = 0.0, unit_roundoff: float = 2.0 ** -24) -> float:
    """REORDER_INVARIANCE.md section 4: tol = delta/2 + gamma_{2K}; fp32 unit roundoff by default. (Two computed
    softmaxes over differently ordered sums differ by a ratio of 2K rounding factors, not K + 2.)"""
    n = 2 * k
    return delta_logit / 2 + n * unit_roundoff / (1 - n * unit_roundoff)


def assert_invariant(score_fn: ScoreFn, state: Any, options: Sequence[Option], tol: float | None = None,
                     relabel: bool = True, **kw: Any) -> list[dict]:
    tol = fp_tolerance(len(options)) if tol is None else tol
    reps = [check_reorder(score_fn, state, options, **kw)]
    if relabel:
        reps.append(check_relabel(score_fn, state, options, **kw))
    for r in reps:
        assert r["argmax_equal"] and r["max_abs_dp"] <= tol, r
    return reps


# ------------------------------------------------------------------------------------------------ toy heads
def _feat(text: str, dim: int = 8) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    return [(h[i] - 127.5) / 64.0 for i in range(dim)]


def _softmax(z: Sequence[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = math.fsum(e)  # exactly rounded: order-free normaliser
    return [v / s for v in e]


def equivariant_head(state: str, options: Sequence[Option]) -> list[float]:
    """z_i = f(state, content_i): no position, no id, softmax over the set. Satisfies C1-C3."""
    s = _feat("state|" + state)
    return _softmax([sum(a * b for a, b in zip(s, _feat("opt|" + c))) for _id, c in options])


def position_head(state: str, options: Sequence[Option]) -> list[float]:
    """Adds a per-slot bias (G16's u_k / beta_k pattern): violates C2 for reorder."""
    s = _feat("state|" + state)
    return _softmax([sum(a * b for a, b in zip(s, _feat("opt|" + c))) + 0.7 * k for k, (_id, c) in enumerate(options)])


def id_head(state: str, options: Sequence[Option]) -> list[float]:
    """Reads the id (a letter-logit readout): reorder-invariant, NOT relabel-invariant (violates C2 for ids)."""
    s = _feat("state|" + state)
    return _softmax([sum(a * b for a, b in zip(s, _feat("opt|" + i + "|" + c))) for i, c in options])


def prefix_recurrence_head(state: str, options: Sequence[Option]) -> list[float]:
    """Shared prefix reads the option list in caller order (a recurrence): violates C1."""
    h = _feat("state|" + state)
    for _id, c in options:
        f = _feat("opt|" + c)
        h = [math.tanh(0.6 * a + 0.8 * b) for a, b in zip(h, f)]
    return _softmax([sum(a * b for a, b in zip(h, _feat("opt|" + c))) for _id, c in options])


def canonical_prefix_head(state: str, options: Sequence[Option]) -> list[float]:
    """Same recurrence, but over the options sorted by CONTENT: the C1 repair. Invariant to reorder and relabel."""
    h = _feat("state|" + state)
    for c in sorted(c for _id, c in options):
        f = _feat("opt|" + c)
        h = [math.tanh(0.6 * a + 0.8 * b) for a, b in zip(h, f)]
    return _softmax([sum(a * b for a, b in zip(h, _feat("opt|" + c))) for _id, c in options])


def id_sorted_prefix_head(state: str, options: Sequence[Option]) -> list[float]:
    """Prefix sorted by ID (what the four evaluated models do): reorder-invariant, relabel breaks it."""
    h = _feat("state|" + state)
    for _id, c in sorted(options):
        f = _feat("opt|" + c)
        h = [math.tanh(0.6 * a + 0.8 * b) for a, b in zip(h, f)]
    return _softmax([sum(a * b for a, b in zip(h, _feat("opt|" + c))) for _id, c in options])


def naive_sum_head(state: str, options: Sequence[Option]) -> list[float]:
    """Equivariant scores, but the normaliser is a plain left-to-right float sum (non-associative)."""
    s = _feat("state|" + state)
    z = [sum(a * b for a, b in zip(s, _feat("opt|" + c))) * 3.0 for _id, c in options]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    tot = 0.0
    for v in e:
        tot += v
    return [v / tot for v in e]


def _cases(n: int = 12, seed: int = 0) -> list[tuple[str, list[Option]]]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        k = rng.choice([2, 3, 4, 5, 7, 9])
        out.append((f"state-{i}", [(f"opt_{j + 1}", f"content-{i}-{rng.randrange(10 ** 6)}") for j in range(k)]))
    return out


# ------------------------------------------------------------------------------------------------ tests
@pytest.mark.parametrize("head", [equivariant_head, canonical_prefix_head])
def test_invariant_heads_pass_reorder_and_relabel_exactly(head):
    for state, opts in _cases():
        for r in assert_invariant(head, state, opts, tol=0.0):
            assert r["max_abs_dp"] == 0.0 and r["flips"] == 0  # bitwise: same ops per option + fsum normaliser


@pytest.mark.parametrize("head", [position_head, prefix_recurrence_head])
def test_order_dependent_heads_fail_reorder(head):
    reps = [check_reorder(head, s, o) for s, o in _cases()]
    assert max(r["max_abs_dp"] for r in reps) > 1e-3
    assert sum(r["flips"] for r in reps) > 0
    with pytest.raises(AssertionError):
        for s, o in _cases():
            assert_invariant(head, s, o, relabel=False)


@pytest.mark.parametrize("head", [id_head, id_sorted_prefix_head])
def test_id_dependent_heads_pass_reorder_but_fail_relabel(head):
    for s, o in _cases():
        r = check_reorder(head, s, o)
        assert r["max_abs_dp"] == 0.0 and r["flips"] == 0
    rel = [check_relabel(head, s, o) for s, o in _cases()]
    assert max(r["max_abs_dp"] for r in rel) > 1e-3
    assert sum(r["flips"] for r in rel) > 0


def test_float_nonassociativity_is_within_the_stated_tolerance():
    worst = 0.0
    for s, o in _cases(30, seed=1):
        # Python floats are fp64: check against gamma_{2K} at the fp64 unit roundoff, the arithmetic actually used
        # (the fp32 default is ~1e9 times looser and could not fail)
        for r in assert_invariant(naive_sum_head, s, o, tol=fp_tolerance(len(o), unit_roundoff=2.0 ** -53)):
            worst = max(worst, r["max_abs_dp"])
            assert r["flips"] == 0
    assert worst > 0.0  # non-vacuous: the left-to-right sum really is order-dependent on these cases
    assert worst <= fp_tolerance(9, unit_roundoff=2.0 ** -53)


def test_all_k_factorial_orders_for_small_k():
    _s, opts = "s", [(f"o{i}", f"c{i}") for i in range(4)]
    assert check_reorder(equivariant_head, _s, opts)["n_checked"] == math.factorial(4) - 1


def test_ties_are_decided_by_content_not_position():
    # duplicate content: equal scores; the argmax SET (by content) is order-free, so no spurious flip
    opts = [("a", "same"), ("b", "same"), ("c", "other")]
    r = check_reorder(equivariant_head, "s", opts)
    assert r["flips"] == 0 and r["max_abs_dp"] == 0.0


def test_harness_validates_score_fn_output():
    with pytest.raises(ValueError):
        check_reorder(lambda s, o: [1.0], "s", [("a", "x"), ("b", "y")])
    with pytest.raises(ValueError):
        check_reorder(lambda s, o: {"zz": 1.0}, "s", [("a", "x"), ("b", "y")])
    r = check_reorder(lambda s, o: {i: (1.0 if c == "x" else 0.0) for i, c in o}, "s", [("a", "x"), ("b", "y")])
    assert r["max_abs_dp"] == 0.0


# ------------------------------------------------------------------------------------------------ typed fixture
# REORDER_INVARIANCE.md section 2T. An ORDINAL/SCORE question rendered with the level value as option CONTENT
# ("level 2: <rubric>"), behind handles. Handles here are jevkit-style "0".."K-1": they coincide with the level value as
# strings, which is exactly what makes a handle-reading head tempting. Symmetry (ii) permutes the handles among the
# options with content fixed; symmetry (i) permutes the display order; (iii) renames the level VALUE (not a symmetry).
_LEVEL = re.compile(r"^level (\d+): (.*)$", re.S)


def _norm(text: str) -> str:
    """(C5) whitespace normalisation: the content the head reads is a fixed function of the option text alone."""
    return " ".join(text.split())


def render_level(value: int, rubric: str) -> str:
    return f"level {value}: {rubric}"


def _target(state: str) -> int:
    return int(re.search(r"target=(\d+)", state).group(1))


def _ordinal_score(state: str, value: int, rubric: str) -> float:
    # the level value dominates (1.5 per squared step); the rubric adds a small content term (|.| <= 0.32)
    s = _feat("state|" + state)
    return -1.5 * (value - _target(state)) ** 2 + 0.01 * sum(a * b for a, b in zip(s, _feat("rubric|" + rubric)))


def _displayed(options: Sequence[Option]) -> list[str]:
    """Render the option list as a prompt would: a separator after every option except the last displayed one (a
    position leak through the raw text that (C5) removes)."""
    return [c + ("\n" if k < len(options) - 1 else "") for k, (_h, c) in enumerate(options)]


def ordinal_content_head(state: str, options: Sequence[Option]) -> list[float]:
    """Typed rendering: level value and rubric both read from the normalised CONTENT; the handle is never read."""
    z = []
    for c in _displayed(options):
        m = _LEVEL.match(_norm(c))
        z.append(_ordinal_score(state, int(m.group(1)), m.group(2)))
    return _softmax(z)


def ordinal_unnormalised_head(state: str, options: Sequence[Option]) -> list[float]:
    """Same, but reads the raw displayed text (no C5): the trailing separator differs by position."""
    s = _feat("state|" + state)
    z = []
    for c in _displayed(options):
        m = _LEVEL.match(c)
        z.append(_ordinal_score(state, int(m.group(1)), m.group(2)) + 0.01 * sum(a * b for a, b in zip(s, _feat(c))))
    return _softmax(z)


def ordinal_handle_head(state: str, options: Sequence[Option]) -> list[float]:
    """Reads the level value THROUGH THE HANDLE (int(handle)): display-order invariant, not handle-relabel invariant."""
    z = []
    for h, c in options:
        m = _LEVEL.match(_norm(c))
        z.append(_ordinal_score(state, int(h), m.group(2)))
    return _softmax(z)


def ordinal_value_dropping_head(state: str, options: Sequence[Option]) -> list[float]:
    """Id-agnostic AND level-agnostic: reads only the rubric text. Invariant under (i) and (ii), but it has thrown the
    level value away (the JEV3-SYM seed 0 failure mode: an accuracy cost, not an invariance failure)."""
    s = _feat("state|" + state)
    z = []
    for c in _displayed(options):
        m = _LEVEL.match(_norm(c))
        z.append(0.01 * sum(a * b for a, b in zip(s, _feat("rubric|" + m.group(2)))))
    return _softmax(z)


def _ordinal_cases(n: int = 40, seed: int = 7) -> list[tuple[str, list[Option]]]:
    """Rubric texts are deliberately uninformative about the level; the state names the target level."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        k = rng.choice([3, 4, 5])
        t = rng.randrange(k)
        rubrics = [f"rubric-{i}-{rng.randrange(10 ** 6)}  text" for _ in range(k)]  # double space: C5 matters
        out.append((f"ordinal-{i} target={t}", [(str(v), render_level(v, rubrics[v])) for v in range(k)]))
    return out


def check_handle_permutation(score_fn: ScoreFn, state: Any, options: Sequence[Option], n_relabels: int = 8,
                             seed: int = 0, tie_margin: float = 0.0) -> dict:
    """Symmetry (ii) with the most adversarial sigma: permute the EXISTING handles among the options (content fixed),
    so the new handles still look like level values. Maps back through sigma and compares per-content decisions."""
    options = list(options)
    content = dict(options)
    base = _per_id(options, score_fn(state, options))
    dec = _argmax_set(base, content, tie_margin)
    rng = random.Random(f"handleperm|{seed}")
    handles = [h for h, _c in options]
    maxdp, flips, n = 0.0, 0, 0
    for _ in range(n_relabels):
        perm = handles[:]
        while perm == handles:
            rng.shuffle(perm)
        sigma = dict(zip(handles, perm))
        o2 = [(sigma[h], c) for h, c in options]
        d2 = _per_id(o2, score_fn(state, o2))
        back = {h: d2[sigma[h]] for h in base}
        maxdp = max(maxdp, max(abs(back[h] - base[h]) for h in base))
        flips += _argmax_set(back, content, tie_margin) != dec
        n += 1
    return {"kind": "handle_permutation", "n_checked": n, "max_abs_dp": maxdp, "flips": flips,
            "argmax_equal": flips == 0}


def test_typed_ordinal_level_as_content_is_bitwise_invariant_under_display_and_handle_symmetries():
    for s, o in _ordinal_cases():
        for r in assert_invariant(ordinal_content_head, s, o, tol=0.0):  # (i) all orders / shuffles; (ii) fresh ids
            assert r["max_abs_dp"] == 0.0 and r["flips"] == 0
        r = check_handle_permutation(ordinal_content_head, s, o)  # (ii) handles permuted among themselves
        assert r["max_abs_dp"] == 0.0 and r["flips"] == 0


def test_typed_ordinal_handle_reading_head_passes_display_order_but_fails_relabel():
    for s, o in _ordinal_cases():
        r = check_reorder(ordinal_handle_head, s, o)
        assert r["max_abs_dp"] == 0.0 and r["flips"] == 0
    rel = [check_handle_permutation(ordinal_handle_head, s, o) for s, o in _ordinal_cases()]
    assert max(r["max_abs_dp"] for r in rel) > 0.5  # the level value moved with the handle
    assert sum(r["flips"] for r in rel) > 0


def test_typed_ordinal_needs_whitespace_normalisation():
    reps = [check_reorder(ordinal_unnormalised_head, s, o) for s, o in _ordinal_cases()]
    assert max(r["max_abs_dp"] for r in reps) > 0.0  # the last-displayed option's text differs: C5 violated


def test_level_value_renaming_is_not_a_symmetry():
    # (iii) tau swaps level values 0 <-> target in the CONTENT (rubric and handle kept): a different question
    changed = 0
    for s, o in _ordinal_cases():
        t = _target(s)
        if t == 0:
            continue
        tau = {v: v for v in range(len(o))}
        tau[0], tau[t] = t, 0
        o2 = [(h, render_level(tau[int(h)], _LEVEL.match(c).group(2))) for h, c in o]
        p, p2 = _per_id(o, ordinal_content_head(s, o)), _per_id(o2, ordinal_content_head(s, o2))
        changed += max(p, key=p.get) != max(p2, key=p2.get)
    assert changed > 0  # the head follows the level value, as it must: invariance under tau would be a defect


def test_dropping_level_values_keeps_invariance_but_costs_accuracy():
    cases = _ordinal_cases()
    for s, o in cases:
        assert_invariant(ordinal_value_dropping_head, s, o, tol=0.0)
        assert check_handle_permutation(ordinal_value_dropping_head, s, o)["flips"] == 0

    def acc(head):
        hits = 0
        for s, o in cases:
            p = _per_id(o, head(s, o))
            hits += _LEVEL.match(_norm(dict(o)[max(p, key=p.get)])).group(1) == str(_target(s))
        return hits / len(cases)

    assert acc(ordinal_content_head) == 1.0
    assert acc(ordinal_value_dropping_head) <= 0.6  # near chance (1/3 to 1/5): the information is gone


@pytest.mark.skipif(not os.environ.get("SSDM_INV_HEAD"), reason="set SSDM_INV_HEAD=module:function to check a head")
def test_plugged_head():
    mod, fn = os.environ["SSDM_INV_HEAD"].split(":")
    head = getattr(importlib.import_module(mod), fn)
    tol = float(os.environ.get("SSDM_INV_TOL", "0"))
    for s, o in _cases():
        assert_invariant(head, s, o, tol=tol)
