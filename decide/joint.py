"""Joint (case-level) readout, R4 ``v3j``: score the questions of one request together when the request's own structure
declares a coupling between them. Score-only: the input is the backend's own per-question logits for THIS request and
the request text; never gold, teachers, data fields or other rows.

Why: case-exact benchmarks (BFCL tool selection, SATA-Bench, Home appliances) count a case only when every question in
it is right, so the per-question argmax is not the case argmax whenever the questions are coupled.

The rule (pre-declared 2026-10-08, one rule, no per-benchmark switch) -- a *candidate block*:
  * the request has n >= 2 questions, every one binary with affirmative/negative option ids {"yes", "no"} (choice) or
    {"true", "false"} (noul), and
  * every instruction, minus its last non-empty line (the candidate), is the same non-empty stem.
Such a request asks one question ("mark each candidate tool ...", "which categories can be assigned ...") over a
candidate list, so the answer set is presupposed non-empty: the constraint is ``#affirmative >= 1``. Requests that do
not match are returned unchanged (identity, byte-for-byte logits).

Decodes (both exact, over the cardinality-constrained joint of the independent per-question distributions):
  ``map``   joint MAP under lo <= #yes <= hi (lo = 1, hi = n)            -- the R4 (b) constrained joint argmax
  ``card``  joint MAP under #yes == clamp(round(sum_i p_i(yes)), lo, hi) -- the R4 (c) expected-cardinality decode
Output: per-question log MAX-MARGINALS, log-normalized: m_i(v) = max over feasible x with x_i = v of sum_j log p_j(x_j).
Their per-question argmax is the joint MAP (unique MAP), so ``choice`` = argmax still holds (decide.schema), and they
are proper distributions (ECE stays defined). When the constraint is slack (>= 2 questions already answered yes) they
equal the input distributions exactly; with exactly one yes the MAP is unchanged and only that question sharpens.
"""
from __future__ import annotations

import math
from typing import Any

from .schema import canonical_json, canonical_question_ids

JOINT_RULE_ID = "ssdm.joint-candidate-block.v1"
DECODES = ("map", "card")
AFFIRM = {"choice": ("yes", "no"), "noul": ("true", "false")}


def _stem(instr: Any) -> str | None:
    text = instr if isinstance(instr, str) else canonical_json(instr)
    lines = [ln.rstrip() for ln in text.strip().splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) < 2:
        return None
    stem = "\n".join(lines[:-1]).strip()
    return stem or None


def candidate_block(request: dict[str, Any]) -> list[str] | None:
    """Question ids (canonical order) of the request if it is a candidate block, else None."""
    qs = request.get("questions") or {}
    ids = canonical_question_ids(qs)
    if len(ids) < 2:
        return None
    stems = set()
    for qid in ids:
        q = qs[qid]
        t = q.get("type")
        if t not in AFFIRM:
            return None
        if t == "choice" and set(map(str, (q.get("criteria") or {}))) != set(AFFIRM["choice"]):
            return None
        s = _stem(q.get("instructions"))
        if s is None:
            return None
        stems.add(s)
    return ids if len(stems) == 1 else None


def _lognorm(d: dict[str, float]) -> dict[str, float]:
    m = max(d.values())
    lz = m + math.log(sum(math.exp(v - m) for v in d.values()))
    return {o: v - lz for o, v in d.items()}


def _best_with(deltas: list[float], base: float, lo: int, hi: int, fix: tuple[int, int] | None) -> float:
    """max over k in [lo, hi] of base + sum of the k best deltas, optionally with variable i forced to yes (1)/no (0)."""
    if fix is None:
        free, k0, add = sorted(deltas, reverse=True), 0, 0.0
    else:
        i, v = fix
        free = sorted((d for j, d in enumerate(deltas) if j != i), reverse=True)
        k0, add = (1, deltas[i]) if v else (0, 0.0)
    best, acc = -math.inf, base + add
    for k in range(0, len(free) + 1):
        if k > 0:
            acc += free[k - 1]
        if lo <= k + k0 <= hi:
            best = max(best, acc)
    return best


def cardinality_max_marginals(lp_yes: list[float], lp_no: list[float], lo: int, hi: int) -> list[tuple[float, float]]:
    """Exact log max-marginals (yes, no) per variable under lo <= #yes <= hi; inputs are normalized log-probs."""
    n = len(lp_yes)
    if not 0 <= lo <= hi <= n:
        raise ValueError("infeasible cardinality bounds")
    base = sum(lp_no)
    deltas = [y - z for y, z in zip(lp_yes, lp_no)]
    return [(_best_with(deltas, base, lo, hi, (i, 1)), _best_with(deltas, base, lo, hi, (i, 0))) for i in range(n)]


def joint_logits(request: dict[str, Any], logits: dict[str, dict[str, float]], decode: str = "map"
                 ) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """(logits, diagnostics). Identity (the same dict objects) unless the request is a candidate block."""
    if decode not in DECODES:
        raise ValueError(f"decode must be one of {DECODES}")
    ids = candidate_block(request)
    if ids is None:
        return logits, {"rule": JOINT_RULE_ID, "applied": False}
    pos, neg = [], []
    for qid in ids:
        a, b = AFFIRM[request["questions"][qid]["type"]]
        lp = _lognorm({a: float(logits[qid][a]), b: float(logits[qid][b])})
        pos.append(lp[a])
        neg.append(lp[b])
    n = len(ids)
    lo, hi = 1, n
    if decode == "card":
        k = int(math.floor(sum(math.exp(v) for v in pos) + 0.5))
        lo = hi = min(max(k, 1), n)
    mm = cardinality_max_marginals(pos, neg, lo, hi)
    out = dict(logits)
    changed = []
    for qid, (my, mn), y, z in zip(ids, mm, pos, neg):
        a, b = AFFIRM[request["questions"][qid]["type"]]
        out[qid] = _lognorm({a: my, b: mn})
        if (my > mn) != (y > z):
            changed.append(qid)
    return out, {"rule": JOINT_RULE_ID, "applied": True, "decode": decode, "n": n, "bounds": [lo, hi],
                 "changed": changed}
