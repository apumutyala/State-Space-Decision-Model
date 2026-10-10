"""Per-family LABEL-COLLAPSE detection: a NON-GATING warning (never changes a score, a verdict or a gate).

Motivation: D10w predicts RAGTruth "hallucinated" on 0% of rows (gold 42%); the Index alone does not say so.

Rule (pre-declared, 2026-10-08):
  * group = one Index family (benchmark catalog_id) x one question type with a fixed label set: a ``choice`` question's
    label set is its criteria keys, a ``noul`` question's is {true, false} (predicted ``noul >= 0.5`` = true, exactly as
    ``harness.score._pred``); questions whose criteria keys differ are different groups (per-row label spaces such as
    hashed keys or chess moves therefore form tiny groups and are never flagged);
  * counted questions = questions of the rows scored (rows that have a result) whose gold is not None and whose result
    is ``status == "ok"`` with an answer for that question (a refused row has no predicted class);
  * for every label: gold share and predicted share over the group's counted questions;
  * COLLAPSE if a label with gold share in [10%, 90%] has a predicted share of exactly 0% or exactly 100%;
    NEAR-COLLAPSE if instead its predicted share is < 2% or > 98%;
  * only groups with >= 20 counted questions are flagged.
Gold labels come only from the scorer's row metadata (``row["questions"]``, ``row["expected"]``), read-only.
"""
from __future__ import annotations

from typing import Any, Iterable

GOLD_LO, GOLD_HI = 0.10, 0.90
NEAR_LO, NEAR_HI = 0.02, 0.98
MIN_N = 20
RULE = ("per (benchmark, question type, fixed label set): COLLAPSE if a label with gold share in [10%, 90%] has "
        "predicted share exactly 0% or 100%; NEAR-COLLAPSE if < 2% or > 98%; >= 20 counted questions; counted = "
        "gold not None and result status ok; NON-GATING (never changes a score or a gate)")


def _names() -> dict[int, str]:
    try:
        import json

        from harness import config

        d = json.loads((config.MANIFEST_DIR / "families-v1.json").read_text(encoding="utf-8"))
        return {int(k): v.get("name", str(k)) for k, v in d["benchmarks"].items()}
    except Exception:  # noqa: BLE001 - a name is cosmetic
        return {}


def _label(qtype: str, value: Any) -> str | None:
    if value is None:
        return None
    if qtype == "noul":
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return "true" if value >= 0.5 else "false"
        return str(value).lower()
    return str(value)


def tally(rows: Iterable[dict], results: dict[str, dict]) -> dict[tuple, dict]:
    """{(catalog_id, qtype, labels): {"labels", "gold": {label: n}, "pred": {label: n}, "n"}}."""
    groups: dict[tuple, dict] = {}
    for r in rows:
        ev = r["_evaluation"]
        res = results.get(ev["run_id"])
        if not isinstance(res, dict) or res.get("status") != "ok":
            continue
        answers = ((res.get("response") or {}).get("answers")) or {}
        for key, q in (r.get("questions") or {}).items():
            qtype = q.get("type")
            if qtype not in ("choice", "noul"):
                continue
            gold = _label(qtype, (r.get("expected") or {}).get(key))
            a = answers.get(key)
            if gold is None or not isinstance(a, dict):
                continue
            if qtype == "choice":
                labels = tuple(sorted(str(k) for k in (q.get("criteria") or {})))
                pred = a.get("choice")
                pred = None if pred is None else str(pred)
            else:
                labels = ("false", "true")
                pred = a.get("noul")
                pred = None if not isinstance(pred, (int, float)) else ("true" if pred >= 0.5 else "false")
            if pred is None or not labels:
                continue
            g = groups.setdefault((int(ev["catalog_id"]), qtype, labels),
                                  {"labels": labels, "gold": {}, "pred": {}, "n": 0})
            g["n"] += 1
            g["gold"][gold] = g["gold"].get(gold, 0) + 1
            g["pred"][pred] = g["pred"].get(pred, 0) + 1
    return groups


def detect(rows: Iterable[dict], results: dict[str, dict]) -> dict[str, Any]:
    """The ``warnings`` block: label_collapse / label_near_collapse lists + counts. Never raises on odd predictions."""
    names = _names()
    collapse, near = [], []
    groups = tally(rows, results)
    for (cid, qtype, labels), g in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        n = g["n"]
        if n < MIN_N:
            continue
        for lab in sorted(set(labels) | set(g["gold"])):
            gs = g["gold"].get(lab, 0) / n
            ps = g["pred"].get(lab, 0) / n
            if not (GOLD_LO <= gs <= GOLD_HI):
                continue
            ent = {"family": names.get(cid, str(cid)), "benchmark": cid, "question_type": qtype, "label": lab,
                   "gold_share": round(gs, 4), "pred_share": round(ps, 4), "n": n,
                   "direction": "never_predicted" if ps < 0.5 else "always_predicted"}
            if ps in (0.0, 1.0):
                collapse.append(ent)
            elif ps < NEAR_LO or ps > NEAR_HI:
                near.append(ent)
    fams = lambda xs: sorted({e["benchmark"] for e in xs})  # noqa: E731
    return {"label_collapse": collapse, "label_collapse_count": len(collapse),
            "label_collapse_benchmarks": fams(collapse),
            "label_near_collapse": near, "label_near_collapse_count": len(near),
            "label_near_collapse_benchmarks": fams(near),
            "groups_checked": sum(1 for g in groups.values() if g["n"] >= MIN_N),
            "label_collapse_rule": RULE, "gating": False}


def safe_detect(rows, results) -> dict[str, Any]:
    """``detect`` for job reports: an exception becomes a recorded error, never a failed job."""
    try:
        return detect(rows, results)
    except Exception as e:  # noqa: BLE001 - a warning must never break a measurement
        return {"label_collapse": [], "label_collapse_count": None, "error": f"{type(e).__name__}: {e}",
                "label_collapse_rule": RULE, "gating": False}
