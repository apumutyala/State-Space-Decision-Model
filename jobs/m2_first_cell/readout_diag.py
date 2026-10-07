"""Readout diagnostics on the Index DEV subset (first trained cell; coordinator request 2026-10-06 after job-tkfaj).

job-tkfaj: the stock RWKV7-13.3B option-likelihood readout scored 24.9 on the 2,008-row subset, with knowledge at 8.7
despite MMLU ~0.77, and RWKV7 smoke runs showed first-option bias. Two cheap diagnostics separate "the readout is
order/label biased" from "the model lacks the knowledge", for BOTH arms (stock parent, trained child):

1. ``position_bias`` (free; from the scored results): over every choice question with a single gold option, how often
   the chosen option is the first / last PRESENTED one (decide renders choice options sorted by id,
   ``decide.schema.canonical_options``) against how often the gold is, accuracy by gold position, and the most-chosen
   labels against their gold frequency.
2. ``order_diagnostic`` (one small extra pass per arm): ``DIAG_N`` subset rows chosen by sha256 order, each re-asked
   twice with every choice option relabelled ``"NN-<id>"``: FORWARD numbers the options in their canonical order,
   REVERSED numbers them backwards, so the same options are presented in opposite orders under the same label
   pattern. An order-invariant readout gets the same accuracy and the same answer in both; the gap and the agreement
   rate measure order sensitivity. (Reversing the dict order alone would change nothing: decide sorts options by id.)
Neither diagnostic enters any criterion; they are reported beside the Index scores.
"""
from __future__ import annotations

import collections
import hashlib
import math
from typing import Any, Callable, Sequence

DIAG_N = 160
DIAG_SEED = 20261006
CONDITIONS = ("forward", "reversed")


def _gold_choice_questions(row: dict) -> list[tuple[str, list[str], str]]:
    """(question key, canonical option ids, gold id) for choice questions with one gold id among >= 2 options."""
    out = []
    for k, q in sorted(row["questions"].items()):
        if q.get("type") != "choice" or not isinstance(q.get("criteria"), dict):
            continue
        ids = sorted(str(o) for o in q["criteria"])
        gold = (row.get("expected") or {}).get(k)
        if isinstance(gold, str) and gold in ids and len(ids) >= 2:
            out.append((k, ids, gold))
    return out


def _choice(results: dict, row: dict, key: str) -> str | None:
    r = results.get(row["_evaluation"]["run_id"]) or {}
    if r.get("status") != "ok":
        return None
    a = ((r.get("response") or {}).get("answers") or {}).get(key) or {}
    c = a.get("choice")
    return str(c) if c is not None else None


def position_bias(rows: Sequence[dict], results: dict, top: int = 8) -> dict[str, Any]:
    n = acc = first_c = last_c = first_g = last_g = 0
    inv_k = 0.0
    by_gold_pos = {"first": [0, 0], "last": [0, 0], "middle": [0, 0]}
    chosen_lab, gold_lab = collections.Counter(), collections.Counter()
    by_k = collections.defaultdict(lambda: [0, 0, 0])  # K bucket -> n, chose first, gold first
    unanswered = 0
    areas = _areas()
    by_area = collections.defaultdict(lambda: [0, 0, 0, 0])  # area -> n, correct, chose first, gold first
    for row in rows:
        e = row["_evaluation"]
        area = areas.get(e.get("catalog_id"), areas.get(str(e.get("catalog_id")), "unknown"))
        for key, ids, gold in _gold_choice_questions(row):
            c = _choice(results, row, key)
            if c is None or c not in ids:
                unanswered += 1
                continue
            k = len(ids)
            ci, gi = ids.index(c), ids.index(gold)
            n += 1
            acc += c == gold
            first_c += ci == 0
            last_c += ci == k - 1
            first_g += gi == 0
            last_g += gi == k - 1
            inv_k += 1 / k
            pos = "first" if gi == 0 else ("last" if gi == k - 1 else "middle")
            by_gold_pos[pos][0] += 1
            by_gold_pos[pos][1] += c == gold
            ba = by_area[area]
            ba[0] += 1
            ba[1] += c == gold
            ba[2] += ci == 0
            ba[3] += gi == 0
            chosen_lab[c] += 1
            gold_lab[gold] += 1
            b = str(k) if k <= 4 else ("5-16" if k <= 16 else ">16")
            by_k[b][0] += 1
            by_k[b][1] += ci == 0
            by_k[b][2] += gi == 0
    f = (lambda a, b: a / b if b else float("nan"))
    return {"n_questions": n, "unanswered_or_invalid": unanswered, "accuracy": f(acc, n),
            "p_chosen_first": f(first_c, n), "p_gold_first": f(first_g, n), "p_chosen_last": f(last_c, n),
            "p_gold_last": f(last_g, n), "uniform_first_rate": f(inv_k, n),
            "accuracy_by_gold_position": {p: {"n": v[0], "accuracy": f(v[1], v[0])} for p, v in by_gold_pos.items()},
            "by_k": {b: {"n": v[0], "p_chosen_first": f(v[1], v[0]), "p_gold_first": f(v[2], v[0])}
                     for b, v in sorted(by_k.items())},
            "by_area": {a: {"n": v[0], "accuracy": f(v[1], v[0]), "p_chosen_first": f(v[2], v[0]),
                            "p_gold_first": f(v[3], v[0])} for a, v in sorted(by_area.items())},
            "top_chosen_labels": [{"label": lab, "chosen": cnt, "gold": gold_lab.get(lab, 0)}
                                  for lab, cnt in chosen_lab.most_common(top)]}


def _areas() -> dict:
    try:
        from harness.suite import area_of

        return area_of()
    except Exception:  # noqa: BLE001 - the kit spec is unavailable (tests without the kit): no area split
        return {}


def diag_rows(index_rows: Sequence[dict], n: int = DIAG_N, seed: int = DIAG_SEED) -> list[dict]:
    """The pre-registered diagnostic rows: rows with >= 1 gold choice question, first ``n`` in sha256(seed:run_id)."""
    ok = [r for r in index_rows if _gold_choice_questions(r)]
    ok.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['_evaluation']['run_id']}".encode()).hexdigest())
    return ok[:n]


def relabel(row: dict, condition: str) -> tuple[dict, dict[str, dict[str, str]]]:
    """The decide request of ``row`` with every gold choice question's options relabelled ``NN-<id>`` (forward or
    reversed numbering); returns (request, {question key: {new id: original id}})."""
    if condition not in CONDITIONS:
        raise ValueError(condition)
    qs = {k: dict(q) for k, q in row["questions"].items()}
    back = {}
    for key, ids, _ in _gold_choice_questions(row):
        k = len(ids)
        width = max(2, len(str(k)))
        new = {o: f"{(j + 1 if condition == 'forward' else k - j):0{width}d}-{o}" for j, o in enumerate(ids)}
        crit = row["questions"][key]["criteria"]
        qs[key]["criteria"] = {new[str(o)]: v for o, v in crit.items()}
        back[key] = {v: o for o, v in new.items()}
    return {"model": "ssdm-diag", "state": row["state"], "questions": qs}, back


def diag_requests(rows: Sequence[dict]) -> list[dict]:
    return [relabel(r, c)[0] for c in CONDITIONS for r in rows]


def order_diagnostic(decide: Callable[[dict], dict], rows: Sequence[dict]) -> dict[str, Any]:
    """Per-question outcomes in both conditions (original ids) + summary. Errors count as wrong / not agreeing."""
    per: dict[str, dict] = {}
    for cond in CONDITIONS:
        for row in rows:
            req, back = relabel(row, cond)
            resp = decide(req)
            ok = (resp.get("ssdm") or {}).get("status") == "ok" and "answers" in resp
            for key, ids, gold in _gold_choice_questions(row):
                qid = f"{row['_evaluation']['run_id']}/{key}"
                c = ((resp.get("answers") or {}).get(key) or {}).get("choice") if ok else None
                orig = back[key].get(str(c)) if c is not None else None
                presented_first = (ids[0] if cond == "forward" else ids[-1])
                per.setdefault(qid, {"gold": gold, "k": len(ids)})[cond] = {
                    "choice": orig, "correct": orig == gold, "chose_presented_first": orig == presented_first}
    return {"rows": len(rows), "questions": per, "summary": summarize_order(per)}


def summarize_order(per: dict[str, dict]) -> dict[str, Any]:
    n = len(per)
    if not n:
        return {"n_questions": 0}
    m = lambda f: sum(f(v) for v in per.values()) / n  # noqa: E731
    acc_f, acc_r = m(lambda v: v["forward"]["correct"]), m(lambda v: v["reversed"]["correct"])
    return {"n_questions": n, "acc_forward": acc_f, "acc_reversed": acc_r, "acc_gap": acc_f - acc_r,
            "agreement": m(lambda v: v["forward"]["choice"] is not None
                           and v["forward"]["choice"] == v["reversed"]["choice"]),
            "p_chose_presented_first": {c: m(lambda v, c=c: v[c]["chose_presented_first"]) for c in CONDITIONS},
            "uniform_first_rate": m(lambda v: 1 / v["k"]),
            "both_correct": m(lambda v: v["forward"]["correct"] and v["reversed"]["correct"])}


def paired_order(child: dict, parent: dict, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Child - parent on the identical diagnostic questions: order-averaged accuracy and agreement (row-cluster
    bootstrap, rows = run ids)."""
    import numpy as np

    qs = sorted(set(child["questions"]) & set(parent["questions"]))
    if not qs or set(child["questions"]) != set(parent["questions"]):
        return {"identical_questions": False}
    cl = [q.rsplit("/", 1)[0] for q in qs]
    uniq = sorted(set(cl))
    idx = {c: i for i, c in enumerate(uniq)}
    ci = np.array([idx[c] for c in cl])

    def vals(d, f):
        return np.array([f(d["questions"][q]) for q in qs], dtype=float)

    acc = lambda v: (v["forward"]["correct"] + v["reversed"]["correct"]) / 2  # noqa: E731
    agr = lambda v: float(v["forward"]["choice"] is not None and v["forward"]["choice"] == v["reversed"]["choice"])  # noqa: E731
    out = {"identical_questions": True, "n_questions": len(qs), "n_rows": len(uniq)}
    rng = np.random.default_rng(seed)
    for name, f in (("order_avg_accuracy", acc), ("agreement", agr)):
        d = vals(child, f) - vals(parent, f)
        draws = []
        for _ in range(n_boot):
            w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[ci].astype(float)
            if w.sum() > 0:
                draws.append(float((w * d).sum() / w.sum()))
        lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (math.nan, math.nan)
        out[name] = {"child": float(vals(child, f).mean()), "parent": float(vals(parent, f).mean()),
                     "diff": {"point": float(d.mean()), "lo": float(lo), "hi": float(hi), "n_boot": len(draws)}}
    return out
