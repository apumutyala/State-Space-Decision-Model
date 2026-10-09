"""Yes/no balance of BINARY questions (a trained item each): Index DEV rows (eval side) or a training draw.

Binary question = ``noul``, or a two-option ``choice`` whose options are a positive / negative pair. Polarity comes from
the option id (yes/true/supported/entailment/correct vs no/false/...), else from the option DESCRIPTION, using the
positive/negative description texts the training sources themselves use (learned from rows whose ids are semantic, so
re-styled ids in a schema-permutation draw are still classified). Output: per family and aggregate yes share, item counts.

    python -X utf8 -I scripts/b_schema_binary_balance.py --repo <worktree> --records <jsonl.gz> [--learn <jsonl.gz>] --out <json>
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

POS = {"yes", "true", "y", "correct", "supported", "entailment", "entailed"}
NEG = {"no", "false", "n", "incorrect", "not_supported", "not_entailment", "contradiction", "not_entailed", "refuted"}


def norm(x) -> str:
    return " ".join(str(x).casefold().split())


def learn(paths) -> dict[str, bool]:
    """description text -> polarity, from two-option questions with semantic positive/negative ids"""
    pol: dict[str, bool] = {}
    for p in paths:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                for q in r["questions"].values():
                    c = q.get("criteria")
                    if q["type"] == "choice" and isinstance(c, dict) and len(c) == 2:
                        ids = {norm(o): d for o, d in c.items()}
                        if any(i in POS for i in ids) and any(i in NEG for i in ids):
                            for i, d in ids.items():
                                if d is not None:
                                    pol.setdefault(norm(d), i in POS)
                    if q["type"] == "noul" and isinstance(c, dict):
                        for k, d in c.items():
                            pol.setdefault(norm(d), k == "true")
    return pol


def polarity(q: dict, lab: str, pol: dict[str, bool]) -> bool | None:
    """True (positive gold) / False (negative gold) / None (not a binary question)"""
    if q["type"] == "noul":
        return str(lab).lower() == "true"
    c = q.get("criteria")
    if q["type"] != "choice" or not isinstance(c, dict) or len(c) != 2:
        return None
    o = {str(k): d for k, d in c.items()}
    ids = {norm(k) for k in o}
    if any(i in POS for i in ids) and any(i in NEG for i in ids):
        return norm(lab) in POS
    ps = {k: pol.get(norm(d)) if d is not None else None for k, d in o.items()}
    if None in ps.values() or len(set(ps.values())) != 2:
        return None
    return bool(ps[str(lab)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--learn", action="append", default=[])
    ap.add_argument("--labels", default="labels", help="label field: labels (training) or expected (Index rows)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.repo)
    pol = learn(a.learn or [a.records])
    fam = collections.defaultdict(collections.Counter)
    items = collections.Counter()
    with gzip.open(a.records, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            key = f"{r.get('source', '')}/{r.get('family', '')}".strip("/")
            labs = r.get(a.labels) or {}
            for qid, q in r["questions"].items():
                items[key] += 1
                if qid not in labs:
                    continue
                p = polarity(q, labs[qid], pol)
                if p is not None:
                    fam[key]["yes" if p else "no"] += 1
    tot = collections.Counter()
    for c in fam.values():
        tot.update(c)
    n_items = sum(items.values())
    rep = {"records": a.records, "n_items": n_items, "n_binary_items": sum(tot.values()),
           "binary_share_of_items": round(sum(tot.values()) / max(1, n_items), 4),
           "aggregate_yes_share": round(tot["yes"] / max(1, sum(tot.values())), 4),
           "by_family": {k: {"binary": sum(c.values()), "yes_share": round(c["yes"] / sum(c.values()), 4),
                             "items": items[k], "item_share": round(items[k] / n_items, 4)}
                         for k, c in sorted(fam.items(), key=lambda kv: -sum(kv[1].values()))},
           "items_by_family": {k: {"items": v, "item_share": round(v / n_items, 4)} for k, v in items.most_common()},
           "n_polarity_descriptions": len(pol)}
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_items", "n_binary_items", "binary_share_of_items", "aggregate_yes_share")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
