"""Refuse any built API catalog that reproduces an Index API-Bank catalog (coordinator / agent-a, v3b).

For every candidate row (``--records``) the API-name set of its options is compared with the option set of EVERY Index
API-Bank eval row (DEV, TEST and held-out, read through ``train.doc_overlap_guard.iter_eval_rows`` from a checkout of
claude/b-doc-overlap-gate given by ``--guard-root``: TEST rows by manifest id, the harness unseal gate is never invoked).
Names are compared case-folded with non-alphanumerics removed. Writes aggregate numbers only (no eval text or names):
per-row max Jaccard, histogram, max overall and per split, and the ids of rows at or above ``--refuse`` (default 0.8).

    python -X utf8 scripts/b_schema_eval_catalog_overlap.py --guard-root <checkout> --records <levers.jsonl.gz>
        --out <report.json> --refuse-ids <ids.txt>
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path


def _canon(n) -> str:
    return re.sub(r"[^a-z0-9]", "", str(n).casefold())


def _optset(q: dict) -> frozenset[str]:
    crit = q.get("criteria") or {}
    return frozenset(_canon(v if v not in (None, "") else k) for k, v in crit.items())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--guard-root", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refuse-ids", required=True)
    ap.add_argument("--refuse", type=float, default=0.8)
    a = ap.parse_args(argv)
    sys.path.insert(0, a.guard_root)
    from train.doc_overlap_guard import iter_eval_rows

    ev = []  # (split, option set)
    n_rows_by_split = collections.Counter()
    for split, rid, fam, r in iter_eval_rows():
        if "api-bank" not in str(fam or "").casefold():
            continue
        for q in (r.get("questions") or {}).values():
            if isinstance(q, dict) and q.get("type") == "choice":
                ev.append((split, _optset(q)))
                n_rows_by_split[split] += 1
    if not ev:
        raise SystemExit("no API-Bank eval rows found (fail closed)")
    k_hist = collections.Counter(len(s) for _, s in ev)
    rows, refuse = [], []
    hist = collections.Counter()
    best_by_split = collections.defaultdict(float)
    with gzip.open(a.records, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            (q,) = r["questions"].values()
            s = _optset(q)
            best = 0.0
            for split, e in ev:
                j = len(s & e) / len(s | e)
                if j > best_by_split[split]:
                    best_by_split[split] = j
                best = max(best, j)
            rows.append(best)
            hist[f"{int(best * 10) / 10:.1f}"] += 1
            if best >= a.refuse:
                refuse.append(r["id"])
    Path(a.refuse_ids).write_text("".join(i + "\n" for i in refuse), encoding="utf-8")
    rep = {"rule": f"API-name Jaccard of each built catalog vs every Index API-Bank eval option set; refuse >= {a.refuse}",
           "guard_root_commit": None, "n_eval_option_sets": len(ev), "eval_rows_by_split": dict(n_rows_by_split),
           "eval_k_hist": dict(sorted(k_hist.items())), "n_candidates": len(rows), "max_jaccard": round(max(rows), 4),
           "max_jaccard_by_split": {k: round(v, 4) for k, v in best_by_split.items()},
           "per_row_max_hist": dict(sorted(hist.items())), "n_refused": len(refuse),
           "refused_ids_sha256": hashlib.sha256("".join(i + "\n" for i in refuse).encode()).hexdigest()}
    try:
        import subprocess
        rep["guard_root_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                                  cwd=a.guard_root).stdout.strip()
    except Exception:
        pass
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
