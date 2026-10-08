"""Diversity / leakage audit of a schema-permutation draw (streamed; bounded RAM).

Reports: rows per source item (fraction of rows sharing a source item, max variants per item, histogram), exact
key-free-text collisions (``data.b1.dedup_keys``, the builder's and b1_check's near-dup keys) within vs across source items, and a MinHash near-duplicate
rate (word 5-shingles of the key-free full text, 64 hashes, 16 bands x 4; a candidate pair counts when its estimated
Jaccard >= 0.8) over a seeded sample, split into within-item pairs (expected: permutations of one item) and
cross-item pairs (the leakage-relevant number). Also per-transformation and per-source counts from ``aug.schema_synth``.

    python -X utf8 -I scripts/b_schema_audit.py --repo <worktree> --draw-dir <dir> --out <json> [--sample 30000]
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

NPERM, BANDS, ROWS = 64, 16, 4
MASK = (1 << 61) - 1


def shingles(text: str, k: int = 5) -> set[int]:
    w = text.split()
    if len(w) < k:
        return {hash(" ".join(w))}
    return {int.from_bytes(hashlib.blake2b(" ".join(w[i:i + k]).encode(), digest_size=8).digest(), "little")
            for i in range(len(w) - k + 1)}


def minhash(sh: set[int], params) -> tuple[int, ...]:
    return tuple(min(((a * x + b) & MASK) for x in sh) for a, b in params)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--draw-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--n-expected", type=int, default=300000, help="rows in the draw (sets the sampling rate)")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.repo)
    from train.cohort import record_text_keys, record_texts
    from data.b_schema import root_items
    from data.b1 import dedup_keys

    rng = random.Random(a.seed)
    params = [(rng.getrandbits(61) | 1, rng.getrandbits(61)) for _ in range(NPERM)]
    reuse = collections.Counter()
    owner: dict[str, str] = {}
    within = across = 0
    ops = collections.Counter()
    ops_src = collections.defaultdict(collections.Counter)
    by_src = collections.Counter()
    n = 0
    sample = []  # (root, minhash)
    path = Path(a.draw_dir) / "records.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            ss = r.get("aug", {}).get("schema_synth", {})
            root = ss.get("root") or sorted(root_items(r))[0]  # base draws: the row's own source item
            reuse[root] += 1
            by_src[r["source"]] += 1
            for o in ss.get("ops", []):
                key = "ids" if o.startswith("ids:") else o.split(":")[0]
                ops[key] += 1
                ops_src[r["source"]][key] += 1
            for k in dedup_keys(r):  # the builder's / b1_check's near-dup keys (empty-state rows: full text only)
                prev = owner.get(k)
                if prev is None:
                    owner[k] = root
                elif prev == root:
                    within += 1
                else:
                    across += 1
            if rng.random() < a.sample / a.n_expected:
                full, _ = record_texts(r)
                sample.append((root, minhash(shingles(full), params)))
    buckets = collections.defaultdict(list)
    for i, (_, mh) in enumerate(sample):
        for b in range(BANDS):
            buckets[(b, mh[b * ROWS:(b + 1) * ROWS])].append(i)
    pairs = set()
    for idx in buckets.values():
        if 1 < len(idx) <= 200:
            for x in range(len(idx)):
                for y in range(x + 1, len(idx)):
                    pairs.add((idx[x], idx[y]))
    nd_within = nd_across = 0
    rows_nd_across = set()
    for x, y in pairs:
        est = sum(p == q for p, q in zip(sample[x][1], sample[y][1])) / NPERM
        if est >= 0.8:
            if sample[x][0] == sample[y][0]:
                nd_within += 1
            else:
                nd_across += 1
                rows_nd_across |= {x, y}
    hist = collections.Counter(reuse.values())
    rep = {"draw": str(path), "n_rows": n, "source_items": len(reuse),
           "rows_sharing_a_source_item_frac": round(sum(c for c in reuse.values() if c > 1) / n, 4),
           "max_variants_per_item": max(reuse.values()), "variants_per_item_hist": dict(sorted(hist.items())),
           "keyfree_text_collisions": {"keys": "data.b1.dedup_keys", "within_item": within, "across_items": across},
           "minhash_near_dup": {"sample_rows": len(sample), "threshold_jaccard": 0.8, "pairs_within_item": nd_within,
                                "pairs_across_items": nd_across,
                                "rows_with_cross_item_near_dup_frac": round(len(rows_nd_across) / max(1, len(sample)), 5)},
           "by_source": dict(by_src.most_common()), "ops_rows": dict(ops.most_common()),
           "ops_by_source": {s: dict(c.most_common()) for s, c in sorted(ops_src.items())}}
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n_rows", "source_items", "rows_sharing_a_source_item_frac",
                                          "max_variants_per_item", "keyfree_text_collisions", "minhash_near_dup")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
