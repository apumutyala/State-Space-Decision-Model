"""Exact match of every normalised LINE / sentence (>= 4 words) of training rows against every line of the eval side
(JevBench public items + every Index suite row). Catches short copied questions (AmbigNQ / NQ-style) that the
13-gram (>= 13 words) and leaf (>= 6 words, whole leaf) rules miss. Ids and counts only.

    python -X utf8 scripts/regate_line_exact.py --records A.jsonl.gz [B.jsonl.gz ...] --out rep.json
"""
import argparse, collections, json, re, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import regate_jev2a as RG

SPLIT = re.compile(r"[\n\r]+|(?<=[.?!])\s+")
MIN_W = 4


def units(lvs):
    for s in lvs:
        for u in SPLIT.split(s):
            u = re.sub(r"^(user|question|user question|assistant reply|user message)\s*:\s*", "", u.strip(), flags=re.I)
            n = RG.norm_leaf(u)
            if len(n.split()) >= MIN_W:
                yield n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--records", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tr = collections.defaultdict(set)  # line hash -> row ids
    src = {}
    for p in a.records:
        for r in RG.recs(Path(p)):
            src[r["id"]] = r.get("source")
            for u in units(RG.row_leaves(r["state"], r["questions"])):
                tr[RG.h64(u)].add(r["id"])
    from jevkit.items import load_public, overlap_guard_rows
    from train import doc_overlap_guard as DG
    hit_rows = collections.defaultdict(set)
    df = collections.Counter()

    def probe(side, fam, row):
        for u in set(units(RG.row_leaves(row.get("state"), row.get("questions")))):
            h = RG.h64(u)
            if h in tr:
                df[h] += 1
                for i in tr[h]:
                    hit_rows[i].add(f"{side}:{fam}")

    for split, _, fam, row in overlap_guard_rows(load_public()):
        probe("JEVBENCH", fam, row)
    for split, _, fam, row in DG.iter_eval_rows():
        probe(f"INDEX_{split}", fam, row)
    by = collections.Counter()
    for i, s in hit_rows.items():
        for x in s:
            by[f"{src[i]} <- {x}"] += 1
    # generic lines (in many train rows or many eval rows) are templates; report separately
    out = {"rule": f"normalised line / sentence >= {MIN_W} words, exact", "n_train_rows_hit": len(hit_rows),
           "by_source_evalside": dict(by.most_common(60)),
           "rows": {i: sorted(s) for i, s in sorted(hit_rows.items())},
           "hit_line_train_df": sorted((len(tr[h]) for h in df), reverse=True)[:30],
           "hit_line_eval_df": sorted(df.values(), reverse=True)[:30]}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))


if __name__ == "__main__":
    main()
