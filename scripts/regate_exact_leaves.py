"""EVERY training row that shares a verbatim normalised leaf (>= 6 words) with a JevBench public item or any Index suite
row (DEV / TEST / HELDOUT). Ids only; eval text is never written.

    python -X utf8 scripts/regate_exact_leaves.py --records R.jsonl.gz --out rep.json [--ids-out ids.txt]
"""
import argparse, collections, hashlib, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import regate_jev2a as RG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ids-out")
    ap.add_argument("--no-index", action="store_true")
    a = ap.parse_args()
    from jevkit.items import load_public, overlap_guard_rows
    ev = collections.defaultdict(set)  # leaf hash -> {(side, family)}
    item_df = collections.Counter()

    def add(side, fam, row):
        seen = set()
        for s in RG.row_leaves(row.get("state"), row.get("questions")):
            ns = RG.norm_leaf(s)
            if len(ns.split()) >= RG.EXACT_MIN_WORDS:
                h = RG.h64(ns)
                ev[h].add((side, str(fam)))
                seen.add(h)
        if side == "JEVBENCH":
            item_df.update(seen)

    for split, _, fam, row in overlap_guard_rows(load_public()):
        add("JEVBENCH", fam, row)
    if not a.no_index:
        from train import doc_overlap_guard as DG
        for split, _, fam, row in DG.iter_eval_rows():
            add(f"INDEX_{split}", fam, row)
    hit = {}
    by = collections.Counter()
    gold = 0
    for r in RG.recs(Path(a.records)):
        sides = set()
        q = r["questions"]["q"]
        for s in RG.row_leaves(r["state"], r["questions"]):
            ns = RG.norm_leaf(s)
            if len(ns.split()) >= RG.EXACT_MIN_WORDS and RG.h64(ns) in ev:
                sides |= ev[RG.h64(ns)]
        if sides:
            hit[r["id"]] = sorted(f"{x}:{y}" for x, y in sides)
            for x, y in sides:
                by[f"{RG.fam(r)}/{r.get('source')} <- {x}:{y}"] += 1
            gd = (q.get("criteria") or {}).get(r["labels"]["q"])
            if isinstance(gd, str) and RG.h64(RG.norm_leaf(gd)) in ev:
                gold += 1
    ids = sorted(hit)
    sha = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest() if ids else None
    out = {"records": a.records, "records_sha256": RG.sha_file(Path(a.records)), "rule": "normalised leaf >= 6 words, "
           "verbatim equal to a leaf of a JevBench public item or an Index suite row", "n_rows": len(ids),
           "rows_gold_is_matched_leaf": gold, "by_family_source_evalside": dict(by), "ids_sha256": sha,
           "rows": hit}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    if a.ids_out:
        Path(a.ids_out).write_bytes(("\n".join(ids) + "\n").encode())
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}))


if __name__ == "__main__":
    main()
