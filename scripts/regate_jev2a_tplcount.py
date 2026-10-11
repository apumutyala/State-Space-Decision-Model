import sys, json, collections
from pathlib import Path
REPO = Path(r"C:\Users\apumu\research\CHLabs\ssdm-wt\ev-jev2a-regate")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import regate_jev2a as RG
from jevkit.items import load_public, overlap_guard_rows
# every >=6-word normalised leaf of the JevBench items, with item count (hash only)
cnt = collections.Counter()
for _, _, fam, row in overlap_guard_rows(load_public()):
    seen = set()
    for s in RG.row_leaves(row["state"], row["questions"]):
        ns = RG.norm_leaf(s)
        if len(ns.split()) >= RG.EXACT_MIN_WORDS:
            seen.add(RG.h64(ns))
    cnt.update(seen)
rows_hit = collections.Counter(); gold_is_tpl = 0; ids = []
by_item_count = collections.Counter()
for r in RG.recs(Path(sys.argv[1])):
    q = r["questions"]["q"]
    hit = False
    for s in RG.row_leaves(r["state"], r["questions"]):
        h = RG.h64(RG.norm_leaf(s))
        if len(RG.norm_leaf(s).split()) >= RG.EXACT_MIN_WORDS and h in cnt:
            hit = True
            by_item_count[cnt[h]] += 1
    if hit:
        rows_hit[RG.fam(r) + "/" + r["source"]] += 1
        ids.append(r["id"])
        g = r["labels"]["q"]
        gd = (q.get("criteria") or {}).get(g)
        if isinstance(gd, str) and RG.h64(RG.norm_leaf(gd)) in cnt:
            gold_is_tpl += 1
print(json.dumps({"train_rows_with_jevbench_exact_leaf": sum(rows_hit.values()), "by_family_source": rows_hit,
                  "matched_leaf_jevbench_item_df_hist": dict(by_item_count),
                  "rows_where_gold_option_is_that_leaf": gold_is_tpl, "example_ids": ids[:10]}))
