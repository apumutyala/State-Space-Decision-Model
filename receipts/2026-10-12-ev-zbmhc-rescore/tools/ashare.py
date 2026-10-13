"""A-share check (A-PREREG-JEV3-LM GOLD-LETTER (b)-(d)): predicted-A vs gold-A under the canonical order,
recomputed independently: canonical order from evidence's gold_letter_canonical.py @1ccdc676 (not the lead's
canon_shim), slice option texts from the slice records, public from the pinned jevkit items.
Usage: python -I -X utf8 ashare.py <repo> <slice records.jsonl.gz> <calib_probs.jsonl> <public preds.jsonl> <out.json>"""
import gzip, hashlib, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gold_letter_canonical as G  # noqa: E402

repo, rec_p, calib_p, pub_p, out_p = sys.argv[1:6]
sys.path.insert(0, repo)
LET = G.LET

def share(letters, n):
    c = Counter(letters)
    return {LET[j]: round(c.get(j, 0) / n, 4) for j in range(max(c) + 1)} if n else {}

# ---- slice
recs = {}
ids = []
with gzip.open(rec_p, "rt", encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line)
        ids.append(r["id"])
        recs[r["id"]] = r
ids_sha = hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()
ids_sha_nonl = hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()
calib = [json.loads(l) for l in open(calib_p, encoding="utf-8")]
rows = []
order_mismatch = gold_mismatch = argmax_ties = 0
for c in calib:
    if c["type"] != "choice":
        continue
    rid, qid = c["id"].rsplit("/", 1)
    r = recs[rid]
    q = r["questions"][qid]
    opts = G.options_of(q)
    order, ties = G.canonical(opts)
    if list(c["option_ids"]) != order:
        order_mismatch += 1
    gold_id = str(r["labels"][qid])
    if c["option_ids"][c["gold"]] != gold_id:
        gold_mismatch += 1
    p = c["probs"]
    m = max(p)
    if sum(1 for x in p if x == m) > 1:
        argmax_ties += 1
    pred_id = c["option_ids"][p.index(m)]
    rows.append({"fam": r["aug"]["jev1"]["family8"], "K": len(opts), "gold": order.index(gold_id),
                 "pred": order.index(pred_id)})

def block(rs):
    n = len(rs)
    g = share([x["gold"] for x in rs], n)
    pr = share([x["pred"] for x in rs], n)
    gA = sum(x["gold"] == 0 for x in rs) / n
    pA = sum(x["pred"] == 0 for x in rs) / n
    acc = sum(x["gold"] == x["pred"] for x in rs) / n
    return {"n": n, "gold_A": round(gA, 4), "pred_A": round(pA, 4), "A_excess_pp": round(100 * (pA - gA), 2),
            "flag_gt5pp": (pA - gA) > 0.05, "acc": round(acc, 4), "gold_share": g, "pred_share": pr,
            "acc_pred_A": round(sum(x["gold"] == 0 for x in rs if x["pred"] == 0) / max(1, sum(x["pred"] == 0 for x in rs)), 4)}

res = {"slice_records_ids_sha256": ids_sha, "slice_records_ids_sha256_nonewline": ids_sha_nonl,
       "slice_n_records": len(ids), "slice_choice_n": len(rows),
       "calib_option_order_vs_evidence_canonical_mismatch": order_mismatch,
       "calib_gold_vs_records_mismatch": gold_mismatch, "argmax_ties": argmax_ties,
       "slice_choice_all": block(rows),
       "slice_choice_without_ambiguous": block([x for x in rows if x["fam"] != "ambiguous"]),
       "slice_choice_ambiguous_only": block([x for x in rows if x["fam"] == "ambiguous"])}

# ---- public
from jevkit.items import load_public  # noqa: E402
items = {it.id: it for it in load_public()}
preds = {json.loads(l)["id"]: json.loads(l) for l in open(pub_p, encoding="utf-8")}
prow = []
tie_items = 0
for iid, it in items.items():
    if it.type != "choice":
        continue
    opts = G.options_of(it.question)
    order, ties = G.canonical(opts)
    if max(ties.values()) > 1:
        tie_items += 1
    a = preds[iid]["answer"]
    probs = a["probabilities"]
    m = max(probs.values())
    pred_id = min((o for o in order if probs.get(o) == m), key=order.index)
    prow.append({"fam": it.family, "K": len(opts), "gold": order.index(str(it.expected)), "pred": order.index(pred_id),
                 "pred_choice_field": order.index(a["choice"])})
res["public_choice"] = block(prow)
res["public_choice_items_with_tie_set"] = tie_items
res["public_pred_choice_field_vs_argmax_mismatch"] = sum(x["pred"] != x["pred_choice_field"] for x in prow)
Path(out_p).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ("n", "gold_A", "pred_A", "A_excess_pp", "flag_gt5pp", "acc") if kk in v}) for k, v in res.items()}, indent=1))
