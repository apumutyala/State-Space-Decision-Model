"""job-8vjve (q9_feature_cache stock_train @06e9733): which of its 20k training rows' records are text twins / passage
near-duplicates of records the PREVIOUS cohort v3 (0c0309b9) held, i.e. the records the rebuilt v3 now excludes on
their account. Positions are seed-0 draw positions."""
import gzip, hashlib, json, subprocess, sys
REPO = r"C:\Users\apumu\research\CHLabs\ssdm-wt\cohort-v3"
sys.path.insert(0, REPO)
import numpy as np
from train.cohort import record_text_keys
from train.cohort_v3 import HELDOUT_EXCLUDED_150K, SHINGLE_FRAC, ShingleIndex, draw_key, sampled_shingles
D = r"C:\Users\apumu\research\CHLabs\ssdm-data\m2\build\sample-150000-seed20261006"
TOK = r"C:\Users\apumu\research\CHLabs\ssdm-data\tokenizers\Qwen--Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a"
old = subprocess.run(["git", "-C", REPO, "show", "cb94dba:train/cohorts/m2-dev-cohort-v3.jsonl.gz"], capture_output=True).stdout
old_ids = {e["id"] for e in (json.loads(x) for x in gzip.decompress(old).decode("utf-8").split("\n")[1:] if x.strip())
           if e.get("pool") == "m2-150k"}
new_ids = {e["id"] for e in (json.loads(x) for x in gzip.open(REPO + r"\train\cohorts\m2-dev-cohort-v3.jsonl.gz", "rt",
                                                                 encoding="utf-8").read().split("\n")[1:] if x.strip())
           if e.get("pool") == "m2-150k"}
man = [json.loads(x)["id"] for x in gzip.open(D + r"\sample-150000-seed20261006.manifest.jsonl.gz", "rt", encoding="utf-8") if x.strip()]
order = sorted(set(man) - set(HELDOUT_EXCLUDED_150K), key=draw_key)[:24000]
pos = {i: k for k, i in enumerate(order)}
recs = {}
with gzip.open(D + r"\records.jsonl.gz", "rt", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r["id"] in pos or r["id"] in old_ids:
            recs[r["id"]] = r
# the stock_train rows: records in draw order, keep_item rows (Qwen3.5 template + tokenizer), until 20,000 rows
from train.data import items_from_records
from train.run_cell import keep_item, parent_encoder, to_train_record
enc = parent_encoder("q9af", TOK)
rows, used = 0, []
for i in order:
    if rows >= 20000:
        break
    its = items_from_records([to_train_record(recs[i])], enc, layout="ordered", template="jrt-chatml-v1")
    k = sum(1 for x in its if keep_item(x.row, 16, 2048))
    if k:
        used.append(i)
        rows += min(k, 20000 - rows)
cut = pos[used[-1]]
removed = sorted(old_ids - new_ids)
tk_used = {}
for i in used:
    for k in record_text_keys(recs[i]):
        tk_used.setdefault(k, []).append(i)
idx = ShingleIndex()
for i in used:
    idx.add(i, sampled_shingles(recs[i]))
idx.freeze()
hits = []
for v in removed:
    tw = sorted({j for k in record_text_keys(recs[v]) for j in tk_used.get(k, [])}, key=pos.get)
    ref, frac = idx.best(sampled_shingles(recs[v]))
    if tw or (ref is not None and frac >= SHINGLE_FRAC):
        hits.append({"old_v3_id": v, "text_twins_in_job_rows": [[j, pos[j]] for j in tw],
                     "passage_ref_in_job_rows": [ref, pos[ref], round(frac, 3)] if ref and frac >= SHINGLE_FRAC else None})
job_recs = sorted({j for h in hits for j, _ in h["text_twins_in_job_rows"]}
                  | {h["passage_ref_in_job_rows"][0] for h in hits if h["passage_ref_in_job_rows"]}, key=pos.get)
out = {"job": "job-8vjve", "code_sha": "06e9733", "stock_train_rows": rows, "stock_train_records": len(used),
       "last_record_draw_position": cut,
       "old_v3_sha256": "0c0309b9ec96dec8907e5c421925924d70a4df0f87f9a3dae3dbd2e5701cc32f",
       "n_old_v3_records_removed_by_rebuild": len(removed),
       "n_removed_with_a_twin_or_passage_in_job_rows": len(hits),
       "n_text_twin": sum(1 for h in hits if h["text_twins_in_job_rows"]),
       "n_passage_only": sum(1 for h in hits if not h["text_twins_in_job_rows"]),
       "job_records_involved_by_draw_position": [[j, pos[j]] for j in job_recs],
       "new_v3_overlap_with_job_rows": {"by_id": len(new_ids & set(used)),
                                        "by_text": sum(1 for i in new_ids if recs.get(i) is not None and record_text_keys(recs[i]) & set(tk_used))},
       "pairs": hits}
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("pairs", "job_records_involved_by_draw_position")}, indent=1))
