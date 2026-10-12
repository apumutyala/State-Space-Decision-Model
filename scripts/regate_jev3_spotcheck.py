"""Print one row per jev3 sub-family (training rows) with the source's own human label, for a hand gold check."""
import gzip, json, sys
from pathlib import Path
DATA = Path(r"C:\Users\apumu\research\CHLabs\ssdm-data")
P = DATA / "m2" / "build" / "bschema-draw-jev3" / "records.jsonl.gz"
want = ["prm_step", "prm_all", "hs2_bar_helpfulness", "aegis_message", "aegis_reply", "ambignq_multi", "ambignq_single",
        "tatqa_choice", "timeqa_yn", "pqa_answerable", "pqa_count"]
aeg = {x["id"]: x for x in json.load(open(DATA / "m2/sources/dl-2026-10-10/aegis2/train.json", encoding="utf-8"))}
hs2 = [json.loads(l) for l in gzip.open(DATA / "m2/sources/dl-2026-10-10/helpsteer2/train.jsonl.gz", "rt", encoding="utf-8")]
prm = {}
for l in gzip.open(DATA / "raw/jev3/prm800k/prm_cache.jsonl.gz", "rt", encoding="utf-8"):
    x = json.loads(l)
    if "unique_id" in x:
        prm[x["unique_id"]] = x
skip = int(sys.argv[1]) if len(sys.argv) > 1 else 0
seen = {}
for l in gzip.open(P, "rt", encoding="utf-8"):
    r = json.loads(l); m = r["aug"]["jev1"]; sub = m["sub"]
    if sub not in want or seen.get(sub, 0) > skip:
        continue
    seen[sub] = seen.get(sub, 0) + 1
    if seen[sub] <= skip:
        continue
    st = r["state"]
    print("=" * 20, r["id"], sub, {k: v for k, v in m.items() if k not in ("prov", "item_keys")})
    if isinstance(st, dict):
        for k, v in st.items():
            print(f"[{k}]", str(v)[:1800])
    else:
        print(st[:1500] + (" ... " + st[-1200:] if len(st) > 2700 else st[1500:2700]))
    print("Q:", json.dumps(r["questions"]["q"], ensure_ascii=False)[:900]); print("GOLD:", r["labels"]["q"])
    if r["source"] == "aegis2":
        x = aeg[r["source_id"]]; print("SOURCE:", {k: x.get(k) for k in ("prompt_label", "prompt_label_source", "response_label", "response_label_source", "violated_categories")})
    if r["source"] == "helpsteer2":
        x = hs2[int(r["source_id"])]; print("SOURCE:", {k: x.get(k) for k in ("helpfulness", "correctness", "coherence", "complexity", "verbosity")})
    if r["source"] == "prm800k":
        x = prm[r["source_id"]]; print("SOURCE ratings:", x["ratings"], x["finish_reason"])
