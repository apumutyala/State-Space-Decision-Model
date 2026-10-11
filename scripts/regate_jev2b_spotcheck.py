import gzip, json, collections, sys
P = r"C:\Users\apumu\research\CHLabs\ssdm-data\m2\build\bschema-draw-jev2b\records.jsonl.gz"
rates = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
shown = set()
want_sub = {"probability": "prob_bayes_band", "trap": "trap_count", "ambiguous": "ambig_email_count",
            "tradeoff": "tradeoff_weighted_count", "adversarial": "adv_severity"}
extra = {"prob_indep", "trap_premise"}
for l in gzip.open(P, "rt", encoding="utf-8"):
    r = json.loads(l)
    f = r["aug"]["jev1"]["family8"]; q = r["questions"]["q"]; g = r["labels"]["q"]
    if q["type"] in ("score", "noul"):
        rates[f][q["type"]][g] += 1
    sub = r["aug"]["jev1"]["sub"]
    if (want_sub.get(f) == sub and f not in shown) or (sub in extra and sub not in shown):
        shown.add(f if want_sub.get(f) == sub else sub)
        print("=" * 30, r["id"], json.dumps({k: v for k, v in r["aug"]["jev1"].items()}))
        print(json.dumps(r["state"], ensure_ascii=False)[:2500])
        print(json.dumps(q, ensure_ascii=False)[:1200]); print("GOLD", g)
print("RATES")
for f, d in rates.items():
    for t, c in d.items():
        n = sum(c.values()); print(f, t, {k: round(v / n, 4) for k, v in sorted(c.items())})
