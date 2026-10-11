import gzip, json, sys, collections
for p in sys.argv[1:3]:
    subs = collections.Counter()
    fam = collections.Counter()
    for line in gzip.open(p, "rt", encoding="utf-8"):
        r = json.loads(line)
        aug = r.get("aug") or {}
        s = None
        for v in aug.values():
            if isinstance(v, dict) and "sub" in v:
                s = v["sub"]
        subs[str(s)] += 1
        fam[r.get("family")] += 1
    print(p, "prob_indep rows:", sum(c for k, c in subs.items() if "prob_indep" in k))
    print(" subs:", dict(subs.most_common(60)))
preds = [json.loads(x) for x in open(sys.argv[3], encoding="utf-8") if x.strip()]
toks = [(p.get("usage") or {}).get("input_tokens") for p in preds]
print("public-231 input_tokens max", max(toks), "n>4096", sum(t > 4096 for t in toks), "n", len(toks))
