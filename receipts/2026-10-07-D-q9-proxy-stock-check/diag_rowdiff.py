import json, sys, collections
def load(p):
    return {r["run_id"]: r for r in map(json.loads, open(p, encoding="utf-8"))}
a = load(sys.argv[1]); b = load(sys.argv[2])
print("ids equal", set(a)==set(b))
st = collections.Counter((a[k]["status"], b[k]["status"]) for k in a)
print("status (cache, 5cmat)", dict(st))
nq=nd=nmiss=0; bycat=collections.Counter(); pdiff=[]
for k in a:
    ra=(a[k].get("response") or {}).get("answers") or {}
    rb=(b[k].get("response") or {}).get("answers") or {}
    if set(ra)!=set(rb): nmiss+=1
    for q in ra:
        if q not in rb: continue
        nq+=1
        ca, cb = ra[q].get("choice"), rb[q].get("choice")
        if ca!=cb: nd+=1; bycat[a[k]["catalog_id"]]+=1
        pa, pb = ra[q].get("probabilities") or {}, rb[q].get("probabilities") or {}
        if pa and pb and set(pa)==set(pb): pdiff.append(max(abs(pa[o]-pb[o]) for o in pa))
print("questions compared", nq, "choice diffs", nd, "rows w/ question-set mismatch", nmiss)
print("diffs by catalog", bycat.most_common(10))
pdiff.sort(); n=len(pdiff)
print("max|dp| quantiles", [round(pdiff[int(n*q)],4) for q in (0.5,0.9,0.99)], round(pdiff[-1],4))
errs=[k for k in a if (a[k]["status"],b[k]["status"])!=("ok","ok")][:5]
for k in errs: print(k, a[k]["status"], b[k]["status"], str(b[k].get("error") or b[k].get("response",""))[:200])
