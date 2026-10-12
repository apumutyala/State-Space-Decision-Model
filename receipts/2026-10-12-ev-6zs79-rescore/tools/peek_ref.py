import collections
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
rows = [json.loads(x) for x in open(sys.argv[1], encoding="utf-8") if x.strip()]
c = collections.Counter()
for r in rows:
    c[(r["task_id"].split("-")[0], type(r["predicted"]).__name__, tuple(sorted(r["probs"]))[:3] if r.get("probs") else None)] += 1
for k, v in sorted(c.items(), key=lambda x: -x[1])[:15]:
    print(v, k)
print(sum(r["correct"] for r in rows), collections.Counter((r["task_id"].split("-")[0], r["correct"]) for r in rows))
noul = [r for r in rows if isinstance(r["predicted"], bool) or r["predicted"] in ("yes", "no")][:2]
print(json.dumps(noul)[:800])
