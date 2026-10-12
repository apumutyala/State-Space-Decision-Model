import collections
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
for p in sys.argv[1:]:
    rows = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
    bad = [r for r in rows if r.get("status") != "ok"]
    print("##", p[-60:], len(rows), "non-ok", len(bad))
    print(collections.Counter((r.get("error") or "")[:140] for r in bad))
    print([r["id"] for r in bad])
