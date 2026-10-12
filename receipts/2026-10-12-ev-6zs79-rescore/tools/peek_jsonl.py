import collections
import json
import sys

sys.stdout.reconfigure(encoding="utf-8")
rows = [json.loads(x) for x in open(sys.argv[1], encoding="utf-8") if x.strip()]
print("n", len(rows))
print(json.dumps(rows[0])[:1500])
keys = collections.Counter(k for r in rows for k in r)
print(keys)
