import collections, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_rows, load_results, SOURCES, index_of
rows = load_rows()
info = collections.defaultdict(lambda: dict(rows=0, q=collections.Counter(), tok=[]))
for r in rows:
    ev = r["_evaluation"]; i = info[ev["catalog_id"]]; i["name"] = ev["dataset"]; i["rows"] += 1; i["tok"].append(ev.get("proxy_tokens", 0))
    for k, q in r["questions"].items():
        if r["expected"][k] is None: continue
        t = "bin" if q["type"] != "choice" else ("k2" if len(q["criteria"]) == 2 else "k3-16" if len(q["criteria"]) <= 16 else "k>16")
        i["q"][t] += 1
for n in sorted(info):
    i = info[n]; t = sorted(i["tok"])
    print(n, i["name"], i["rows"], dict(i["q"]), "tok med", t[len(t)//2], "max", t[-1], ">16k", sum(x > 16384 for x in t))
for k, p in SOURCES.items():
    res = load_results(p)
    miss = sum(r["_evaluation"]["run_id"] not in res for r in rows)
    print(k, len(res), "missing", miss, round(index_of(rows, res)["index"], 3))
