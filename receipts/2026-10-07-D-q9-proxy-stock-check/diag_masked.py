import json, sys
sys.path.insert(0, r"C:/Users/apumu/research/CHLabs/ssdm-wt/q9fcl-run")
import train.proxy_loop as PL, train.proxy_rule as R
a_p, b_p, out = sys.argv[1:4]
b = {r["run_id"]: r for r in map(json.loads, open(b_p, encoding="utf-8"))}
rows = [json.loads(l) for l in open(a_p, encoding="utf-8")]
with open(out, "w", encoding="utf-8", newline="\n") as f:
    for r in rows:
        if b[r["run_id"]]["status"] != "ok":
            r = dict(b[r["run_id"]])  # take 5cmat's failed row verbatim
        f.write(json.dumps(r) + "\n")
sub = PL._subset_path(R.SELECT_SUBSET)
pt = PL.score(out, sub, 2000)["slice_index"]["index"]["point"]
print("cache readout with 5cmat's 65 failed rows masked:", round(pt, 3), "delta vs 38.27:", round(pt - 38.27, 3))
c = PL.compare(out, b_p, sub, 2000); print("paired diff vs 5cmat:", {k: round(v, 3) for k, v in c["diff"].items() if isinstance(v, float)})
