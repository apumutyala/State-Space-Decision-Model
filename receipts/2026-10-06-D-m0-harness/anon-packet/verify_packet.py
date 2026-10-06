"""Recompute the DEV aggregates from the anonymous packet alone (reviewer side).

    python -X utf8 -I receipts/2026-10-06-D-m0-harness/anon-packet/verify_packet.py > my-packet.json
    python -c "import json,sys; a,b=(json.load(open(p)) for p in sys.argv[1:]); print('MATCH' if a==b else 'DIFF')" \
        my-packet.json receipts/2026-10-06-D-m0-harness/anon-packet/expected-verify.json

Needs only numpy, the pinned kit (for `harness` imports) and this packet; no suite, no DEV rows,
no manifests. Checks: (1) every per-benchmark raw/skill and the index recomputed from the
summed cluster statistics equal the official kit's outputs stored in the packet; (2) the
paired source-case bootstrap (B=2000, seed 0) on the packet. Bootstrap numbers are specific
to this packet's cluster shuffle; they agree with reproduce.py statistically, not digit for
digit. Values rounded to 4 decimals (gaps to 1e-9)."""
import json
import sys
from pathlib import Path

from harness import packet

p = packet.load(Path(__file__).with_name("packet.json.gz"))
pairs = [
    ("independent-0.60", "base-0.55"),
    ("base+11pct_rows_fixed", "base-0.55"),
    ("base+4pct_rows_fixed", "base-0.55"),
    ("base+11pct_cases_fixed", "base-0.55"),
    ("base-0.55", "base-0.55"),
]
r = packet.recompute(p, pairs=pairs, B=2000, seed=0)
out = {"benchmarks": len(p["benchmarks"]), "clusters": sum(b["n_clusters"] for b in p["benchmarks"].values()), "systems": {}, "pairs": {}}
for name, e in r["systems"].items():
    out["systems"][name] = {
        "index": round(e["index"], 4),
        "official_index": round(p["official_reference"][name]["balanced_skill"], 4),
        "parity_ok": e["max_abs_skill_gap_vs_official"] < 1e-9 and e["max_abs_raw_gap_vs_official"] < 1e-9 and e["index_gap_vs_official"] < 1e-9,
    }
for name, e in r["pairs"].items():
    d = e["diff"]
    out["pairs"][name] = {"diff": round(d["point"], 4), "ci95": [round(d["lo"], 4), round(d["hi"], 4)], "se": round(d["se"], 4), "mde": round(e["mde"]["table"][2]["mde"], 4) if e["mde"]["available"] else None, "mde_available": e["mde"]["available"]}
json.dump(out, sys.stdout, indent=1, sort_keys=True)
print()
