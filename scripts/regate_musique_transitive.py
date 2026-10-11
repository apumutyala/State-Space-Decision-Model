"""MuSiQue provenance by transitivity when the train file is gone: every MuSiQue row of draw B must render documents
that a row of an already-verified draw A (same base id; REGATE_JEV2A checked A against musique_ans_v1.0_train.jsonl,
sha 83a75b1e) rendered. Rows without such a twin are reported as UNVERIFIED.

    python -X utf8 scripts/regate_musique_transitive.py --verified A/records.jsonl.gz --draw B/records.jsonl.gz --out o.json
"""
import argparse, json, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import regate_jev2a as RG


def docs(r):
    st = r["state"]
    d = st.get("documents") if isinstance(st, dict) else None
    return frozenset(RG.norm_leaf(x.get("text", "")) for x in d) if d else None


ap = argparse.ArgumentParser()
ap.add_argument("--verified", required=True); ap.add_argument("--draw", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
ver = {}
for r in RG.recs(Path(a.verified)):
    if r.get("source") == "musique":
        ver.setdefault(r["aug"]["b1_view"]["base_id"], set()).update(docs(r) or ())
n = ok = 0; noid = []; nodoc = []; prefix_bad = []
for r in RG.recs(Path(a.draw)):
    if r.get("source") != "musique":
        continue
    n += 1
    b = r["aug"]["b1_view"]["base_id"]; d = docs(r) or frozenset()
    sid = str(r.get("source_id"))
    if sid.split("__")[0] not in ("2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3"):
        prefix_bad.append(r["id"])
    if r.get("source_split") != "train":
        prefix_bad.append(r["id"] + " (split)")
    if b not in ver:
        noid.append(r["id"])
    elif d and d <= ver[b]:
        ok += 1
    elif isinstance(r["state"], str) and sum(1 for x in ver[b] if x and x in RG.norm_leaf(r["state"])) >= 2:
        ok += 1  # prose-rendered state: the verified twin's documents appear verbatim (>= 2 of them) in the text
    else:
        nodoc.append(r["id"])
out = {"n_musique_rows": n, "verified_twin_and_docs_subset": ok, "base_id_not_in_verified_draw": len(noid),
       "base_id_in_verified_but_docs_differ": len(nodoc), "unverified_examples": (noid + nodoc)[:20],
       "unexpected_id_prefix_or_split": prefix_bad[:20], "verified_draw": a.verified}
Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out))
