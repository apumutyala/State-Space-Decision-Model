"""Print rule lines + case + question + gold of a few jev2c(-v2) rows for a hand-check (training rows, synthetic).

    python -X utf8 scripts/regate_jev2c_spotcheck.py <records.jsonl.gz> [n_per_type]
"""
import gzip, json, re, sys

n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 1
seen = {}
doms = set()
for l in gzip.open(sys.argv[1], "rt", encoding="utf-8"):
    r = json.loads(l)
    q = r["questions"]["q"]; t = q["type"]; m = r["aug"]["jev1"]
    if seen.get(t, 0) >= n_per or m.get("domain") in doms:
        continue
    seen[t] = seen.get(t, 0) + 1; doms.add(m.get("domain"))
    st = r["state"]
    print("=" * 20, r["id"], json.dumps(m))
    # target chapter text: split on 'Chapter' headings, keep the target one
    parts = re.split(r"\n(?=Chapter \d+)", st)
    tc = m.get("target_chapter")
    tgt = [p for p in parts if p.startswith(f"Chapter {tc}")] if tc else []
    print((tgt[0] if tgt else "[target chapter not split; first 1200 chars]\n" + st[:1200])[:3000])
    i = st.find("Case record")
    print("CASE:", st[i:i + 800] if i >= 0 else st[-800:])
    print("Q:", json.dumps(q, ensure_ascii=False)); print("GOLD:", r["labels"]["q"])
    if len(seen) >= 3 and all(v >= n_per for v in seen.values()):
        break
