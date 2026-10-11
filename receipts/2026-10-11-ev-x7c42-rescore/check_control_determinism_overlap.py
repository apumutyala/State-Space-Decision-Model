import gzip, json, sys, hashlib
ctrl, g4, c9, cx, tr, ca, l9, lx = sys.argv[1:9]
rj = lambda p: [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
a = {p["id"]: p["answer"] for p in rj(ctrl)}
b = {p["id"]: p["answer"] for p in rj(g4)}
mx, nmx, kinds = 0.0, None, {}
for i in a:
    x, y = a[i], b[i]
    if x.get("type") == "noul":
        d = abs(x["noul"] - y["noul"])
    else:
        px, py = x.get("probabilities") or {}, y.get("probabilities") or {}
        d = max((abs(px[k] - py.get(k, 0)) for k in px), default=0.0)
    kinds.setdefault(x.get("type"), []).append(d)
    if d > mx:
        mx, nmx = d, i
for k, v in kinds.items():
    v = sorted(v)
    print(k, "n", len(v), "max|dp|", v[-1], "median", v[len(v) // 2], "n>1e-3", sum(t > 1e-3 for t in v), "n>1e-2", sum(t > 1e-2 for t in v))
print("max", mx, nmx, json.dumps(a[nmx])[:300], json.dumps(b[nmx])[:300])
print("sample keys ctrl", list(a[next(iter(a))].keys()))
j9 = json.load(open(c9)); jx = json.load(open(cx))
print("records sha 9fy9p", j9["data"]["records_sha256"], j9["data"]["train_set"].get("drawn_ids_sha256"), "x7c42", jx["data"]["records_sha256"], jx["data"]["train_set"].get("drawn_ids_sha256"))
L9, LX = rj(l9), rj(lx)
print("loss first 6 x7c42", [round(r["loss"], 4) for r in LX[:6]]); print("loss first 6 9fy9p", [round(r["loss"], 4) for r in L9[:6]])
print("keys", list(LX[0].keys()))
# state overlap slice/train
def nt(x):
    s = json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(" ".join(s.casefold().split()).encode()).hexdigest()
T = {}
for line in gzip.open(tr, "rt", encoding="utf-8"):
    r = json.loads(line); T.setdefault(nt(r.get("state")), []).append(r)
for line in gzip.open(ca, "rt", encoding="utf-8"):
    r = json.loads(line)
    h = nt(r.get("state"))
    if h in T:
        t = T[h][0]
        print("STATE-OVERLAP slice", r["id"], r.get("family"), "| train", t["id"], t.get("family"), "| state chars", len(json.dumps(r.get("state"))),
              "| q equal", r["questions"] == t["questions"], "| labels", r.get("labels"), t.get("labels"))
