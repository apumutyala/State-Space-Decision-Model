"""job-q668q integrity checks not in ev_rescore_q668q.py (run with python -I; artifact files are untrusted data).

argv: <art final dir> <train records.jsonl.gz> <calib records.jsonl.gz> <local draw-report.json>
      <x7c42 slice_floor_rows.jsonl> <x7c42 train records.jsonl.gz (bschema-draw-jev2h-final)>
"""
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

art, tr, ca, drep, x7rows, x7tr = (Path(p) for p in sys.argv[1:7])
out = art / "out"
rj = lambda p: [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
rgz = lambda p: [json.loads(x) for x in gzip.open(p, "rt", encoding="utf-8") if x.strip()]
sha = lambda b: hashlib.sha256(b).hexdigest()

print("== host")
for line in open(art / "logs" / "run.log", encoding="utf-8", errors="replace"):
    if "HOST" in line:
        print(" run.log:", line.strip()[:200])
hv = json.load(open(out / "host-verify.json", encoding="utf-8"))
print(" host-verify revision", hv["revision"], "resolved", hv["resolved_sha"])
sv = [x.strip() for x in open(out / "snapshot-verify.txt", encoding="utf-8") if x.strip()]
print(" snapshot-verify", sum(x.endswith(": OK") for x in sv), "/", len(sv), "OK")
for f in ("LICENSE", "README.md"):
    print(" license copy", f, sha((out / "license" / f).read_bytes()), "host-verify", hv["license"][f]["sha256"])

print("== exclusion lists")
cm = json.load(open(out / "CONTEXT_MANIFEST.json", encoding="utf-8"))
for k, v in cm["files_sha256"].items():
    if "exclude" in k or "provenance_unverified" in k:
        print(" CONTEXT_MANIFEST", k, v)
dr = json.load(open(drep, encoding="utf-8"))
for k in ("global_exclude", "excluded_subs", "n_train", "n_calib", "train_ids_sha256", "calib_ids_sha256", "force_calib"):
    if k in dr:
        print(" draw-report", k, json.dumps(dr[k])[:400])

print("== records vs in-job pins")
cell = json.load(open(out / "cell.json", encoding="utf-8"))
pin = cell["data"]["train_set"]["files_uncompressed_sha256"]
print(" in-job records uncompressed", pin["records"], "local", sha(gzip.open(tr).read()))
print(" in-job drawn_ids_sha256", cell["data"]["train_set"]["drawn_ids_sha256"])
print(" local calib uncompressed", sha(gzip.open(ca).read()))

print("== prob_indep / subs")
T, C = rgz(tr), rgz(ca)
for nm, R in (("train", T), ("calib", C)):
    subs = collections.Counter()
    for r in R:
        s = None
        for v in (r.get("aug") or {}).values():
            if isinstance(v, dict) and "sub" in v:
                s = v["sub"]
        subs[str(s)] += 1
    fam = collections.Counter(r.get("family") for r in R)
    print(f" {nm}: n {len(R)} prob_indep rows {sum(c for k, c in subs.items() if 'prob_indep' in k)}; families {len(fam)}")
tm = cell["config"]["train_set"].get("type_mix")
print(" type_mix (in-job)", tm)

print("== public-231 input tokens")
for arm in ("stock_shim", "native_shimcal"):
    toks = [(p.get("usage") or {}).get("input_tokens") for p in rj(out / "screen" / "jevkit" / f"{arm}.preds.jsonl")]
    print(f" {arm}: n {len(toks)} max {max(toks)} n>4096 {sum(t > 4096 for t in toks)}")
print(" train no_truncation", cell["data_checks"])

print("== slice-train identical STATE (content = state+questions overlap is 0; this is state alone)")
def nt(x):
    s = json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha(" ".join(s.casefold().split()).encode())
idx = {}
for r in T:
    idx.setdefault(nt(r.get("state")), []).append(r)
hits = []
for r in C:
    h = nt(r.get("state"))
    if h in idx:
        t = idx[h][0]
        hits.append((r, t))
byfam = collections.Counter(r.get("family") for r, _ in hits)
qeq = sum(r["questions"] == t["questions"] for r, t in hits)
leq = sum(r.get("labels") == t.get("labels") for r, t in hits)
chars = sorted(len(json.dumps(r.get("state"))) for r, _ in hits)
print(f" n {len(hits)} by family {dict(byfam)}; questions equal {qeq}; labels equal {leq}; state chars min/med/max "
      f"{chars[0] if chars else None}/{chars[len(chars)//2] if chars else None}/{chars[-1] if chars else None}")
for r, t in hits[:6]:
    print("  e.g. slice", r["id"], "| train", t["id"], "| q", json.dumps(list(r["questions"].values())[0])[:120],
          "| train q", json.dumps(list(t["questions"].values())[0])[:120])
# the same check on x7c42's train draw (it reported 3)
idx7 = {nt(r.get("state")) for r in rgz(x7tr)}
print(" for reference, x7c42 train draw: slice rows with an identical state", sum(nt(r.get("state")) in idx7 for r in C))

print("== slice step-0 column vs x7c42 (same untrained host, same 3,600 decisions)")
a = {r["id"]: r for r in rj(out / "screen" / "jevkit" / "slice_floor_rows.jsonl")}
b = {r["id"]: r for r in rj(x7rows)}
diff = [i for i in a if bool(a[i]["step0"]) != bool(b[i]["step0"])]
print(f" ids equal {set(a) == set(b)}; step0 top-1 q668q {sum(bool(r['step0']) for r in a.values())} x7c42 "
      f"{sum(bool(r['step0']) for r in b.values())}; per-id step0 disagreements {len(diff)}")
for i in diff[:6]:
    print("  ", i, a[i]["family"], "q668q", a[i]["step0"], "x7c42", b[i]["step0"])
print(" row keys", sorted(next(iter(a.values())).keys()))

print("== train log")
lg = [r for r in rj(out / "train_log.jsonl") if "loss" in r]
print(f" loss rows {len(lg)} last step {lg[-1].get('step')} rows seen ~{lg[-1].get('step') * 8} of {len(T)} "
      f"({lg[-1].get('step') * 8 / len(T):.3f} epoch)")
