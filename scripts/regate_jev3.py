"""jev3-specific re-gate checks (evidence lane): identity + ids sha, per-row provenance against the source files that are
on disk, slice-vs-train disjointness, cross-draw duplicates against jev1 / jev2. Counts and TRAINING ids only.

    python -X utf8 scripts/regate_jev3.py --out jev3-checks.json
"""
import argparse, collections, gzip, hashlib, json, re, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
import regate_jev2a as RG
from train.cohort import record_content_hash

DATA = Path(r"C:\Users\apumu\research\CHLabs\ssdm-data")
BUILD = DATA / "m2" / "build"
DL1 = DATA / "m2" / "sources" / "dl-2026-10-10"
RAW3 = DATA / "raw" / "jev3"
DRAWS = {"train": "bschema-draw-jev3", "slice": "bschema-draw-jev3-slice"}
OTHER = ["bschema-draw-jev1", "bschema-draw-jev2a-v2", "bschema-draw-jev2b", "bschema-draw-jev2c-v2",
         "bschema-draw-jev2a", "bschema-draw-jev2c"]


def canon(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_unc(p):
    h = hashlib.sha256()
    with gzip.open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def state_key(r):
    return hashlib.sha256(" | ".join(RG.norm_leaf(x) for x in RG.leaves(r["state"])).encode()).hexdigest()[:24]


def keys(r):
    m = r["aug"]["jev1"]
    pv = m.get("prov") or {}
    doc = None
    if r.get("source") == "privacyqa":
        mm = re.match(r"DocID (.*?) / QueryID", pv.get("row_key", ""))
        doc = mm.group(1) if mm else None
    return {"id": r["id"], "source_item": r["aug"]["b1_view"]["base_id"], "group": m.get("group"),
            "content": record_content_hash(r), "state": state_key(r), "policy_doc": doc}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = {"identity": {}, "prov": {}, "disjoint": {}, "cross_draw": {}}
    rows = {}
    for role, b in DRAWS.items():
        d = BUILD / b
        ident = {}
        for f in ("records.jsonl.gz", f"{b}.manifest.jsonl.gz", f"{b}.content.jsonl.gz", f"{b}.lens.jsonl.gz"):
            p = d / f
            ident[f] = sha_unc(p) if p.exists() else "MISSING"
        rows[role] = list(RG.recs(d / "records.jsonl.gz"))
        ids = sorted(r["id"] for r in rows[role])
        ident["n"] = len(ids); ident["unique_ids"] = len(set(ids))
        ident["ids_sha256_no_trailing_nl"] = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        ident["families"] = dict(collections.Counter(r["aug"]["jev1"]["family8"] for r in rows[role]))
        ident["sources"] = dict(collections.Counter(r.get("source") for r in rows[role]))
        out["identity"][role] = ident
    # ---------------- provenance
    aeg = {x["id"]: x for x in json.load(open(DL1 / "aegis2" / "train.json", encoding="utf-8"))}
    hs2 = [json.loads(l) for l in gzip.open(DL1 / "helpsteer2" / "train.jsonl.gz", "rt", encoding="utf-8")]
    hs3 = {}
    for l in gzip.open(RAW3 / "helpsteer3" / "hs3_cache.jsonl.gz", "rt", encoding="utf-8"):
        x = json.loads(l)
        if "_meta" not in x:
            hs3[(str(x["line"]), int(x["k"]))] = x
    prm = {}
    for l in gzip.open(RAW3 / "prm800k" / "prm_cache.jsonl.gz", "rt", encoding="utf-8"):
        x = json.loads(l)
        if "_meta" not in x:
            prm[x["unique_id"]] = x
    for role, rs in rows.items():
        c = collections.defaultdict(collections.Counter)
        ex = collections.defaultdict(list)
        for r in rs:
            src = r.get("source"); m = r["aug"]["jev1"]; pv = m.get("prov") or {}
            g = r["labels"]["q"]
            cc = c[src]
            cc["rows"] += 1
            if not (pv.get("split") == "train" and r.get("source_split") == "train"):
                cc["split_not_train"] += 1; ex[f"{src}:split"].append(r["id"])
            if not (r.get("license") and pv.get("license") == r.get("license")):
                cc["licence_missing_or_mismatch"] += 1
            if src == "aegis2":
                x = aeg.get(r["source_id"])
                if x is None:
                    cc["not_in_train_file"] += 1; ex["aegis2:missing"].append(r["id"]); continue
                cc["row_sha256_match" if hashlib.sha256(canon(x).encode()).hexdigest() == pv["row_sha256"]
                   else "row_sha256_MISMATCH"] += 1
                t = m.get("target")
                if t == "reply":
                    want = x.get("response_label") == "unsafe"
                    cc["reply_label_src_human" if x.get("response_label_source") == "human" else "reply_label_src_NOT_human"] += 1
                elif t == "category":
                    mm = re.search(r"'([^']+)'", r["questions"]["q"]["instructions"])
                    cat = mm.group(1) if mm else None
                    vc = [v.strip() for v in (x.get("violated_categories") or "").split(",") if v.strip()]
                    want = x.get("prompt_label") == "unsafe" and cat in vc
                    cc["prompt_label_src_" + str(x.get("prompt_label_source"))] += 1
                else:
                    want = x.get("prompt_label") == "unsafe"
                    cc["prompt_label_src_" + str(x.get("prompt_label_source"))] += 1
                cc["gold_matches_source_label" if (g == "true") == want else "gold_MISMATCH"] += 1
                if (g == "true") != want:
                    ex["aegis2:gold"].append(r["id"])
            elif src == "helpsteer2":
                i = int(r["source_id"])
                x = hs2[i] if i < len(hs2) else None
                if x is None:
                    cc["not_in_train_file"] += 1; continue
                cc["row_sha256_match" if hashlib.sha256(canon(x).encode()).hexdigest() == pv["row_sha256"]
                   else "row_sha256_MISMATCH"] += 1
                v = int(x[m["attribute"]])
                cc["gold_matches_rating" if (g == "true") == (v >= 3) and v != 2 else "gold_MISMATCH"] += 1
            elif src == "helpsteer3":
                line, k = r["source_id"].rsplit("-", 1)
                x = hs3.get((line, int(k)))
                if x is None:
                    cc["not_in_builder_cache"] += 1; continue
                cc["row_sha256_match_builder_cache" if hashlib.sha256(canon({"line_sha256": x["line_sha256"], "k": x["k"]}).encode()).hexdigest() == pv["row_sha256"]
                   else "row_sha256_MISMATCH"] += 1
                cc["gold_matches_cache_verdict" if (g == "true") == bool(x["good"]) else "gold_MISMATCH"] += 1
                cc["official_file_not_on_disk"] += 1
            elif src == "prm800k":
                uid = r["source_id"]
                cc["unique_id_train_prefix" if uid.startswith("train/") else "unique_id_NOT_TRAIN"] += 1
                if not uid.startswith("train/"):
                    ex["prm:test"].append(r["id"])
                x = prm.get(uid)
                if x is None:
                    cc["not_in_builder_cache"] += 1; continue
                req = r["state"]["request"] if isinstance(r["state"], dict) else None
                cc["problem_equals_cache" if req == x["problem"] else "problem_MISMATCH"] += 1
                cc["official_file_not_on_disk"] += 1
            else:
                cc["official_file_not_on_disk"] += 1
        out["prov"][role] = {"by_source": {k: dict(v) for k, v in c.items()}, "examples": {k: v[:10] for k, v in ex.items()}}
    # ---------------- slice vs train
    K = {role: [keys(r) for r in rs] for role, rs in rows.items()}
    dj = {}
    for k in ("id", "source_item", "group", "content", "state", "policy_doc"):
        tr = {x[k] for x in K["train"] if x[k]}
        sl = [x for x in K["slice"] if x[k] and x[k] in tr]
        dj[k] = {"shared": len(sl), "examples": [x["id"] for x in sl[:10]]}
    st_all = collections.Counter(x["state"] for role in K for x in K[role])
    dj["state_shared_any_two_rows_train_plus_slice"] = sum(v - 1 for v in st_all.values() if v > 1)
    out["disjoint"] = dj
    # ---------------- cross draw
    jev3_keys = {k: {} for k in ("id", "source_item", "content", "state")}
    for role in K:
        for x in K[role]:
            for k in jev3_keys:
                jev3_keys[k].setdefault(x[k], f"{role}:{x['id']}")
    for b in OTHER:
        cnt = collections.Counter(); exs = collections.defaultdict(list)
        for r in RG.recs(BUILD / b / "records.jsonl.gz"):
            x = {"id": r["id"], "source_item": r["aug"]["b1_view"]["base_id"], "content": record_content_hash(r),
                 "state": state_key(r)}
            for k in jev3_keys:
                if x[k] in jev3_keys[k]:
                    cnt[k] += 1
                    if len(exs[k]) < 8:
                        exs[k].append([jev3_keys[k][x[k]], x["id"]])
        out["cross_draw"][b] = {"hits": dict(cnt), "examples": dict(exs)}
        print(b, dict(cnt), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()})[:8000])


if __name__ == "__main__":
    main()
