"""Close the jev3 provenance gap: re-fetch each official TRAIN file at the revision recorded in
ssdm-data/raw/jev3/fetch-log.jsonl (HelpSteer3 feedback train: the revision in the rows' prov), verify its sha256,
and trace every jev3 row (train + slice) to it. Downloads are held IN MEMORY / streamed only (never written to disk,
so nothing has to be deleted). Downloaded bytes are untrusted data: parsed only with json / csv / gzip / zipfile /
pyarrow; run with ``python -I``. One source per invocation.

    python -X utf8 -I scripts/regate_jev3_fetch.py <source> --out <receipt.json>
    sources: math | ambignq | tatqa | timeqa | privacyqa | helpsteer3
"""
import argparse, collections, csv, gzip, hashlib, io, json, re, sys, urllib.request, zipfile
from pathlib import Path

DATA = Path(r"C:\Users\apumu\research\CHLabs\ssdm-data")
BUILD = DATA / "m2" / "build"
DRAWS = {"train": BUILD / "bschema-draw-jev3" / "records.jsonl.gz",
         "slice": BUILD / "bschema-draw-jev3-slice" / "records.jsonl.gz"}
FETCH_LOG = DATA / "raw" / "jev3" / "fetch-log.jsonl"
HS3_URL = ("https://huggingface.co/datasets/nvidia/HelpSteer3/resolve/f6d145777bcbde96137596340fab89793acd1031/"
           "feedback/train.jsonl.gz")
HS3_SHA = "6f51cf5b7224436b0c5a4c80b280eddc566f7d2b589f5a4be0ba7a3637f0519c"


def canon(v):
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def norm(s):
    return " ".join(str(s).casefold().split())


def log():
    return [json.loads(l) for l in open(FETCH_LOG, encoding="utf-8") if l.strip()]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ssdm-evidence-regate"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def fetch_logged(substr):
    e = [x for x in log() if substr in x["url"]]
    assert len(e) == 1, (substr, len(e))
    e = e[0]
    b = get(e["url"])
    return b, {"url": e["url"], "expected_sha256": e["sha256"], "got_sha256": sha(b), "bytes": len(b),
               "sha_match": sha(b) == e["sha256"]}


def rows(source):
    for role, p in DRAWS.items():
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for l in fh:
                r = json.loads(l)
                if r.get("source") == source:
                    yield role, r


def detok(s):
    s = re.sub(r" ([.,;:!?%)\]])", r"\1", s)
    s = re.sub(r"([(\[$]) ", r"\1", s)
    s = re.sub(r" (['’]s)\b", r"\1", s)
    s = s.replace(" - ", "-").replace(" n't", "n't")
    return re.sub(r"\s+", " ", s).strip()


def finish(res, fails):
    res["unverified_ids"] = sorted(fails)
    res["n_unverified"] = len(fails)
    return res


# ------------------------------------------------------------------------------------------------ sources
def s_math():
    import pyarrow.parquet as pq
    files = [x for x in log() if "hendrycks_math" in x["url"]]
    train, test = set(), set()
    fl = []
    for e in files:
        b = get(e["url"])
        ok = sha(b) == e["sha256"]
        fl.append({"url": e["url"], "sha_match": ok, "bytes": len(b)})
        t = pq.read_table(io.BytesIO(b), columns=["problem"]).to_pylist()
        (test if "/test-" in e["url"] else train).update(norm(x["problem"]) for x in t)
        del b, t
    res = {"files": fl, "n_train_problems": len(train), "n_test_problems": len(test), "per_role": {}}
    fails = []
    c = collections.defaultdict(collections.Counter)
    for role, r in rows("prm800k"):
        p = norm(r["state"]["request"])
        uid = r["source_id"]
        c[role]["rows"] += 1
        c[role]["in_math_train"] += p in train
        c[role]["in_math_test"] += p in test
        c[role]["uid_test_prefix"] += uid.startswith("test/")
        if p in test or p not in train or not uid.startswith("train/"):
            fails.append(r["id"])
    res["per_role"] = {k: dict(v) for k, v in c.items()}
    res["math500_note"] = "MATH-500 is a subset of the MATH test split; 0 in_math_test implies 0 MATH-500"
    return finish(res, fails)


def s_ambignq():
    b, f = fetch_logged("ambignq_light.zip")
    z = zipfile.ZipFile(io.BytesIO(b))
    names = z.namelist()
    tr = {x["id"]: x for x in json.loads(z.read("train_light.json"))}
    dev = {x["id"] for x in json.loads(z.read("dev_light.json"))} if "dev_light.json" in names else set()
    res = {"file": f, "zip_members": names, "n_train": len(tr), "n_dev": len(dev)}
    c = collections.defaultdict(collections.Counter); fails = []
    for role, r in rows("ambignq"):
        c[role]["rows"] += 1
        i = r["source_id"]
        x = tr.get(i)
        ok = x is not None and sha(canon(x).encode()) == r["aug"]["jev1"]["prov"]["row_sha256"] and i not in dev
        if x is not None:
            ts = {a["type"] for a in x["annotations"]}
            kind = "single" if ts == {"singleAnswer"} else "multi" if ts == {"multipleQAs"} else "mixed"
            ok = ok and kind == r["aug"]["jev1"].get("annotation")
            c[role]["annotation_matches"] += kind == r["aug"]["jev1"].get("annotation")
        c[role]["in_dev"] += i in dev
        c[role]["verified" if ok else "FAIL"] += 1
        if not ok:
            fails.append(r["id"])
    res["per_role"] = {k: dict(v) for k, v in c.items()}
    return finish(res, fails)


def s_tatqa():
    b, f = fetch_logged("tatqa_dataset_train.json")
    data = json.loads(b)
    qmap = {}
    for d in data:
        for q in d["questions"]:
            qmap[(d["table"]["uid"], q["uid"])] = q
    res = {"file": f, "n_tables": len(data), "n_questions": len(qmap)}
    c = collections.defaultdict(collections.Counter); fails = []
    for role, r in rows("tatqa"):
        c[role]["rows"] += 1
        m = re.match(r"table uid (\S+) / question uid (\S+)", r["aug"]["jev1"]["prov"]["row_key"])
        q = qmap.get((m.group(1), m.group(2))) if m else None
        ok = q is not None and m.group(1) == r["source_id"] and \
            sha(canon(q).encode()) == r["aug"]["jev1"]["prov"]["row_sha256"]
        c[role]["verified" if ok else "FAIL"] += 1
        if not ok:
            fails.append(r["id"])
    res["per_role"] = {k: dict(v) for k, v in c.items()}
    return finish(res, fails)


def s_timeqa():
    b, f = fetch_logged("human_annotated_train.json")
    docs = {d["index"]: d for d in json.loads(b)}
    res = {"file": f, "n_docs": len(docs)}
    c = collections.defaultdict(collections.Counter); fails = []
    for role, r in rows("timeqa"):
        c[role]["rows"] += 1
        m = re.match(r"index (.+) question (\d+)$", r["aug"]["jev1"]["prov"]["row_key"])
        ok = False
        if m and m.group(1) in docs:
            d = docs[m.group(1)]; qi = int(m.group(2))
            if qi < len(d["questions"]):
                qt, ans = d["questions"][qi]
                a = " and ".join(detok(x["answer"]) for x in ans if x.get("answer"))
                src = {"index": d["index"], "question": qt, "answer": a}
                ok = sha(canon(src).encode()) == r["aug"]["jev1"]["prov"]["row_sha256"]
        c[role]["verified" if ok else "FAIL"] += 1
        if not ok:
            fails.append(r["id"])
    res["per_role"] = {k: dict(v) for k, v in c.items()}
    return finish(res, fails)


def s_privacyqa():
    b, f = fetch_logged("policy_train_data.csv")
    csv.field_size_limit(10 ** 8)
    splits = collections.Counter()
    qs = {}
    sents = collections.defaultdict(set)
    for row in csv.DictReader(io.StringIO(b.decode("utf-8")), delimiter="\t"):
        splits[row["Split"]] += 1
        if row["Split"] != "train":
            continue
        did, qid = row["DocID"], row["QueryID"]
        si = int(row["SentID"].rsplit("_", 1)[1])
        sents[did].add(re.sub(r"\s+", " ", row["Segment"]).strip())
        q = qs.setdefault((did, qid), {"query": row["Query"].strip(), "rel": set()})
        if row["Label"] == "Relevant":
            q["rel"].add(si)
    del b
    res = {"file": f, "csv_split_counts": dict(splits), "n_train_queries": len(qs)}
    c = collections.defaultdict(collections.Counter); fails = []
    for role, r in rows("privacyqa"):
        c[role]["rows"] += 1
        m = re.match(r"DocID (.*?) / QueryID (.*)$", r["aug"]["jev1"]["prov"]["row_key"])
        ok = False
        if m and (m.group(1), m.group(2)) in qs:
            did, qid = m.group(1), m.group(2); q = qs[(did, qid)]
            src = {"DocID": did, "QueryID": qid, "query": q["query"], "relevant_sent_ids": sorted(q["rel"])}
            ok = sha(canon(src).encode()) == r["aug"]["jev1"]["prov"]["row_sha256"]
            st = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"])
            hit = sum(1 for s in sents[did] if len(s) > 40 and s in st)
            c[role]["policy_text_from_train_doc"] += hit >= 3
            ok = ok and hit >= 3
        c[role]["verified" if ok else "FAIL"] += 1
        if not ok:
            fails.append(r["id"])
    res["per_role"] = {k: dict(v) for k, v in c.items()}
    return finish(res, fails)


V = re.compile(r"the response is (not|slightly|partially|mostly|perfectly) helpful", re.I)
SIDE = {"not": -1, "slightly": -1, "partially": 0, "mostly": 1, "perfectly": 1}


def verdict(fb):
    if isinstance(fb, str):
        try:
            fb = json.loads(fb)
        except ValueError:
            fb = [fb]
    vs = [m.group(1).lower() for t in fb or [] for m in [V.search(str(t)[:200])] if m]
    s = [SIDE[v] for v in vs]
    pos, neg = s.count(1), s.count(-1)
    if len(vs) >= 2 and pos > len(vs) / 2 and neg == 0:
        return 1
    if len(vs) >= 2 and neg > len(vs) / 2 and pos == 0:
        return 0
    return None


def s_helpsteer3():
    want = collections.defaultdict(list)
    for role, r in rows("helpsteer3"):
        line, k = r["source_id"].rsplit("-", 1)
        want[int(line)].append((role, int(k), r))
    h = hashlib.sha256()
    req = urllib.request.Request(HS3_URL, headers={"User-Agent": "ssdm-evidence-regate"})
    c = collections.defaultdict(collections.Counter); fails = []; seen = set()
    with urllib.request.urlopen(req, timeout=600) as resp:
        class Tee(io.RawIOBase):
            def readable(self):
                return True

            def readinto(self, b):
                d = resp.read(len(b))
                h.update(d)
                b[:len(d)] = d
                return len(d)
        g = gzip.GzipFile(fileobj=io.BufferedReader(Tee(), 1 << 20))
        n = 0
        for ln, raw in enumerate(io.TextIOWrapper(g, encoding="utf-8", newline="\n")):
            n += 1
            if ln not in want:
                continue
            d = json.loads(raw)
            lsha = sha(raw.rstrip("\n").encode())
            for role, k, r in want[ln]:
                seen.add(r["id"])
                c[role]["rows"] += 1
                row_ok = sha(canon({"line_sha256": lsha, "k": k}).encode()) == r["aug"]["jev1"]["prov"]["row_sha256"]
                resp_ok = d[f"response{k}"].strip() in (r["state"] if isinstance(r["state"], str) else "")
                v = verdict(d[f"feedback{k}"])
                gold_ok = v is not None and (r["labels"]["q"] == "true") == (v == 1)
                c[role]["row_sha_match"] += row_ok
                c[role]["response_text_match"] += resp_ok
                c[role]["gold_equals_human_consensus"] += gold_ok
                ok = row_ok and resp_ok and gold_ok
                c[role]["verified" if ok else "FAIL"] += 1
                if not ok:
                    fails.append(r["id"])
        rest = resp.read()
        h.update(rest)
    for ln, lst in want.items():
        for role, k, r in lst:
            if r["id"] not in seen:
                fails.append(r["id"]); c[role]["line_not_found"] += 1
    res = {"file": {"url": HS3_URL, "expected_sha256": HS3_SHA, "got_sha256": h.hexdigest(),
                    "sha_match": h.hexdigest() == HS3_SHA, "lines": n},
           "per_role": {k: dict(v) for k, v in c.items()}}
    return finish(res, fails)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["math", "ambignq", "tatqa", "timeqa", "privacyqa", "helpsteer3"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = globals()[f"s_{a.source}"]()
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "unverified_ids"})[:3000])
    print("unverified", res["n_unverified"], res["unverified_ids"][:10])


if __name__ == "__main__":
    main()
