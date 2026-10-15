"""SPLIT-PROVENANCE gate for bschema-draw-jev3 (FAMILY_SPEC_JEV3.md; coordinator 2026-10-11): every row traces to its
dataset's TRAIN split by id / hash, with dataset revision + licence recorded per row; PRM800K rows must be MATH TRAIN
problems and never MATH TEST (incl. MATH-500), checked by normalised problem text against the official MATH files.

    python -X utf8 -I scripts/jev3_prov_gate.py --records R1.jsonl.gz [R2 ...] --out prov-gate.json [--drop-ids-out f]

MATH reference: EleutherAI/hendrycks_math @21a5633873b6a120296cce3e2df9d5550074f4a3 (7 subjects, train + test parquet,
fetched with scripts/jev3_fetch.py into ssdm-data/raw/jev3/math/). Needs pyarrow (m2-data venv).
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from data import jev3  # noqa: E402

MATH_REV = "21a5633873b6a120296cce3e2df9d5550074f4a3"
SUBJ = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra",
        "precalculus"]


def nprob(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def math_sets(root: Path) -> tuple[dict, dict]:
    import pyarrow.parquet as pq
    out, shas = {"train": set(), "test": set()}, {}
    for sp in ("train", "test"):
        for s in SUBJ:
            p = root / f"{s}-{sp}.parquet"
            shas[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
            for x in pq.read_table(p, columns=["problem"]).column("problem").to_pylist():
                out[sp].add(nprob(x))
    return out, shas


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-ids-out")
    a = ap.parse_args(argv)
    raw = jev3.RAW
    M, mshas = math_sets(raw / "math")
    # PRM800K cache level: how many cached (math_splits train/*) problems are MATH train / MATH test by text
    cache = [json.loads(l) for l in gzip.open(raw / "prm800k" / "prm_cache.jsonl.gz", "rt", encoding="utf-8")]
    meta = cache[0]["_meta"]
    cache = cache[1:]
    cstat = collections.Counter()
    for d in cache:
        p = nprob(d["problem"])
        cstat["in_math_train"] += p in M["train"]
        cstat["in_math_test"] += p in M["test"]
        cstat["in_neither"] += p not in M["train"] and p not in M["test"]
    amb = zipfile.ZipFile(raw / "ambigqa" / "ambignq_light.zip")
    amb_train = {x["id"] for x in json.loads(amb.read("train_light.json"))}
    amb_dev = {x["id"] for x in json.loads(amb.read("dev_light.json"))}
    tat = {d["table"]["uid"]: {q["uid"] for q in d["questions"]} for d in
           json.load(open(raw / "tatqa" / "tatqa_dataset_train.json", encoding="utf-8"))}
    tq = {d["index"]: d for d in json.load(open(raw / "timeqa" / "human_annotated_train.json", encoding="utf-8"))}
    csv.field_size_limit(10 ** 8)
    pqa = set()
    with open(raw / "privacyqa" / "policy_train_data.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["Split"] == "train":
                pqa.add((row["DocID"], row["QueryID"]))
    used = jev3._jev1_used()
    hs_n = sum(1 for _ in gzip.open(jev3.DL1 / "helpsteer2" / "train.jsonl.gz", "rt", encoding="utf-8"))
    ae_ids = {str(x["id"]) for x in json.load(open(jev3.DL1 / "aegis2" / "train.json", encoding="utf-8"))}
    # HelpSteer3 feedback TRAIN (streamed; the cache keeps line no + line sha256 of train.jsonl.gz rows)
    hs3 = set()
    for l in gzip.open(raw / "helpsteer3" / "hs3_cache.jsonl.gz", "rt", encoding="utf-8"):
        d = json.loads(l)
        if "_meta" not in d:
            hs3.add((d["line"], d["k"]))
    fails = collections.defaultdict(list)
    by = collections.Counter()
    rec_sha = {}
    prm_rows = collections.Counter()
    for path in a.records:
        rec_sha[path] = hashlib.sha256(gzip.open(path, "rb").read()).hexdigest()
        for line in gzip.open(path, "rt", encoding="utf-8"):
            r = json.loads(line)
            rid, src, j = r["id"], r["source"], r["aug"]["jev1"]
            pv = j.get("prov") or {}
            by[src] += 1
            want = jev3.SRC.get(src)
            if not want or r.get("source_split") != "train" or pv.get("split") != "train" or \
                    pv.get("license") != want["license"] or pv.get("revision") != want["revision"] or \
                    pv.get("dataset") != want["dataset"] or not pv.get("row_key") or not pv.get("row_sha256"):
                fails["prov_fields"].append(rid)
                continue
            sid = r["source_id"]
            ok = True
            if src == "prm800k":
                p = nprob(r["state"]["request"])
                prm_rows["rows"] += 1
                if p in M["test"]:
                    prm_rows["math_test_match"] += 1
                    ok = False
                if p not in M["train"]:
                    prm_rows["not_math_train"] += 1
                    ok = False
                if not str(j.get("math_unique_id", "")).startswith("train/"):
                    ok = False
            elif src == "ambignq":
                ok = sid in amb_train and sid not in amb_dev
            elif src == "tatqa":
                tuid, quid = re.match(r"table uid (\S+) / question uid (\S+)", pv["row_key"]).groups()
                ok = sid == tuid and tuid in tat and quid in tat[tuid]
            elif src == "timeqa":
                idx = re.match(r"index (.+) question (\d+)$", pv["row_key"])
                ok = bool(idx) and idx.group(1) in tq and int(idx.group(2)) < len(tq[idx.group(1)]["questions"])
            elif src == "privacyqa":
                m = re.match(r"DocID (.+) / QueryID (.+)$", pv["row_key"])
                ok = bool(m) and (m.group(1), m.group(2)) in pqa
            elif src == "helpsteer2":
                ok = 0 <= int(sid) < hs_n and sid not in used["helpsteer2"]
            elif src == "helpsteer3":
                m = re.match(r"feedback/train\.jsonl\.gz line (\d+) response([12])$", pv["row_key"])
                ok = bool(m) and (int(m.group(1)), int(m.group(2))) in hs3
            elif src == "aegis2":
                ok = sid in ae_ids and sid not in used["aegis2"]
            else:
                ok = False
            if not ok:
                fails[f"split_trace:{src}"].append(rid)
    n_fail = sum(len(v) for v in fails.values())
    rep = {"records": rec_sha, "rows_by_source": dict(by), "fails": {k: len(v) for k, v in fails.items()},
           "fail_ids": {k: v[:50] for k, v in fails.items()}, "pass": n_fail == 0,
           "prm800k": {"stream_stats": meta["stats"], "stream_sha256": meta["stream_sha256"],
                       "cached_problems": len(cache), "cache_vs_official_math": dict(cstat),
                       "rows_vs_official_math": dict(prm_rows),
                       "rule": "problem text (whitespace-normalised, casefolded) in MATH TRAIN and NOT in MATH TEST "
                               "(MATH-500 is a subset of MATH TEST); PRM800K math_splits unique_id starts with train/",
                       "math_reference": {"repo": "EleutherAI/hendrycks_math", "revision": MATH_REV, "files": mshas,
                                          "n_train": len(M["train"]), "n_test": len(M["test"])}},
           "ambignq_dev_ids_excluded": len(amb_dev)}
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    if a.drop_ids_out:
        Path(a.drop_ids_out).write_text("\n".join(sorted({x for v in fails.values() for x in v})) + "\n",
                                        encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("rows_by_source", "fails", "pass")}),
          json.dumps({k: v for k, v in rep["prm800k"].items() if k != "math_reference"}))
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
