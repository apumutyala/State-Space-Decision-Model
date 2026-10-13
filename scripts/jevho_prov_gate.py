"""jev-heldout gate: split provenance. Every row's source key must resolve to its stated NON-TRAIN split file (or, for
tool_selection, to a Glaive train row no m2 build ever used; for the rule engines, to an engine index >= 10,000,000),
and the sha256 of the canonical source record must equal the row's ``aug.jev1.prov.row_sha256``.

    python -X utf8 -I scripts/jevho_prov_gate.py --records <records.jsonl.gz> --out <json> [--drop-ids-out <txt>]

Also: PRM800K problems must be MATH-test problems (math_splits/test.jsonl unique_id test/*); AmbigNQ dev ids must be
absent from train_light.json; MASSIVE rows must carry partition test / dev in the tarball. Prints counts only.
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
import tarfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def canon(v) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(v) -> str:
    return hashlib.sha256(canon(v).encode()).hexdigest()


def main(argv=None) -> int:
    from data import jev_heldout as H
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop-ids-out")
    a = ap.parse_args(argv)
    rows = [json.loads(l) for l in gzip.open(a.records, "rt", encoding="utf-8")]
    need = collections.Counter(r["source"] for r in rows)
    R = H.RAW
    idx: dict[str, dict] = {}
    if need["helpsteer2"]:
        idx["helpsteer2"] = {i: sha(json.loads(l)) for i, l in
                             enumerate(gzip.open(R / "helpsteer2" / "validation.jsonl.gz", "rt", encoding="utf-8"))}
    if need["aegis2"]:
        idx["aegis2"] = {(s, str(x["id"])): sha(x) for s in ("validation", "test")
                         for x in json.load(open(R / "aegis2" / f"{s}.json", encoding="utf-8"))}
    if need["prm800k"]:
        test_uid = {json.loads(l)["unique_id"] for l in open(R / "prm800k" / "math_splits_test.jsonl", encoding="utf-8")}
        lines = {}
        for fn in ("phase1_test.jsonl", "phase2_test.jsonl"):
            for ln, raw in enumerate(open(R / "prm800k" / fn, encoding="utf-8")):
                lines[(fn, ln)] = hashlib.sha256(raw.rstrip("\n").encode()).hexdigest()
        idx["prm800k"] = lines
    if need["ambignq"]:
        z = zipfile.ZipFile(R / "ambigqa" / "ambignq_light.zip")
        dev = {x["id"]: sha(x) for x in json.loads(z.read("dev_light.json"))}
        train_ids = {x["id"] for x in json.loads(z.read("train_light.json"))}
        idx["ambignq"] = dev
    if need["tatqa"]:
        tq = {}
        for d in json.load(open(R / "tatqa" / "tatqa_dataset_dev.json", encoding="utf-8")):
            for q in d["questions"]:
                tq[(d["table"]["uid"], q["uid"])] = sha(q)
        idx["tatqa"] = tq
    if need["timeqa"]:
        ti = {}
        for d in json.load(open(R / "timeqa" / "human_annotated_test.json", encoding="utf-8")):
            for qi, (qt, ans) in enumerate(d["questions"]):
                ti[(d["index"], qi)] = qt
        idx["timeqa"] = ti
    if need["privacyqa"]:
        csv.field_size_limit(10 ** 8)
        pq = set()
        with open(R / "privacyqa" / "policy_test_data.csv", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row["Split"] == "test":
                    pq.add((row["DocID"], row["QueryID"]))
        idx["privacyqa"] = pq
    if need["boolq"]:
        idx["boolq"] = {json.loads(l)["row"]: sha(json.loads(l)) for l in
                        gzip.open(R / "boolq" / "validation.jsonl.gz", "rt", encoding="utf-8")}
    if need["musique"]:
        idx["musique"] = {json.loads(l)["id"]: hashlib.sha256(l.rstrip("\n").encode()).hexdigest()
                          for l in open(R / "musique" / "musique_ans_v1.0_dev.jsonl", encoding="utf-8")}
    if need["massive"]:
        t = tarfile.open(H.MASSIVE_TAR, "r:gz")
        for m in t:
            if m.name.endswith("/en-US.jsonl"):
                idx["massive"] = {x["id"]: (x["partition"], sha(x)) for x in map(json.loads, t.extractfile(m))}
                break
    if need["sst5"]:
        idx["sst5"] = {(s, i): sha(json.loads(l)) for s in ("test", "dev")
                       for i, l in enumerate(open(R / "sst5" / f"{s}.jsonl", encoding="utf-8")) if l.strip()}
    if need["glaive_fc"]:
        used = set(json.loads((R / "glaive" / "used-ids.json").read_text(encoding="utf-8"))["ids"])
        idx["glaive_fc"] = {json.loads(l)["i"]: json.loads(l)["row_sha256"]
                            for l in gzip.open(R / "glaive" / "candidates.jsonl.gz", "rt", encoding="utf-8")}
    fails = collections.defaultdict(collections.Counter)
    bad_ids, ok = [], collections.Counter()
    for r in rows:
        s, pv, fam = r["source"], r["aug"]["jev1"]["prov"], r["aug"]["jev1"]["family8"]
        rk, rs = pv["row_key"], pv["row_sha256"]
        why = None
        if r["source_split"] == "train" or pv["split"] == "train":
            why = "train_split"
        elif s == "helpsteer2":
            i = int(re.search(r"row (\d+)", rk).group(1))
            why = None if idx[s].get(i) == rs else "unresolved_or_sha"
        elif s == "aegis2":
            sp, i = re.match(r"(\w+)\.json id (.+)$", rk).groups()
            why = None if idx[s].get((sp, i)) == rs and sp == r["source_split"] else "unresolved_or_sha"
        elif s == "prm800k":
            fn, ln = re.match(r"(\S+) line (\d+)$", rk).groups()
            src_sha = idx[s].get((fn, int(ln)))
            want = sha({"line_sha256": src_sha}) if src_sha else None
            uid = r["aug"]["jev1"].get("math_unique_id", "")
            why = None if (want == rs and uid in test_uid and uid.startswith("test/")) else "unresolved_or_not_math_test"
        elif s == "ambignq":
            i = re.search(r"id (\S+)$", rk).group(1)
            why = None if idx[s].get(i) == rs and i not in train_ids else "unresolved_or_in_train"
        elif s == "tatqa":
            tu, qu = re.match(r"table uid (\S+) / question uid (\S+)$", rk).groups()
            why = None if idx[s].get((tu, qu)) == rs else "unresolved_or_sha"
        elif s == "timeqa":
            ix, qi = re.match(r"index (.+) question (\d+)$", rk).groups()
            why = None if (ix, int(qi)) in idx[s] else "unresolved"
        elif s == "privacyqa":
            did, qid = re.match(r"DocID (.+) / QueryID (.+)$", rk).groups()
            why = None if (did, qid) in idx[s] else "unresolved"
        elif s == "boolq":
            i = int(re.search(r"row (\d+)", rk).group(1))
            why = None if idx[s].get(i) == rs else "unresolved_or_sha"
        elif s == "musique":
            i = re.search(r"dev id (\S+)$", rk).group(1)
            src_sha = idx[s].get(i)
            why = None if src_sha and sha({"line_sha256": src_sha}) == rs else "unresolved_or_sha"
        elif s == "massive":
            i = re.search(r"id (\S+) \(partition (\w+)\)", rk).groups()
            got = idx[s].get(i[0])
            why = None if got and got[0] == i[1] == r["source_split"] and got[0] in ("test", "dev") and got[1] == rs \
                else "unresolved_or_partition"
        elif s == "sst5":
            sp, i = re.match(r"(\w+)\.jsonl line (\d+)$", rk).groups()
            why = None if idx[s].get((sp, int(i))) == rs else "unresolved_or_sha"
        elif s == "glaive_fc":
            i = int(re.search(r"row (\d+)", rk).group(1))
            got = idx[s].get(i)
            why = None if got and sha({"row_sha256": got}) == rs and i not in used else "unresolved_or_used"
        elif s == "jevho_synth":
            i = int(re.search(r"index (\d+)$", rk).group(1))
            why = None if i >= H.SYN_OFFSET and r["source_split"] == "synthetic" else "engine_index_overlap"
        else:
            why = "unknown_source"
        if why:
            fails[fam][why] += 1
            bad_ids.append(r["id"])
        else:
            ok[f"{s}:{r['source_split']}"] += 1
    rep = {"gate": "jev-heldout split provenance (non-train split file + row sha; PRM800K MATH-test; AmbigNQ dev not "
                   "in train; MASSIVE partition; Glaive unused; engine index offset)", "n": len(rows),
           "ok_by_source_split": dict(ok), "fails_by_family": {k: dict(v) for k, v in fails.items()},
           "n_fail": len(bad_ids), "pass": not bad_ids,
           "records_sha256": hashlib.sha256(gzip.open(a.records).read()).hexdigest()}
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    if a.drop_ids_out:
        Path(a.drop_ids_out).write_text("".join(i + "\n" for i in bad_ids), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n", "n_fail", "fails_by_family", "pass")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
