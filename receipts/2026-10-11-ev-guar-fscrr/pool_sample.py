"""B-PREREG-GUAR-FSCRR, the unseen split: a seeded, answer-free sample of the Decision Index 0.2.1 POOL.

The pool = every scoreable 0.2.1 index row that is in neither the DEV nor the TEST manifest (harness/slicing.py
leaves >= 10% of every benchmark, plus every case too large for a slice, in an unused pool). Slices take whole
source cases, so no pool row shares a source case with DEV or TEST. No model has ever been run on a pool row, and
no guarantee analysis has used one. TEST is touched only through its manifest's run ids (membership, never content).

    python -X utf8 -I pool_sample.py <checkout with harness/> <out dir>

Streams the suite twice (laptop RAM is tight; peak ~ the union-find over pool link keys + the selected raw lines).

Procedure (fixed in B-PREREG-GUAR-FSCRR before any prediction on these rows exists):
  1. eligible = pool rows with >= 1 gold decision (expected[key] not null; gold presence only, never its value);
  2. source cases over the pool rows = harness.cases components (same link keys as the slices);
  3. per benchmark b: quota q_b = number of DEV rows of b with >= 1 gold decision (the sample mirrors DEV's
     composition); cases with more than max(1, int(0.25 q_b)) eligible rows are skipped (the slices' case cap);
     cases ordered by sha256("B-PREREG-GUAR-FSCRR|pool|{b}:{case}") ascending; whole cases are taken while the
     running row count is < q_b (so the last case may overshoot); fewer eligible rows than q_b -> all of them.
Outputs: pool-sample-rows.jsonl.gz (rows in run_id order, byte-for-byte the suite lines), pool-sample-manifest.jsonl.gz
(slicing.manifest_entries fields), pool_sample.json (counts, quotas, ids sha256, input shas).
"""
import collections
import gzip
import hashlib
import json
import os
import sys

sys.path.insert(0, sys.argv[1])
from harness import cases, config, slicing  # noqa: E402
from harness.suite import dumps, index_rows, load_suite, row_sha256  # noqa: E402

SALT = "B-PREREG-GUAR-FSCRR|pool"
CASE_SHARE = 0.25


def n_gold(r):
    return sum(1 for k in r["questions"] if r["expected"].get(k) is not None)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ids_sha256(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def main():
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    dev_m = slicing.read_manifest(config.MANIFEST_DIR / "dev-manifest.jsonl.gz")
    test_ids = {m["run_id"] for m in slicing.read_manifest(config.MANIFEST_DIR / "test-manifest.jsonl.gz")}
    dev_ids = {m["run_id"] for m in dev_m}
    del dev_m
    assert not dev_ids & test_ids
    # DEV quotas: rows with >= 1 gold decision per benchmark (materialised DEV rows, streamed)
    quota = collections.Counter()
    with gzip.open(config.SLICE_ROWS_DIR / "dev-rows.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if n_gold(r) >= 1:
                quota[r["_evaluation"]["catalog_id"]] += 1
    # pass 1: pool rows, union-find over link keys (harness.cases.case_ids, inlined to stream)
    uf, groups = cases._UF(), set()
    info = {}  # run_id -> (catalog, group_id, n_gold)
    seen_dev = seen_test = 0
    for r in index_rows(load_suite()):
        e = r["_evaluation"]
        rid = e["run_id"]
        if rid in dev_ids:
            seen_dev += 1
            continue
        if rid in test_ids:
            seen_test += 1
            continue
        g = ("G", e["catalog_id"], str(e["group_id"]))
        groups.add(g)
        for k in cases.link_keys(r):
            uf.union(g, ("K",) + k)
        info[rid] = (e["catalog_id"], str(e["group_id"]), n_gold(r))
    assert seen_dev == len(dev_ids) and seen_test == len(test_ids), (seen_dev, seen_test)
    members = collections.defaultdict(list)
    for g in groups:
        members[uf.find(g)].append(g)
    del uf, groups
    case_of_group = {}
    for comp in members.values():
        cid = f"{comp[0][1]}:{min(m[2] for m in comp)}"
        for m in comp:
            case_of_group[(m[1], m[2])] = cid
    del members
    # selection
    by_case = collections.defaultdict(list)
    pool_by_bench = collections.Counter()
    for rid, (b, gid, ng) in info.items():
        pool_by_bench[b] += 1
        if ng >= 1:
            by_case[(b, case_of_group[(b, gid)])].append(rid)
    by_bench = collections.defaultdict(list)
    for (b, cid), rids in by_case.items():
        by_bench[b].append((hashlib.sha256(f"{SALT}|{b}:{cid}".encode()).hexdigest(), cid, sorted(rids)))
    chosen, table = {}, {}
    for b in sorted(set(by_bench) | set(quota)):
        q = quota.get(b, 0)
        limit = max(1, int(CASE_SHARE * q))
        cands = sorted(by_bench.get(b, []))
        eligible = sum(len(x[2]) for x in cands)
        skipped = sum(len(x[2]) for x in cands if len(x[2]) > limit)
        n = n_cases = 0
        for _, cid, rids in cands:
            if n >= q:
                break
            if len(rids) > limit:
                continue
            for rid in rids:
                chosen[rid] = cid
            n += len(rids)
            n_cases += 1
        table[str(b)] = {"pool_rows": pool_by_bench.get(b, 0), "pool_rows_with_gold": eligible, "dev_quota": q,
                         "max_case_rows": limit, "rows_in_oversized_cases": skipped, "taken_rows": n,
                         "taken_cases": n_cases}
    # pass 2: write the chosen rows byte-for-byte (kit dumps of the parsed row = harness.suite.write_rows, the slice convention)
    lines = {}
    for r in index_rows(load_suite()):
        rid = r["_evaluation"]["run_id"]
        if rid in chosen:
            lines[rid] = (dumps(r), row_sha256(r), r["_evaluation"])
    assert set(lines) == set(chosen)
    rows_path = os.path.join(out_dir, "pool-sample-rows.jsonl.gz")
    man_path = os.path.join(out_dir, "pool-sample-manifest.jsonl.gz")
    h_man = hashlib.sha256()
    with gzip.GzipFile(rows_path, "wb", mtime=0) as gr, gzip.GzipFile(man_path, "wb", mtime=0) as gm:
        for rid in sorted(lines):
            text, rsha, e = lines[rid]
            gr.write((text + "\n").encode())
            entry = {"run_id": rid, "catalog_id": e["catalog_id"], "group_id": str(e["group_id"]), "track": e["track"],
                     "proxy_tokens": e["proxy_tokens"], "length_bin": config.length_bin(e["proxy_tokens"]),
                     "payload_sha256": e["payload_sha256"], "row_sha256": rsha, "case_id": chosen[rid]}
            line = (json.dumps(entry, separators=(",", ":"), sort_keys=True) + "\n").encode()
            h_man.update(line)
            gm.write(line)
    rep = {"prereg": "B-PREREG-GUAR-FSCRR", "salt": SALT, "case_share": CASE_SHARE, "edition": config.EDITION,
           "dev_manifest_sha256": sha_file(config.MANIFEST_DIR / "dev-manifest.jsonl.gz"),
           "test_manifest_sha256": sha_file(config.MANIFEST_DIR / "test-manifest.jsonl.gz"),
           "pool_rows": len(info), "pool_ids_sha256": ids_sha256(info),
           "sample_rows": len(chosen), "sample_cases": len(set(chosen.values())),
           "sample_ids_sha256": ids_sha256(chosen), "manifest_text_sha256": h_man.hexdigest(),
           "rows_file_sha256": sha_file(rows_path), "manifest_file_sha256": sha_file(man_path),
           "dev_quota_total": sum(quota.values()), "by_benchmark": table}
    with open(os.path.join(out_dir, "pool_sample.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "by_benchmark"}, indent=1))


if __name__ == "__main__":
    main()
