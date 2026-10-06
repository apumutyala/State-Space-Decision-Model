"""Deterministic, stratified, disjoint DEV / TEST slices of the 0.2.1 index rows.

Unit of sampling: a *source case* (`harness.cases`): the connected component of rows that share
a kit group, a source key (contest, trial, event, song, rater, HotpotQA question, ...), a
byte-identical payload or a byte-identical non-trivial state. No source case straddles DEV
and TEST, so identical payloads are deduplicated across slices by construction. A case larger
than `max_case_share` (default 25%) of its benchmark's slice quota cannot enter either slice
and stays in the pool (NLI4CT trial chains and ForecastBench joint-event chains); this keeps
slice sizes on target at the cost of under-sampling the most interlinked sources.

Allocation:
1. Per area, a row target (`area_target`, default 2,400) is spread equally over the
   area's benchmarks, capped at `cap` (default 45%) of each benchmark's rows, with the
   shortfall water-filled onto benchmarks that still have headroom. DEV and TEST get the
   same quota; at least 10% of every benchmark stays in an unused pool.
2. Inside a benchmark, the quota is split over request-length bins in proportion to the
   rows in each bin (largest remainder).
3. Inside a (benchmark, length-bin) stratum (bin = longest request of the case), cases are
   ordered by sha256(f"{seed}:{catalog_id}:{case_id}"); TEST takes cases from the front until
   its row quota is met, DEV takes the next cases.
4. Slice rows and manifests are ordered by run_id, independent of loader order.

Benchmarks missing from the local build (HLE is gated) are allocated with their published
row count, so adding them later does not move any other benchmark's slice.
"""

import collections
import hashlib
import json

from harness import cases as cases_mod
from harness import config
from harness.suite import area_of, index_ids, row_sha256, spec

# Published scoreable rows (0.2.1, after exclusions) for benchmarks that may be absent locally.
PUBLISHED_ROWS = {45: 501}


def group_key(row):
    e = row["_evaluation"]
    return e["catalog_id"], str(e["group_id"])


def priority(seed, catalog_id, group_id):
    return hashlib.sha256(f"{seed}:{catalog_id}:{group_id}".encode()).hexdigest()


def group_bin(rows):
    return config.length_bin(max(r["_evaluation"]["proxy_tokens"] for r in rows))


def water_fill(sizes, total, cap):
    """Equal split of `total` over benchmarks, each capped at floor(cap * size)."""
    caps = {b: int(cap * n) for b, n in sizes.items()}
    alloc = {b: 0 for b in sizes}
    remaining = total
    open_ = set(sizes)
    while remaining > 0 and open_:
        share = remaining / len(open_)
        done = set()
        for b in sorted(open_):
            add = min(share, caps[b] - alloc[b])
            alloc[b] += add
            remaining -= add
            if alloc[b] >= caps[b] - 1e-9:
                done.add(b)
        if not done:
            break
        open_ -= done
    return {b: int(round(v)) for b, v in alloc.items()}


def largest_remainder(quota, weights):
    total = sum(weights.values())
    if total == 0:
        return {k: 0 for k in weights}
    shares = {k: quota * w / total for k, w in weights.items()}
    out = {k: int(v) for k, v in shares.items()}
    for k in sorted(shares, key=lambda k: (-(shares[k] - out[k]), k))[: quota - sum(out.values())]:
        out[k] += 1
    return out


def quotas(sizes, area_target, cap):
    areas = collections.defaultdict(dict)
    amap = area_of()
    for b, n in sizes.items():
        areas[amap[b]][b] = n
    out = {}
    for a, s in areas.items():
        out.update(water_fill(s, area_target, cap))
    return out


def build(rows, seed=config.SLICE_SEED, area_target=2400, cap=0.45, max_case_share=0.25):
    """Return ({"test": [...rows], "dev": [...rows]}, allocation table)."""
    ids = set(index_ids())
    rows = [r for r in rows if r["_evaluation"]["catalog_id"] in ids]
    case_of = cases_mod.case_of(rows)
    by_case = collections.OrderedDict()
    for r in rows:
        by_case.setdefault((r["_evaluation"]["catalog_id"], case_of(r)), []).append(r)
    by_bench = collections.defaultdict(list)
    for (n, cid), rs in by_case.items():
        by_bench[n].append((cid, rs))
    sizes = {b: sum(len(rs) for _, rs in by_bench[b]) if b in by_bench else PUBLISHED_ROWS.get(b, 0) for b in ids}
    q = quotas(sizes, area_target, cap)
    out = {"test": [], "dev": []}
    table = {}
    for b in sorted(by_bench):
        limit = max(1, int(max_case_share * q[b]))
        strata = collections.defaultdict(list)
        too_big = 0
        for cid, rs in by_bench[b]:
            if len(rs) > limit:
                too_big += len(rs)
                continue
            strata[group_bin(rs)].append((priority(seed, b, cid), cid, rs))
        bin_rows = {k: sum(len(rs) for _, _, rs in v) for k, v in strata.items()}
        per_bin = largest_remainder(q[b], bin_rows)
        taken = {"test": collections.Counter(), "dev": collections.Counter()}
        cases_taken = collections.Counter()
        for k in sorted(strata):
            ordered = sorted(strata[k])
            i = 0
            for name in ("test", "dev"):
                n = 0
                while i < len(ordered) and n < per_bin[k]:
                    rs = ordered[i][2]
                    out[name].extend(rs)
                    n += len(rs)
                    i += 1
                    cases_taken[name] += 1
                taken[name][k] = n
        table[b] = {"rows": sizes[b], "cases": len(by_bench[b]), "quota": q[b], "max_case_rows": limit, "rows_in_oversized_cases": too_big, "bins": dict(sorted(bin_rows.items())), "test": dict(taken["test"]), "dev": dict(taken["dev"]), "test_cases": cases_taken["test"], "dev_cases": cases_taken["dev"]}
    for b in ids - set(by_bench):
        table[b] = {"rows": sizes[b], "quota": q[b], "missing_locally": True}
    for name in out:
        out[name].sort(key=lambda r: r["_evaluation"]["run_id"])
    return out, table


def manifest_entries(rows):
    for r in sorted(rows, key=lambda r: r["_evaluation"]["run_id"]):
        e = r["_evaluation"]
        yield {
            "run_id": e["run_id"],
            "catalog_id": e["catalog_id"],
            "group_id": str(e["group_id"]),
            "track": e["track"],
            "proxy_tokens": e["proxy_tokens"],
            "length_bin": config.length_bin(e["proxy_tokens"]),
            "payload_sha256": e["payload_sha256"],
            "row_sha256": row_sha256(r),
        }


def summarize(rows):
    amap = area_of()
    by_area = collections.Counter()
    by_bench = collections.Counter()
    by_bin = collections.Counter()
    by_area_bin = collections.defaultdict(collections.Counter)
    groups = set()
    for r in rows:
        e = r["_evaluation"]
        a = amap[e["catalog_id"]]
        lb = config.length_bin(e["proxy_tokens"])
        by_area[a] += 1
        by_bench[e["catalog_id"]] += 1
        by_bin[lb] += 1
        by_area_bin[a][lb] += 1
        groups.add((e["catalog_id"], str(e["group_id"])))
    case_of = cases_mod.case_of(rows)
    n_cases = len({(r["_evaluation"]["catalog_id"], case_of(r)) for r in rows})
    order = [a["id"] for a in spec()["areas"]]
    return {
        "rows": len(rows),
        "groups": len(groups),
        "source_cases": n_cases,
        "benchmarks": len(by_bench),
        "by_area": {a: by_area[a] for a in order},
        "by_length_bin": {b: by_bin[b] for b in config.LENGTH_BIN_LABELS},
        "by_area_and_length_bin": {a: {b: by_area_bin[a][b] for b in config.LENGTH_BIN_LABELS} for a in order},
        "by_benchmark": {str(k): v for k, v in sorted(by_bench.items())},
    }


def write_manifest(path, rows):
    """Gzipped JSONL of IDs and hashes only (no state, questions or gold). Returns sha256 of the uncompressed text."""
    import gzip

    path.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    with gzip.GzipFile(path, "wb", mtime=0) as g:
        for entry in manifest_entries(rows):
            line = (json.dumps(entry, separators=(",", ":"), sort_keys=True) + "\n").encode()
            h.update(line)
            g.write(line)
    return h.hexdigest()


def read_manifest(path):
    import gzip

    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def select(rows, manifest):
    """Materialise a slice from the suite and check every row hash against the manifest."""
    want = {m["run_id"]: m for m in manifest}
    found = {}
    for r in rows:
        m = want.get(r["_evaluation"]["run_id"])
        if m is None:
            continue
        if row_sha256(r) != m["row_sha256"]:
            raise ValueError(f"row hash mismatch for {m['run_id']}")
        found[m["run_id"]] = r
    if len(found) != len(want):
        raise ValueError(f"manifest lists {len(want)} rows, suite provided {len(found)}")
    return [found[m["run_id"]] for m in manifest]
