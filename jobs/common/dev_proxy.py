"""Pre-declared stratified PROXY of the 2,008-row m1 DEV subset, for screening trained cells cheaply (B-2).

A trained cell's in-job Index eval on the 2,008-row subset (``jobs/common/dev_subset_m1.json``) is ~1 h of child
scoring; a ~600-row proxy scored with the SAME frozen scorer cuts it to ~1/3 so cells can be screened, and only the
winners get the full subset + full DEV. The proxy is a subset of the m1 subset, in the same file shape, so every
``jobs.common.dev_subset`` command takes it as ``--subset jobs/common/dev_proxy600.json``.

Construction (fixed rule, declared BEFORE any validation; uses the DEV rows for case linking + lengths, never any
model output):
  1. strata = the Index benchmarks (catalog_id; all 38 kept, so the slice index renormalises nothing away) x
     request-length bins (``config.LENGTH_BIN_LABELS``; a case's bin = the bin of its longest row);
  2. unit = a source CASE (``harness.cases.case_ids`` over the 2,008 rows: linked kit groups = the scorer's bootstrap
     cluster), so a case is always whole and no group straddles the boundary;
  3. per-benchmark row quota q_n (largest remainder, iteratively capped at the benchmark's rows) from a pre-declared
     allocation ``design``:
       rows   q_n ~ rows_n                    (the m1 subset's own proportions, scaled down)
       weight q_n ~ w_n                       (the edition's effective Index weight, ``harness.score.weights``)
       mix    q_n ~ (rows_n / R + w_n / W) / 2
     with q_n >= min_per_bench = max(4, round(n_target / 100));
  4. inside a benchmark, q_n is split over its length bins in proportion to rows (largest remainder); inside a
     (benchmark, bin) stratum cases are taken in sha256(f"{seed}:{catalog_id}:{case_id}") order while the stratum
     is below its quota (the last case may overshoot by its own size).
Validation (leave-one-system-out over ~15 scored systems; design choice by held-out systems only):
``analysis/proxy_validate.py``. Receipt: ``receipts/dev_proxy600/``.

CLI (needs the kit + the private DEV rows):
  python -X utf8 -m jobs.common.dev_proxy build --n 600 --design weight --seed proxy-v1 --out jobs/common/dev_proxy600.json
  python -X utf8 -m jobs.common.dev_proxy check [--proxy jobs/common/dev_proxy600.json]
Scoring: ``python -X utf8 -m jobs.common.dev_subset score|compare --subset jobs/common/dev_proxy600.json ...``.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROXY600 = HERE / "dev_proxy600.json"
#: the committed proxy files by row mode (``rows: "proxy600"``), each pinned by its ids_sha256
PROXIES = {"proxy600": (PROXY600, "7d42c398336e0934e4fba8d7d78bda07d0073967fc92a8af7b3a14253a9185e6")}
SEED = "proxy-v1"
DESIGNS = ("rows", "weight", "mix")
LENGTH_BINS = ("<128", "128-511", "512-2047", "2048-8191", ">=8192")


def min_per_bench(n_target: int) -> int:
    return max(4, round(n_target / 100))


def priority(seed, catalog_id, case_id):
    return hashlib.sha256(f"{seed}:{catalog_id}:{case_id}".encode()).hexdigest()


def largest_remainder(total, weights):
    s = sum(weights.values())
    if total <= 0 or s == 0:
        return {k: 0 for k in weights}
    raw = {k: total * w / s for k, w in weights.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), str(k)))[: total - sum(out.values())]:
        out[k] += 1
    return out


def bench_quotas(rows_by_bench: dict, weight_by_bench: dict, n_target: int, design: str) -> dict:
    """{catalog_id: row quota}: the design's shares, floored at min_per_bench, capped at the benchmark's rows; the
    rows a cap or floor frees / takes are re-spread over the still-free benchmarks (iterated to a fixed point)."""
    if design not in DESIGNS:
        raise ValueError(f"design {design!r}: one of {DESIGNS}")
    R, W = sum(rows_by_bench.values()), sum(weight_by_bench[n] for n in rows_by_bench)
    share = {n: {"rows": rows_by_bench[n] / R, "weight": weight_by_bench[n] / W,
                 "mix": (rows_by_bench[n] / R + weight_by_bench[n] / W) / 2}[design] for n in rows_by_bench}
    m = min_per_bench(n_target)
    fixed: dict = {}
    while True:
        free = [n for n in rows_by_bench if n not in fixed]
        q = largest_remainder(n_target - sum(fixed.values()), {n: share[n] for n in free})
        newly = {}
        for n in free:
            lo, hi = min(m, rows_by_bench[n]), rows_by_bench[n]
            if q[n] < lo:
                newly[n] = lo
            elif q[n] > hi:
                newly[n] = hi
        if not newly:
            return {**fixed, **q}
        fixed.update(newly)


def case_table(rows: list[dict]) -> dict:
    """{(catalog_id, case_id): {"rows": [run_id], "bin": length bin of its longest row}} over ``rows``; lengths
    (``proxy_tokens``, ``length_bin``) from the committed DEV manifest."""
    from harness.cases import case_ids
    from jobs.common.dev_subset import read_manifest

    man = {m["run_id"]: m for m in read_manifest()}
    cid = case_ids(rows)
    out: dict = {}
    for r in sorted(rows, key=lambda r: r["_evaluation"]["run_id"]):
        e = r["_evaluation"]
        m = man[e["run_id"]]
        k = (e["catalog_id"], cid[(e["catalog_id"], str(e["group_id"]))])
        c = out.setdefault(k, {"rows": [], "tok": -1, "bin": None})
        c["rows"].append(e["run_id"])
        if m["proxy_tokens"] > c["tok"]:
            c["tok"], c["bin"] = m["proxy_tokens"], m["length_bin"]
    return out


def build_proxy(cases: dict, weight_by_bench: dict, n_target: int, design: str, seed=SEED) -> list[str]:
    """cases: ``case_table`` of the 2,008 subset rows. Returns the sorted run_ids of the proxy (whole cases)."""
    by_bench = collections.defaultdict(list)
    for (n, c), v in cases.items():
        by_bench[n].append((c, v))
    rows_by_bench = {n: sum(len(v["rows"]) for _, v in cs) for n, cs in by_bench.items()}
    q = bench_quotas(rows_by_bench, weight_by_bench, n_target, design)
    chosen = []
    for n, cs in sorted(by_bench.items()):
        bins = collections.Counter()
        for _, v in cs:
            bins[v["bin"]] += len(v["rows"])
        bq = largest_remainder(q[n], bins)
        for b in sorted(bins, key=str):
            got = 0
            for c, v in sorted((x for x in cs if x[1]["bin"] == b), key=lambda x: priority(seed, n, x[0])):
                if got >= bq[b]:
                    break
                chosen += v["rows"]
                got += len(v["rows"])
    return sorted(chosen)


def subset_rows(subset_path=None) -> list[dict]:
    """The 2,008 m1 subset's DEV rows (kit + private rows needed)."""
    from jobs.common import dev_subset as D

    sub = D.load_subset(subset_path or D.SUBSET)
    return D._rows(sub, "0.2.1")


def load_proxy(name_or_path="proxy600", check_manifest=True) -> dict:
    """A proxy file, refused unless its run_ids match its ids_sha256, its pinned sha (for a named row mode), its DEV
    manifest, and are a subset of the m1 subset it was built from."""
    from jobs.common import dev_subset as D

    if name_or_path in PROXIES:
        path, pin = PROXIES[name_or_path]
    else:
        path, pin = Path(name_or_path), None
    d = D.load_subset(path, check_manifest)
    if pin is not None and d["ids_sha256"] != pin:
        raise SystemExit(f"{path}: ids_sha256 {d['ids_sha256']} != pinned {pin}")
    base = D.load_subset(D.SUBSET, check_manifest)
    if d.get("base_subset_ids_sha256") != base["ids_sha256"] or not set(d["run_ids"]) <= set(base["run_ids"]):
        raise SystemExit(f"{path}: not a subset of {D.SUBSET.name}")
    return d


def proxy_ids(name_or_path="proxy600") -> set[str]:
    return set(load_proxy(name_or_path)["run_ids"])


def main(argv=None):
    p = argparse.ArgumentParser(prog="dev_proxy")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build")
    s.add_argument("--n", type=int, default=600)
    s.add_argument("--design", choices=DESIGNS, default="weight")
    s.add_argument("--seed", default=SEED)
    s.add_argument("--out", default=str(PROXY600))
    s = sub.add_parser("check")
    s.add_argument("--proxy", default="proxy600")
    a = p.parse_args(argv)
    if a.cmd == "check":
        d = load_proxy(a.proxy)
        print(json.dumps({"ids_sha256": d["ids_sha256"], "rows": len(d["run_ids"])}))
        return 0
    from harness import score
    from jobs.common import dev_subset as D

    base = D.load_subset()
    rows = D._rows(base, "0.2.1")
    ids = build_proxy(case_table(rows), score.weights(edition="0.2.1"), a.n, a.design, a.seed)
    entries = D.read_manifest()
    out = {"_comment": "Pre-declared stratified proxy of the 2,008-row m1 DEV subset (jobs/common/dev_proxy.py; validated "
                       "by analysis/proxy_validate.py, receipt receipts/dev_proxy600/). IDs only; rows stay private. Same "
                       "shape as dev_subset_m1.json, so `jobs.common.dev_subset score|compare --subset <this>` scores it. "
                       "A SCREEN: winners get the full subset + full DEV.",
           "seed": a.seed, "n_target": a.n, "design": a.design, "min_per_bench": min_per_bench(a.n),
           "base_subset": "jobs/common/dev_subset_m1.json", "base_subset_ids_sha256": base["ids_sha256"],
           **D.manifest_hashes(), "ids_sha256": D.ids_sha256(ids), "summary": D.summary(entries, ids), "run_ids": ids}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"ids_sha256": out["ids_sha256"], "rows": len(ids), "design": a.design}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
