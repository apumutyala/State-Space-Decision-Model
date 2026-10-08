"""Per-Index-row latency / cost comparison: Clef 27B, Clef-flash, stock RWKV7-13.3B readouts, trained cell D1.

CPU only, $0. Reads each arm's per-row ``total_wall_ms`` (the kit's wall time around one ``decide`` request: batch 1,
one request at a time) from its landed artifact, straight out of the tar / tgz members (nothing is extracted, nothing
is executed; run with ``python -I``). Rows are matched by ``run_id`` on the 2,008-row Index DEV subset:

* ``own``    every ``ok`` row of that arm (its own denominator);
* ``common`` the rows that are ``ok`` in EVERY arm of the table (identical ids; the headline).

Per arm: median / p95 / mean row latency, total scoring minutes, rows/min, $/1k rows at the billed H100 rate
($0.0495 / GPU-min, ``jobs/clef_baseline/price.py`` RATE_USD_PER_MIN), proxy tokens/s (the kit's tokenizer-free
``proxy_tokens``, the same numerator for every arm) and model tokens/s where the arm reports ``usage.input_tokens``
(Clef tokenizer vs RWKV World tokenizer: NOT comparable across families), and Index-per-dollar.

Confounds the table cannot remove are listed in the report (analysis/COST_LATENCY_2026-10-08.md).

    python -I -X utf8 analysis/cost_latency.py [--config arms.json] [--json out.json] [--md out.md]
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterator

RATE_USD_PER_MIN = 0.0495  # SF batch H100 billed rate (jobs/clef_baseline/price.py)
H = "C:/Users/apumu"

#: arm -> where its per-row results live. ``src``: [archive, member, member, ...] (nested tar/tgz members) or a plain
#: path. ``fmt``: "jsonl" (kit results, one row per line) or "units" (m2 units json, ``index_results`` dict).
#: ``exclude_from``: a substitute.provenance.json whose ``substituted_run_ids`` are NOT this arm's own rows.
DEFAULT_ARMS: list[dict[str, Any]] = [
    {"arm": "clef27b_fast", "label": "Clef 27B, fast fla GDN (job-d2u54 art-syxu6)", "family": "clef",
     "gdn": "fast", "index_subset": 62.14, "index_src": "d2u54 score_decide.json",
     "src": [f"{H}/clef_d2u54_art/art-syxu6.tar", "./clef-index-subset/index/decide/results.jsonl"]},
    {"arm": "clef27b_ref", "label": "Clef 27B, torch REFERENCE GDN (job-zfexa art-p93d4, full DEV)", "family": "clef",
     "gdn": "reference", "index_subset": 62.09, "index_src": "zfexa score_decide_subset.json",
     "src": [f"{H}/clef_zfexa_land/dl/art-p93d4.tar", "./clef-index-full/index/decide/results.jsonl"]},
    {"arm": "flash_fast", "label": "Clef-flash, fast fla GDN (job-sgihj, via ftpnf's pinned copy)", "family": "clef",
     "gdn": "fast", "index_subset": 55.78, "index_src": "sgihj on the subset (CANDIDATES.md)",
     "src": f"{H}/clefl/clefl_ftpnf/x/clef-flash-index-overlength/index/decide_32k_substituted/results.jsonl",
     "exclude_from": f"{H}/clefl/clefl_ftpnf/x/clef-flash-index-overlength/index/decide_32k_substituted/"
                     "substitute.provenance.json"},
    {"arm": "flash_ref", "label": "Clef-flash, torch REFERENCE GDN (job-t5yg2 art-7pyk2, full DEV)", "family": "clef",
     "gdn": "reference", "index_subset": 55.55, "index_src": "t5yg2 score_decide_subset.json",
     "src": f"{H}/clef_t5yg2_land/dl/clef-flash-index-full/index/decide/results.jsonl"},
    {"arm": "rwkv7_stock_v2", "label": "RWKV7-13.3B stock readout g1-v2 (job-zhyvg art-6bhnu)", "family": "rwkv7",
     "gdn": "n/a (chunked WKV)", "index_subset": 27.31, "index_src": "zhyvg score.json",
     "src": [f"{H}/r1_zhyvg_land/art/phases/final.tgz", "out/index/stock/results.jsonl"]},
    {"arm": "rwkv7_stock_v3w", "label": "RWKV7-13.3B stock readout best code-label v3w (job-fgu2c art-ivu28)",
     "family": "rwkv7", "gdn": "n/a (chunked WKV)", "index_subset": 31.07, "index_src": "fgu2c v3w score.json",
     "src": [f"{H}/r3_fgu2c_land/dl/art-ivu28.tar", "./phases/final.tgz", "out/index/v3w/results.jsonl"]},
    {"arm": "rwkv7_parent_c7pq2", "label": "RWKV7-13.3B stock readout g1-v1, D1's parent arm (seeded, art-bm3e6)",
     "family": "rwkv7", "gdn": "n/a (chunked_lean WKV)", "index_subset": 24.85,
     "index_src": "c7pq2 first_cell_result parent_stock_readout", "fmt": "units",
     "src": [f"{H}/rescore_c7pq2/dl/phases/final.tgz", "out/parent_arm/units-index.json"]},
    {"arm": "d1_child", "label": "RWKV7-13.3B trained cell D1 (job-c7pq2 art-bm3e6, LoRA r64 + fork head)",
     "family": "rwkv7", "gdn": "n/a (chunked_lean WKV)", "index_subset": 35.12,
     "index_src": "c7pq2 first_cell_result child", "fmt": "units",
     "src": [f"{H}/rescore_c7pq2/dl/phases/final.tgz", "out/units-child-index.json"]},
]


def _open(src) -> io.BufferedIOBase:
    """A plain path, or [archive, member, ...] read through nested tar / tgz members (never extracted)."""
    if isinstance(src, str):
        return open(src, "rb")
    fobj: Any = open(src[0], "rb")
    for member in src[1:]:
        tf = tarfile.open(fileobj=fobj, mode="r:*")
        m = tf.getmember(member)
        if not m.isfile():
            raise ValueError(f"{member}: not a regular file")
        fobj = tf.extractfile(m)
    return fobj


def _slim(r: dict[str, Any]) -> dict[str, Any]:
    usage = (r.get("response") or {}).get("usage") or {}
    return {"status": r.get("status"), "ms": r.get("total_wall_ms"), "proxy": r.get("proxy_tokens"),
            "model_tokens": usage.get("input_tokens")}


def iter_rows(arm: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    with _open(arm["src"]) as f:
        if arm.get("fmt") == "units":
            d = json.load(f)
            for rid, r in d["index_results"].items():
                yield rid, _slim(r)
            return
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                yield r["run_id"], _slim(r)


def load_arm(arm: dict[str, Any], keep: set[str] | None) -> dict[str, dict[str, Any]]:
    excl: set[str] = set()
    if arm.get("exclude_from"):
        excl = set(json.loads(Path(arm["exclude_from"]).read_text(encoding="utf-8"))["substituted_run_ids"])
    out = {}
    for rid, r in iter_rows(arm):
        if (keep is None or rid in keep) and rid not in excl:
            out[rid] = r
    return out


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def metrics(rows: dict[str, dict[str, Any]], ids: list[str], index: float | None) -> dict[str, Any]:
    ms = [float(rows[i]["ms"]) for i in ids]
    tot_s = sum(ms) / 1000.0
    proxy = sum(int(rows[i]["proxy"] or 0) for i in ids)
    mt = [rows[i]["model_tokens"] for i in ids]
    usd = tot_s / 60.0 * RATE_USD_PER_MIN
    out = {"n": len(ids), "median_ms": statistics.median(ms), "p95_ms": _pct(ms, 0.95), "mean_ms": tot_s * 1000 / len(ids),
           "total_min": tot_s / 60.0, "rows_per_min": len(ids) / (tot_s / 60.0),
           "usd_per_1k_rows": usd / len(ids) * 1000.0, "proxy_tok_per_s": proxy / tot_s,
           "model_tok_per_s": (sum(mt) / tot_s) if all(isinstance(x, int) for x in mt) else None,
           "s_per_1k_proxy_tokens": tot_s / proxy * 1000.0}
    if index is not None:
        out["index_per_usd_1k_rows"] = index / out["usd_per_1k_rows"]
    return out


def run(arms: list[dict[str, Any]]) -> dict[str, Any]:
    # the 2,008-row DEV subset = D1's own Index rows (every arm below ran exactly these ids or a superset)
    ref = next(a for a in arms if a["arm"] == "d1_child")
    subset = {rid for rid, _ in iter_rows(ref)}
    data = {a["arm"]: load_arm(a, subset) for a in arms}
    ok = {k: {i for i, r in v.items() if r["status"] == "ok" and isinstance(r["ms"], (int, float))} for k, v in data.items()}
    common = sorted(set.intersection(*ok.values()))
    res: dict[str, Any] = {"subset_rows": len(subset), "common_ok_rows": len(common), "rate_usd_per_min": RATE_USD_PER_MIN,
                           "arms": {}}
    for a in arms:
        k = a["arm"]
        res["arms"][k] = {"label": a["label"], "gdn": a["gdn"], "family": a["family"], "index_subset": a["index_subset"],
                          "index_src": a["index_src"], "rows_present": len(data[k]), "own": metrics(data[k], sorted(ok[k]),
                                                                                                    a["index_subset"]),
                          "common": metrics(data[k], common, a["index_subset"])}
    # ratios on the common rows: >1 means the arm is SLOWER / costlier than the comparator
    for comp in ("clef27b_fast", "flash_fast"):
        if comp not in res["arms"]:
            continue
        c = res["arms"][comp]["common"]
        for k, v in res["arms"].items():
            m = v["common"]
            v.setdefault("vs", {})[comp] = {"total_time_ratio": m["total_min"] / c["total_min"],
                                            "median_ratio": m["median_ms"] / c["median_ms"],
                                            "p95_ratio": m["p95_ms"] / c["p95_ms"]}
    # D1 child vs its parent, row by row (the 2.8x lever)
    if "d1_child" in data and "rwkv7_parent_c7pq2" in data:
        both = sorted(ok["d1_child"] & ok["rwkv7_parent_c7pq2"])
        rr = [data["d1_child"][i]["ms"] / data["rwkv7_parent_c7pq2"][i]["ms"] for i in both]
        res["d1_vs_parent"] = {"n": len(both), "sum_ratio": sum(data["d1_child"][i]["ms"] for i in both)
                               / sum(data["rwkv7_parent_c7pq2"][i]["ms"] for i in both),
                               "median_row_ratio": statistics.median(rr)}
    return res


def markdown(res: dict[str, Any]) -> str:
    lines = [f"Common ok rows (identical run_ids, ok in every arm): **{res['common_ok_rows']}** of {res['subset_rows']}.",
             "", "| arm | GDN path | Index (subset) | median ms | p95 ms | rows/min | $/1k rows | proxy tok/s | "
             "time vs Clef 27B fast | time vs Clef-flash fast |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, v in res["arms"].items():
        m = v["common"]
        vs = v.get("vs", {})
        lines.append(f"| {k} | {v['gdn']} | {v['index_subset']:.2f} | {m['median_ms']:.0f} | {m['p95_ms']:.0f} | "
                     f"{m['rows_per_min']:.1f} | {m['usd_per_1k_rows']:.3f} | {m['proxy_tok_per_s']:.0f} | "
                     f"{vs.get('clef27b_fast', {}).get('total_time_ratio', float('nan')):.2f}x | "
                     f"{vs.get('flash_fast', {}).get('total_time_ratio', float('nan')):.2f}x |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", help="JSON list of arms (default: the landed local artifacts)")
    ap.add_argument("--json", help="write the full result here")
    ap.add_argument("--md", help="write the markdown table here")
    a = ap.parse_args(argv)
    arms = json.loads(Path(a.config).read_text(encoding="utf-8")) if a.config else DEFAULT_ARMS
    res = run(arms)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    md = markdown(res)
    if a.md:
        Path(a.md).write_text(md + "\n", encoding="utf-8")
    print(md)
    if "d1_vs_parent" in res:
        print(json.dumps(res["d1_vs_parent"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
