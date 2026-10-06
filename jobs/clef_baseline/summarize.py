"""Stage S5 of the Clef baseline job: composite native results, latency, G1 audit, result JSON + table.

    python -X utf8 jobs/clef_baseline/summarize.py composite --decide R --native N --out C
    python -X utf8 jobs/clef_baseline/summarize.py report --out-dir OUT --release clef-flash --job A|B > result.json

Never edits a results file in place; the composite is a new file whose provenance is in its sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

WARMUP_ROWS = 10  # Index latency-v1 excludes 10 untimed warm-up rows


def rows(path: Path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def composite(a) -> int:
    dec, nat = rows(Path(a.decide)), {r["run_id"]: r for r in rows(Path(a.native))}
    out = [nat.get(r["run_id"], r) for r in dec]
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    side = {"decide": a.decide, "decide_sha256": sha(Path(a.decide)), "native": a.native,
            "native_sha256": sha(Path(a.native)), "replaced_run_ids": sorted(set(nat) & {r["run_id"] for r in dec}),
            "rule": "rows in probes.native_arm_run_ids use Clef's own systemone: ids differ, Clef truncates, ssdm encoder "
                    "refuses, >255 options, or the adapter rejects before scoring (validate_legacy / max_questions / max_options_per_question)"}
    if a.probes and Path(a.probes).exists():
        enc = json.loads(Path(a.probes).read_text(encoding="utf-8")).get("encodings", {})
        side["categories"] = {k: (len(v) if isinstance(v, list) else v) for k, v in enc.items()}
        missing = sorted(set(json.loads(Path(a.probes).read_text(encoding="utf-8"))["native_arm_run_ids"]) - set(nat))
        side["native_rows_missing_from_native_run"] = missing  # must be [] (kit writes every row, incl. errors)
    (p.parent / "composite.provenance.json").write_text(json.dumps(side, indent=1), encoding="utf-8")
    return 0


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))] if xs else None


def latency(res):
    ok = [r for r in res if r.get("status") == "ok"][WARMUP_ROWS:]
    ms = [r["total_wall_ms"] for r in ok if r.get("total_wall_ms") is not None]
    return {"n": len(ms), "median_ms": statistics.median(ms) if ms else None, "p95_ms": pct(ms, 0.95),
            "note": "DEV rows, H100 80GB, batch 1, one request at a time; NOT the board's RTX PRO 6000 latency-v1 "
                    "sample. Cloudflare's own card: Clef-flash 38.8 / 122.4 ms, Clef 209.3 / 238.6 ms (median/p95)."}


def g1_audit(res):
    status = {}
    bad = 0
    for r in res:
        status[r.get("status")] = status.get(r.get("status"), 0) + 1
        raw = (r.get("raw_output") or {}).get("raw_probabilities")
        if r.get("status") == "ok" and raw is not None:
            bad += sum(abs(sum(p.values()) - 1.0) > 1e-9 or any(not math.isfinite(x) or x < 0 for x in p.values())
                       for p in raw.values())
    return {"status_counts": status, "invalid_raw_distributions": bad, "rows": len(res)}


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def report(a) -> int:
    o = Path(a.out_dir)
    probes = load(o / "probes.json") or {}
    res_dec = rows(o / "index" / "decide" / "results.jsonl")
    idx_dec = load(o / "index" / "score_decide.json")
    idx_nat = load(o / "index" / "score_native_composite.json")
    ece = load(o / "index" / "ece_decide.json")
    saf = load(o / "safety" / "summary.json") or {}
    arms = saf.get("arms", {})

    def fam(arm, f):
        return (arms.get(arm, {}).get("families") or {}).get(f)

    result = {
        "ok": True, "release": a.release, "job": a.job,
        "parity_ok": (probes.get("parity") or {}).get("ok"),
        "index_dev_decide": idx_dec, "index_dev_native_composite": idx_nat, "ece_decide": ece,
        "latency_decide": latency(res_dec) if res_dec else None,
        "g1_audit": g1_audit(res_dec) if res_dec else None,
        "g8": probes.get("g8"), "g2b_chatml_spoof": probes.get("g2b_chatml_spoof"),
        "encodings": {k: (len(v) if isinstance(v, list) else v) for k, v in (probes.get("encodings") or {}).items()},
        "max_memory_allocated_gib": probes.get("max_memory_allocated_gib"),
        "safety": {"arms": {n: {"clean": arms[n].get("clean"), "status_counts": arms[n].get("status_counts"),
                                **{f: fam(n, f) for f in ("hijack", "d1_A", "d1_B", "gcg_steer", "gcg_hijack")}}
                            for n in arms}, "arm_vs_arm": saf.get("arm_vs_arm"), "gcg": saf.get("gcg")},
    }
    json.dump(result, sys.stdout, indent=1, default=str)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("composite")
    c.add_argument("--decide", required=True)
    c.add_argument("--native", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--probes", default=None)
    r = sub.add_parser("report")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--release", required=True)
    r.add_argument("--job", required=True)
    a = ap.parse_args(argv)
    return composite(a) if a.cmd == "composite" else report(a)


if __name__ == "__main__":
    sys.exit(main())
