"""Stage S5 of the Clef baseline job: composite native results, latency, G1 audit, result JSON + table.

    python -X utf8 jobs/clef_baseline/summarize.py composite --decide R --native N --out C
    python -X utf8 jobs/clef_baseline/summarize.py report --out-dir OUT --release clef-flash --job A|B|I|T [--rows subset|full] > result.json

Exit codes: 0 ok | 3 a results file is truncated / malformed / has duplicate run_ids | 4 inputs incomplete (decide rows !=
probes' row count, native-arm rows missing). run_job.sh maps 3 and 4 to its exit 17.

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


#: exit codes of this script (run_job.sh maps both to its exit 17, "results file unusable")
EXIT_BAD_FILE = 3     # truncated tail / malformed line / duplicate run_id (harness.jsonl.ResultsFileError)
EXIT_INCOMPLETE = 4   # a file is well-formed but misses rows it must have


class Incomplete(Exception):
    pass


def rows(path: Path, required: bool = True):
    """Strict JSONL (harness/jsonl.py): split on newline only, so U+2028 inside a JSON string (job-sgihj, GPQA-Diamond)
    never cuts a record; a partial last line or an unparseable line raises instead of being skipped or crashing."""
    from harness.jsonl import read_jsonl_strict

    if not path.exists() and not required:
        return []
    return read_jsonl_strict(path)


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _unique(rs, path):
    seen = set()
    for r in rs:
        if r["run_id"] in seen:
            from harness.jsonl import ResultsFileError

            raise ResultsFileError("duplicate", path, f"run_id {r['run_id']} twice")
        seen.add(r["run_id"])
    return seen


def composite(a) -> int:
    """Decide results with Clef's own systemone substituted on probes.native_arm_run_ids. Fail-closed: every input is
    read strictly, the decide file must hold exactly the rows the probes enumerated (``encodings.rows``, when given)
    and the native file must hold every native-arm id. Nothing is written unless all checks pass."""
    dpath, npath = Path(a.decide), Path(a.native)
    dec = rows(dpath)
    nat_rows = rows(npath)
    dec_ids = _unique(dec, dpath)
    nat = {r["run_id"]: r for r in nat_rows}
    _unique(nat_rows, npath)
    probes = json.loads(Path(a.probes).read_text(encoding="utf-8")) if a.probes and Path(a.probes).exists() else None
    if a.probes and probes is None:
        raise Incomplete(f"--probes {a.probes} missing")
    side = {"decide": a.decide, "decide_sha256": sha(dpath), "decide_rows": len(dec), "native": a.native,
            "native_sha256": sha(npath), "native_rows": len(nat),
            "rule": "rows in probes.native_arm_run_ids use Clef's own systemone: ids differ, Clef truncates, ssdm encoder "
                    "refuses, >255 options, or the adapter rejects before scoring (validate_legacy / max_questions / max_options_per_question)"}
    if probes is not None:
        enc = probes.get("encodings", {})
        side["categories"] = {k: (len(v) if isinstance(v, list) else v) for k, v in enc.items()}
        want_rows = enc.get("rows")
        if isinstance(want_rows, int) and want_rows != len(dec):
            raise Incomplete(f"decide results have {len(dec)} rows, probes enumerated {want_rows}: partial decide run")
        need = set(probes["native_arm_run_ids"])
        missing = sorted(need - set(nat))
        side["native_rows_missing_from_native_run"] = missing
        if missing:
            raise Incomplete(f"{len(missing)} native-arm rows missing from {npath} (e.g. {missing[:3]}): partial native run")
        stray = sorted(need - dec_ids)
        if stray:
            raise Incomplete(f"{len(stray)} native-arm ids are not in the decide results (e.g. {stray[:3]})")
        replace = need
    else:
        replace = set(nat)
    out = [nat[r["run_id"]] if r["run_id"] in replace else r for r in dec]
    side["replaced_run_ids"] = sorted(replace & dec_ids)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8", newline="\n")
    (p.parent / "composite.provenance.json").write_text(json.dumps(side, indent=1), encoding="utf-8")
    return 0


def substitute(a) -> int:
    """The 32k sensitivity arm (JOB_SPEC 4c): ``base`` results with every row of ``rows`` (the over-length rows re-read
    at max_length 32768) put in place of the base row with the same run_id. Strict: every sensitivity row must exist
    in the base, ``--expect`` rows exactly, nothing written otherwise."""
    bpath, rpath = Path(a.base), Path(a.rows)
    base, sens = rows(bpath), rows(rpath)
    ids = _unique(sens, rpath)
    base_ids = _unique(base, bpath)
    if a.expect is not None and len(sens) != a.expect:
        raise Incomplete(f"{rpath}: {len(sens)} sensitivity rows, expected {a.expect}: partial run")
    stray = sorted(ids - base_ids)
    if stray:
        raise Incomplete(f"{len(stray)} sensitivity rows not in the base results (e.g. {stray[:3]})")
    by = {r["run_id"]: r for r in sens}
    out = [by.get(r["run_id"], r) for r in base]
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8", newline="\n")
    prov = {"base": a.base, "base_sha256": sha(bpath), "rows": a.rows, "rows_sha256": sha(rpath),
            "substituted_run_ids": sorted(ids),
            "status_before": _count(r.get("status") for r in base if r["run_id"] in ids),
            "status_after": _count(r.get("status") for r in sens)}
    (p.parent / "substitute.provenance.json").write_text(json.dumps(prov, indent=1), encoding="utf-8")
    return 0


def _count(xs):
    c = {}
    for x in xs:
        c[x] = c.get(x, 0) + 1
    return c


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
    res_dec = rows(o / "index" / "decide" / "results.jsonl", required=False)
    idx_dec = load(o / "index" / "score_decide.json")
    idx_nat = load(o / "index" / "score_native_composite.json")
    ece = load(o / "index" / "ece_decide.json")
    saf = load(o / "safety" / "summary.json") or {}
    arms = saf.get("arms", {})

    def fam(arm, f):
        return (arms.get(arm, {}).get("families") or {}).get(f)

    teach = load(o / "teacher" / f"teacher-{a.release}.identity.json")
    sel = load(o / "teacher" / "selection.json") or {}
    result = {
        "ok": True, "release": a.release, "job": a.job, "rows": a.rows if a.job == "I" else None,
        "dry_run": str(a.dry_run) == "1",
        "parity_ok": (probes.get("parity") or {}).get("ok"),
        # fast GDN kernels vs torch reference, non-vacuous (PR #43 B1); job B is gated on s1_ok
        "kernel_path_ok": (probes.get("kernel_path") or {}).get("ok"),
        "kernel_path": {k: (probes.get("kernel_path") or {}).get(k) for k in
                        ("n", "gdn_layers_found", "gdn_layers_already_reference", "gdn_layers_switched",
                         "max_abs_dp", "argmax_flips", "fail_reasons")},
        "s1_ok": (probes.get("parity") or {}).get("ok") is True and (probes.get("kernel_path") or {}).get("ok") is True,
        # headline Clef number = the native composite: Clef's own systemone (silent truncation at its shipped
        # 16,384) on the rows our decide path scores exhausted; decide and the 32k sensitivity arm alongside (4c)
        "headline": {"metric": "index_dev_native_composite", "edition": "0.2.1", "value": idx_nat,
                     "why": "Clef as shipped answers over-length rows on a truncated state; the decide path scores "
                            "them exhausted (= wrong), which is one-directional against Clef"},
        "index_dev_decide": idx_dec, "index_dev_native_composite": idx_nat, "ece_decide": ece,
        "index_dev_decide_03": load(o / "index" / "score_decide_03.json"),
        "index_dev_native_composite_03": load(o / "index" / "score_native_composite_03.json"),
        "sensitivity_32k": {"index_dev_decide_32k_substituted": load(o / "index" / "score_decide_32k_substituted.json"),
                            "reference_decide": load(o / "index" / "score_reference_decide.json"),
                            "compare_32k_vs_16k": load(o / "index" / "compare_32k_vs_16k.json"),
                            "provenance": load(o / "index" / "decide_32k_substituted" / "substitute.provenance.json")},
        "paired_compare": ("present" if (o / "index" / "compare_clef_vs_flash.json").exists()
                           else ("absent" if a.release == "clef" and a.job == "I" else "n/a")),
        "compare_clef_composite_vs_flash": load(o / "index" / "compare_clef_composite_vs_flash.json"),
        "index_dev_decide_on_m1_subset": load(o / "index" / "score_decide_subset.json"),
        "compare_decide_vs_native": load(o / "index" / "compare_decide_vs_native.json"),
        "compare_clef_vs_flash": load(o / "index" / "compare_clef_vs_flash.json"),
        "composite_provenance": load(o / "index" / "native_composite" / "composite.provenance.json"),
        "rate_guard": {k: load(o / "env" / f"rate_guard_{k}.json") for k in ("index", "export")},
        "preflight_index": load(o / "env" / "preflight_index.json"),
        "parity_timing_n": len(probes.get("parity_timing") or []),
        "latency_decide": latency(res_dec) if res_dec else None,
        "g1_audit": g1_audit(res_dec) if res_dec else None,
        "g8": probes.get("g8"), "g2b_chatml_spoof": probes.get("g2b_chatml_spoof"),
        "encodings": {k: (len(v) if isinstance(v, list) else v) for k, v in (probes.get("encodings") or {}).items()},
        "max_memory_allocated_gib": probes.get("max_memory_allocated_gib"),
        "safety": {"arms": {n: {"clean": arms[n].get("clean"), "status_counts": arms[n].get("status_counts"),
                                **{f: fam(n, f) for f in ("hijack", "d1_A", "d1_B", "gcg_steer", "gcg_hijack")}}
                            for n in arms}, "arm_vs_arm": saf.get("arm_vs_arm"), "gcg": saf.get("gcg")},
        "teacher": None if teach is None else {
            **teach, "selection": {k: sel.get(k) for k in ("pointer", "pointer_sha256", "n_set", "n_drawn",
                                                           "drawn_ids_sha256", "cohort_hits", "guard_refused",
                                                           "heldout_scan", "files_uncompressed_sha256",
                                                           "frozen_family_refused", "heldout_draw_v1_sha256",
                                                           "frozen_families")},
            "guard": (sel.get("guard") or {}).get("content_index"), "guard_note": (sel.get("guard") or {}).get("note")},
    }
    if a.job == "I" and a.release == "clef" and result["paired_compare"] == "absent" and not result["dry_run"]:
        raise Incomplete("RELEASE=clef index job without the paired Clef - Clef-flash comparison")
    if a.job == "T" and teach is None:
        raise Incomplete("job T without teacher identity")
    # job-ftpnf: the platform refuses a $GMN_RESULT_PATH file over 32 KiB ("result_error: exceeds 32 KiB") and then
    # keeps no result for depends_on / get_job. The full report goes to report.json next to it; result.json is compact.
    Path(a.out_dir, "report.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    json.dump(compact_result(result), sys.stdout, indent=1, default=str)
    return 0


RESULT_MAX_BYTES = 32 * 1024 - 1024  # the platform limit is 32 KiB; keep 1 KiB of slack


def _slim(v):
    """Scores / compares -> their index point estimate + CI; everything else unchanged (recursively)."""
    if isinstance(v, dict):
        if "slice_index" in v and isinstance(v["slice_index"], dict):
            return {"index": v["slice_index"].get("index"), "rows": v.get("rows"), "edition": v.get("edition")}
        if "diff" in v and isinstance(v["diff"], dict):
            return {"diff_index": v["diff"].get("index")}
        return {k: _slim(x) for k, x in v.items()}
    if isinstance(v, list) and len(v) > 20:
        return {"n": len(v), "first": v[:3]}
    return v


def compact_result(result: dict) -> dict:
    """result.json for the platform (<= RESULT_MAX_BYTES): index points and CIs, gates, counts; the full report is
    report.json in the output directory."""
    out = _slim(result)
    out["full_report"] = "report.json"
    if len(json.dumps(out, indent=1, default=str).encode()) > RESULT_MAX_BYTES:
        keep = ("ok", "release", "job", "rows", "dry_run", "s1_ok", "parity_ok", "kernel_path_ok", "headline",
                "index_dev_decide", "index_dev_native_composite", "paired_compare", "full_report")
        out = {k: out.get(k) for k in keep} | {"truncated": "result.json over 31 KiB; see report.json"}
        if len(json.dumps(out, indent=1, default=str).encode()) > RESULT_MAX_BYTES:  # PR #106 review: re-check
            out = {k: out.get(k) for k in ("ok", "release", "job", "rows", "dry_run", "s1_ok", "parity_ok",
                                           "kernel_path_ok", "paired_compare", "full_report")} | {
                "truncated": "result.json over 31 KiB even when trimmed; see report.json"}
            out = {k: (v if len(json.dumps(v, default=str)) <= 512 else "see report.json") for k, v in out.items()}
    assert len(json.dumps(out, indent=1, default=str).encode()) <= RESULT_MAX_BYTES
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("composite")
    c.add_argument("--decide", required=True)
    c.add_argument("--native", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--probes", default=None)
    t = sub.add_parser("substitute")
    t.add_argument("--base", required=True)
    t.add_argument("--rows", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--expect", type=int, default=None)
    r = sub.add_parser("report")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--release", required=True)
    r.add_argument("--job", required=True)
    r.add_argument("--rows", default="full")
    r.add_argument("--dry-run", default="0")
    a = ap.parse_args(argv)
    from harness.jsonl import ResultsFileError

    try:
        return {"composite": composite, "substitute": substitute, "report": report}[a.cmd](a)
    except ResultsFileError as e:
        print(f"summarize {a.cmd}: results file unusable: {e}", file=sys.stderr)
        return EXIT_BAD_FILE
    except Incomplete as e:
        print(f"summarize {a.cmd}: incomplete inputs: {e}", file=sys.stderr)
        return EXIT_INCOMPLETE


if __name__ == "__main__":
    sys.exit(main())
