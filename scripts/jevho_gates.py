"""jev-heldout gate bookkeeping (counts and ids only; no item text).

  dropids  union of the ids every PRE gate (on the candidate pool) failed -> one drop file for jevho_build.py final
           python -X utf8 -I scripts/jevho_gates.py dropids --pre <receipt>/pre --work <work dir> --out <drop.txt>
  summary  GATES.json from the FINAL-record gate receipts (every gate 0, shape + tolerance + chi2 pass)
           python -X utf8 -I scripts/jevho_gates.py summary --receipt <receipt>
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def J(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def gate_ids(d: Path) -> dict:
    """ids per gate from the report files (None = the gate's report is missing, i.e. the gate did not run)."""
    out = {}
    sf = J(d / "strict-fp.json")
    out["strict_full_row_index_fp"] = None if sf is None else sorted(sf.get("hits") or [])
    el = J(d / "exact-leaves.json")
    out["exact_leaves_ge6"] = None if el is None else sorted(el.get("rows") or [])
    le = J(d / "line-exact.json")
    out["line_exact_ge4"] = None if le is None else sorted(le.get("rows") or [])
    doc = J(d / "doc-overlap-guard.json")
    if doc is not None:
        ids = set(doc.get("fail_ids") or []) | set((doc.get("jevbench") or {}).get("fail_ids") or [])
        if doc.get("n_fail") and not ids:
            raise SystemExit("doc-overlap-guard: failures without ids")
        out["doc_overlap"] = sorted(ids)
    else:
        out["doc_overlap"] = None
    atk = J(d / "attack-guard.json")
    if atk is not None and atk.get("n_hit") and not atk.get("hit_ids"):
        raise SystemExit("attack-guard: hits without ids")
    out["attack_guard"] = None if atk is None else sorted(atk.get("hit_ids") or [])
    jev = J(d / "jevbench-guard-rescan.json")
    if jev is not None:
        jd = next(iter(jev["draws"].values()))
        out["jevbench_rescan"] = sorted(jd.get("jevbench_training_gate_fail_ids") or [])
    else:
        out["jevbench_rescan"] = None
    return out


def cmd_dropids(a) -> int:
    d, w = Path(a.pre), Path(a.work)
    ids = gate_ids(d)
    extra = {}
    for name in ("drop-disjoint.txt", "drop-disjoint-final1.txt", "drop-prov.txt", "drop-strictfp.txt", "drop-leaves.txt"):
        p = w / name
        if p.exists():
            extra[name] = sorted({x.strip() for x in p.read_text(encoding="utf-8").split() if x.strip()})
    od = w / "drop-overlap13.txt"  # regate_jev2a overlap --dump-hit-ids dir: rows sharing any 13-gram with an eval side
    if od.is_dir():
        for f in sorted(od.glob("*.ids.txt")):
            extra[f"overlap13:{f.name}"] = sorted({x.strip() for x in f.read_text(encoding="utf-8").split() if x.strip()})
    allids = sorted(set().union(*(v for v in ids.values() if v), *extra.values()))
    Path(a.out).write_text("".join(x + "\n" for x in allids), encoding="utf-8")
    fam_of = {}
    for line in gzip.open(w / "pool.jsonl.gz", "rt", encoding="utf-8"):
        r = json.loads(line)
        fam_of[r["id"]] = r["aug"]["jev1"]["family8"]
    per = {k: dict(collections.Counter(fam_of.get(x, "?") for x in v)) for k, v in {**ids, **extra}.items() if v}
    rep = {"gates_run": {k: v is not None for k, v in ids.items()}, "by_gate": {k: (None if v is None else len(v))
                                                                               for k, v in ids.items()},
           "files": {k: len(v) for k, v in extra.items()}, "by_gate_and_family": per, "n_drop": len(allids),
           "n_drop_by_family": dict(collections.Counter(fam_of.get(x, "?") for x in allids)),
           "sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest()}
    (d / "drop-ids-summary.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep))
    return 0


def cmd_summary(a) -> int:
    rd = Path(a.receipt)
    fd = rd / "final"
    br = J(rd / "build-report.json")
    ids = gate_ids(fd)
    dj, pv, rows, shape = J(fd / "disjoint-jev.json"), J(fd / "prov-gate.json"), J(fd / "row-gates.json"), \
        J(fd / "shape-gate.json")
    doc = J(fd / "doc-overlap-guard.json")
    z = {"jev_disjoint_drop_rows (ids/source items/row keys/content/state + human-source text)":
         None if dj is None else dj["n_rows_dropped"],
         "jev_disjoint_key_hits": None if dj is None else dj["key_hits"],
         "engine_rows_template_line_overlap_reported": None if dj is None else
         dj["engine_rows_template_line_overlap_not_dropped"],
         "provenance_fail": None if pv is None else pv["n_fail"],
         **{f"{k}_rows": (None if v is None else len(v)) for k, v in ids.items()},
         "doc_overlap_index_fail": None if doc is None else doc.get("n_fail"),
         "doc_overlap_jevbench_fail": None if doc is None else (doc.get("jevbench") or {}).get("n_fail"),
         "row_gates": None if rows is None else {"pass": rows["pass"], "fails": rows["fails"],
                                                 "frozen_refused": rows["harness_heldout"]["refused"],
                                                 "d1b_hits": rows["d1b_heldout"].get("n_hits"),
                                                 "cohort_identity": rows["cohort_identity"],
                                                 "index_fingerprint": rows["index_fingerprint"]}}
    ran = {k: v is not None for k, v in z.items()}
    zero = (dj is not None and dj["n_rows_dropped"] == 0 and dj["key_hits"] == 0 and pv is not None and pv["n_fail"] == 0
            and all(v is not None and len(v) == 0 for v in ids.values())
            and doc is not None and not doc.get("n_fail") and not (doc.get("jevbench") or {}).get("n_fail")
            and rows is not None and rows["pass"])
    tol = br["tolerance_gate"]
    tol_ok = all(t["count_ok"] and t["type_share_ok"] and t["types_subset"] and t["string_share_ok"] and
                 t["noul_criteria_all"] for t in tol.values())
    chi_ok = all(br["gold_position_gate_p_ge_0.001"].values())
    g = {"set": br["set"], "n": br["n"], "seal": br["seal"], "families": br["families"], "types": br["types"],
         "choice_K": br["choice_K"], "sources": br["sources"], "tiers": br["tiers"],
         "build_check_stage": "jev1_build.cmd_check (gold-once, Checker cohort v2/v3 + SplitGuard + harness.heldout + "
                              "Index DEV/TEST content/text/item text, Index fp item tier, JevBench fp + exact segment, "
                              "attack_guard)" + (" RAN" if br["input"] == "checked.jsonl.gz" else " NOT RUN (stage)"),
         "pre_gates_drop_ids": J(rd / "pre" / "drop-ids-summary.json"),
         "final_gates_must_be_0": z, "final_gates_ran": ran, "final_gates_all_zero": bool(zero),
         "tolerance_gate": tol, "tolerance_all_ok": tol_ok, "gold_position_chi2": br["gold_position_chi2"],
         "gold_position_ok": chi_ok,
         "shape_gate": None if shape is None else {"all_pass": shape["all_pass"],
                                                   "families": {f: {"pass": v["pass"], "checks": v["checks"]}
                                                                for f, v in shape["families"].items()}},
         "short_families": br["short_families"]}
    g["all_pass"] = bool(zero and tol_ok and chi_ok and shape is not None and shape["all_pass"] and
                         not br["short_families"] and br["input"] == "checked.jsonl.gz")
    (rd / "GATES.json").write_text(json.dumps(g, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": g["all_pass"], "zero": zero, "ran": ran, "tol_ok": tol_ok, "chi_ok": chi_ok,
                      "shape": None if shape is None else shape["all_pass"]}))
    return 0 if g["all_pass"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dropids")
    d.add_argument("--pre", required=True)
    d.add_argument("--work", required=True)
    d.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--receipt", required=True)
    a = ap.parse_args(argv)
    return {"dropids": cmd_dropids, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
