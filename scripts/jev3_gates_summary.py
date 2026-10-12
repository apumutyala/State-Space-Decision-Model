"""jev3 gate bookkeeping.

  dropids  union of the ids every PRE gate (on the checked pool) failed -> one drop file for ``jev3_build.py final``
           python -X utf8 -I scripts/jev3_gates_summary.py dropids --pre <receipt>/pre --out <drop.txt>
  summary  GATES.json from the POST gate receipts on the final train + slice records (every gate must be 0, the shape
           gate must pass for every family, slice/train disjoint by id / source item / group / content / state)
           python -X utf8 -I scripts/jev3_gates_summary.py summary --receipt <receipt>
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def J(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def gate_ids(d: Path) -> dict[str, list[str]]:
    out = {}
    pv = J(d / "prov-gate.json")
    out["prov"] = sorted({x for v in pv["fail_ids"].values() for x in v})
    # (fail_ids are truncated at 50 per reason; the full list is the --drop-ids-out file, read by dropids)
    out["strict_fp"] = sorted(J(d / "strict-fp.json")["hits"])
    out["exact_leaves"] = sorted(J(d / "exact-leaves.json")["rows"])
    doc = J(d / "doc-overlap-guard.json")
    ids = set(doc.get("fail_ids") or []) | set(doc.get("jevbench", {}).get("fail_ids") or [])
    if doc["n_fail"] and not ids:
        raise SystemExit("doc-overlap-guard: failures without ids")
    out["doc_overlap"] = sorted(ids)
    atk = J(d / "attack-guard.json")
    if atk["n_hit"] and not atk.get("hit_ids"):
        raise SystemExit("attack-guard: hits without ids")
    out["attack"] = sorted(atk.get("hit_ids") or [])
    jev = J(d / "jevbench-guard-rescan.json")
    jd = next(iter(jev["draws"].values()))
    out["jevbench"] = sorted(jd.get("jevbench_training_gate_fail_ids") or [])
    return out


def cmd_dropids(a) -> int:
    d = Path(a.pre)
    ids = gate_ids(d)
    w = Path(a.out).parent
    extra = {}
    for name in ("drop-prov.txt", "drop-strictfp.txt", "drop-leaves.txt", "drop-hs3.txt"):
        p = w / name
        if p.exists():
            extra[name] = sorted({x.strip() for x in p.read_text(encoding="utf-8").split() if x.strip()})
    allids = sorted(set().union(*ids.values(), *extra.values()))
    Path(a.out).write_text("\n".join(allids) + ("\n" if allids else ""), encoding="utf-8")
    fam_of = {}
    if a.checked:
        import gzip
        for line in gzip.open(a.checked, "rt", encoding="utf-8"):
            r = json.loads(line)
            fam_of[r["id"]] = r["aug"]["jev1"]["family8"]
    per_fam = {k: dict(collections.Counter(fam_of.get(x, "?") for x in v)) for k, v in {**ids, **extra}.items() if v}
    rep = {"by_gate": {k: len(v) for k, v in ids.items()}, "files": {k: len(v) for k, v in extra.items()},
           "by_gate_and_family": per_fam, "n_drop": len(allids),
           "n_drop_by_family": dict(collections.Counter(fam_of.get(x, "?") for x in allids)),
           "sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest()}
    (d / "drop-ids-summary.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rep))
    return 0


def cmd_summary(a) -> int:
    rd = Path(a.receipt)
    br = J(rd / "build-report.json")
    g = {"build": br["build"], "slice": br["slice"], "n_train": br["n_train"], "n_slice": br["n_slice"],
         "records_uncompressed_sha256": {k: v["records"] for k, v in br["uncompressed_sha256"].items()},
         "ids_sha256": br["ids_sha256"], "build_check_stage": {
             "what": "jev1_build.cmd_check unchanged: gold-once, Checker (cohort v2/v3, SplitGuard, harness.heldout, "
                     "Index DEV/TEST content/text/item text), production Index Fingerprint (item tier), JevBench "
                     "fingerprint + exact segment, attack_guard; failing rows DROPPED before final",
             "n_pool": br["check"]["n_pool"], "n_kept": br["check"]["n_kept"], "drops_by_family": br["check"]["reasons"]},
         "pre_gates_drop_ids": J(rd / "pre" / "drop-ids-summary.json") if (rd / "pre" / "drop-ids-summary.json").exists() else None,
         "post_gates_must_be_0": {}}
    # strict full-row Index fp on the checked pool BEFORE spec amendment A4 (template rewording), from the first pre run
    # (scratch log p_pre_strictfp.log, 2026-10-11 15:27); after A4 see pre_gates_drop_ids.by_gate_and_family
    g["strict_full_row_fp_pre_gate_before_A4"] = {
        "n_rows_checked": 11800, "n_rows_hit": 748,
        "matcher_hits_by_family": {"judge_hard/exact": 29, "judge_hard/minhash": 29, "ambiguous/minhash": 47,
                                   "ambiguous/exact": 29, "ambiguous/edit": 1, "temporal_numeric/exact": 77,
                                   "temporal_numeric/minhash": 114, "temporal_numeric/edit": 127,
                                   "long_policy/exact": 438, "long_policy/minhash": 441},
        "attribution_own_template_segments": {"long_policy 'none of them' (score level)": 438,
                                              "temporal_numeric 'that is the right answer' (noul criterion)": 117,
                                              "judge_hard 'is it correct' (instruction)": 29}}
    ok = True
    for role in ("train", "slice"):
        d = rd / role
        ids = gate_ids(d)
        rows = J(d / "row-gates.json")
        pv = J(d / "prov-gate.json")
        doc = J(d / "doc-overlap-guard.json")
        z = {"provenance_fail": sum(pv["fails"].values()), "provenance_prm800k": pv["prm800k"],
             "strict_full_row_index_fp_hits": J(d / "strict-fp.json")["n_hit"], "exact_leaf_ge6_rows": len(ids["exact_leaves"]),
             "doc_overlap_index_fail": doc["n_fail"], "doc_overlap_jevbench_fail": doc["jevbench"]["n_fail"],
             "doc_overlap_index_sha256": doc.get("index_sha256"), "attack_guard_hits": len(ids["attack"]),
             "jevbench_rescan_fail": len(ids["jevbench"]), "row_gates_fails": rows["fails"],
             "row_gates_pass": rows["pass"], "frozen_heldout_refused": rows["harness_heldout"]["refused"],
             "d1b_heldout_hits": rows["d1b_heldout"].get("n_hits"), "cohort_identity": rows["cohort_identity"],
             "index_fingerprint": rows["index_fingerprint"], "generators_checked": rows["generators"]}
        z["pass"] = bool(z["provenance_fail"] == 0 and z["strict_full_row_index_fp_hits"] == 0 and
                         z["exact_leaf_ge6_rows"] == 0 and z["doc_overlap_index_fail"] == 0 and
                         z["doc_overlap_jevbench_fail"] == 0 and z["attack_guard_hits"] == 0 and
                         z["jevbench_rescan_fail"] == 0 and rows["pass"] and z["frozen_heldout_refused"] == 0 and
                         not z["d1b_heldout_hits"])
        ok &= z["pass"]
        g["post_gates_must_be_0"][role] = z
    fam = br["families"]
    g["gold_position_chi2_presented_train"] = {f: fam[f]["train"]["gold_position_chi2_presented"] for f in fam}
    g["gold_position_gate_p_ge_0.001"] = br["gold_position_gate_p_ge_0.001"]
    ok &= all(br["gold_position_gate_p_ge_0.001"].values())
    g["state_dedupe_dropped_by_family"] = br["state_dedupe_dropped_by_family"]
    g["slice_train_disjoint_counts"] = br["slice_train_disjoint_counts"]
    g["within_split_duplicate_states"] = br["within_split_duplicate_states"]
    ok &= not any(br["slice_train_disjoint_counts"].values()) and not any(br["within_split_duplicate_states"].values())
    tok = {}
    for f, v in fam.items():
        t = v["train"]["tokens_shim_qwen35_4b"]
        lo, hi = t["t_window"]
        tok[f] = {**t, "in_window": lo <= t["min"] and t["max"] <= hi}
        ok &= tok[f]["in_window"]
    g["tokens_jrt_h2o_shim_qwen35_4b_train"] = tok
    s = J(rd / "shape-gate.json")
    g["shape_gate"] = {"all_pass": s["all_pass"], "families": {f: {"pass": v["pass"], "checks": v["checks"],
                                                                   "draw": v["draw"], "public": v["public"]}
                                                               for f, v in s["families"].items()}}
    ok &= bool(s["all_pass"])
    g["rows_by_family"] = {f: {"train": fam[f]["train"]["rows"], "slice": fam[f]["slice"]["rows"]} for f in fam}
    g["types_by_family_train"] = {f: fam[f]["train"]["types"] for f in fam}
    g["label_rates_train"] = {f: fam[f]["train"]["label_rates"] for f in fam}
    g["all_pass"] = bool(ok)
    (rd / "GATES.json").write_text(json.dumps(g, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": g["all_pass"], "post": {r: v["pass"] for r, v in g["post_gates_must_be_0"].items()},
                      "shape": {f: v["pass"] for f, v in g["shape_gate"]["families"].items()},
                      "chi2_ok": br["gold_position_gate_p_ge_0.001"]}))
    return 0 if g["all_pass"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dropids")
    d.add_argument("--pre", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--checked")
    s = sub.add_parser("summary")
    s.add_argument("--receipt", required=True)
    a = ap.parse_args(argv)
    return {"dropids": cmd_dropids, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
