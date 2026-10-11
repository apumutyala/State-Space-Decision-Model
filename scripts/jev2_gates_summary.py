"""GATES.json + token-template-identity.json + sources.sha256 for one jev2 stage receipt (reads the gate receipts).

    python -X utf8 -I scripts/jev2_gates_summary.py --stage a
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
DATA = Path(r"C:\Users\apumu\research\CHLabs\ssdm-data")
SOURCES = {"a": [("musique", DATA / "raw/jev2/musique/musique_ans_v1.0_train.jsonl"),
                 ("gsm8k", DATA / "m2/sources/gsm8k/train-00000-of-00001.parquet"),
                 ("boolq", DATA / "m2/sources/boolq/train-00000-of-00001.parquet"),
                 ("glaive_fc", DATA / "m2/sources/glaive_fc/0000.parquet")],
           "b": []}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["a", "b"])
    a = ap.parse_args(argv)
    build = f"bschema-draw-jev2{a.stage}"
    rd = REPO / "receipts" / "2026-10-10-jev2-draw" / build
    work = DATA / "m2" / "build" / f"{build}-work"
    J = lambda n: json.loads((rd / n).read_text(encoding="utf-8"))  # noqa: E731
    br, doc, atk, jev, rows = (J("build-report.json"), J("doc-overlap-guard.json"), J("attack-guard.json"),
                               J("jevbench-guard-rescan.json"), J("row-gates.json"))
    chk = json.loads((work / "check-report.json").read_text(encoding="utf-8"))
    jd = next(iter(jev["draws"].values()))
    ident = br["token_template"]["identity"]
    (rd / "token-template-identity.json").write_text(json.dumps(ident) + "\n", encoding="utf-8")
    lines = [f"{sha(p)}  {name}  {p.as_posix()}" for name, p in SOURCES[a.stage] if p.exists()]
    if lines:
        (rd / "sources.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fam = br["families"]
    g = {"build": build, "n_records": br["n_records"], "records_uncompressed_sha256": br["uncompressed_sha256"]["records"],
         "row_gates_in_build_check_stage": {
             "what": "scripts/jev2_build.py check = jev1_build.cmd_check (gold-once, Checker: cohort v2/v3 + SplitGuard "
                     "+ harness.heldout + Index DEV/TEST content/text/item text, production Index Fingerprint, JevBench "
                     "fingerprint + exact segment, attack_guard); failing rows are DROPPED before final",
             "n_pool": chk["n_pool"], "n_kept": chk["n_kept"], "drops_by_family": chk["reasons"],
             "index_fingerprint_identity": chk["index_fingerprint"]["identity"],
             "jevbench_identity": chk["jevbench_fingerprint"]["identity"]},
         "final_draw_rescans_must_be_0": {
             "doc_overlap_guard_index_fail": doc["n_fail"], "doc_overlap_guard_index_sha256": doc["index_sha256"],
             "doc_overlap_guard_jevbench_fail": doc["jevbench"]["n_fail"],
             "doc_overlap_guard_jevbench_exact_fail": doc["jevbench"]["n_fail_exact"],
             "attack_guard_hits": atk["n_hit"], "attack_guard_templates_sha256": atk["templates_sha256"],
             "jevbench_guard_rescan": {k: jd[k] for k in ("jevbench_doc_fail", "jevbench_exact_segment_rows",
                                                          "jevbench_training_gate_fail", "jevbench_fingerprint_rows_hit",
                                                          "fingerprint_records_checked")},
             "row_gates_fails": rows["fails"], "row_gates_pass": rows["pass"],
             "frozen_heldout_refused": rows["harness_heldout"]["refused"], "generators_checked": rows["generators"],
             "d1b_heldout_hits": rows["d1b_heldout"].get("n_hits"),
             "cohort_identity": rows["cohort_identity"], "index_fingerprint": rows["index_fingerprint"],
             "gold_once_fail": br["gold_once_fail"]},
         "gold_position_chi2_presented": {f: fam[f]["gold_position_chi2_presented"] for f in fam},
         "gold_position_gate_p_ge_0.001": br["gold_position_gate_p_ge_0.001"],
         "tokens_jrt_h2o_shim_qwen35_4b": {f: fam[f]["tokens_shim_qwen35_4b"] for f in fam},
         "token_template_identity": ident}
    z = g["final_draw_rescans_must_be_0"]
    g["all_pass"] = bool(z["doc_overlap_guard_index_fail"] == 0 and z["doc_overlap_guard_jevbench_fail"] == 0
                         and z["attack_guard_hits"] == 0 and all(v == 0 for k, v in z["jevbench_guard_rescan"].items()
                                                                if k != "fingerprint_records_checked")
                         and z["row_gates_pass"] and z["frozen_heldout_refused"] == 0 and not z["d1b_heldout_hits"]
                         and z["gold_once_fail"] == 0 and all(g["gold_position_gate_p_ge_0.001"].values())
                         and all(t["n_over_2048"] == 0 for t in g["tokens_jrt_h2o_shim_qwen35_4b"].values()))
    (rd / "GATES.json").write_text(json.dumps(g, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": g["all_pass"], "chi2_p": {f: v["p"] for f, v in g["gold_position_chi2_presented"].items()}}))
    return 0 if g["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
