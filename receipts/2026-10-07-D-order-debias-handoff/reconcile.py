"""CPU-only reconciliation of PR #151; no model/checkpoint imports or calls.

Run with Python -I. Arguments: original_repo units-index.json rows.jsonl.gz
recomputed.json output.json. The original receipt is read without modification.
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def endpoint_excess(k, f_first, r_first):
    return (f_first + r_first) / 2 - 1 / k


def main():
    repo, units_path, rows_path, rerun_path, output_path = map(Path, sys.argv[1:])
    receipt = repo / "receipts/2026-10-07-D-order-debias"
    saved = json.loads((receipt / "analysis.json").read_text(encoding="utf-8"))
    rerun = json.loads(rerun_path.read_text(encoding="utf-8"))
    assert saved == rerun, "Original analysis did not reproduce exactly"
    assert digest(rerun_path) == digest(receipt / "analysis.json"), "Raw re-score bytes changed"
    # The original Windows script writes CRLF. Preserve those raw bytes outside
    # git, and save the review copy as LF so git's whitespace check remains useful.
    Path(output_path).with_name("recomputed.json").write_text(
        rerun_path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    ix = json.loads(units_path.read_text(encoding="utf-8"))
    with gzip.open(rows_path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    rows_by_id = {r["_evaluation"]["run_id"]: r for r in rows}
    assert len(rows_by_id) == len(rows) == 2008
    assert set(rows_by_id) == set(ix["index_results"])
    sys.path.insert(0, str(repo))
    from jobs.m2_first_cell import readout_diag as rd

    assert rd.position_bias(rows, ix["index_results"]) == json.loads(
        (receipt / "position_bias.json").read_text(encoding="utf-8"))
    selected = rd.diag_rows(rows)
    expected_qids = {f"{r['_evaluation']['run_id']}/{q}"
                     for r in selected for q, _, _ in rd._gold_choice_questions(r)}
    assert expected_qids == set(ix["order_diag"]["questions"])
    by_k = collections.defaultdict(collections.Counter)
    benchmarks_by_k = collections.defaultdict(collections.Counter)
    for qid, v in ix["order_diag"]["questions"].items():
        rid, key = qid.rsplit("/", 1)
        ids = sorted(rows_by_id[rid]["questions"][key]["criteria"])
        assert len(ids) == v["k"]
        assert set(v) == {"forward", "reversed", "gold", "k"}
        for cond, first in (("forward", ids[0]), ("reversed", ids[-1])):
            assert set(v[cond]) == {"choice", "correct", "chose_presented_first"}
            assert v[cond]["chose_presented_first"] == (v[cond]["choice"] == first)
            assert v[cond]["correct"] == (v[cond]["choice"] == v["gold"])
        bucket = str(len(ids)) if len(ids) <= 4 else ("5-16" if len(ids) <= 16 else ">16")
        c = by_k[bucket]
        benchmarks_by_k[bucket][rows_by_id[rid]["_evaluation"].get("dataset", "unknown")] += 1
        c["questions"] += 1
        fc, rc = v["forward"]["choice"], v["reversed"]["choice"]
        c["both_valid"] += fc in ids and rc in ids
        c["forward_invalid"] += fc not in ids
        c["reversed_invalid"] += rc not in ids
        c["choice_changes"] += fc != rc
        c["first_in_both_orders"] += fc == ids[0] and rc == ids[-1]
        c["last_in_both_orders"] += fc == ids[-1] and rc == ids[0]
        c["same_original_choice"] += fc is not None and fc == rc

    def latency(result_list):
        v = np.array([r["total_wall_ms"] for r in result_list], dtype=float)
        return {"n": len(v), "sum_seconds": float(v.sum() / 1000),
                "median_ms": float(np.median(v)), "mean_ms": float(v.mean()),
                "p80_ms": float(np.percentile(v, 80))}

    all_latency = latency(list(ix["index_results"].values()))
    ok_latency = latency([r for r in ix["index_results"].values() if r["status"] == "ok"])
    # An order-invariant model always choosing the same original endpoint has
    # nonzero "position_effect" for k>2. A balanced mixture of first-following
    # and last-following models has zero endpoint excess at k=2 despite changes.
    fixed_endpoint = {str(k): endpoint_excess(k, 1, 0) for k in (2, 3, 4, 77)}
    assert fixed_endpoint["77"] > 0.48
    assert endpoint_excess(2, 0.5, 0.5) == 0
    out = {
        "schema": "ssdm.order-debias-handoff.v1",
        "source_head": "478aec24f19656243ec8e347bf1563acb188c5b8",
        "parent_integration_base": "ee0adb70029fc80c65ee6072c307873034052362",
        "hardware": "local Windows CPU; no model calls; CUDA_VISIBLE_DEVICES empty",
        "cost_usd": 0,
        "input_files": {str(p): {"sha256": digest(p), "bytes": p.stat().st_size}
                        for p in (units_path, rows_path, receipt / "analysis.py",
                                  receipt / "analysis.json", receipt / "position_bias.json", rerun_path)},
        "validation": {"analysis_exact_json_equality": True, "raw_analysis_byte_equality": True,
                       "position_bias_exact_json_equality": True,
                       "diagnostic_selection_and_schema": True, "counterexamples": True},
        "source_receipt_seed": 0, "source_receipt_bootstraps": 2000,
        "rows": len(rows), "diagnostic_rows": len(selected),
        "diagnostic_questions": len(expected_qids), "by_k": dict(by_k),
        "benchmarks_by_k": dict(benchmarks_by_k),
        "forward_request_latency_all": all_latency, "forward_request_latency_ok": ok_latency,
        "whole_index_stage_seconds_including_order_diag": ix["seconds"],
        "remaining_stage_seconds_after_index_request_wall_sum": ix["seconds"] - all_latency["sum_seconds"],
        "pricing_at_recorded_rate_not_new_launch_quote": {
            "usd_per_h100_minute": 0.0495,
            "original_stage_minutes_including_diag": ix["seconds"] / 60,
            "original_stage_compute_usd_including_diag": ix["seconds"] / 60 * 0.0495,
            "forward_request_wall_minutes": all_latency["sum_seconds"] / 60,
            "forward_request_wall_compute_usd": all_latency["sum_seconds"] / 60 * 0.0495,
            "note": "No setup/build/finalize/retry pricing and no new-order latency measurement; not a spend cap."},
        "counterexample_fixed_original_endpoint": fixed_endpoint,
        "next_experiment": {
            "mixed_original_plus_reversed_relabelled": "one new full pass; tests a combined relabelling/order ensemble",
            "symmetric_relabelled_forward_plus_reversed": "two new full passes; save both probability maps",
            "position_only": "requires explicit presentation independent of continuation IDs or balanced neutral controls",
            "status": "not implemented or launched"},
    }
    Path(output_path).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: out[k] for k in ("validation", "by_k", "forward_request_latency_all",
                                       "whole_index_stage_seconds_including_order_diag",
                                       "pricing_at_recorded_rate_not_new_launch_quote")}, indent=2))


if __name__ == "__main__":
    main()
