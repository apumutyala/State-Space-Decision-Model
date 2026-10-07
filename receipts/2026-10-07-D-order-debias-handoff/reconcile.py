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


def matched_probability_pair_available(first, second):
    """Metadata guard only; it does not validate or score probability contents."""
    return (first["probabilities_saved"] and second["probabilities_saved"]
            and {first["condition"], second["condition"]} == {"forward", "reversed"}
            and first["label_scheme"] == second["label_scheme"]
            and first["rows"] == second["rows"]
            and first["run_ids_sha256"] == second["run_ids_sha256"])


def validate_interpretation(receipt):
    """Hold this historical receipt's claim scope; new measurements need a new receipt."""
    if receipt["schema"] != "ssdm.order-debias-handoff.v1":
        raise ValueError("unsupported receipt schema")
    interpretation = receipt["interpretation"]
    if interpretation["first_choice_statistic"] != "endpoint_excess_not_isolated_position_effect":
        raise ValueError("two endpoint orders cannot isolate position sensitivity")
    if interpretation["index_stage_timer_scope"] != "index_plus_order_diagnostic":
        raise ValueError("the stage timer includes extra diagnostic requests")
    if interpretation["forward_latency_source"] != "index_results.total_wall_ms":
        raise ValueError("forward latency must come from request timers")
    passes = receipt["recorded_passes"]
    available = any(matched_probability_pair_available(a, b)
                    for i, a in enumerate(passes) for b in passes[i + 1:])
    if interpretation["matched_probability_pair_available"] != available or available:
        raise ValueError("this receipt has no matched probability pair")
    if (interpretation["probability_averaging_status"] != "unmeasured"
            or interpretation["new_full_passes_for_symmetric_relabelled_pair"] != 2):
        raise ValueError("matched averaging requires two new full relabelled passes")
    timed = receipt["forward_request_latency_all"]
    if not np.isclose(timed["mean_ms"], timed["sum_seconds"] * 1000 / timed["n"]):
        raise ValueError("mean request latency disagrees with request timer sum")
    if receipt["whole_index_stage_seconds_including_order_diag"] <= timed["sum_seconds"]:
        raise ValueError("stage timer must retain the extra diagnostic work")


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
    def run_ids_sha256(ids):
        return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()

    diag_ids = [r["_evaluation"]["run_id"] for r in selected]
    recorded_passes = [{"condition": "forward", "label_scheme": "original", "rows": len(rows),
                        "probabilities_saved": True, "run_ids_sha256": run_ids_sha256(rows_by_id)}]
    recorded_passes.extend({"condition": condition, "label_scheme": "numeric-prefix/v1", "rows": len(selected),
                            "probabilities_saved": False, "run_ids_sha256": run_ids_sha256(diag_ids)}
                           for condition in ("forward", "reversed"))
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
        "recorded_passes": recorded_passes,
        "interpretation": {
            "first_choice_statistic": "endpoint_excess_not_isolated_position_effect",
            "probability_averaging_status": "unmeasured",
            "matched_probability_pair_available": False,
            "new_full_passes_for_symmetric_relabelled_pair": 2,
            "index_stage_timer_scope": "index_plus_order_diagnostic",
            "forward_latency_source": "index_results.total_wall_ms"},
        "next_experiment": {
            "mixed_original_plus_reversed_relabelled": "one new full pass; tests a combined relabelling/order ensemble",
            "symmetric_relabelled_forward_plus_reversed": "two new full passes; save both probability maps",
            "position_only": "requires explicit presentation independent of continuation IDs or balanced neutral controls",
            "status": "not implemented or launched"},
    }
    validate_interpretation(out)
    Path(output_path).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: out[k] for k in ("validation", "by_k", "forward_request_latency_all",
                                       "whole_index_stage_seconds_including_order_diag",
                                       "pricing_at_recorded_rate_not_new_launch_quote")}, indent=2))


if __name__ == "__main__":
    main()
