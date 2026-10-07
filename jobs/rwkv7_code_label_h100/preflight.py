"""Measured production-row rate and real-shape memory admission; no fallback rates."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VOLUME = ROOT / "receipts/2026-10-07-D-code-label-v2-fork/dev-volume.json"
REFERENCE_TOK_S = 5360.0  # job-mhc3h, jobs/common/readout_cost.py
REFERENCE_CALL_S = 0.098
PLANNING_TOK_S = 3757.39821244465  # inherited planning estimate; not a qualification measurement
PLANNING_CALL_S = 0.5997345447540283  # old 64x64 tail stress bound, not a current-v2 measurement
INHERITED_SMALL_CALL_S = 0.11704301834106445
RATE_MARGIN = 0.95
PEAK_FRACTION = 0.90
MIN_FREE_BYTES = 5 * 1024 ** 3
TIMED_REPEATS = 2
PARITY_ATOL = 0.002  # preregistered absolute probability tolerance; bf16 diagnostic only


def reference_seconds(volume):
    return volume["positions"] / REFERENCE_TOK_S + volume["calls"] * REFERENCE_CALL_S


def planning_seconds(volume):
    return volume["positions"] / PLANNING_TOK_S + volume["calls"] * PLANNING_CALL_S


def probe_ids(volume):
    per = volume["per_row"]
    fields = {"v1": ("max_prompt_tokens", "positions"),
              "v2": ("max_prompt_tokens", "max_pack_positions", "n_options", "prefix_tokens")}
    return sorted({max((i for i in sorted(per) if per[i]["admitted"][arm]), key=lambda i: per[i][arm][field])
                   for arm, names in fields.items() for field in names})


def evaluate(probe, volume, cap_min):
    if probe.get("synthetic") or probe.get("gpu", "").find("H100") < 0:
        raise ValueError("real H100 probe required")
    parity = probe.get("parity_max_abs")
    if type(parity) not in (float, int) or not math.isfinite(parity) or not 0 <= parity <= PARITY_ATOL:
        raise ValueError("production fork/recompute parity failed")
    samples = probe["samples"]
    required = {(arm, rid) for arm in ("v1", "v2") for rid in probe_ids(volume)}
    if {(s["arm"], s["run_id"]) for s in samples} != required or len(samples) != len(required):
        raise ValueError("incomplete real-shape probe coverage")
    factors = {"v1": [], "v2": []}
    for sample in samples:
        seconds = sample["seconds"]
        if (len(seconds) != TIMED_REPEATS or any(type(s) not in (float, int) or not math.isfinite(s) or s <= 0 for s in seconds)
                or not 0 < sample["peak_reserved_bytes"] <= PEAK_FRACTION * probe["total_memory_bytes"]
                or sample["min_free_bytes"] < MIN_FREE_BYTES):
            raise ValueError("invalid timings or real-shape memory stop")
        baseline = reference_seconds(volume["per_row"][sample["run_id"]][sample["arm"]])
        factors[sample["arm"]].append(max(seconds) / baseline)
    # Take the slowest warmed real-shape sample per arm, not an optimistic fit.
    factors = {arm: max(values) for arm, values in factors.items()}
    projected = {a: reference_seconds(volume["totals"]["admitted"][a]) * factors[a] / 60 for a in factors}
    if sum(projected.values()) > RATE_MARGIN * cap_min:
        raise ValueError("measured full workload exceeds scoring cap")
    return {"ok": True, "rate_factors": factors, "projected_minutes": projected,
            "total_projected_minutes": sum(projected.values()), "cap_minutes": cap_min,
            "rate_margin": RATE_MARGIN, "probe_sha256_required": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from jobs.rwkv7_code_label_screen.runner import write_json
    params = json.loads((Path(__file__).with_name("job_params.json")).read_text())
    try:
        result = evaluate(json.loads(args.probe.read_text()), json.loads(VOLUME.read_text()), params["phase_caps_minutes"]["screen"])
    except Exception as exc:
        write_json(args.out, {"ok": False, "error_type": type(exc).__name__})
        return 15
    write_json(args.out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
