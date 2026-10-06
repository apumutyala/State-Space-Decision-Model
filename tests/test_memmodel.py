"""CPU tests: memory/FLOP model arithmetic, calibration against the paid-run observations (job-c4zkm), the 9B plan
(micro-batch 4 x 4096 with accumulation) and the MFU-derived stop thresholds.

Run: python -m tests.test_memmodel   (or pytest)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from convert import memmodel as M
from convert.align import Guard

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs/qual_h100.json").read_text(encoding="utf-8"))
D9, D08 = M.QWEN35_9B, M.QWEN35_0P8B
# job-c4zkm observations (out/qual_9b.json, out/qual_0p8b.json)
OBS = {"w9": 17.9076352, "w08": 1.50480896, "s1_9b_peak": 68.92, "s1_08_peak": 11.8, "s2_08_peak": 5.70}


def rel(a, b):
    return abs(a - b) / b


def test_param_counts_match_loaded_weights():
    assert rel(M.weights_bytes(D9) / 1e9, OBS["w9"]) < 0.005
    assert rel(M.weights_bytes(D08) / 1e9, OBS["w08"]) < 0.005
    assert 67.0e6 < D9.p_gdn < 67.8e6 and 10.4e6 < D08.p_gdn < 10.7e6
    assert D9.attn_layers == [3, 7, 11, 15, 19, 23, 27, 31] and D08.attn_layers == [3, 7, 11, 15, 19, 23]


def test_calibration_reproduces_observed_peaks():
    # old code: all aligned GDN graphs alive at once, full batch
    s1_9b = M.stage1_peak(D9, 8 * 4096, graphs_alive=8, frag=1.0) / 1e9
    s1_08 = M.stage1_peak(D08, 16 * 1024, graphs_alive=6, frag=1.0) / 1e9
    s2_08 = M.stage2_peak(D08, 16 * 1024, 0, n_attached=6, frag=1.0) / 1e9  # old path: full student forward
    assert rel(s1_9b, OBS["s1_9b_peak"]) < 0.03
    assert rel(s1_08, OBS["s1_08_peak"]) < 0.05
    assert OBS["s2_08_peak"] <= s2_08 <= 1.3 * OBS["s2_08_peak"]  # conservative (upper bound) for stage 2


def test_9b_plan_fits_and_keeps_tokens_per_step():
    m = CFG["models"]["9b"]
    assert m["batch"] == 8 and m["seq_len"] == 4096 and m["batch"] % m["micro_batch"] == 0
    first = D9.attn_layers[0] if m["stage2_layer"] == "first" else D9.attn_layers[-1]
    p = M.plan(D9, m["batch"], m["seq_len"], m["micro_batch"], first)
    assert p["tokens_per_step"] == 8 * 4096 and p["accum_steps"] * p["tokens_per_micro_batch"] == p["tokens_per_step"]
    for st in ("stage1", "stage2"):
        assert p[st]["peak_gb"] <= 60.0 < m[st]["guard"]["max_peak_gb"]  # decision-rule memory line, with margin
    # the OOM configuration sits within ~5% of the 75.2 GB process cap (allocator headroom gone); the fix uses < half
    cap = 0.95 * 79.18
    assert M.stage1_peak(D9, 8 * 4096, graphs_alive=8) / 1e9 > 0.95 * cap
    assert max(p["stage1"]["peak_gb"], p["stage2"]["peak_gb"]) < 0.5 * cap
    # memory is monotone in the micro-batch
    peaks = [M.plan(D9, 8, 4096, mb, first)["stage1"]["peak_gb"] for mb in (1, 2, 4, 8)]
    assert peaks == sorted(peaks)


def test_flops_and_thresholds():
    f1 = M.flops_stage1(D9, D9.attn_layers)
    f2 = M.flops_stage2(D9, [D9.attn_layers[0]])
    assert rel(f1, 2 * D9.n_body + 6 * 8 * D9.p_gdn) < 1e-9  # last aligned layer is the last layer
    assert 55e9 < f2 < 65e9 and f2 < 10 * 8.5e9  # the old 10N = 85 GF/token over-counted the 9B body
    assert rel(M.min_tok_s(f1, 0.125), 0.125 * 989e12 / f1) < 1e-12
    g = Guard({"min_mfu": 0.125}, "s2", flop_per_token=f2)
    assert rel(g.min_tok_s, 0.5 * 0.25 * 989e12 / f2) < 1e-12  # = 50% of the 25%-MFU prediction
    for st in ("stage1", "stage2"):
        assert CFG["models"]["9b"][st]["guard"]["min_mfu"] == 0.125 and "min_tok_s" not in CFG["models"]["9b"][st]["guard"]
    try:
        Guard({"min_mfu": 0.125}, "x")
        raise AssertionError("min_mfu without flop_per_token must raise")
    except ValueError:
        pass


TESTS = [test_param_counts_match_loaded_weights, test_calibration_reproduces_observed_peaks,
         test_9b_plan_fits_and_keeps_tokens_per_step, test_flops_and_thresholds]


def main():
    res = {}
    for t in TESTS:
        try:
            t()
            res[t.__name__] = True
        except Exception as e:  # noqa: BLE001
            res[t.__name__] = f"FAIL {e!r}"
    m = CFG["models"]["9b"]
    p = M.plan(D9, m["batch"], m["seq_len"], m["micro_batch"], D9.attn_layers[0])
    res["plan_9b"] = {k: ({kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()}
                          if isinstance(v, dict) else v) for k, v in p.items()}
    res["n_tests"], res["n_passed"] = len(TESTS), sum(v is True for v in res.values())
    res["PASS"] = res["n_passed"] == len(TESTS)
    print(json.dumps(res, indent=1))
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
