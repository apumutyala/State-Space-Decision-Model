"""Mode latency_ab (jobs/m2_first_cell): params, preflight price, the row-by-row A/B comparison and its validator."""

from __future__ import annotations

import json
from pathlib import Path

from jobs.m2_first_cell import fulldev_eval as F

ROOT = Path(__file__).resolve().parents[1]
AB = ROOT / "jobs" / "m2_first_cell" / "job_params_latency_ab.json"


def _row(rid, ms, probs, status="ok"):
    r = {"run_id": rid, "status": status, "total_wall_ms": ms}
    if status == "ok":
        r["response"] = {"answers": {q: {"type": "choice", "choice": max(p, key=p.get), "probabilities": p}
                                     for q, p in probs.items()}}
    return r


def _write(p: Path, rows):
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_params_price_both_variants_under_5_usd_and_inside_the_cap():
    p = json.loads(AB.read_text(encoding="utf-8"))
    assert p["mode"] == "latency_ab" and p["eval_only"] == "fulldev" and p["rows"] == "smoke"
    assert p["child_variants"] == ["base", "fast"] and not p.get("fulldev_parent_arm")
    assert sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"] == p["max_duration_minutes"]
    est = F.estimate(p)
    assert est["ok"] and est["minutes"]["fulldev"] <= 0.95 * p["phase_caps_minutes"]["fulldev"]
    exp = sum(p["expected_minutes"].values()) * p["batch_usd_per_min"]
    assert exp <= 5.0, exp
    assert "latency_ab" in p["submit"]["command"]
    assert (ROOT / p["cell_config"]).is_file()
    for f in p["context"]["manifest_files"]:
        assert (ROOT / f).is_file(), f


def test_ab_compare_counts_flips_dp_and_speedup(tmp_path):
    base, cand = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    rows_a = [_row(f"r{i}", 1000.0, {"q": {"x": 0.7, "y": 0.3}}) for i in range(12)]
    rows_b = [_row(f"r{i}", 250.0, {"q": {"x": 0.69, "y": 0.31}}) for i in range(12)]
    _write(base, rows_a)
    _write(cand, rows_b)
    d = F.ab_compare(base, cand)
    assert d["rows_both"] == 12 and d["decision_flips"] == 0 and abs(d["max_abs_dp"] - 0.01) < 1e-9
    assert d["outputs_match"] and abs(d["speedup_sum"] - 4.0) < 1e-9 and F.valid_ab(d)
    rows_b[3] = _row("r3", 250.0, {"q": {"x": 0.2, "y": 0.8}})  # one flipped decision
    _write(cand, rows_b)
    d = F.ab_compare(base, cand)
    assert d["decision_flips"] == 1 and not d["outputs_match"] and F.valid_ab(d)  # valid run, outputs differ


def test_valid_ab_refuses_missing_rows(tmp_path):
    base, cand = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write(base, [_row("r0", 10.0, {"q": {"x": 1.0}}), _row("r1", 10.0, {"q": {"x": 1.0}})])
    _write(cand, [_row("r0", 5.0, {"q": {"x": 1.0}})])
    assert not F.valid_ab(F.ab_compare(base, cand))
    assert not F.valid_ab({"schema": "other"})
