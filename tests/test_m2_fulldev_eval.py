"""Mode fulldev_eval (jobs/m2_first_cell/fulldev_eval.py): CPU checks of the pieces that decide what is scored and
priced (no model, no kit data): the over-length cap, the plan-only refusal pass, latency, pricing, the validator and
the runner wiring."""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

from jobs.m2_first_cell import fulldev_eval as F

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "jobs" / "m2_first_cell" / "job_params_fulldev_eval.json"


class _TokBackend:
    """Plans len(state text) 'tokens'; score() must never run in a plan pass."""
    name = "tok"

    def revision(self):
        return {"revision_id": "tok@test"}

    def plan(self, request):
        from decide.backends.base import Plan

        n = len(json.dumps(request["state"]))
        return Plan(inference_calls=1, max_call_input_tokens=n, total_input_tokens=n)

    def score(self, request, check_deadline):
        raise AssertionError("plan pass reached score()")


def _rows(long_n=20_000):
    rows = []
    for i in range(4):
        st = {"note": "x" * (40 if i != 2 else long_n)}
        rows.append({"_evaluation": {"run_id": f"SYNTH:{i}", "group_id": f"g{i}", "catalog_id": 32},
                     "state": st, "questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}})
    return rows


def test_default_cap_refuses_a_long_row_and_the_raised_request_cap_reads_it():
    from decide import Decider

    dec = Decider(_TokBackend())
    rows = _rows()
    p16 = F.plan_rows(dec, rows, 16_384)
    assert [r["run_id"] for r in p16["refused"]] == ["SYNTH:2"] and p16["refused"][0]["code"] == "input_tokens_exceeded"
    p64 = F.plan_rows(dec, rows, F.DEFAULT_MAX_INPUT_TOKENS)
    assert p64["n_refused"] == 0 and p64["n_over_16384"] == 1 and p64["max_row_tokens"] > 16_384


def test_limited_sets_only_the_input_cap():
    seen = []
    F.limited(lambda req: seen.append(req) or {}, 65_536)({"model": "m", "state": {}, "questions": {}})
    assert seen[0]["ssdm"] == {"limits": {"max_input_tokens": 65_536}} and seen[0]["model"] == "m"


def test_latency_matches_the_clef_row_definition():
    res = [{"status": "ok", "total_wall_ms": float(i)} for i in range(1, 111)] + [{"status": "exhausted",
                                                                                   "total_wall_ms": 1e9}]
    lat = F.latency(res)
    assert lat["n"] == 100 and lat["median_ms"] == 60.5 and lat["p95_ms"] == 105.0  # rows 11..110, nearest rank


def test_committed_lengths_cover_full_dev_and_the_cap_clears_the_longest_row():
    L = json.loads(F.LENGTHS.read_text(encoding="utf-8"))
    p = json.loads(PARAMS.read_text(encoding="utf-8"))
    assert L["n_rows"] == F.N_DEV == len(L["lengths"]) and sum(L["lengths"].values()) == L["total_tokens"]
    assert p["max_input_tokens"] > L["max_tokens"] and L["overlength51"]["n"] == F.N_OVERLENGTH
    assert set(F.overlength_ids()) <= set(L["lengths"])


def test_estimate_prices_from_measured_rates_and_refuses_a_template_mismatch():
    p = json.loads(PARAMS.read_text(encoding="utf-8"))
    est = F.estimate(p)
    assert est["ok"] and 380 < est["minutes"]["fulldev"] < 0.95 * p["phase_caps_minutes"]["fulldev"]
    slow = copy.deepcopy(p)
    slow["measured_rates_reference"]["child_s_per_1k_tokens"] *= 1.5
    assert not F.estimate(slow)["ok"]
    bad = copy.deepcopy(p)
    bad["readout_template"] = "jrt-rwkv-g1-v2f"
    with pytest.raises(ValueError, match="readout_template"):
        F.estimate(bad)


def test_params_are_consistent_with_the_runner_and_the_shared_loader():
    from jobs.common.costs import schedule_minutes
    from jobs.m2_first_cell.checkpoint_loader import checkpoint_block

    p = json.loads(PARAMS.read_text(encoding="utf-8"))
    assert p["mode"] == "fulldev_eval" and p["eval_only"] == "fulldev"
    assert p["cell_config"] == "jobs/m2_first_cell/m2_first_cell_rwkv7_fulldev_eval.json"
    assert schedule_minutes(p) == p["max_duration_minutes"]
    assert checkpoint_block(p)["artifact"] == "art-bm3e6"
    assert "<" in p["submit"]["env"]["CKPT_URL"]  # a placeholder: the signed URL is never committed
    cfg = json.loads((ROOT / p["cell_config"]).read_text(encoding="utf-8"))
    base = json.loads((ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7.json").read_text(encoding="utf-8"))
    assert cfg == base and cfg["template"] == p["readout_template"]  # D1 = the base cell
    sh = (ROOT / "jobs/m2_first_cell/first_cell_job.sh").read_text(encoding="utf-8")
    assert "jobs.m2_first_cell.fulldev_eval:valid_result" in sh and "checkpoint_loader install" in sh
    assert subprocess.run(["bash", "-n", str(ROOT / "jobs/m2_first_cell/first_cell_job.sh")]).returncode == 0


def _result(**over):
    ci = {"point": 40.0, "lo": 38.0, "hi": 42.0}
    d = {"schema": F.SCHEMA, "verdict": "measured", "rows": F.N_DEV,
         "arms": {"child": {"status": {"ok": F.N_DEV}, "full": {"index": ci}, "excl_overlength51": {"index": ci},
                            "latency": {"median_ms": 500.0, "p95_ms": 9000.0}}},
         "checkpoint": {"ckpt_sha256": "a" * 64, "loaded": {"sha256": "a" * 64}}}
    d.update(over)
    return d


def test_valid_result_requires_every_row_and_the_pinned_checkpoint():
    assert F.valid_result(_result())
    assert not F.valid_result(_result(checkpoint={"ckpt_sha256": "a" * 64, "loaded": {"sha256": "b" * 64}}))
    short = _result()
    short["arms"]["child"]["status"] = {"ok": F.N_DEV - 1}
    assert not F.valid_result(short)
    nan = _result()
    nan["arms"]["child"]["full"]["index"] = {"point": float("nan"), "lo": 0, "hi": 1}
    assert not F.valid_result(nan)
