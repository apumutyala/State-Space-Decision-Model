"""Clef 27B Index + teacher-export jobs: pricing, caps, registration and the measured-rate preflight (CPU, no data)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from jobs.clef_baseline import price as P
from jobs.common import costs

ROOT = Path(__file__).resolve().parents[2]
SPEC = (ROOT / "jobs/clef_baseline/JOB_SPEC.md").read_text(encoding="utf-8")
TREE = (ROOT / "jobs/DECISION_TREE.md").read_text(encoding="utf-8")
RUNNER = (ROOT / "jobs/clef_baseline/run_job.sh").read_text(encoding="utf-8")
NAMES = {"clef_index_subset": "job_params_index_subset.json", "clef_index_full": "job_params_index_full.json",
         "clef_index_full_reference": "job_params_index_full_reference.json",
         "clef_teacher_flash": "job_params_teacher_clef_flash.json", "clef_teacher_clef": "job_params_teacher_clef.json",
         "clef_overlength_flash": "job_params_overlength_flash.json"}


def test_flops_ratio_from_pinned_configs():
    assert P.matmul_params_per_token(P.CONFIGS["clef"]) == 24_350_556_160
    assert P.matmul_params_per_token(P.CONFIGS["clef-flash"]) == 6_918_504_448
    assert round(P.flops_ratio(), 2) == 3.52
    # revisions are the ones the runner and clef_backend pin
    from baselines.clef_backend import RELEASES

    for r in ("clef", "clef-flash"):
        assert P.CONFIGS[r]["revision"] == RELEASES[r]["revision"]
        assert f"{RELEASES[r]['revision']}" in RUNNER


def test_sgihj_rate_model_reproduces_the_receipt():
    # the fitted per-row model sums to the measured 18.86 summed minutes over the job's 12,053 ok rows (+51 unsupported)
    assert abs(P.decide_minutes("clef-flash", 12053, P.SGIHJ["decide_tokens"]) - 18.81) < 0.1
    # planning at the card's 5.39x latency ratio (PR #98 review); the FLOPs ratio is the optimistic reference
    assert round(P.PLANNING_RATIO, 2) == 5.39
    assert abs(P.projection("clef", "dev_full")["decide_min"] - 101.6) < 0.2
    assert abs(P.projection("clef", "dev_full")["flops_ratio_min"] - 66.3) < 0.2
    assert abs(P.projection("clef", "dev_subset")["decide_min"] - 17.1) < 0.2


@pytest.mark.parametrize("name", list(NAMES))
def test_caps_margins_schedule_and_costs(name):
    p = costs.load(name)
    assert costs.PARAMS[name] == f"jobs/clef_baseline/{NAMES[name]}"
    c = P.check_params(p)
    assert c["schedule_ok"] and c["min_margin"] >= 1.3  # every phase cap >= 1.3x its expected minutes
    assert p["build_timeout_minutes"] >= 60 and p["batch_usd_per_min"] == costs.rates.BATCH_USD_PER_MIN
    assert costs.cap_ok(p) and costs.expected_usd(p) < costs.worst_case_usd(p)
    assert f"${costs.worst_case_usd(p):.2f}" in SPEC and f"${costs.expected_usd(p):.2f}" in SPEC
    assert f"jobs/clef_baseline/{NAMES[name]}" in TREE and f"${costs.worst_case_usd(p):.2f}" in TREE
    assert p["decision_rule"] is None and p["runner"] == "jobs/clef_baseline/run_job.sh"


def test_index_caps_cover_the_projection_and_the_rate_guard_overshoot():
    # projection = planning-ratio (5.39x) decide minutes + load; the live linear projection overshoots by <= 1.32 (job-sgihj
    # row order, after 12% of the phase); it must stay under rate_guard's 95% of the cap at the expected rate
    for name, vol, over in (("clef_index_subset", "dev_subset", 1.32), ("clef_index_full", "dev_full", 1.28)):
        p = costs.load(name)
        proj = P.projection("clef", vol)
        exp = proj["decide_min"] + proj["load_min"]
        assert p["expected_minutes"]["index"] >= exp
        assert exp * over <= 0.95 * p["phase_caps_minutes"]["index"]


def test_rederive_full_caps_from_the_subset_measurement():
    full = costs.load("clef_index_full")
    sub_model = P.decide_minutes("clef-flash", 2008, P.VOLUMES["dev_subset"]["tokens"])
    at = lambda s: {"elapsed_min": s * sub_model + 1.6, "rows_done": 2008, "startup_s": 96.0}  # noqa: E731
    ok = P.rederive(at(5.39), full)  # the planning case fits the committed cap
    assert ok["ok"] and abs(ok["projected_full_index_min"] - 103.2) < 0.5
    slow = P.rederive(at(7.0), full)  # a slower box: the full job's cap must be raised before it is submitted
    assert not slow["ok"] and slow["required_index_cap_min"] > full["phase_caps_minutes"]["index"]
    with pytest.raises(ValueError):
        P.rederive({"elapsed_min": 10, "rows_done": 100}, full)


def test_sensitivity_phase_caps():
    for name, vol in (("clef_index_subset", "overlength_subset"), ("clef_index_full", "overlength_all"),
                      ("clef_overlength_flash", "overlength_all")):
        p = costs.load(name)
        rel = "clef-flash" if "flash" in name else "clef"
        pr = P.projection(rel, vol)
        assert p["expected_minutes"]["sens"] >= pr["load_min"] + P.LONG_ROW_FACTOR * pr["decide_min"] - 0.5
    ov = json.loads((ROOT / "jobs/clef_baseline/overlength_rows.json").read_text(encoding="utf-8"))
    assert ov["n"] == len(ov["rows"]) == 51 and ov["max_tokens"] <= ov["max_length_sensitivity"] == 32768
    assert sum(r["tokens"] for r in ov["rows"]) == P.VOLUMES["overlength_all"]["tokens"]


def test_teacher_caps_cover_the_projection():
    for name, rel in (("clef_teacher_flash", "clef-flash"), ("clef_teacher_clef", "clef")):
        p = costs.load(name)
        proj = P.projection(rel, "teacher_draw_24k")
        assert p["expected_minutes"]["export"] >= proj["decide_min"] + proj["load_min"]
        assert p["phase_caps_minutes"]["export"] >= 1.3 * p["expected_minutes"]["export"]


def test_joint_gate_and_dependencies():
    led = costs.branch_ledger()
    assert led["joint_gate_fits"] and led["joint_gate_worst"] == 61.74  # qual + Clef-flash GCG B (48.51 at the pre-2026-10-08 rate)
    # Clef-27B Index (re-priced at 5.39x) is a separately authorised line, still counted against granted credit
    assert led["clef27b_index_authorised_worst"] == round(sum(costs.worst_case_usd(costs.load(n))
                                                              for n in costs.CLEF27B_INDEX_JOBS), 2)
    assert all(b["fits_granted_credit"] for b in led["branches"].values())
    full = costs.load("clef_index_full")
    assert full["submit"]["depends_on"]["require"] == "result.s1_ok == true and result.ok == true"
    assert costs.load("clef_teacher_clef")["submit"]["depends_on"]["require"] == "result.s1_ok == true"
    # job C no longer waits on A-flash result.ok (job-sgihj never wrote it)
    assert 'require":"result.ok == true"}' not in SPEC.split("- **C-27b.**")[1].split("\n")[0]


def test_runner_guards_dev_only_and_exit_codes():
    assert "jobs.common.rate_guard run" in RUNNER and "--exclude-startup" in RUNNER
    assert "jobs.clef_baseline.price preflight" in RUNNER
    assert "SSDM_UNSEAL_TEST" not in RUNNER.replace("(no SSDM_UNSEAL_TEST)", "")
    assert "--slice test" not in RUNNER
    for code in ("fail 17", "fail 18", "fail 20"):
        assert code in RUNNER
    assert not re.search(r"[\\]n[ \t]+--", RUNNER)


def test_preflight_measured_rate():
    def probes(factor):
        rows = [{"run_id": str(i), "ms": factor * (P.SGIHJ["row_ms_intercept"] + P.SGIHJ["row_ms_per_token"] * t),
                 "tokens": t} for i, t in enumerate([500, 1500, 3000, 800] * 16)]
        return {"release": "clef", "parity_timing": rows}

    ok = P.preflight(probes(3.5), "dev_full", 100)  # a box at the FLOPs prediction passes
    assert ok["ok"] and ok["measured"] and abs(ok["speed_vs_flash_sgihj"] - 3.5) < 1e-6
    slow = P.preflight(probes(6.0), "dev_full", 100)  # 6x Clef-flash: 113 min projected > 95
    assert not slow["ok"] and slow["projected_min"] > 95
    assert P.preflight({"release": "clef"}, "dev_full", 100)["measured"] is False


def test_preflight_cli_exit_codes(tmp_path):
    rows = [{"run_id": str(i), "ms": 6.0 * (32.0 + 0.04 * 1000), "tokens": 1000} for i in range(10)]
    f = tmp_path / "probes.json"
    f.write_text(json.dumps({"release": "clef", "parity_timing": rows}), encoding="utf-8")
    args = ["preflight", "--probes", str(f), "--volume", "dev_full", "--cap-min", "100", "--out", str(tmp_path / "o.json")]
    assert P.main(args) == P.EXIT_TOO_SLOW
    assert P.main(args + ["--advisory"]) == 0


def test_rate_guard_exclude_startup_projection():
    from jobs.common import rate_guard as rg

    # 100 of 1000 rows written 60 s after a 300 s load: naive 36 min; startup-excluded 15 min
    assert round(rg.project(100, 360, 1000), 1) == 60.0
    assert round(rg.project(100, 360, 1000, startup_s=300), 1) == 15.0


def test_teacher_scope_matches_m2_first_cell_draw():
    s = json.loads((ROOT / "jobs/clef_baseline/teacher_scope.json").read_text(encoding="utf-8"))
    assert s["pointer"] == "data/manifests/sample-150000-seed20261006.pointer.json"
    assert (ROOT / s["pointer"]).exists()
    assert s["draw"] == {**s["draw"], "n_records": 24000, "seed": 0}
    assert s["cohort_sha256"] == (ROOT / "train/cohorts/m2-dev-cohort-v2.sha256").read_text().split()[0]
    from baselines.clef_teacher import draw_ids
    from train.run_cell import split_ids

    ids = [f"m2:x:{i}" for i in range(50)] + s["heldout_excluded"]["ids"]
    got = draw_ids(ids, s["heldout_excluded"]["ids"], 10, 0)
    assert got == split_ids(sorted(set(ids) - set(s["heldout_excluded"]["ids"])), 10, 0, 0)[0]
    assert not set(got) & set(s["heldout_excluded"]["ids"])


def test_preflight_excludes_one_time_compile_spikes_job_5mw9h():
    """L-26: job-5mw9h stopped at index_preflight (exit 18): 4 of 61 parity rows were 11.5-14.7 s Triton compiles for
    new sequence-length shapes, which the mean-based rate extrapolated to the whole subset (34.0 min > 95% of 27)."""
    f = ROOT / "tests/clef/fixtures/job_5mw9h_parity_timing.json"
    probes = json.loads(f.read_text(encoding="utf-8"))
    r = P.preflight(probes, "dev_subset", 27)
    # tightened (PR #106 review): 3 spikes are confirmed by a later fast row of comparable length; the 10,595-token
    # row has no fast repeat, so it stays IN the rate: s ~5.23x, projection ~18.7 min, still < 0.95 x 27 = 25.65
    assert r["ok"] and len(r["compile_spikes"]) == 3 and all(x["ms"] > 11000 for x in r["compile_spikes"])
    assert r["slow_rows_unconfirmed"] == 1 and 10595 not in [x["tokens"] for x in r["compile_spikes"]]
    assert abs(r["speed_vs_flash_sgihj"] - 5.23) < 0.05 and abs(r["projected_min"] - 18.7) < 0.15
    assert r["projected_min"] < 0.95 * 27 and not r["too_many_spikes"]
    assert 0.55 < r["warmup_min_charged_once"] < 0.65  # the confirmed spikes' time is still charged, once
    # a uniformly slow box is still stopped: no row is a spike when every row is slow
    slow = {"release": "clef", "parity_timing": [dict(x, ms=x["ms"] * 1.0) for x in probes["parity_timing"][:3]] +
            [dict(x, ms=12.0 * (P.SGIHJ["row_ms_intercept"] + P.SGIHJ["row_ms_per_token"] * x["tokens"]))
             for x in probes["parity_timing"][3:]]}
    s = P.preflight(slow, "dev_subset", 27)
    assert not s["ok"] and s["compile_spikes"] == []



def _model_ms(x):
    return P.SGIHJ["row_ms_intercept"] + P.SGIHJ["row_ms_per_token"] * x["tokens"]


def _fixture():
    return json.loads((ROOT / "tests/clef/fixtures/job_5mw9h_parity_timing.json").read_text(encoding="utf-8"))


def test_preflight_slow_long_rows_with_no_fast_repeat_are_not_excluded():
    """PR #106 review, scenario 1: a box that is steadily 30x Clef-flash on rows > 5k tokens (a slow long-sequence
    path, not a compile) must fail closed. The untightened rule excluded those rows and passed (10.9 vs truly 51 min)."""
    base = _fixture()["parity_timing"]
    rows = [dict(x, ms=x["ms"] if i < 3 else (30.0 if x["tokens"] > 5000 else 2.5) * _model_ms(x))
            for i, x in enumerate(base)]
    r = P.preflight({"release": "clef", "parity_timing": rows}, "dev_subset", 27)
    assert not r["ok"] and r["compile_spikes"] == [] and r["slow_rows_unconfirmed"] > 0


def test_preflight_more_than_10pct_excluded_is_not_ok():
    """PR #106 review, scenario 2: 25% of rows 5x the median (bimodal), each followed by fast rows of the same length.
    Even if every slow row is 'confirmed', excluding more than 10% of the sample fails closed."""
    base = _fixture()["parity_timing"]
    fast = [dict(x, ms=x["ms"] if i < 3 else 2.5 * _model_ms(x)) for i, x in enumerate(base)]
    k = int(0.25 * (len(fast) - 3))
    rows = fast[:3] + [dict(x, ms=x["ms"] * 5) for x in fast[3:3 + k]] + fast[3 + k:] +         [dict(x, ms=2.5 * _model_ms(x)) for x in fast[3:3 + k]]  # fast repeats of the same shapes, later
    r = P.preflight({"release": "clef", "parity_timing": rows}, "dev_subset", 27)
    assert r["too_many_spikes"] and not r["ok"]


def test_preflight_zero_median_fails_closed():
    rows = [{"run_id": str(i), "ms": 0.0 if i < 40 else 500.0, "tokens": 1000} for i in range(64)]
    r = P.preflight({"release": "clef", "parity_timing": rows}, "dev_subset", 27)
    assert r["ok"] is False and "median" in r["note"]


# ----------------------------------------------------------------------------- GDN=reference (JOB_SPEC 4d, job-jxevr)
def test_reference_caps_cover_4x_the_measured_fast_rate_and_the_rederive_rule():
    full, ref = costs.load("clef_index_full"), costs.load("clef_index_full_reference")
    assert ref["env"]["CLEF_GDN_PATH"] == "reference" and ref["submit"]["env"] == ref["env"]
    assert P.REFERENCE_SLOWDOWN_FLOOR == 4.0
    proj_fast = 46.2  # price.rederive on job-d2u54's rate_guard_index.json (2.373x Clef-flash), startup included
    proj = P.REFERENCE_SLOWDOWN_FLOOR * proj_fast
    assert ref["expected_minutes"]["index"] >= 0.97 * proj
    assert ref["phase_caps_minutes"]["index"] >= max(1.3 * proj, 1.28 * proj / 0.95) * 0.97  # rederive's rule
    for ph in ("sens", "native"):  # the other reference phases: caps >= 4x the fast params' expected minutes
        assert ref["phase_caps_minutes"][ph] >= 4 * full["expected_minutes"][ph]
    assert ref["max_duration_minutes"] == sum(ref["phase_caps_minutes"].values()) + ref["reserve_minutes"]


def test_reference_preflight_uses_the_reference_arms_own_timing():
    # art-g7mqf-shaped probes: no per_row timing -> live guard only, never a multiplier on compile-spiked fast timing
    assert P.preflight({"parity_timing": [{"ms": 35000, "tokens": 4590}] * 8}, "dev_full", 250, "reference")["measured"] is False
    rows = [{"run_id": f"r{i}", "tokens": 1000, "fast_ms": 100.0, "ref_ms": 400.0} for i in range(16)]
    r = P.preflight({"kernel_path": {"per_row": rows}, "release": "clef"}, "dev_full", 250, "reference")
    assert r["measured"] and r["n_rows"] == 15 and r["ref_over_fast"] == 4.0
    s = 400.0 / (P.SGIHJ["row_ms_intercept"] + P.SGIHJ["row_ms_per_token"] * 1000)
    assert abs(r["speed_vs_flash_sgihj"] - round(s, 3)) < 1e-9 and r["ok"] == (r["projected_min"] <= 0.95 * 250)
    slow = [{**x, "ref_ms": 4000.0} for x in rows]
    assert P.preflight({"kernel_path": {"per_row": slow}}, "dev_full", 250, "reference")["ok"] is False
