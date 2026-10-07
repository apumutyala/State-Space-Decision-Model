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
         "clef_teacher_flash": "job_params_teacher_clef_flash.json", "clef_teacher_clef": "job_params_teacher_clef.json"}


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
    assert abs(P.projection("clef", "dev_full")["decide_min"] - 66.3) < 0.2
    assert abs(P.projection("clef", "dev_subset")["decide_min"] - 11.1) < 0.2


@pytest.mark.parametrize("name", list(NAMES))
def test_caps_margins_schedule_and_costs(name):
    p = costs.load(name)
    assert costs.PARAMS[name] == f"jobs/clef_baseline/{NAMES[name]}"
    c = P.check_params(p)
    assert c["schedule_ok"] and c["min_margin"] >= 1.3  # every phase cap >= 1.3x its expected minutes
    assert p["build_timeout_minutes"] >= 60 and p["batch_usd_per_min"] == 0.0495
    assert costs.cap_ok(p) and costs.expected_usd(p) < costs.worst_case_usd(p)
    assert f"${costs.worst_case_usd(p):.2f}" in SPEC and f"${costs.expected_usd(p):.2f}" in SPEC
    assert f"jobs/clef_baseline/{NAMES[name]}" in TREE and f"${costs.worst_case_usd(p):.2f}" in TREE
    assert p["decision_rule"] is None and p["runner"] == "jobs/clef_baseline/run_job.sh"


def test_index_caps_cover_the_projection_and_the_rate_guard_overshoot():
    # projection = FLOPs-scaled decide minutes + load; the live linear projection overshoots by <= 1.32 (job-sgihj
    # row order, after 12% of the phase); it must stay under rate_guard's 95% of the cap at the expected rate
    for name, vol, over in (("clef_index_subset", "dev_subset", 1.32), ("clef_index_full", "dev_full", 1.28)):
        p = costs.load(name)
        proj = P.projection("clef", vol)
        exp = proj["decide_min"] + proj["load_min"]
        assert p["expected_minutes"]["index"] >= exp
        assert exp * over <= 0.95 * p["phase_caps_minutes"]["index"]


def test_teacher_caps_cover_the_projection():
    for name, rel in (("clef_teacher_flash", "clef-flash"), ("clef_teacher_clef", "clef")):
        p = costs.load(name)
        proj = P.projection(rel, "teacher_draw_24k")
        assert p["expected_minutes"]["export"] >= proj["decide_min"] + proj["load_min"]
        assert p["phase_caps_minutes"]["export"] >= 1.3 * p["expected_minutes"]["export"]


def test_joint_gate_and_dependencies():
    led = costs.branch_ledger()
    assert led["joint_gate_fits"] and led["joint_gate_worst"] == 74.54
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
