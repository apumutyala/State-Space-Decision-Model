"""CPU tests for the rung-2 qualification job's gating, cost cap and schedule (issue #3 / PR #9 Track L findings).

Run: python -m tests.test_qual_job
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile

from jobs.qual import finalize
from scripts import decide_9b

P = json.load(open("jobs/qual/job_params.json"))
CFG = json.load(open("configs/qual_h100.json"))
RUNNER = open("jobs/qual/qual_job.sh", encoding="utf-8").read()
SPEC = open("jobs/qual/JOB_SPEC.md", encoding="utf-8").read()
DOCKER = open("jobs/qual/Dockerfile", encoding="utf-8").read()

GOOD = {"verdict": "re-price-lean-plan", "stage2_mfu": 0.2, "peak_gb": 50.0, "stopped": None}


def _fin(rc, export_fail, decision, stale_pass=False):
    with tempfile.TemporaryDirectory() as d:
        dec = os.path.join(d, "decision.json")
        if decision is not None:
            with open(dec, "w") as f:
                f.write(decision if isinstance(decision, str) else json.dumps(decision))
        if stale_pass:
            open(os.path.join(d, "QUAL_PASS"), "w").write("stale")
        code = finalize.main([os.path.join(d, "result.json"), str(rc), "{}", str(export_fail), "1", dec, d])
        res = json.load(open(os.path.join(d, "result.json")))
        return code, os.path.exists(os.path.join(d, "QUAL_PASS")), res["exit_code"]


def t_finalize_pass():
    return _fin(0, 0, GOOD) == (0, True, 0)


def t_finalize_missing_decision():
    return _fin(0, 0, None, stale_pass=True) == (9, False, 9)


def t_finalize_malformed_decision():
    return _fin(0, 0, "{not json") == (9, False, 9)


def t_finalize_unknown_or_unmeasured_verdict():
    a = _fin(0, 0, {**GOOD, "verdict": "launch-everything"})
    b = _fin(0, 0, {"verdict": "no-measurement (stage 2 did not run)", "stage2_mfu": None, "peak_gb": None})
    c = _fin(0, 0, {**GOOD, "stage2_mfu": None})
    return a == b == c == (9, False, 9)


def t_finalize_export_fail():
    return _fin(0, 1, GOOD) == (6, False, 6)


def t_finalize_propagates_stop_codes():
    return all(_fin(rc, 0, GOOD, stale_pass=True) == (rc, False, rc) for rc in (2, 3, 4, 5, 7, 8))


def t_decide_exit_codes():
    with tempfile.TemporaryDirectory() as d:
        ok = os.path.join(d, "ok.json")
        bad = os.path.join(d, "bad.json")
        json.dump({"stage1": {"peak_gb": 40.0}, "progressive": {"steps": [{"steady_tok_s": 3500.0, "peak_gb": 45.0, "flop_per_token": 6.18e10}]}},
                  open(ok, "w"))
        json.dump({"stage1": {}}, open(bad, "w"))
        r_ok = decide_9b.main(ok, "configs/qual_h100.json", os.path.join(d, "d1.json"))
        r_bad = decide_9b.main(bad, "configs/qual_h100.json", os.path.join(d, "d2.json"))
    return r_ok == 0 and r_bad == 9


def t_cost_cap_includes_build():
    rate, cap = P["batch_usd_per_min"], P["hard_cap_usd"]
    attempts = 1 + max(P["max_restarts"], P["assumed_restart_floor"])
    worst = (P["build_timeout_minutes"] + attempts * P["max_duration_minutes"]) * rate
    return worst <= cap and P["build_timeout_minutes"] > 0


def t_schedule_fits_duration():
    c = P["phase_caps_minutes"]
    total = c["setup"] + c["A_per_test"] * c["A_tests"] + c["B"] + c["C_weight_wait"] + c["C"] + P["reserve_minutes"]
    s2_9b = CFG["models"]["9b"]["stage2"]["guard"]["max_minutes"]
    s2_08 = CFG["models"]["0p8b"]["stage2"]["guard"]["max_minutes"]
    return total <= P["max_duration_minutes"] and s2_9b + 30 <= c["C"] and s2_08 + 10 <= c["B"]


def t_runner_gating_static():
    no_m0 = not re.search(r"phase 0|debias|safety_eval|/opt/ssdm/(decide|steer)", RUNNER)
    no_direct_pass = "QUAL_PASS" not in RUNNER.replace("# /output/QUAL_PASS is written ONLY", "").split("finish()")[0] \
        and not re.search(r"touch [^\n]*QUAL_PASS", RUNNER)
    decision_gated = re.search(r'decide_9b\.py[^\n]*\n?[^\n]*r=\$\?\s*\ngate "D:decision" \$r\s*\n\[ \$r -eq 0 \] \|\| fail 9',
                               RUNNER) is not None
    caps_from_params = all(f"timeout ${{{v}}}m" in RUNNER for v in ("CAP_A", "CAP_B", "CAP_C")) \
        and not re.search(r"timeout \d+m", RUNNER)
    tests_gated = all(t in RUNNER for t in ("test_gdn", "test_init_shapes", "test_progressive_resume", "test_guard",
                                            "test_qual_job"))
    finalize_used = "jobs/qual/finalize.py" in RUNNER
    image_pure = not re.search(r"decide|steer|mamba", DOCKER)
    return all([no_m0, no_direct_pass, decision_gated, caps_from_params, tests_gated, finalize_used, image_pure])


def t_spec_requires_cap_and_revalidation():
    s = SPEC.lower()
    return ("set_workspace_cap" in SPEC and "re-run `validate_job` after setting the cap" in s and "required" in s
            and "build_timeout_minutes" in SPEC and str(P["max_duration_minutes"]) in SPEC)


TESTS = [t_finalize_pass, t_finalize_missing_decision, t_finalize_malformed_decision,
         t_finalize_unknown_or_unmeasured_verdict, t_finalize_export_fail, t_finalize_propagates_stop_codes,
         t_decide_exit_codes, t_cost_cap_includes_build, t_schedule_fits_duration, t_runner_gating_static,
         t_spec_requires_cap_and_revalidation]


def main():
    res = {}
    for t in TESTS:
        try:
            res[t.__name__] = bool(t())
        except Exception as e:  # noqa: BLE001
            res[t.__name__] = f"ERROR {e!r}"
    res["n_tests"] = len(TESTS)
    res["n_passed"] = sum(v is True for v in res.values())
    res["PASS"] = res["n_passed"] == len(TESTS)
    print(json.dumps(res, indent=1))
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
