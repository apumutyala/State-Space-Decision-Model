"""jobs/cell_step_profile PRE-REGISTERED decision rule (rule.py). Pins the constants and every verdict branch.

    python -X utf8 -m pytest -q tests/test_cell_step_rule.py
"""
from __future__ import annotations

import json

from jobs.cell_step_profile import rule as R


def test_preregistered_constants_are_pinned():
    assert R.VARIANTS == ("baseline", "ckpt", "ckpt_rows", "ckpt_rows_lean")
    assert R.FLAGS == {  # amendment A1 (before any run): row_tokens 8192 on the row variants
        "baseline": {"act_ckpt": False, "row_batch": 1, "row_tokens": 0, "rwkv7_wkv": "chunked"},
        "ckpt": {"act_ckpt": True, "row_batch": 1, "row_tokens": 0, "rwkv7_wkv": "chunked"},
        "ckpt_rows": {"act_ckpt": True, "row_batch": 8, "row_tokens": 8192, "rwkv7_wkv": "chunked"},
        "ckpt_rows_lean": {"act_ckpt": True, "row_batch": 8, "row_tokens": 8192, "rwkv7_wkv": "chunked_lean"},
    }
    assert (R.WARMUP_STEPS, R.TIMED_STEPS, R.N_STEPS) == (3, 20, 23)
    assert R.MAX_PEAK_GB == 76.0 and R.LOSS_REL_MAX == 1e-3
    assert R.LADDER == (1024, 1536, 2048, 3072, 4096) and R.LADDER_K == 10 and R.LADDER_CONT_LEN == 16
    assert R.CELL_T_LADDER == (512, 1024, 2048) and R.CELL_MINUTES == 190 and R.ROWS_PER_STEP == 8
    assert R.ALLOC_CONF == "expandable_segments:True"


def _rec(v, tok=300.0, oom=(), peak=60.0, loss0=2.0, ladder_fit=4096, skipped=(), nonfinite=(), alloc=None, flags=None):
    steps = []
    for i in range(R.N_STEPS):
        s = {"i": i, "timed": i >= R.WARMUP_STEPS, "tokens": 4000}
        if i in skipped:
            s["skipped"] = "deadline"
        elif i in oom:
            s.update(oom=True, peak_gb=79.0)
        else:
            loss = float("nan") if i in nonfinite else (loss0 if i == 0 else 1.5)
            s.update(s=4000 / tok, tok_s=tok, loss=loss, finite=i not in nonfinite, peak_gb=peak)
        steps.append(s)
    ladder = {str(t): ({"ok": True, "peak_gb": 50.0} if t <= ladder_fit else {"ok": False, "oom": "x", "peak_gb": 79.0})
              for t in R.LADDER}
    return {"variant": v, "flags": R.FLAGS[v] if flags is None else flags, "complete": True, "steps": steps,
            "ladder": ladder, "alloc_conf": R.ALLOC_CONF if alloc is None else alloc}


def _all(**over):
    recs = {"baseline": _rec("baseline", tok=260.0, oom=(0, 1, 2, 4), peak=84.0, ladder_fit=1024),
            "ckpt": _rec("ckpt", tok=200.0, peak=40.0),
            "ckpt_rows": _rec("ckpt_rows", tok=1000.0, peak=55.0),
            "ckpt_rows_lean": _rec("ckpt_rows_lean", tok=1500.0, peak=55.0)}
    recs.update(over)
    return recs


def test_adopt_fastest_qualifying_variant_and_projection():
    d = R.decide(_all())
    assert d["verdict"] == "ADOPT" and d["chosen"] == "ckpt_rows_lean"
    assert d["cell_config"] == {"act_ckpt": True, "row_batch": 8, "row_tokens": 8192, "rwkv7_wkv": "chunked_lean",
                                "t_ladder": [512, 1024, 2048]}
    p = d["projection"]
    assert p["median_step_s"] == 4000 / 1500.0 and p["steps_190min"] == int(190 * 60 / (4000 / 1500.0))
    assert p["rows_190min"] == 8 * p["steps_190min"]
    assert d["variants"]["baseline"]["qualifies"] is False  # OOM + peak
    assert d["variants"]["ckpt_rows_lean"]["loss"]["basis"] == "cpu_equivalence_tests"  # baseline step 0 OOMed
    assert abs(d["speedup_vs_baseline"] - 1500 / 260) < 1e-9
    assert R.valid_decision(json.loads(json.dumps(d)))


def test_each_disqualifier():
    for bad in (dict(oom=(7,)), dict(peak=76.01), dict(skipped=(22,)), dict(nonfinite=(5,)), dict(alloc=""),
                dict(flags={"act_ckpt": True, "row_batch": 8, "row_tokens": 16384, "rwkv7_wkv": "chunked_lean"})):
        d = R.decide(_all(ckpt_rows_lean=_rec("ckpt_rows_lean", tok=1500.0, **bad)))
        assert d["chosen"] == "ckpt_rows", bad
    d = R.decide(_all(ckpt_rows_lean=_rec("ckpt_rows_lean", tok=1500.0, peak=76.0)))  # <= 76 is inside
    assert d["chosen"] == "ckpt_rows_lean"
    d = R.decide(_all(ckpt_rows_lean=None))
    assert d["chosen"] == "ckpt_rows" and d["variants"]["ckpt_rows_lean"]["reasons"] == ["missing"]
    short = _rec("ckpt_rows_lean", tok=1500.0)
    short["steps"] = short["steps"][:20]
    assert R.decide(_all(ckpt_rows_lean=short))["chosen"] == "ckpt_rows"


def test_loss_clause_against_completed_baseline_step0():
    base = _rec("baseline", tok=260.0, oom=(5,), peak=70.0, loss0=2.0)
    d = R.decide(_all(baseline=base, ckpt_rows_lean=_rec("ckpt_rows_lean", tok=1500.0, loss0=2.0 * (1 + 1.1e-3))))
    assert d["chosen"] == "ckpt_rows" and d["variants"]["ckpt_rows_lean"]["loss"]["basis"] == "baseline_step0"
    d = R.decide(_all(baseline=base, ckpt_rows_lean=_rec("ckpt_rows_lean", tok=1500.0, loss0=2.0 * (1 + 0.9e-3))))
    assert d["chosen"] == "ckpt_rows_lean"
    v0 = _rec("ckpt_rows_lean", tok=1500.0)
    v0["steps"][0] = {"i": 0, "timed": False, "tokens": 4000, "oom": True, "peak_gb": 79.0}
    d = R.decide(_all(baseline=base, ckpt_rows_lean=v0))
    assert d["variants"]["ckpt_rows_lean"]["loss"]["ok"] is False


def test_tie_breaks_on_peak_then_order():
    d = R.decide(_all(ckpt_rows=_rec("ckpt_rows", tok=1500.0, peak=50.0)))
    assert d["chosen"] == "ckpt_rows"  # same speed, lower peak
    d = R.decide(_all(ckpt_rows=_rec("ckpt_rows", tok=1500.0, peak=55.0)))
    assert d["chosen"] == "ckpt_rows"  # full tie: earlier variant


def test_cap_t_when_nothing_qualifies_and_no_fit():
    recs = {v: _rec(v, oom=(3,), ladder_fit=t) for v, t in
            zip(R.VARIANTS, (1024, 2048, 1536, 0))}
    d = R.decide(recs)
    assert d["verdict"] == "CAP_T" and d["chosen"] == "ckpt" and d["t_fit"] == 2048
    assert d["cell_config"]["t_ladder"] == [512, 1024, 2048] and d["cell_config"]["t_cap"] == 2048
    recs["ckpt"] = _rec("ckpt", oom=(3,), ladder_fit=1536)
    d = R.decide(recs)
    assert d["t_fit"] == 1536 and d["cell_config"]["t_ladder"] == [512, 1024]
    # a gap in the ladder: 1024 OOM, 1536 fit -> nothing counts (the ladder stops at the first failure)
    gap = _rec("ckpt", oom=(3,), ladder_fit=4096)
    gap["ladder"]["1024"] = {"ok": False, "oom": "x"}
    assert R.summarize(gap)["t_fit"] is None
    d = R.decide({v: _rec(v, oom=(3,), ladder_fit=0) for v in R.VARIANTS})
    assert d["verdict"] == "NO_FIT" and d["chosen"] is None and R.valid_decision(d)


def test_adopt_with_small_ladder_keeps_note():
    d = R.decide(_all(ckpt_rows_lean=_rec("ckpt_rows_lean", tok=1500.0, ladder_fit=1536)))
    assert d["chosen"] == "ckpt_rows_lean" and d["cell_config"]["t_ladder"] == [512, 1024] and "note" in d


def test_valid_decision_rejects_malformed():
    assert not R.valid_decision({})
    assert not R.valid_decision({"rule": "cell_step_profile v1-A1", "verdict": "MAYBE", "variants": {}, "chosen": None})
    d = R.decide(_all())
    d["chosen"] = None
    assert not R.valid_decision(d)


def test_job_params_cost_and_schedule():
    from pathlib import Path

    from jobs.common import costs

    root = Path(__file__).resolve().parents[1]
    p = costs.load("cell_step_profile")
    caps, exp = p["phase_caps_minutes"], p["expected_minutes"]
    assert set(caps) == {"setup", "download_wait", "profile", *R.VARIANTS} == set(exp)
    assert sum(caps.values()) + p["reserve_minutes"] == p["max_duration_minutes"]
    assert all(caps[k] >= 1.3 * exp[k] for k in caps), {k: caps[k] / exp[k] for k in caps}
    assert p["build_timeout_minutes"] == 60 and p["max_restarts"] == 0 and p["assumed_restart_floor"] >= 1
    assert p["chip"] == "h100" and costs.cap_ok(p) and costs.schedule_ok(p)
    spec = (root / "jobs/cell_step_profile/JOB_SPEC.md").read_text(encoding="utf-8")
    assert f"${costs.worst_case_usd(p):.2f}" in spec and f"${costs.expected_usd(p):.2f}" in spec
    sh = (root / "jobs/cell_step_profile/cell_step_job.sh").read_text(encoding="utf-8")
    assert "for V in " + " ".join(R.VARIANTS) + "; do" in sh
    assert f"PYTORCH_CUDA_ALLOC_CONF={R.ALLOC_CONF}" in sh
