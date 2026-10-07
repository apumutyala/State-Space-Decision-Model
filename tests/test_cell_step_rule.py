"""jobs/cell_step_profile PRE-REGISTERED decision rule (rule.py, v1-A2). Pins the constants and every verdict branch.

    python -X utf8 -m pytest -q tests/test_cell_step_rule.py
"""
from __future__ import annotations

import json

from jobs.cell_step_profile import rule as R


def test_preregistered_constants_are_pinned():
    # A2 (after independent review, before any run): run order fastest-expected first, ckpt (memory-only) last
    assert R.VARIANTS == ("baseline", "ckpt_rows_lean", "ckpt_rows", "ckpt")
    assert R.FLAGS == {  # amendment A1 (before any run): row_tokens 8192 on the row variants
        "baseline": {"act_ckpt": False, "row_batch": 1, "row_tokens": 0, "rwkv7_wkv": "chunked"},
        "ckpt": {"act_ckpt": True, "row_batch": 1, "row_tokens": 0, "rwkv7_wkv": "chunked"},
        "ckpt_rows": {"act_ckpt": True, "row_batch": 8, "row_tokens": 8192, "rwkv7_wkv": "chunked"},
        "ckpt_rows_lean": {"act_ckpt": True, "row_batch": 8, "row_tokens": 8192, "rwkv7_wkv": "chunked_lean"},
    }
    assert (R.WARMUP_STEPS, R.TIMED_STEPS, R.N_STEPS, R.MIN_TIMED_COMPLETED) == (3, 20, 23, 15)
    assert R.MAX_PEAK_GB == 76.0 and R.LOSS_REL_MAX == 5e-3 and R.PROBE_STEP == 22
    assert R.LADDER == (1024, 1536, 2048, 3072, 4096) and R.LADDER_K == 10 and R.LADDER_CONT_LEN == 16
    assert R.CELL_T_LADDER == (512, 1024, 2048) and R.CELL_MINUTES == 190 and R.ROWS_PER_STEP == 8
    assert R.ALLOC_CONF == "expandable_segments:True" and R.RULE_ID == "cell_step_profile v1-A2"
    assert R.VERDICTS == ("ADOPT", "CAP_T", "NO_FIT", "INCONCLUSIVE")


def _rec(v, tok=300.0, oom=(), peak=60.0, probe=2.0, ladder_fit=4096, skipped=(), nonfinite=(), alloc=None,
         flags=None, tok_by_step=None):
    steps = []
    for i in range(R.N_STEPS):
        s = {"i": i, "timed": i >= R.WARMUP_STEPS, "tokens": 4000}
        if i in skipped:
            s["skipped"] = "deadline"
        elif i in oom:
            s.update(oom=True, peak_gb=79.0)
        else:
            t = (tok_by_step or {}).get(i, tok)
            loss = float("nan") if i in nonfinite else 1.5
            s.update(s=4000 / t, tok_s=t, loss=loss, finite=i not in nonfinite, peak_gb=peak)
        steps.append(s)
    ladder = {str(t): ({"ok": True, "peak_gb": 50.0} if t <= ladder_fit else {"ok": False, "oom": "x", "peak_gb": 79.0})
              for t in R.LADDER}
    pr = {"loss": probe, "finite": True} if probe is not None else {"error": "OutOfMemoryError"}
    return {"variant": v, "flags": R.FLAGS[v] if flags is None else flags, "complete": True, "steps": steps,
            "ladder": ladder, "alloc_conf": R.ALLOC_CONF if alloc is None else alloc, "probe": pr}


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
    assert d["variants"]["ckpt_rows_lean"]["loss"] == {"ok": True, "basis": "probe", "rel": 0.0}
    assert abs(d["speedup_vs_baseline"] - 1500 / 260) < 1e-9
    assert R.valid_decision(json.loads(json.dumps(d)))


def test_each_disqualifier():
    for bad in (dict(oom=(7,)), dict(oom=(1,)), dict(peak=76.01), dict(nonfinite=(5,)), dict(alloc=""),
                dict(skipped=tuple(range(3, 9))),  # 14 of 20 timed completed < 15
                dict(flags={"act_ckpt": True, "row_batch": 8, "row_tokens": 16384, "rwkv7_wkv": "chunked_lean"})):
        d = R.decide(_all(ckpt_rows_lean=_rec("ckpt_rows_lean", tok=1500.0, **bad)))
        assert d["chosen"] == "ckpt_rows", bad
    d = R.decide(_all(ckpt_rows_lean=_rec("ckpt_rows_lean", tok=1500.0, peak=76.0)))  # <= 76 is inside
    assert d["chosen"] == "ckpt_rows_lean"
    d = R.decide(_all(ckpt_rows_lean=None))
    assert d["chosen"] == "ckpt_rows" and d["variants"]["ckpt_rows_lean"]["reasons"] == ["missing"]


def test_deadline_skips_are_reported_not_disqualifying_down_to_15():
    rec = _rec("ckpt", tok=200.0, skipped=(18, 19, 20, 21, 22))  # 15 of 20 timed completed
    s = R.summarize(rec)
    assert s["qualifies_ex_loss"] and s["skipped_steps"] == [18, 19, 20, 21, 22] and s["completed_timed"] == 15
    killed = _rec("ckpt", tok=200.0)
    killed["steps"] = killed["steps"][:18]  # a timeout killed the process: unrecorded steps count as not run
    s = R.summarize(killed)
    assert s["qualifies_ex_loss"] and s["skipped_steps"] == [18, 19, 20, 21, 22]
    killed["steps"] = killed["steps"][:17]
    assert not R.summarize(killed)["qualifies_ex_loss"]


def test_rank_on_matched_rows_only():
    # lean is slower on the steps both completed but looks faster on steps only it completed (ckpt_rows skipped them)
    fast_tail = {i: 3000.0 for i in range(13, 23)}
    lean = _rec("ckpt_rows_lean", tok=900.0, tok_by_step=fast_tail)
    rows = _rec("ckpt_rows", tok=1000.0, skipped=tuple(range(18, 23)))
    d = R.decide(_all(ckpt_rows_lean=lean, ckpt_rows=rows, ckpt=_rec("ckpt", tok=200.0, skipped=tuple(range(18, 23)))))
    assert d["matched_steps"] == list(range(3, 18))
    assert d["variants"]["ckpt_rows_lean"]["median_tok_s"] > d["variants"]["ckpt_rows"]["median_tok_s"]
    assert d["chosen"] == "ckpt_rows"  # matched: 1000 vs median(900 x 10, 3000 x 5) = 900


def test_probe_loss_clause():
    for rel, ok in ((4.9e-3, True), (5.1e-3, False), (-4.9e-3, True)):
        d = R.decide(_all(ckpt_rows_lean=_rec("ckpt_rows_lean", tok=1500.0, probe=2.0 * (1 + rel))))
        assert (d["chosen"] == "ckpt_rows_lean") is ok, rel
        assert d["variants"]["ckpt_rows_lean"]["loss"]["basis"] == "probe"
    d = R.decide(_all(ckpt_rows_lean=_rec("ckpt_rows_lean", tok=1500.0, probe=None)))  # variant probe failed
    assert d["chosen"] == "ckpt_rows" and d["variants"]["ckpt_rows_lean"]["loss"]["ok"] is False
    # the baseline's probe failed: the clause is inconclusive for everyone -> INCONCLUSIVE, never a pass
    d = R.decide(_all(baseline=_rec("baseline", tok=260.0, oom=(0,), probe=None)))
    assert d["verdict"] == "INCONCLUSIVE" and d["chosen"] is None and R.valid_decision(d)
    assert all(d["variants"][v]["loss"]["basis"] == "inconclusive_baseline_probe" for v in R.VARIANTS)
    d = R.decide(_all(baseline=None))
    assert d["verdict"] == "INCONCLUSIVE"


def test_tie_breaks_on_peak_then_order():
    d = R.decide(_all(ckpt_rows=_rec("ckpt_rows", tok=1500.0, peak=50.0)))
    assert d["chosen"] == "ckpt_rows"  # same speed, lower peak
    d = R.decide(_all(ckpt_rows=_rec("ckpt_rows", tok=1500.0, peak=55.0)))
    assert d["chosen"] == "ckpt_rows_lean"  # full tie: earlier variant in the A2 order


def test_cap_t_when_nothing_qualifies_and_no_fit():
    fits = {"baseline": 1024, "ckpt_rows_lean": 1536, "ckpt_rows": 0, "ckpt": 2048}
    recs = {v: _rec(v, oom=(3,), ladder_fit=t) for v, t in fits.items()}
    d = R.decide(recs)
    assert d["verdict"] == "CAP_T" and d["chosen"] == "ckpt" and d["t_fit"] == 2048
    assert d["cell_config"]["t_ladder"] == [512, 1024, 2048] and d["cell_config"]["t_cap"] == 2048
    recs["ckpt"] = _rec("ckpt", oom=(3,), ladder_fit=1024)
    d = R.decide(recs)
    assert d["chosen"] == "ckpt_rows_lean" and d["t_fit"] == 1536 and d["cell_config"]["t_ladder"] == [512, 1024]
    gap = _rec("ckpt", oom=(3,), ladder_fit=4096)  # 1024 OOM, the rest fit -> nothing counts
    gap["ladder"]["1024"] = {"ok": False, "oom": "x"}
    assert R.summarize(gap)["t_fit"] is None
    d = R.decide({v: _rec(v, oom=(3,), ladder_fit=0) for v in R.VARIANTS})
    assert d["verdict"] == "NO_FIT" and d["chosen"] is None and R.valid_decision(d)


def test_adopt_with_small_ladder_keeps_note():
    d = R.decide(_all(ckpt_rows_lean=_rec("ckpt_rows_lean", tok=1500.0, ladder_fit=1536)))
    assert d["chosen"] == "ckpt_rows_lean" and d["cell_config"]["t_ladder"] == [512, 1024] and "note" in d


def test_valid_decision_rejects_malformed():
    assert not R.valid_decision({})
    assert not R.valid_decision({"rule": R.RULE_ID, "verdict": "MAYBE", "variants": {}, "chosen": None})
    d = R.decide(_all())
    d["chosen"] = None
    assert not R.valid_decision(d)
    d = R.decide(_all())
    d["rule"] = "cell_step_profile v1-A1"
    assert not R.valid_decision(d)


def test_job_params_cost_and_schedule():
    from pathlib import Path

    from jobs.common import costs

    root = Path(__file__).resolve().parents[1]
    p = costs.load("cell_step_profile")
    caps, exp = p["phase_caps_minutes"], p["expected_minutes"]
    assert set(caps) == {"setup", "download_wait", "profile", *R.VARIANTS} == set(exp)
    assert sum(caps.values()) + p["reserve_minutes"] == p["max_duration_minutes"]
    assert all(caps[k] >= 1.5 * exp[k] for k in caps), {k: caps[k] / exp[k] for k in caps}  # A2: >= 1.5x
    assert p["build_timeout_minutes"] == 60 and p["max_restarts"] == 0 and p["assumed_restart_floor"] >= 1
    assert p["chip"] == "h100" and costs.cap_ok(p) and costs.schedule_ok(p)
    spec = (root / "jobs/cell_step_profile/JOB_SPEC.md").read_text(encoding="utf-8")
    assert f"${costs.worst_case_usd(p):.2f}" in spec and f"${costs.expected_usd(p):.2f}" in spec
    sh = (root / "jobs/cell_step_profile/cell_step_job.sh").read_text(encoding="utf-8")
    assert "for V in " + " ".join(R.VARIANTS) + "; do" in sh
    assert f"PYTORCH_CUDA_ALLOC_CONF={R.ALLOC_CONF}" in sh
