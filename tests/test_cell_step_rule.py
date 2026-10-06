"""jobs/cell_step_profile PRE-REGISTERED decision rule (rule.py). Pins the constants and every verdict branch.

    python -X utf8 -m pytest -q tests/test_cell_step_rule.py
"""
from __future__ import annotations

import json

from jobs.cell_step_profile import rule as R


def test_preregistered_constants_are_pinned():
    assert R.VARIANTS == ("baseline", "ckpt", "ckpt_rows8", "ckpt_rows8_lean")
    assert R.FLAGS == {
        "baseline": {"act_ckpt": False, "row_batch": 1, "rwkv7_wkv": "chunked"},
        "ckpt": {"act_ckpt": True, "row_batch": 1, "rwkv7_wkv": "chunked"},
        "ckpt_rows8": {"act_ckpt": True, "row_batch": 8, "rwkv7_wkv": "chunked"},
        "ckpt_rows8_lean": {"act_ckpt": True, "row_batch": 8, "rwkv7_wkv": "chunked_lean"},
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
            "ckpt_rows8": _rec("ckpt_rows8", tok=1000.0, peak=55.0),
            "ckpt_rows8_lean": _rec("ckpt_rows8_lean", tok=1500.0, peak=55.0)}
    recs.update(over)
    return recs


def test_adopt_fastest_qualifying_variant_and_projection():
    d = R.decide(_all())
    assert d["verdict"] == "ADOPT" and d["chosen"] == "ckpt_rows8_lean"
    assert d["cell_config"] == {"act_ckpt": True, "row_batch": 8, "rwkv7_wkv": "chunked_lean", "t_ladder": [512, 1024, 2048]}
    p = d["projection"]
    assert p["median_step_s"] == 4000 / 1500.0 and p["steps_190min"] == int(190 * 60 / (4000 / 1500.0))
    assert p["rows_190min"] == 8 * p["steps_190min"]
    assert d["variants"]["baseline"]["qualifies"] is False  # OOM + peak
    assert d["variants"]["ckpt_rows8_lean"]["loss"]["basis"] == "cpu_equivalence_tests"  # baseline step 0 OOMed
    assert abs(d["speedup_vs_baseline"] - 1500 / 260) < 1e-9
    assert R.valid_decision(json.loads(json.dumps(d)))


def test_each_disqualifier():
    for bad in (dict(oom=(7,)), dict(peak=76.01), dict(skipped=(22,)), dict(nonfinite=(5,)), dict(alloc=""),
                dict(flags={"act_ckpt": True, "row_batch": 4, "rwkv7_wkv": "chunked_lean"})):
        d = R.decide(_all(ckpt_rows8_lean=_rec("ckpt_rows8_lean", tok=1500.0, **bad)))
        assert d["chosen"] == "ckpt_rows8", bad
    d = R.decide(_all(ckpt_rows8_lean=_rec("ckpt_rows8_lean", tok=1500.0, peak=76.0)))  # <= 76 is inside
    assert d["chosen"] == "ckpt_rows8_lean"
    d = R.decide(_all(ckpt_rows8_lean=None))
    assert d["chosen"] == "ckpt_rows8" and d["variants"]["ckpt_rows8_lean"]["reasons"] == ["missing"]
    short = _rec("ckpt_rows8_lean", tok=1500.0)
    short["steps"] = short["steps"][:20]
    assert R.decide(_all(ckpt_rows8_lean=short))["chosen"] == "ckpt_rows8"


def test_loss_clause_against_completed_baseline_step0():
    base = _rec("baseline", tok=260.0, oom=(5,), peak=70.0, loss0=2.0)
    d = R.decide(_all(baseline=base, ckpt_rows8_lean=_rec("ckpt_rows8_lean", tok=1500.0, loss0=2.0 * (1 + 1.1e-3))))
    assert d["chosen"] == "ckpt_rows8" and d["variants"]["ckpt_rows8_lean"]["loss"]["basis"] == "baseline_step0"
    d = R.decide(_all(baseline=base, ckpt_rows8_lean=_rec("ckpt_rows8_lean", tok=1500.0, loss0=2.0 * (1 + 0.9e-3))))
    assert d["chosen"] == "ckpt_rows8_lean"
    v0 = _rec("ckpt_rows8_lean", tok=1500.0)
    v0["steps"][0] = {"i": 0, "timed": False, "tokens": 4000, "oom": True, "peak_gb": 79.0}
    d = R.decide(_all(baseline=base, ckpt_rows8_lean=v0))
    assert d["variants"]["ckpt_rows8_lean"]["loss"]["ok"] is False


def test_tie_breaks_on_peak_then_order():
    d = R.decide(_all(ckpt_rows8=_rec("ckpt_rows8", tok=1500.0, peak=50.0)))
    assert d["chosen"] == "ckpt_rows8"  # same speed, lower peak
    d = R.decide(_all(ckpt_rows8=_rec("ckpt_rows8", tok=1500.0, peak=55.0)))
    assert d["chosen"] == "ckpt_rows8"  # full tie: earlier variant


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
    d = R.decide(_all(ckpt_rows8_lean=_rec("ckpt_rows8_lean", tok=1500.0, ladder_fit=1536)))
    assert d["chosen"] == "ckpt_rows8_lean" and d["cell_config"]["t_ladder"] == [512, 1024] and "note" in d


def test_valid_decision_rejects_malformed():
    assert not R.valid_decision({})
    assert not R.valid_decision({"rule": "cell_step_profile v1", "verdict": "MAYBE", "variants": {}, "chosen": None})
    d = R.decide(_all())
    d["chosen"] = None
    assert not R.valid_decision(d)
