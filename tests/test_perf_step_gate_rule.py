"""jobs/perf_step_gate pre-registered rule (rule.py) and rate guard (preflight.py): constants pinned, decide() on
synthetic records. CPU.

    python -X utf8 -m pytest -q tests/test_perf_step_gate_rule.py
"""
from __future__ import annotations

import copy

from jobs.perf_step_gate import preflight as P
from jobs.perf_step_gate import rule as R


def test_constants_are_the_pre_registered_ones():
    assert R.VARIANTS == ("greedy", "min_cost", "min_cost_single", "min_cost_single_compile")
    assert R.BASELINE == "greedy"
    assert R.WARMUP_STEPS == 3 and R.TIMED_STEPS == 20 and R.N_STEPS == 23 and R.ROWS_PER_STEP == 8
    assert R.MIN_TIMED_COMPLETED == 15 and R.MAX_PEAK_GB == 76.0
    assert R.GRAD_STEP == 6 and R.GRAD_B_SEED == 20261008 and R.GRAD_B_SCALE == 1e-3
    assert R.LOSS_REL_MAX == 5e-3 and R.GRAD_COS_MIN == 0.999
    assert R.LADDER == (1024, 2048) and R.LADDER_K == 10 and R.LADDER_CONT_LEN == 16
    assert R.CELL_MINUTES == 190 and R.ALLOC_CONF == "expandable_segments:True"
    assert R.RULE_ID == "perf_step_gate v1" and R.VERDICTS == ("ADOPT", "INCONCLUSIVE")
    lean = {"act_ckpt": True, "row_batch": 8, "row_tokens": 8192, "rwkv7_wkv": "chunked_lean"}
    assert R.FLAGS["greedy"] == {**lean, "row_plan": "greedy", "cont_plan": "per_group", "compile_blocks": False}
    assert R.FLAGS["min_cost_single_compile"] == {**lean, "row_plan": "min_cost", "cont_plan": "single",
                                                  "compile_blocks": True}
    assert P.TVUAT_MEDIAN_STEP_S == 8.2475 and P.SLOW_FACTOR == 1.5


def test_grad_step_has_two_groups_under_every_plan():
    """mix step 6 is where min_cost and single-cont differ from greedy (the gradient check is not vacuous)."""
    from jobs.fla_rwkv7 import train_mix
    from train.trainer import row_groups, row_groups_min_cost

    rows = train_mix.load()["rows"][R.GRAD_STEP * 8:(R.GRAD_STEP + 1) * 8]
    L = [r["T"] for r in rows]
    g, m = row_groups(L, 8, 8192), row_groups_min_cost(L, 8, 8192)
    assert len(g) == len(m) == 2 and g != m


def _rec(v, s=8.0, peak=50.0, loss=1.5, **kw):
    steps = [{"i": i, "timed": i >= R.WARMUP_STEPS, "s": s, "peak_gb": peak, "loss": 2.0, "finite": True}
             for i in range(R.N_STEPS)]
    r = {"variant": v, "flags": dict(R.FLAGS[v]), "alloc_conf": R.ALLOC_CONF, "complete": True, "steps": steps,
         "grad": {"loss": loss, "finite": True},
         "ladder": {str(t): {"ok": True, "peak_gb": peak, "s": 10.0} for t in R.LADDER}}
    r.update(kw)
    return r


def _all(**over):
    recs = {"greedy": _rec("greedy", 8.0), "min_cost": _rec("min_cost", 7.5), "min_cost_single": _rec("min_cost_single", 7.0),
            "min_cost_single_compile": _rec("min_cost_single_compile", 5.0)}
    recs.update(over)
    return recs


COS = {"min_cost": 0.99999, "min_cost_single": 0.99998, "min_cost_single_compile": 0.9995}


def test_fastest_qualifier_is_adopted():
    d = R.decide(_all(), COS)
    assert d["verdict"] == "ADOPT" and d["chosen"] == "min_cost_single_compile" and R.valid_decision(d)
    assert abs(d["speedup_vs_baseline"] - 8.0 / 5.0) < 1e-9
    assert d["variants"]["greedy"]["rows_190min"] == (190 * 60 // 8) * 8


def test_numerics_clauses():
    d = R.decide(_all(), {**COS, "min_cost_single_compile": 0.998})
    assert d["chosen"] == "min_cost_single" and not d["variants"]["min_cost_single_compile"]["qualifies"]
    recs = _all(min_cost_single_compile=_rec("min_cost_single_compile", 5.0, loss=1.5 * (1 + 6e-3)))
    assert R.decide(recs, COS)["chosen"] == "min_cost_single"
    recs = _all(min_cost_single_compile=_rec("min_cost_single_compile", 5.0, loss=1.5 * (1 + 4e-3)))
    assert R.decide(recs, COS)["chosen"] == "min_cost_single_compile"
    assert R.decide(_all(), {**COS, "min_cost_single_compile": None})["chosen"] == "min_cost_single"


def test_memory_and_ladder_clauses():
    bad = _rec("min_cost_single", 7.0)
    bad["ladder"]["2048"] = {"ok": False, "oom": True, "peak_gb": 79.0}
    d = R.decide(_all(min_cost_single=bad, min_cost_single_compile=None), COS)
    assert d["chosen"] == "min_cost" and not d["variants"]["min_cost_single"]["qualifies"]
    assert not d["variants"]["min_cost_single_compile"]["present"]
    hot = _rec("min_cost", 7.5, peak=77.0)
    assert not R.decide(_all(min_cost=hot), COS)["variants"]["min_cost"]["qualifies"]


def test_baseline_unusable_is_inconclusive_and_too_few_steps_disqualify():
    g = _rec("greedy")
    g["grad"] = {"error": "OutOfMemoryError"}
    d = R.decide(_all(greedy=g), COS)
    assert d["verdict"] == "INCONCLUSIVE" and d["chosen"] is None and R.valid_decision(d)
    few = _rec("min_cost", 1.0)
    for s in few["steps"][R.WARMUP_STEPS:R.WARMUP_STEPS + 6]:
        s.clear()
        s.update(skipped="deadline")
    few["steps"] = [dict(s, i=i) for i, s in enumerate(few["steps"])]
    assert not R.decide(_all(min_cost=few), COS)["variants"]["min_cost"]["qualifies"]
    wrong = copy.deepcopy(_rec("min_cost", 1.0))
    wrong["flags"]["row_plan"] = "greedy"
    assert not R.decide(_all(min_cost=wrong), COS)["variants"]["min_cost"]["qualifies"]


def test_rate_guard():
    rec = _rec("greedy", 8.3)
    rec["env"] = {"dryrun": False}
    assert P.check(rec)["ok"]
    rec = _rec("greedy", 1.5 * 8.2475 + 0.01)
    rec["env"] = {"dryrun": False}
    assert not P.check(rec)["ok"]
    assert P.check({"steps": [], "env": {"dryrun": False}})["ok"] is False
