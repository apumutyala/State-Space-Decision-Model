"""jobs/perf_step_gate pre-registered rule (rule.py) and rate guard (preflight.py): constants pinned, decide() on
synthetic records. CPU.

    python -X utf8 -m pytest -q tests/test_perf_step_gate_rule.py
"""
from __future__ import annotations

import copy

from jobs.perf_step_gate import preflight as P
from jobs.perf_step_gate import rule as R


def test_constants_are_the_pre_registered_ones():
    assert R.VARIANTS == ("greedy", "fast", "fast_fla", "fast_fla_nocompile")
    assert R.BASELINE == "greedy" and R.SPEED_BASELINE == "fast" and R.FLA_VARIANTS == ("fast_fla", "fast_fla_nocompile")
    assert R.WARMUP_STEPS == 3 and R.TIMED_STEPS == 20 and R.N_STEPS == 23 and R.ROWS_PER_STEP == 8
    assert R.MIN_TIMED_COMPLETED == 15 and R.MAX_PEAK_GB == 76.0
    assert R.GRAD_STEP == 6 and R.GRAD_B_SEED == 20261008 and R.GRAD_B_SCALE == 1e-3
    assert R.LOSS_REL_MAX == 5e-3 and R.GRAD_COS_MIN == 0.995
    assert R.LADDER == (1024, 2048) and R.LADDER_K == 10 and R.LADDER_CONT_LEN == 16
    assert R.CELL_MINUTES == 190 and R.ALLOC_CONF == "expandable_segments:True"
    assert R.RULE_ID == "perf_step_gate v2" and R.VERDICTS == ("ADOPT", "INCONCLUSIVE")
    lean = {"act_ckpt": True, "row_batch": 8, "row_tokens": 8192, "rwkv7_wkv": "chunked_lean"}
    assert R.FLAGS["greedy"] == {**lean, "row_plan": "greedy", "cont_plan": "per_group", "compile_blocks": False}
    fast = {**lean, "row_plan": "min_cost", "cont_plan": "single", "compile_blocks": True}  # = d1_fast's flags
    assert R.FLAGS["fast"] == fast
    assert R.FLAGS["fast_fla"] == {**fast, "rwkv7_wkv": "fla"}
    assert R.FLAGS["fast_fla_nocompile"] == {**fast, "rwkv7_wkv": "fla", "compile_blocks": False}
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
         "grad": {"loss": loss, "finite": True}, "env": {"dryrun": False},
         "fla": {"calls": {"fla": 61 * 23} if v in R.FLA_VARIANTS else {}, "fallbacks": {}},
         "ladder": {str(t): {"ok": True, "peak_gb": peak, "s": 10.0} for t in R.LADDER}}
    r.update(kw)
    return r


def _all(**over):
    recs = {"greedy": _rec("greedy", 8.0), "fast": _rec("fast", 5.6), "fast_fla": _rec("fast_fla", 4.0),
            "fast_fla_nocompile": _rec("fast_fla_nocompile", 5.0)}
    recs.update(over)
    return recs


COS = {"fast": 0.9977, "fast_fla": 0.9960, "fast_fla_nocompile": 0.9965}  # fast: job-2amiy's measured compile cosine


def test_fastest_qualifier_is_adopted():
    d = R.decide(_all(), COS)
    assert d["verdict"] == "ADOPT" and d["chosen"] == "fast_fla" and R.valid_decision(d)
    assert abs(d["speedup_vs_baseline"] - 8.0 / 4.0) < 1e-9 and abs(d["speedup_vs_fast"] - 5.6 / 4.0) < 1e-9
    assert d["fla_speeds_training"] is True
    assert d["variants"]["greedy"]["rows_190min"] == (190 * 60 // 8) * 8
    assert d["variants"]["fast_fla"]["rows_190min"] == (190 * 60 // 4) * 8
    assert abs(d["variants"]["fast"]["speedup_vs_fast"] - 1.0) < 1e-12


def test_fla_slower_than_fast_is_reported():
    recs = _all(fast_fla=_rec("fast_fla", 6.0), fast_fla_nocompile=_rec("fast_fla_nocompile", 7.0))
    d = R.decide(recs, COS)
    assert d["chosen"] == "fast" and d["fla_speeds_training"] is False


def test_numerics_clauses():
    d = R.decide(_all(), {**COS, "fast_fla": 0.994})
    assert d["chosen"] == "fast_fla_nocompile" and not d["variants"]["fast_fla"]["qualifies"]
    assert R.decide(_all(), {**COS, "fast_fla": 0.995})["chosen"] == "fast_fla"  # the bar is inclusive
    recs = _all(fast_fla=_rec("fast_fla", 4.0, loss=1.5 * (1 + 6e-3)))
    assert R.decide(recs, COS)["chosen"] == "fast_fla_nocompile"
    recs = _all(fast_fla=_rec("fast_fla", 4.0, loss=1.5 * (1 + 4e-3)))
    assert R.decide(recs, COS)["chosen"] == "fast_fla"
    assert R.decide(_all(), {**COS, "fast_fla": None})["chosen"] == "fast_fla_nocompile"


def test_fla_must_really_run():
    """A fallback to "chunked" would time chunked under the fla name; no fla call at all likewise. Dry run exempt."""
    fb = _rec("fast_fla", 4.0)
    fb["fla"] = {"calls": {"fla": 100}, "fallbacks": {"packed_batch": 1}}
    d = R.decide(_all(fast_fla=fb), COS)
    assert not d["variants"]["fast_fla"]["qualifies"] and d["chosen"] == "fast_fla_nocompile"
    none = _rec("fast_fla", 4.0)
    none["fla"] = {"calls": {}, "fallbacks": {}}
    assert not R.decide(_all(fast_fla=none), COS)["variants"]["fast_fla"]["qualifies"]
    missing = _rec("fast_fla", 4.0)
    del missing["fla"]
    assert not R.decide(_all(fast_fla=missing), COS)["variants"]["fast_fla"]["qualifies"]
    dry = _rec("fast_fla", 4.0, env={"dryrun": True})
    dry["fla"] = {"calls": {}, "fallbacks": {"not_cuda": 9}}
    assert R.decide(_all(fast_fla=dry), COS)["variants"]["fast_fla"]["qualifies"]
    assert not d["variants"]["fast"].get("reasons")  # the chunked_lean variants are never asked for fla calls


def test_memory_and_ladder_clauses():
    bad = _rec("fast_fla", 4.0)
    bad["ladder"]["2048"] = {"ok": False, "oom": True, "peak_gb": 79.0}
    d = R.decide(_all(fast_fla=bad, fast_fla_nocompile=None), COS)
    assert d["chosen"] == "fast" and not d["variants"]["fast_fla"]["qualifies"]
    assert not d["variants"]["fast_fla_nocompile"]["present"] and d["fla_speeds_training"] is False
    hot = _rec("fast_fla", 4.0, peak=77.0)
    assert not R.decide(_all(fast_fla=hot), COS)["variants"]["fast_fla"]["qualifies"]
    assert R.decide(_all(fast_fla=_rec("fast_fla", 4.0, peak=76.0)), COS)["variants"]["fast_fla"]["qualifies"]


def test_baseline_unusable_is_inconclusive_and_too_few_steps_disqualify():
    g = _rec("greedy")
    g["grad"] = {"error": "OutOfMemoryError"}
    d = R.decide(_all(greedy=g), COS)
    assert d["verdict"] == "INCONCLUSIVE" and d["chosen"] is None and R.valid_decision(d)
    few = _rec("fast_fla", 1.0)
    for s in few["steps"][R.WARMUP_STEPS:R.WARMUP_STEPS + 6]:
        s.clear()
        s.update(skipped="deadline")
    few["steps"] = [dict(s, i=i) for i, s in enumerate(few["steps"])]
    assert not R.decide(_all(fast_fla=few), COS)["variants"]["fast_fla"]["qualifies"]
    wrong = copy.deepcopy(_rec("fast_fla", 1.0))
    wrong["flags"]["rwkv7_wkv"] = "chunked_lean"
    assert not R.decide(_all(fast_fla=wrong), COS)["variants"]["fast_fla"]["qualifies"]
    nofast = _rec("fast", 5.6)
    nofast["grad"] = {"error": "boom"}
    d = R.decide(_all(fast=nofast), COS)
    assert d["verdict"] == "ADOPT" and d["chosen"] == "fast_fla" and d["fla_speeds_training"] is None


def test_rate_guard():
    rec = _rec("greedy", 8.3)
    rec["env"] = {"dryrun": False}
    assert P.check(rec)["ok"]
    rec = _rec("greedy", 1.5 * 8.2475 + 0.01)
    rec["env"] = {"dryrun": False}
    assert not P.check(rec)["ok"]
    assert P.check({"steps": [], "env": {"dryrun": False}})["ok"] is False
