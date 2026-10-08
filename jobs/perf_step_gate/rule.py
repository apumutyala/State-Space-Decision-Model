"""PRE-REGISTERED decision rule of jobs/perf_step_gate (JOB_SPEC.md). Pure; CPU-tested (tests/test_perf_step_gate.py pins
every constant). Do not edit after seeing GPU numbers.

Input: one record per variant (``variant_<name>.json`` from ``jobs.perf_step_gate.gate``):
    {"variant", "flags", "alloc_conf", "complete",
     "grad": {"loss": float, "finite": bool, "file": str} | {"error": str},   # one un-applied step on GRAD_STEP's rows
     "steps": [{"i", "timed", "s", "peak_gb", "loss", "finite", "oom", "skipped"}, ...],
     "ladder": {"2048": {"ok", "oom", "peak_gb", "s"}, ...}}
plus, per non-baseline variant, ``grad_cos`` = cosine(its LoRA gradient, the baseline's) computed by the decide phase.
Output (``decision.json``): verdict ADOPT (the fastest qualifying variant, which may be the baseline itself) |
INCONCLUSIVE (the baseline's own record is unusable), with s/step for every variant.
"""
from __future__ import annotations

import math
import statistics

# ----------------------------------------------------------------------------- PRE-REGISTERED (do not tune on GPU data)
#: run order; each in its own process. "greedy" = job-tvuat's ADOPT config (ckpt_rows_lean), the baseline.
VARIANTS = ("greedy", "min_cost", "min_cost_single", "min_cost_single_compile")
BASELINE = "greedy"
_LEAN = {"act_ckpt": True, "row_batch": 8, "row_tokens": 8192, "rwkv7_wkv": "chunked_lean"}
FLAGS = {
    "greedy": {**_LEAN, "row_plan": "greedy", "cont_plan": "per_group", "compile_blocks": False},
    "min_cost": {**_LEAN, "row_plan": "min_cost", "cont_plan": "per_group", "compile_blocks": False},
    "min_cost_single": {**_LEAN, "row_plan": "min_cost", "cont_plan": "single", "compile_blocks": False},
    "min_cost_single_compile": {**_LEAN, "row_plan": "min_cost", "cont_plan": "single", "compile_blocks": True},
}
#: compile warm-up: the compiled variant compiles on its first shapes; every variant gets the same warm-up steps
WARMUP_STEPS = 3
TIMED_STEPS = 20
N_STEPS = WARMUP_STEPS + TIMED_STEPS  # the 23 optimiser steps of jobs/fla_rwkv7/train_mix_v2.json (8 rows each)
ROWS_PER_STEP = 8
MIN_TIMED_COMPLETED = 15
MAX_PEAK_GB = 76.0  # the cell's pass_rule.max_peak_gb
#: numerics: one optimiser step's loss and LoRA gradient (not applied) on the rows of mix step GRAD_STEP (two row groups
#: under every plan: max_T 2025, so min_cost and single-cont really differ from greedy there), with every lora_B set
#: to GRAD_B_SCALE x N(0, 1) from GRAD_B_SEED first (LoRA B starts at zero, which would leave every lora_A gradient 0)
GRAD_STEP = 6
GRAD_B_SEED = 20261008
GRAD_B_SCALE = 1e-3
LOSS_REL_MAX = 5e-3
GRAD_COS_MIN = 0.999
LADDER = (1024, 2048)  # synthetic rows, K = 10 (continuations of 16 tokens), row_batch copies per step
LADDER_K = 10
LADDER_CONT_LEN = 16
CELL_MINUTES = 190
ALLOC_CONF = "expandable_segments:True"
RULE_ID = "perf_step_gate v1"
VERDICTS = ("ADOPT", "INCONCLUSIVE")


def summarize(rec: dict | None) -> dict:
    if not rec:
        return {"present": False, "reasons": ["missing"], "timed_completed": [], "median_step_s": None,
                "grad_loss": None, "peak_gb": None}
    steps = rec.get("steps") or []
    reasons = []
    oom = [s["i"] for s in steps if s.get("oom")]
    ran = [s for s in steps if not s.get("oom") and not s.get("skipped")]
    nonfinite = [s["i"] for s in ran
                 if not (s.get("finite") and isinstance(s.get("loss"), (int, float)) and math.isfinite(s["loss"]))]
    timed = [s for s in ran if s.get("timed") and s["i"] not in nonfinite and isinstance(s.get("s"), (int, float))]
    lad = rec.get("ladder") or {}
    peak = max([float(s.get("peak_gb") or 0.0) for s in steps] + [float(r.get("peak_gb") or 0.0) for r in lad.values()]
               + [0.0])
    if oom:
        reasons.append(f"oom steps {oom}")
    if nonfinite:
        reasons.append(f"non-finite loss steps {nonfinite}")
    if peak > MAX_PEAK_GB:
        reasons.append(f"peak {peak:.2f} GB > {MAX_PEAK_GB}")
    bad_rungs = [t for t in LADDER if not (lad.get(str(t)) or {}).get("ok")]
    if bad_rungs:
        reasons.append(f"ladder rungs not ok: {bad_rungs}")
    if len(timed) < MIN_TIMED_COMPLETED:
        reasons.append(f"{len(timed)} timed steps completed < {MIN_TIMED_COMPLETED}")
    if rec.get("alloc_conf") != ALLOC_CONF:
        reasons.append(f"PYTORCH_CUDA_ALLOC_CONF {rec.get('alloc_conf')!r} != {ALLOC_CONF!r}")
    if rec.get("flags") != FLAGS.get(rec.get("variant")):
        reasons.append(f"flags {rec.get('flags')} != pre-registered {FLAGS.get(rec.get('variant'))}")
    g = rec.get("grad") or {}
    gl = g.get("loss") if g.get("finite") and isinstance(g.get("loss"), (int, float)) else None
    if gl is None:
        reasons.append(f"no finite gradient-step loss ({g.get('error')})")
    return {"present": True, "reasons": reasons, "oom_steps": oom, "peak_gb": peak, "completed_timed": len(timed),
            "timed_completed": sorted(s["i"] for s in timed), "_timed": {s["i"]: s["s"] for s in timed},
            "median_step_s": statistics.median([s["s"] for s in timed]) if timed else None,
            "warmup_s": sum(float(s.get("s") or 0.0) for s in steps if not s.get("timed")),
            "grad_loss": gl}


def decide(records: dict[str, dict | None], grad_cos: dict[str, float | None]) -> dict:
    """A variant qualifies with: zero OOM, finite losses, every step / ladder peak <= MAX_PEAK_GB, every ladder rung ok,
    >= MIN_TIMED_COMPLETED timed steps, its pre-registered flags and allocator, |grad loss - baseline's| / |baseline's|
    <= LOSS_REL_MAX and grad cosine vs the baseline >= GRAD_COS_MIN (the baseline itself: by definition). Rank by the
    median step seconds over the timed steps every qualifier completed (matched); ties: the earlier variant."""
    summ = {v: summarize(records.get(v)) for v in VARIANTS}
    base = summ[BASELINE]
    out = {"rule": RULE_ID, "variants": summ, "max_peak_gb": MAX_PEAK_GB, "loss_rel_max": LOSS_REL_MAX,
           "grad_cos_min": GRAD_COS_MIN}
    if not base["present"] or base["reasons"]:
        for v in VARIANTS:
            summ[v].pop("_timed", None)
        out.update(verdict="INCONCLUSIVE", chosen=None, matched_steps=[],
                   note=f"baseline {BASELINE} unusable: {base['reasons']}")
        return out
    for v in VARIANTS:
        s = summ[v]
        if v == BASELINE:
            s["loss_rel"], s["grad_cos"], s["numerics_ok"] = 0.0, 1.0, True
            continue
        cos = grad_cos.get(v)
        rel = (abs(s["grad_loss"] - base["grad_loss"]) / max(abs(base["grad_loss"]), 1e-12)
               if s.get("grad_loss") is not None else None)
        s["loss_rel"], s["grad_cos"] = rel, cos
        s["numerics_ok"] = rel is not None and rel <= LOSS_REL_MAX and cos is not None and cos >= GRAD_COS_MIN
        if not s["numerics_ok"]:
            s["reasons"] = s.get("reasons", []) + [f"numerics: loss_rel {rel}, grad_cos {cos}"]
    for v in VARIANTS:
        summ[v]["qualifies"] = summ[v]["present"] and not summ[v]["reasons"]
    ok = [v for v in VARIANTS if summ[v]["qualifies"]]
    matched = sorted(set.intersection(*(set(summ[v]["timed_completed"]) for v in ok)))
    for v in VARIANTS:
        t = summ[v].pop("_timed", {})
        if matched and v in ok:
            summ[v]["matched_median_step_s"] = statistics.median([t[i] for i in matched])
        else:
            summ[v]["matched_median_step_s"] = summ[v].get("median_step_s")
    b = summ[BASELINE]["matched_median_step_s"]
    for v in VARIANTS:
        m = summ[v]["matched_median_step_s"]
        summ[v]["speedup_vs_baseline"] = (b / m) if (b and m) else None
        summ[v]["rows_190min"] = math.floor(CELL_MINUTES * 60 / m) * ROWS_PER_STEP if m else None
    best = sorted(ok, key=lambda v: (summ[v]["matched_median_step_s"], VARIANTS.index(v)))[0]
    out.update(verdict="ADOPT", chosen=best, matched_steps=matched, cell_config=dict(FLAGS[best]),
               speedup_vs_baseline=summ[best]["speedup_vs_baseline"])
    return out


def valid_decision(d) -> bool:
    """finalize.py validator."""
    try:
        return (d.get("rule") == RULE_ID and d.get("verdict") in VERDICTS and set(d["variants"]) == set(VARIANTS)
                and (d["verdict"] == "INCONCLUSIVE") == (d.get("chosen") is None))
    except Exception:  # noqa: BLE001
        return False
