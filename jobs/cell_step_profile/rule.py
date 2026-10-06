"""PRE-REGISTERED decision rule of jobs/cell_step_profile (JOB_SPEC.md "Decision rule"). Pure; CPU-tested
(tests/test_cell_step_rule.py pins every constant). Committed alone, before the harness and before any run; amended
before any run only (A1: row-token budget; A2: after independent review, see JOB_SPEC.md). Do not edit after seeing
GPU numbers.

Input: one record per variant (``variant_<name>.json`` written by ``jobs.cell_step_profile.cell_step``):
    {"variant": name, "flags": {...}, "alloc_conf": str, "complete": bool,
     "probe": {"loss": float, "finite": bool} | {"error": str},          # A2: forward loss on mix step 22, no grad
     "steps": [{"i", "timed", "tokens", "s", "tok_s", "peak_gb", "loss", "finite", "oom", "skipped"}, ...],   # 23
     "ladder": {"1024": {"ok", "oom", "peak_gb", "s"}, ...}}
Output (``decision.json``): verdict ADOPT (a variant qualifies) | CAP_T (none qualifies; the cell caps T) | NO_FIT
(nothing fits even the smallest rung) | INCONCLUSIVE (the baseline's probe loss is missing or not finite), the chosen
variant, the recommended cell config and the projection.
"""
from __future__ import annotations

import math
import statistics

# ----------------------------------------------------------------------------- PRE-REGISTERED (do not tune on GPU data)
#: execution order (A2: fastest expected first, the memory-only fallback last); each runs in its own process.
#: A1 (before any run): the row variants carry row_tokens = 8192 (JOB_SPEC.md)
VARIANTS = ("baseline", "ckpt_rows_lean", "ckpt_rows", "ckpt")
FLAGS = {
    "baseline": {"act_ckpt": False, "row_batch": 1, "row_tokens": 0, "rwkv7_wkv": "chunked"},
    "ckpt": {"act_ckpt": True, "row_batch": 1, "row_tokens": 0, "rwkv7_wkv": "chunked"},
    "ckpt_rows": {"act_ckpt": True, "row_batch": 8, "row_tokens": 8192, "rwkv7_wkv": "chunked"},
    "ckpt_rows_lean": {"act_ckpt": True, "row_batch": 8, "row_tokens": 8192, "rwkv7_wkv": "chunked_lean"},
}
WARMUP_STEPS = 3
TIMED_STEPS = 20
N_STEPS = WARMUP_STEPS + TIMED_STEPS  # the 23 optimiser steps of jobs/fla_rwkv7/train_mix_v2.json (8 rows each)
MIN_TIMED_COMPLETED = 15  # A2: >= 15 of the 20 timed steps completed (deadline skips never disqualify beyond this)
MAX_PEAK_GB = 76.0  # the cell's pass_rule.max_peak_gb
#: A2: GPU loss clause. Forward loss (no grad, eval mode, same seed) on the rows of mix step PROBE_STEP (max_T 772, the
#: baseline fits it), computed in every variant process before training; |variant - baseline| / |baseline| <= 5e-3
PROBE_STEP = 22
LOSS_REL_MAX = 5e-3
LADDER = (1024, 1536, 2048, 3072, 4096)  # synthetic rows, K = 10 (continuations of 16 tokens)
LADDER_K = 10
LADDER_CONT_LEN = 16
CELL_T_LADDER = (512, 1024, 2048)  # the cell's pre-registered t_ladder (m2_first_cell_rwkv7.json)
CELL_MINUTES = 190  # the cell's max_minutes
ROWS_PER_STEP = 8  # batch_size 1 x grad_accum 8
ALLOC_CONF = "expandable_segments:True"  # PYTORCH_CUDA_ALLOC_CONF in every variant (recorded per variant)
RULE_ID = "cell_step_profile v1-A2"


def _step_peak(s: dict) -> float:
    return float(s.get("peak_gb") or 0.0)


def _probe_loss(rec: dict) -> float | None:
    p = rec.get("probe") or {}
    v = p.get("loss")
    return float(v) if p.get("finite") and isinstance(v, (int, float)) and math.isfinite(v) else None


def summarize(rec: dict | None) -> dict:
    """Facts about one variant's record; ``qualifies_ex_loss`` without the loss clause (that needs the baseline)."""
    if not rec:
        return {"present": False, "qualifies_ex_loss": False, "reasons": ["missing"], "timed_completed": [],
                "median_tok_s": None, "median_step_s": None, "probe_loss": None, "t_fit": None, "peak_gb": None}
    steps = rec.get("steps") or []
    reasons = []
    oom = [s["i"] for s in steps if s.get("oom")]
    ran = [s for s in steps if not s.get("oom") and not s.get("skipped")]
    nonfinite = [s["i"] for s in ran
                 if not (s.get("finite") and isinstance(s.get("loss"), (int, float)) and math.isfinite(s["loss"]))]
    # deadline-skipped steps, and steps a killed process never recorded: reported, not disqualifying by themselves
    skipped = sorted({s["i"] for s in steps if s.get("skipped")} | (set(range(N_STEPS)) - {s["i"] for s in steps}))
    peak = max((_step_peak(s) for s in steps), default=0.0)
    timed = [s for s in ran if s.get("timed") and s["i"] not in nonfinite and isinstance(s.get("tok_s"), (int, float))]
    if oom:
        reasons.append(f"oom steps {oom}")
    if nonfinite:
        reasons.append(f"non-finite loss steps {nonfinite}")
    if peak > MAX_PEAK_GB:
        reasons.append(f"peak {peak:.2f} GB > {MAX_PEAK_GB}")
    if len(timed) < MIN_TIMED_COMPLETED:
        reasons.append(f"{len(timed)} timed steps completed < {MIN_TIMED_COMPLETED}")
    if rec.get("alloc_conf") != ALLOC_CONF:
        reasons.append(f"PYTORCH_CUDA_ALLOC_CONF {rec.get('alloc_conf')!r} != {ALLOC_CONF!r}")
    if rec.get("flags") != FLAGS.get(rec.get("variant")):
        reasons.append(f"flags {rec.get('flags')} != pre-registered {FLAGS.get(rec.get('variant'))}")
    fits = [int(t) for t, r in (rec.get("ladder") or {}).items()
            if r.get("ok") and not r.get("oom") and float(r.get("peak_gb") or 0.0) <= MAX_PEAK_GB]
    # a rung counts only if every smaller rung fitted too (the ladder stops at the first failure)
    t_fit = None
    for t in LADDER:
        if t in fits:
            t_fit = t
        else:
            break
    return {"present": True, "qualifies_ex_loss": not reasons, "reasons": reasons, "oom_steps": oom,
            "skipped_steps": skipped, "peak_gb": peak, "completed_timed": len(timed),
            "timed_completed": sorted(s["i"] for s in timed),
            "_timed": {s["i"]: (s["tok_s"], s["s"]) for s in timed},
            "median_tok_s": statistics.median([s["tok_s"] for s in timed]) if timed else None,
            "median_step_s": statistics.median([s["s"] for s in timed]) if timed else None,
            "probe_loss": _probe_loss(rec), "t_fit": t_fit}


def loss_clause(summ: dict, base_probe: float | None) -> dict:
    """A2: the variant's probe loss vs the baseline's (same rows, same seed, forward only). Baseline probe missing or
    not finite -> inconclusive (not a pass)."""
    if base_probe is None:
        return {"ok": False, "basis": "inconclusive_baseline_probe", "rel": None}
    v = summ.get("probe_loss")
    if v is None:
        return {"ok": False, "basis": "probe", "rel": None}
    rel = abs(v - base_probe) / max(abs(base_probe), 1e-12)
    return {"ok": rel <= LOSS_REL_MAX, "basis": "probe", "rel": rel}


def cell_t_ladder(t_fit: int | None) -> list[int]:
    return [t for t in CELL_T_LADDER if t_fit is not None and t <= t_fit]


def projection(median_step_s: float | None) -> dict:
    if not median_step_s:
        return {"steps_190min": None, "rows_190min": None}
    steps = CELL_MINUTES * 60 / median_step_s
    return {"steps_190min": math.floor(steps), "rows_190min": math.floor(steps) * ROWS_PER_STEP,
            "median_step_s": median_step_s}


def decide(records: dict[str, dict | None]) -> dict:
    """``records``: variant name -> its record (None = missing). A variant qualifies with zero OOM on every step it
    attempted, a finite loss on every step it completed, every step's peak <= MAX_PEAK_GB, >= MIN_TIMED_COMPLETED timed
    steps completed, the pre-registered flags and allocator setting, and the probe loss clause. Rank: median tok/s over
    the timed steps completed by ALL qualifying variants (matched rows); ties: lower peak, then the earlier variant.
    None qualifies -> CAP_T at the largest ladder T that fits (variants whose loss clause holds; ties: higher median
    tok/s), else NO_FIT. Baseline probe missing / not finite -> INCONCLUSIVE."""
    summ = {v: summarize(records.get(v)) for v in VARIANTS}
    base = summ["baseline"]
    base_probe = base.get("probe_loss") if base["present"] else None
    for v in VARIANTS:
        s = summ[v]
        s["loss"] = loss_clause(s, base_probe) if s["present"] else {"ok": False, "basis": "missing", "rel": None}
        s["qualifies"] = bool(s["qualifies_ex_loss"] and s["loss"]["ok"])
    ok = [v for v in VARIANTS if summ[v]["qualifies"]]
    matched = sorted(set.intersection(*(set(summ[v]["timed_completed"]) for v in ok))) if ok else []
    for v in VARIANTS:
        t = summ[v].pop("_timed", {})
        if v in ok and matched:
            summ[v]["matched_median_tok_s"] = statistics.median([t[i][0] for i in matched])
            summ[v]["matched_median_step_s"] = statistics.median([t[i][1] for i in matched])
        elif v in ok:  # no timed step completed by every qualifier: each variant's own completed steps
            summ[v]["matched_median_tok_s"] = summ[v]["median_tok_s"]
            summ[v]["matched_median_step_s"] = summ[v]["median_step_s"]
    out = {"rule": RULE_ID, "variants": summ, "max_peak_gb": MAX_PEAK_GB, "loss_rel_max": LOSS_REL_MAX,
           "matched_steps": matched}
    if base_probe is None:
        out.update(verdict="INCONCLUSIVE", chosen=None, cell_config=None, t_fit=None, projection=projection(None),
                   note="the baseline's probe loss is missing or not finite: the GPU loss clause cannot be evaluated")
        return out
    if ok:
        best = sorted(ok, key=lambda v: (-summ[v]["matched_median_tok_s"], summ[v]["peak_gb"], VARIANTS.index(v)))[0]
        s = summ[best]
        out.update(verdict="ADOPT", chosen=best,
                   cell_config={**FLAGS[best], "t_ladder": cell_t_ladder(s["t_fit"]) or list(CELL_T_LADDER)},
                   t_fit=s["t_fit"], projection=projection(s["matched_median_step_s"]),
                   speedup_vs_baseline=(s["median_tok_s"] / base["median_tok_s"]
                                        if base.get("median_tok_s") else None))
        if s["t_fit"] is not None and s["t_fit"] < max(CELL_T_LADDER):
            out["note"] = (f"{best} qualifies on the cell's rows but its K={LADDER_K} ladder fits only T <= {s['t_fit']}: "
                           "keep the cell's own ladder stage (it measures the worst real rows) as the authority")
        return out
    cands = [v for v in VARIANTS if summ[v]["present"] and summ[v]["loss"]["ok"] and summ[v]["t_fit"] is not None]
    if not cands:
        out.update(verdict="NO_FIT", chosen=None, cell_config=None, t_fit=None, projection=projection(None))
        return out
    best = sorted(cands, key=lambda v: (-summ[v]["t_fit"], -(summ[v]["median_tok_s"] or 0.0), VARIANTS.index(v)))[0]
    s = summ[best]
    out.update(verdict="CAP_T", chosen=best, t_fit=s["t_fit"],
               cell_config={**FLAGS[best], "t_ladder": cell_t_ladder(s["t_fit"]) or [s["t_fit"]],
                            "t_cap": s["t_fit"]},
               projection=projection(s["median_step_s"]))
    return out


VERDICTS = ("ADOPT", "CAP_T", "NO_FIT", "INCONCLUSIVE")


def valid_decision(d) -> bool:
    """finalize.py validator: a well-formed decision whose verdict is one of the pre-registered ones."""
    try:
        return (d.get("rule") == RULE_ID and d.get("verdict") in VERDICTS
                and set(d["variants"]) == set(VARIANTS)
                and (d["verdict"] in ("NO_FIT", "INCONCLUSIVE")) == (d.get("chosen") is None))
    except Exception:  # noqa: BLE001
        return False
