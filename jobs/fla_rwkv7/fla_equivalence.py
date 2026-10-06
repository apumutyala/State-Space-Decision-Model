"""fla ``chunk_rwkv7`` vs the vendored eager WKV: numeric equivalence + throughput gate (jobs/fla_rwkv7). GPU phases.

    python -m jobs.fla_rwkv7.fla_equivalence warmup     --out DIR             # compile + autotune every fla variant once
    python -m jobs.fla_rwkv7.fla_equivalence kernel     --out DIR             # synthetic inputs, real 13.3B head shapes
    python -m jobs.fla_rwkv7.fla_equivalence fork       --out DIR             # decision path: state split / fork / pack / pad
    python -m jobs.fla_rwkv7.fla_equivalence model      --out DIR --model-dir SNAP   # real weights: logits, continuation,
                                                                               # captured real WKV inputs (layers 0/30/60)
    python -m jobs.fla_rwkv7.fla_equivalence throughput --out DIR --model-dir SNAP   # prefill 2k/8k/16k, train fwd+bwd 2k
    python -m jobs.fla_rwkv7.fla_equivalence fp32       --out DIR             # optional, last: kernel + fork at fp32
    python -m jobs.fla_rwkv7.fla_equivalence decide     --out DIR             # pre-registered rule -> DIR/decision.json
  mode truth_v2 (RULE v2; the kernel / fork / model / throughput phases above run unchanged first):
    python -m jobs.fla_rwkv7.fla_equivalence truth_v2   --out DIR --model-dir SNAP   # L0/15/30/45/60 vs fp64 truth
    python -m jobs.fla_rwkv7.fla_equivalence train_v2   --out DIR --model-dir SNAP --impl IMPL --deadline-s S
                                                                               # the m2 cell's step, 3 + 20 steps
    python -m jobs.fla_rwkv7.fla_equivalence decide_v2  --out DIR             # RULE v2 -> DIR/decision_v2.json

Compile cost (job-4idg4): every new Triton specialisation of fla's DPLR kernels (activation dtype, safe_gate,
initial state on/off, varlen, backward) is JIT-compiled and autotuned on its first call. In job-4idg4 the bf16 kernel
cases, including those first calls, took about 3 of the 4 minutes. The first fp32 fla call then compiled a whole new
set and was still running 62 s later, when the cap hit. ``warmup`` now pays every bf16 compile once, under its own cap
and timed. Triton's on-disk cache (TRITON_CACHE_DIR, and fla's ``cache_results=True`` autotune cache) carries them to
the later phases, which are separate processes. fp32 only ever set ``allow_fp32`` and never the verdict, so it runs
last and only when time is left.

Every phase writes DIR/<phase>.json after each case, so a phase timeout still leaves what it measured. The rule and the
tolerances below are PRE-REGISTERED (jobs/fla_rwkv7/JOB_SPEC.md quotes them; tests/test_fla_rwkv7.py pins them): do not
edit them after seeing GPU numbers. ``decide`` and the metric helpers are pure and CPU-tested.

Metric: ``rel = ||x - ref||_inf / ||ref||_inf`` per tensor (max over outputs / final state / every gradient), plus cosine
similarity of the flattened tensors. Reference = the vendored ``rwkv7_eager`` (token-serial, fp32 state) at the same
activation dtype; an fp64 token-serial run is recorded as a diagnostic ``truth`` only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

# ----------------------------------------------------------------------------- PRE-REGISTERED (do not tune on GPU data)
#: per activation dtype: max rel error of outputs / final state / gradients vs eager, and min cosine
TOL = {
    "bfloat16": {"out": 2e-2, "state": 2e-2, "grad": 5e-2, "cos": 0.999},
    "float32": {"out": 2e-3, "state": 2e-3, "grad": 5e-3, "cos": 0.99999},
}
#: denominator floor of `rel` (and cosine treated as 1 when both tensors are below it): a gradient that the decay has
#: driven to ~1e-45 (e.g. d/d(initial state) after 1024 tokens) is zero, not a 100% error
ABS_FLOOR = 1e-6
#: real-weights model level (bf16, 61 layers compound): fla vs eager logits must be no worse than the production
#: "chunked" path is vs eager (x MODEL_SLACK, floor MODEL_REL_FLOOR), min per-position cosine and top-1 agreement
MODEL_REL_FLOOR = 2e-2
MODEL_SLACK = 1.5
MODEL_COS_MIN = 0.999
MODEL_TOP1_MIN = 0.98
#: continuation / fork at model level: state handed across calls must reproduce the one-shot logits
CONT_REL_MAX = 2e-2
#: two forks of one state (deep copies) must agree to this (exact equality is recorded too)
FORK_REL_MAX = 1e-5
#: speed: fla / chunked tok/s, on the real model, median of REPS after a warm-up rep
SPEEDUP_MIN = 2.0
LENGTHS = (2048, 8192, 16384)
TRAIN_T = 2048
REPS = 3
#: fla_safe_gate is only preferred over plain fla when it passes numerics AND is this much faster (else plain fla)
SAFE_GATE_MARGIN = 1.10
#: the m2 child evals need >= this end-to-end prefill rate (jobs/PERF_AUDIT.md); reported, not part of the rule
CHILD_EVAL_TOK_S = 12000
VARIANTS = ("fla", "fla_safe_gate")
VERDICTS = ("ADOPT", "ADOPT_READOUT_ONLY", "ADOPT_TRAINING_ONLY", "REJECT_SPEED", "REJECT_NUMERICS", "INCONCLUSIVE")

# 13.3B (jobs/rwkv7/JOB_SPEC.md): 61 layers, hidden 4096, 64 heads x 64
H, N = 64, 64
CAPTURE_LAYERS = (0, 30, 60)
#: warmup lengths, first = the one autotune is frozen at (the keys exclude T): a real length, not a toy one
WARMUP_T = (2048, 1000)
DEVICE = "cuda"  # the weight-free phases also run on CPU (tests/test_fla_rwkv7.py smoke, with a stand-in kernel)


# ----------------------------------------------------------------------------- pure helpers (CPU-tested)
def rel_err(x, ref) -> float:
    d = (x.double() - ref.double()).abs().max().item()
    s = max(ref.double().abs().max().item(), ABS_FLOOR)
    return d / s if math.isfinite(d) else math.inf


def cosine(x, ref) -> float:
    a, b = x.double().flatten(), ref.double().flatten()
    if a.abs().max().item() < ABS_FLOOR and b.abs().max().item() < ABS_FLOOR:
        return 1.0
    na, nb = a.norm().item(), b.norm().item()
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return float((a @ b).item() / (na * nb))


def compare(got: dict, ref: dict) -> dict:
    """{name: tensor} vs {name: tensor} -> per-name rel/cos + the worst per kind (out/state/grad)."""
    per = {}
    for k, r in ref.items():
        g = got[k]
        per[k] = {"rel": rel_err(g, r), "cos": cosine(g, r), "finite": bool(g.isfinite().all().item())}
    worst = {}
    for kind in ("out", "state", "grad"):
        ks = [k for k in per if (k.startswith("grad_") if kind == "grad" else k == kind or k.startswith(kind + "_"))]
        if ks:
            worst[kind] = max(per[k]["rel"] for k in ks)
    worst["cos"] = min(v["cos"] for v in per.values())
    worst["finite"] = all(v["finite"] for v in per.values())
    return {"per": per, "worst": worst}


def case_passes(worst: dict, dtype: str) -> bool:
    t = TOL[dtype]
    if not worst.get("finite", False):
        return False
    for kind in ("out", "state", "grad"):
        v = worst.get(kind)
        if v is not None and not (v <= t[kind]):  # NaN fails
            return False
    return worst.get("cos", 0.0) >= t["cos"]


def model_passes(m: dict) -> bool:
    """m: {'fla': {'rel','cos_min','top1'}, 'chunked': {'rel'}} vs eager logits."""
    f, c = m["fla"], m["chunked"]
    limit = max(MODEL_REL_FLOOR, MODEL_SLACK * c["rel"])
    return bool(f["rel"] <= limit and f["cos_min"] >= MODEL_COS_MIN and f["top1"] >= MODEL_TOP1_MIN)


def _variant_numerics(rep: dict, v: str) -> tuple[bool | None, list[str]]:
    """True/False, or None when a required piece is missing. bf16 decides; fp32 only gates ALLOW_FP32."""
    why, missing = [], []
    for phase in ("kernel", "fork", "captured"):
        cases = [c for c in (rep.get(phase) or {}).get("cases", []) if c.get("impl") == v and c.get("dtype") == "bfloat16"]
        if not cases:
            missing.append(f"{phase}:bf16")
        for c in cases:
            if c.get("error") or not case_passes(c.get("worst", {}), "bfloat16"):
                why.append(f"{phase}:{c.get('name')}")
    model = rep.get("model") or {}
    mdl, ref = model.get("logits", {}).get(v), model.get("logits", {}).get("chunked")
    if not mdl or not ref:
        missing.append("model:logits")
    elif not model_passes({"fla": mdl, "chunked": ref}):
        why.append("model:logits")
    cont = model.get("continuation", {}).get(v)
    if not cont:
        missing.append("model:continuation")
    elif not (cont.get("rel", math.inf) <= CONT_REL_MAX and cont.get("fork_rel", math.inf) <= FORK_REL_MAX):
        why.append("model:continuation")
    if why:
        return False, why
    if missing:
        return None, missing
    return True, []


def _fp32_ok(rep: dict, v: str) -> bool:
    """allow_fp32 needs the WHOLE optional fp32 phase (forward, gradients, fork, packed, padding): fp32.json is written
    after every case, so a timed-out run leaves a partial file whose forward-only cases could all pass. Only the
    ``complete`` flag that phase_fp32 writes after its last case makes the evidence count."""
    ph = rep.get("fp32") or {}
    if ph.get("complete") is not True:
        return False
    cases = [c for c in ph.get("cases", []) if c.get("impl") == v and c.get("dtype") == "float32"]
    return bool(cases) and all(not c.get("error") and case_passes(c.get("worst", {}), "float32") for c in cases)


def _ratios(rep: dict, v: str) -> dict:
    tp = rep.get("throughput") or {}
    pre = tp.get("prefill", {})
    out = {"prefill": {}, "train": None}
    for L in LENGTHS:
        a, b = (pre.get(str(L)) or {}).get(v, {}), (pre.get(str(L)) or {}).get("chunked", {})
        out["prefill"][str(L)] = (a["tok_s"] / b["tok_s"]) if a.get("tok_s") and b.get("tok_s") else None
    tr = tp.get("train", {})
    a, b = tr.get(v, {}), tr.get("chunked", {})
    out["train"] = (a["tok_s"] / b["tok_s"]) if a.get("tok_s") and b.get("tok_s") else None
    return out


def decide(rep: dict) -> dict:
    """The pre-registered rule. rep = {'kernel','fork','captured','model','throughput'} phase JSONs.

    Per fla variant: numerics pass iff every bf16 case (kernel, fork, captured real inputs) is within TOL, the real-model
    logits pass ``model_passes`` and the model continuation/fork passes. Readout speed passes iff fla/chunked prefill
    >= SPEEDUP_MIN at EVERY length in LENGTHS; training speed iff the fwd+bwd ratio at TRAIN_T >= SPEEDUP_MIN.
    ADOPT = numerics + both; ADOPT_READOUT_ONLY / ADOPT_TRAINING_ONLY = numerics + one; REJECT_SPEED = numerics, neither;
    REJECT_NUMERICS = a measured numerics failure; INCONCLUSIVE = a required measurement is missing.
    Chosen variant: plain "fla" unless "fla_safe_gate" also qualifies and is >= SAFE_GATE_MARGIN faster on every
    measured ratio. fp32 never blocks (the model runs bf16); it only decides ``allow_fp32``."""
    per = {}
    for v in VARIANTS:
        num, why = _variant_numerics(rep, v)
        r = _ratios(rep, v)
        pre = list(r["prefill"].values())
        readout = None if any(x is None for x in pre) else all(x >= SPEEDUP_MIN for x in pre)
        train = None if r["train"] is None else r["train"] >= SPEEDUP_MIN
        if num is False:
            verdict = "REJECT_NUMERICS"
        elif num is None or readout is None or train is None:
            verdict = "INCONCLUSIVE"
        elif readout and train:
            verdict = "ADOPT"
        elif readout:
            verdict = "ADOPT_READOUT_ONLY"
        elif train:
            verdict = "ADOPT_TRAINING_ONLY"
        else:
            verdict = "REJECT_SPEED"
        per[v] = {"verdict": verdict, "numerics": num, "numerics_detail": why, "ratios": r,
                  "readout_speed_ok": readout, "train_speed_ok": train, "allow_fp32": _fp32_ok(rep, v)}
    chosen = "fla"
    a, b = per["fla"], per["fla_safe_gate"]
    if b["verdict"].startswith("ADOPT") and (not a["verdict"].startswith("ADOPT") or _faster(b["ratios"], a["ratios"])):
        chosen = "fla_safe_gate"
    best = per[chosen]
    pre16 = (((rep.get("throughput") or {}).get("prefill") or {}).get("2048") or {}).get(chosen, {}).get("tok_s")
    return {"verdict": best["verdict"], "chosen": chosen, "variants": per,
            "rule": {"tol": TOL, "abs_floor": ABS_FLOOR, "model_rel_floor": MODEL_REL_FLOOR, "model_slack": MODEL_SLACK,
                     "model_cos_min": MODEL_COS_MIN, "model_top1_min": MODEL_TOP1_MIN, "cont_rel_max": CONT_REL_MAX,
                     "fork_rel_max": FORK_REL_MAX,
                     "speedup_min": SPEEDUP_MIN, "lengths": list(LENGTHS), "train_t": TRAIN_T,
                     "safe_gate_margin": SAFE_GATE_MARGIN},
            "child_eval_tok_s_target": CHILD_EVAL_TOK_S,
            "chosen_prefill_2048_tok_s": pre16,
            "meets_child_eval_rate": bool(pre16 and pre16 >= CHILD_EVAL_TOK_S)}


def _faster(b: dict, a: dict) -> bool:
    pairs = [(b["prefill"][k], a["prefill"][k]) for k in b["prefill"]] + [(b["train"], a["train"])]
    return all(x is not None and y is not None and x >= SAFE_GATE_MARGIN * y for x, y in pairs)


def _fla_really_ran(phase: dict) -> bool:
    """No fallback (it would run "chunked" under the fla name) and every variant actually reached the kernel."""
    calls = phase.get("fla_calls") or {}
    return not phase.get("fallbacks") and all(calls.get(v, 0) > 0 for v in VARIANTS)


def valid_decision(d) -> bool:
    """finalize.py validator: a verdict from the rule and a chosen variant."""
    return isinstance(d, dict) and d.get("verdict") in VERDICTS and d.get("chosen") in VARIANTS


# ============================================================================= RULE v2 (PRE-REGISTERED 2026-10-06)
# Pre-registered by the coordinator on 2026-10-06, BEFORE any truth_v2 job ran, after job-jn47u (REJECT_NUMERICS from
# the captured-real-input cases alone, where the production "chunked" path failed the same bf16-eager reference).
# v1 above (TOL / decide) is NOT edited; v2 is a separate rule. Text, verbatim: jobs/fla_rwkv7/JOB_SPEC.md "RULE v2".
# Constants pinned by tests/test_fla_rwkv7.py::test_rule_v2_constants_are_pinned. Do not tune them on GPU data.
#: captured real WKV inputs (same 2,048-token prompt and capture construction as job-jn47u: prefix 1024 tokens -> carried
#: state, capture during the next 1024): fwd at T=1024, gradients at the first 512 tokens
TRUTH_V2_LAYERS = (0, 15, 30, 45, 60)
V2_PROMPT_T = 2048
V2_FWD_T = 1024
V2_GRAD_T = 512
#: e(fla) <= V2_SLACK * max(e(chunked), e(eager_bf16)) + ABS_FLOOR on EVERY (case, tensor) pair, and
#: e(fla) <= V2_SLACK * e(chunked) + ABS_FLOOR on at least V2_FRAC_MIN of them
V2_SLACK = 1.10
V2_FRAC_MIN = 0.90
V2_REF_IMPLS = ("chunked", "eager_bf16")
V2_FWD_TENSORS = ("out", "state")
V2_GRAD_TENSORS = ("out", "state") + tuple(f"grad_{n}" for n in ("r", "w_log", "k", "v", "kk", "a", "state"))
#: training speed: the m2 first cell's actual step (train.trainer.Trainer.train_step, cell config), current impl vs fla
V2_CURRENT = "chunked"
V2_TRAIN_WARMUP = 3  # untimed optimiser steps first (fla compile / autotune, allocator warm-up)
V2_TRAIN_STEPS = 20  # timed optimiser steps; median per-step tok/s
#: the current impl may be too slow for the phase cap: its median is over the timed steps it reached (OOM = 0 tok/s),
#: at least this many; below that its speed is missing (INCONCLUSIVE). fla variants must reach all V2_TRAIN_STEPS.
V2_CURRENT_MIN_STEPS = 5
V2_TRAIN_RATIO_MIN = 2.0
V2_PREFILL_RATIO_MIN = 2.0  # at every length in LENGTHS (the unchanged throughput phase)
V2_PEAK_GB_MAX = 76.0  # = the m2 cell's pass_rule max_peak_gb
VERDICTS_V2 = ("ADOPT_FOR_TRAINING", "ADOPT_READOUT_ONLY", "REJECT_NUMERICS", "INCONCLUSIVE", "NOT_ADOPTED")


def v2_required_cases() -> dict:
    """{case name: tensors} the truth_v2 phase must produce."""
    req = {}
    for layer in TRUTH_V2_LAYERS:
        req[f"real_L{layer}_fwd_T{V2_FWD_T}"] = V2_FWD_TENSORS
        req[f"real_L{layer}_grad_T{V2_GRAD_T}"] = V2_GRAD_TENSORS
    return req


def _v2_truth_check(tv: dict | None, v: str) -> tuple[bool | None, list[str], dict]:
    """The truth-reference half of NUMERICS_OK for variant ``v``. True / False, or None when the comparison is
    undefined (a case, tensor, reference error or non-finite reference / truth is missing): INCONCLUSIVE.
    A crash or a non-finite value of ``v`` itself is a failure (as in v1)."""
    why, missing = [], []
    by_name = {c.get("name"): c for c in (tv or {}).get("cases", [])}
    n_pairs = n_tight = 0
    worst = {"pair_fail": [], "tight_fail": []}
    for name, tensors in v2_required_cases().items():
        c = by_name.get(name)
        if c is None:
            missing.append(f"truth_v2:{name}")
            continue
        if not c.get("truth_finite", False):
            missing.append(f"truth_v2:{name}:truth_nonfinite")
            continue
        imps = c.get("impls") or {}
        mine = imps.get(v) or {}
        if mine.get("error"):
            why.append(f"truth_v2:{name}:{v}_error")
            continue
        if mine.get("finite") is False:
            why.append(f"truth_v2:{name}:{v}_nonfinite")
            continue
        if mine.get("finite") is not True:
            missing.append(f"truth_v2:{name}:{v}_missing")
            continue
        bad_refs = [r for r in V2_REF_IMPLS if (imps.get(r) or {}).get("error") or (imps.get(r) or {}).get("finite") is not True]
        if bad_refs:
            missing += [f"truth_v2:{name}:{r}_unusable" for r in bad_refs]
            continue
        for t in tensors:
            ev = (mine.get("e") or {}).get(t)
            ec = (imps["chunked"].get("e") or {}).get(t)
            ee = (imps["eager_bf16"].get("e") or {}).get(t)
            if ev is None or ec is None or ee is None:
                missing.append(f"truth_v2:{name}:{t}")
                continue
            if not all(math.isfinite(x) for x in (ev, ec, ee)):
                if not math.isfinite(ev):
                    why.append(f"truth_v2:{name}:{t}:{v}_nonfinite")
                else:
                    missing.append(f"truth_v2:{name}:{t}:ref_nonfinite")
                continue
            n_pairs += 1
            if not ev <= V2_SLACK * max(ec, ee) + ABS_FLOOR:
                why.append(f"truth_v2:{name}:{t}")
                worst["pair_fail"].append([name, t, ev, ec, ee])
            if ev <= V2_SLACK * ec + ABS_FLOOR:
                n_tight += 1
            else:
                worst["tight_fail"].append([name, t, ev, ec])
    frac = n_tight / n_pairs if n_pairs else None
    detail = {"pairs": n_pairs, "within_slack_of_chunked": n_tight, "frac_within_slack_of_chunked": frac, **worst}
    if why:
        return False, why, detail
    if missing:
        return None, missing, detail
    if frac is None or frac < V2_FRAC_MIN:
        return False, [f"truth_v2:frac_within_slack_of_chunked={frac}"], detail
    return True, [], detail


def _v2_synthetic_check(rep: dict, v: str) -> tuple[bool | None, list[str]]:
    """The unchanged v1 synthetic checks (kernel + fork bf16 cases vs eager under TOL, model logits, model
    continuation/fork), re-run by the same phases. The v1 captured-input cases are NOT part of v2."""
    why, missing = [], []
    for phase in ("kernel", "fork"):
        cases = [c for c in (rep.get(phase) or {}).get("cases", []) if c.get("impl") == v and c.get("dtype") == "bfloat16"]
        if not cases:
            missing.append(f"{phase}:bf16")
        for c in cases:
            if c.get("error") or not case_passes(c.get("worst", {}), "bfloat16"):
                why.append(f"{phase}:{c.get('name')}")
    model = rep.get("model") or {}
    mdl, ref = model.get("logits", {}).get(v), model.get("logits", {}).get("chunked")
    if not mdl or not ref:
        missing.append("model:logits")
    elif not model_passes({"fla": mdl, "chunked": ref}):
        why.append("model:logits")
    cont = model.get("continuation", {}).get(v)
    if not cont:
        missing.append("model:continuation")
    elif not (cont.get("rel", math.inf) <= CONT_REL_MAX and cont.get("fork_rel", math.inf) <= FORK_REL_MAX):
        why.append("model:continuation")
    if why:
        return False, why
    if missing:
        return None, missing
    return True, []


def numerics_ok_v2(rep: dict, v: str) -> dict:
    """NUMERICS_OK(v) under RULE v2: the truth check AND the unchanged synthetic checks. False beats None."""
    t_ok, t_why, t_detail = _v2_truth_check(rep.get("truth_v2"), v)
    s_ok, s_why = _v2_synthetic_check(rep, v)
    if t_ok is False or s_ok is False:
        ok = False
    elif t_ok is None or s_ok is None:
        ok = None
    else:
        ok = True
    return {"ok": ok, "truth": t_ok, "synthetic": s_ok, "why": t_why + s_why, "truth_detail": t_detail}


def train_speed_v2(tr: dict | None, current: bool) -> dict:
    """One impl's training-speed record -> {'tok_s' (median per-step tok/s or None), 'peak_gb', 'oom_steps', ...}.
    A timed step that ran out of memory counts 0 tok/s (it cannot be trained at that shape on this path). The current
    impl's median is over the timed steps it reached (>= V2_CURRENT_MIN_STEPS); an fla variant needs every one of
    V2_TRAIN_STEPS. A fallback or no fla call voids an fla measurement (None)."""
    if not tr:
        return {"tok_s": None, "peak_gb": None, "reason": "missing"}
    steps = [s for s in tr.get("steps", []) if s.get("timed")]
    vals = [0.0 if s.get("oom") else s.get("tok_s") for s in steps if s.get("oom") or s.get("tok_s") is not None]
    oom = sum(1 for s in steps if s.get("oom"))
    out = {"tok_s": None, "peak_gb": tr.get("peak_gb"), "oom_steps": oom, "timed_steps": len(vals)}
    if not current:
        if tr.get("fallbacks") or not (tr.get("fla_calls") or {}).get(tr.get("impl"), 0) > 0:
            out["reason"] = "fla fallback or no fla call"
            return out
        need = V2_TRAIN_STEPS
    else:
        need = V2_CURRENT_MIN_STEPS
    if len(vals) < need:
        out["reason"] = f"{len(vals)} timed steps < {need}"
        return out
    out["tok_s"] = statistics.median(vals[:V2_TRAIN_STEPS])
    return out


def decide_v2(rep: dict) -> dict:
    """RULE v2 (verbatim in JOB_SPEC.md). rep = {'kernel','fork','model','throughput','truth_v2',
    'train_v2': {impl: record}} phase JSONs. Per fla variant:
      REJECT_NUMERICS    iff NUMERICS_OK is false;
      INCONCLUSIVE       iff anything required is missing (numerics undefined, a prefill ratio, the train ratio,
                         the fla peak);
      ADOPT_FOR_TRAINING iff NUMERICS_OK and train ratio >= V2_TRAIN_RATIO_MIN and fla peak <= V2_PEAK_GB_MAX;
      ADOPT_READOUT_ONLY iff NUMERICS_OK and prefill ratio >= V2_PREFILL_RATIO_MIN at every length but train ratio
                         < V2_TRAIN_RATIO_MIN;
      NOT_ADOPTED        every other measured combination (the rule names no verdict for it; reported, not adopted).
    Train ratio = median tok/s (fla variant) / median tok/s (current); +inf when the current impl's median is 0 (OOM)
    and the variant's is not; 0 when the variant's median is 0.
    The top-level verdict is plain fla's; fla_safe_gate gets its own under the same rule."""
    trains = rep.get("train_v2") or {}
    cur = train_speed_v2(trains.get(V2_CURRENT), current=True)
    per = {}
    for v in VARIANTS:
        num = numerics_ok_v2(rep, v)
        pre = _ratios(rep, v)["prefill"]
        sp = train_speed_v2(trains.get(v), current=False)
        if sp["tok_s"] is None or cur["tok_s"] is None:
            ratio = None
        elif sp["tok_s"] == 0:
            ratio = 0.0  # fla could not train the median step either
        elif cur["tok_s"] == 0:
            ratio = math.inf
        else:
            ratio = sp["tok_s"] / cur["tok_s"]
        fla_oom = bool(sp.get("oom_steps"))
        peak_ok = None if sp.get("peak_gb") is None else (sp["peak_gb"] <= V2_PEAK_GB_MAX and not fla_oom)
        readout = None if any(x is None for x in pre.values()) else all(x >= V2_PREFILL_RATIO_MIN for x in pre.values())
        if num["ok"] is False:
            verdict = "REJECT_NUMERICS"
        elif num["ok"] is None or readout is None or ratio is None or peak_ok is None:
            verdict = "INCONCLUSIVE"
        elif ratio >= V2_TRAIN_RATIO_MIN and peak_ok:
            verdict = "ADOPT_FOR_TRAINING"
        elif readout and ratio < V2_TRAIN_RATIO_MIN:
            verdict = "ADOPT_READOUT_ONLY"
        else:
            verdict = "NOT_ADOPTED"
        per[v] = {"verdict": verdict, "numerics": num, "prefill_ratios": pre, "readout_speed_ok": readout,
                  "train": sp, "train_ratio": ratio, "peak_ok": peak_ok}
    return {"rule": "v2", "verdict": per["fla"]["verdict"], "variant": "fla", "variants": per, "current_train": cur,
            "constants": {"layers": list(TRUTH_V2_LAYERS), "fwd_t": V2_FWD_T, "grad_t": V2_GRAD_T, "slack": V2_SLACK,
                          "frac_min": V2_FRAC_MIN, "abs_floor": ABS_FLOOR, "train_warmup": V2_TRAIN_WARMUP,
                          "train_steps": V2_TRAIN_STEPS, "current_min_steps": V2_CURRENT_MIN_STEPS,
                          "train_ratio_min": V2_TRAIN_RATIO_MIN, "prefill_ratio_min": V2_PREFILL_RATIO_MIN,
                          "lengths": list(LENGTHS), "peak_gb_max": V2_PEAK_GB_MAX, "current": V2_CURRENT}}


def valid_decision_v2(d) -> bool:
    """finalize.py validator for the truth_v2 mode."""
    return (isinstance(d, dict) and d.get("rule") == "v2" and d.get("verdict") in VERDICTS_V2
            and all((d.get("variants") or {}).get(v, {}).get("verdict") in VERDICTS_V2 for v in VARIANTS))


# ----------------------------------------------------------------------------- GPU side
def _torch():
    import torch

    return torch


def make_inputs(B: int, T: int, dtype, seed: int, device=None, state_scale: float = 0.5):
    """Model-distributed WKV inputs at the real head shape: w_log = -e^-0.5 * sigmoid(.) (the model's decay law),
    kk unit-norm per head, a in (0, 1), fp32 carried state."""
    torch = _torch()
    g = torch.Generator(device="cpu").manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    r, k, v = rn(B, T, H, N), rn(B, T, H, N), rn(B, T, H, N)
    w_log = -math.exp(-0.5) * torch.sigmoid(rn(B, T, H, N) * 2 - 0.5)
    kk = torch.nn.functional.normalize(rn(B, T, H, N), dim=-1)
    a = torch.sigmoid(rn(B, T, H, N))
    state = rn(B, H, N, N) * state_scale
    device = device or DEVICE
    t = {n: x.to(device=device, dtype=dtype) for n, x in dict(r=r, w_log=w_log, k=k, v=v, kk=kk, a=a).items()}
    t["state"] = state.to(device=device, dtype=torch.float32)
    return t


ARGS = ("r", "w_log", "k", "v", "kk", "a", "state")


def run_impl(impl: str, x: dict, grad: bool, cu_seq_lens=None, seed: int = 1, allow_fp32: bool = True):
    """Run one WKV implementation on inputs x; returns {'out','state'[, 'grad_<arg>']} (detached)."""
    torch = _torch()
    from heads import rwkv7_wkv as W
    from third_party.rwkv7_hf import modeling_rwkv7 as M

    leaves = {n: x[n].detach().clone().requires_grad_(grad) for n in ARGS}
    if impl == "eager":
        o, s = M.rwkv7_eager(*(leaves[n] for n in ARGS), cu_seq_lens=cu_seq_lens)
    elif impl == "chunked":
        o, s = M.rwkv7_parallel(*(leaves[n] for n in ARGS), cu_seq_lens=cu_seq_lens)
    elif impl in VARIANTS:
        chunk, err = W._import_chunk()
        if err:
            raise RuntimeError(err)
        reason = W.fallback_reason(leaves["r"], cu_seq_lens, allow_fp32)
        if reason:
            raise RuntimeError(f"would fall back: {reason}")
        o, s = W.fla_wkv_call(chunk, *(leaves[n] for n in ARGS), cu_seq_lens, safe_gate=impl == "fla_safe_gate")
    else:
        raise ValueError(impl)
    res = {"out": o.detach(), "state": s.detach()}
    if grad:
        g = torch.Generator(device="cpu").manual_seed(seed)
        do = torch.randn(o.shape, generator=g).to(o.device)
        ds = torch.randn(s.shape, generator=g).to(s.device)
        ((o.float() * do).sum() + (s.float() * ds).sum()).backward()
        for n in ARGS:
            res[f"grad_{n}"] = _grad(leaves[n])
    return res


def truth(x: dict, cu_seq_lens=None) -> dict:
    """fp64 token-serial run of the same (rounded) inputs: a diagnostic, not the gate's reference."""
    torch = _torch()
    from third_party.rwkv7_hf import modeling_rwkv7 as M

    d = {n: x[n].double() for n in ARGS}
    if cu_seq_lens is not None:
        return {}
    o, s = M.rwkv7_recurrent(*(d[n] for n in ARGS), compute_dtype=torch.float64)
    return {"out": o, "state": s}


class Report:
    def __init__(self, path: Path, **meta):
        self.path, self.d, self.t0 = path, {"cases": [], **meta}, time.perf_counter()
        self.flush()

    def add(self, case: dict):
        case.setdefault("t_s", round(time.perf_counter() - self.t0, 2))  # elapsed in the phase when the case ended
        self.d["cases"].append(case)
        self.flush()

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.d, indent=1, default=str), encoding="utf-8")


def _cmp_case(rep: Report, name: str, dtype_name: str, x: dict, grad: bool, cu=None, impls=("chunked",) + VARIANTS,
              with_truth: bool = False):
    torch = _torch()
    ref = run_impl("eager", x, grad, cu)
    tr = truth(x, cu) if with_truth else {}
    for impl in impls:
        case = {"name": name, "impl": impl, "dtype": dtype_name, "grad": grad,
                "shape": list(x["r"].shape), "cu_seq_lens": None if cu is None else cu.tolist()}
        try:
            got = run_impl(impl, x, grad, cu)
            c = compare(got, ref)
            case.update(worst=c["worst"], per=c["per"], passes=case_passes(c["worst"], dtype_name))
            if tr:
                case["vs_fp64"] = {k: rel_err(got[k], tr[k]) for k in tr}
        except Exception as e:  # noqa: BLE001  recorded; decide() treats it as a failure of that case
            case["error"] = f"{type(e).__name__}: {e}"[:500]
        rep.add(case)
    if tr:
        rep.add({"name": name, "impl": "eager", "dtype": dtype_name, "diag": True,
                 "vs_fp64": {k: rel_err(ref[k], tr[k]) for k in tr}})
    torch.cuda.empty_cache()


def phase_warmup(out: Path, dtypes=("bfloat16",)):
    """Pay fla's JIT compile + autotune once per specialisation the later phases hit: each variant x {initial state +
    final state, packed varlen without initial state} x {forward, forward+backward} x T in {2048, 1000}. fla's autotune keys are block
    sizes, not T, so the FIRST call's T fixes the cached config for every length: it is a real length (2048), not a
    toy one, and 1000 covers the non-multiple-of-16 specialisation. Each first call and a second call are timed (compile cost =
    first - second)."""
    torch = _torch()
    rep = Report(out / "warmup.json", env=_env(), triton_cache_dir=os.environ.get("TRITON_CACHE_DIR"))
    import itertools

    # Triton specialises integer args on divisibility by 16, so warm both kinds of T (the cases use 1000, 1537, ...)
    for dt, T in itertools.product(dtypes, WARMUP_T):
        x = make_inputs(1, T, getattr(torch, dt), 1)
        cu = torch.tensor([0, 30, T], device=DEVICE)
        for impl, grad, packed in itertools.product(VARIANTS, (False, True), (False, True)):
            case = {"name": f"warm_T{T}_{'packed' if packed else 'state'}_{'bwd' if grad else 'fwd'}",
                    "impl": impl, "dtype": dt, "diag": True}
            try:
                ts = []
                for _ in range(2):
                    if DEVICE == "cuda":
                        torch.cuda.synchronize()
                    t = time.perf_counter()
                    run_impl(impl, x, grad, cu if packed else None)
                    if DEVICE == "cuda":
                        torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t)
                case.update(first_s=ts[0], second_s=ts[1], compile_s=ts[0] - ts[1])
            except Exception as e:  # noqa: BLE001  recorded; the kernel phase will fail the same call
                case["error"] = f"{type(e).__name__}: {e}"[:500]
            rep.add(case)


def _kernel_cases(rep: Report, dt: str):
    torch = _torch()
    dtype = getattr(torch, dt)
    for B, T, seed in ((2, 2048, 11), (2, 1000, 12)):  # 1000: not a multiple of any chunk size (padding path)
        _cmp_case(rep, f"fwd_B{B}_T{T}", dt, make_inputs(B, T, dtype, seed), grad=False, with_truth=True)
    _cmp_case(rep, "grad_B2_T512", dt, make_inputs(2, 512, dtype, 13), grad=True)
    # short: d/d(initial state) is still far from zero here (fla's dh0 path, used by every fork)
    _cmp_case(rep, "grad_B2_T24", dt, make_inputs(2, 24, dtype, 14), grad=True)


def phase_kernel(out: Path):
    rep = Report(out / "kernel.json", env=_env())
    _kernel_cases(rep, "bfloat16")
    _p640(rep)


def phase_fp32(out: Path):
    """Optional and last: the kernel + fork cases at fp32. They only decide ``allow_fp32`` (never the verdict); a
    missing or partial fp32.json leaves ALLOW_FP32 = False."""
    rep = Report(out / "fp32.json", env=_env(), complete=False)
    _kernel_cases(rep, "float32")
    _fork_cases(rep, "float32")
    rep.d["complete"] = True  # written only after the last case: _fp32_ok refuses a partial (timed-out) file
    rep.flush()


def _p640(rep: Report):
    """fla #640 is the gated chunk_bwd_dqkwg in fla/ops/common/chunk_o.py. Record that the RWKV7 (DPLR) path never
    references it, from fla's own source as installed, and that a bf16 backward on this GPU is finite."""
    import inspect

    import fla.ops.generalized_delta_rule.dplr.chunk as dplr
    import fla.ops.rwkv7.chunk as rc

    src = inspect.getsource(dplr) + inspect.getsource(rc)
    rep.d["p640"] = {"dplr_references_chunk_bwd_dqkwg": "chunk_bwd_dqkwg" in src,
                     "dplr_imports_common_chunk_o": "fla.ops.common.chunk_o" in src}
    rep.flush()


def phase_fork(out: Path):
    """The decision path: prefix state -> continuation, K-way fork from one state (grads into the prefix), packed
    rows (cu_seq_lens, incl. a 1-token segment), right padding that must leave the state untouched."""
    _fork_cases(Report(out / "fork.json", env=_env()), "bfloat16")


def _fork_cases(rep: Report, dt: str):
    torch = _torch()
    for dt in (dt,):
        dtype = getattr(torch, dt)
        # 1) split: impl(x[:P]) -> s ; impl(x[P:], s) == eager(x) on the tail and the final state
        x = make_inputs(1, 1537, dtype, 21)
        P = 1500
        for impl in ("chunked",) + VARIANTS:
            case = {"name": "split_P1500_L37", "impl": impl, "dtype": dt}
            try:
                full = run_impl("eager", x, False)
                a = run_impl(impl, _slice(x, 0, P), False)
                b = run_impl(impl, {**_slice(x, P, None), "state": a["state"]}, False)
                c = compare({"out": b["out"], "state": b["state"]}, {"out": full["out"][:, P:], "state": full["state"]})
                case.update(worst=c["worst"], passes=case_passes(c["worst"], dt))
            except Exception as e:  # noqa: BLE001
                case["error"] = f"{type(e).__name__}: {e}"[:500]
            rep.add(case)
        # 2) fork: one prefix state, K=4 tails, gradients flow through the shared state into the prefix inputs
        for impl in ("chunked",) + VARIANTS:
            case = {"name": "fork_P1024_K4_L24_grad", "impl": impl, "dtype": dt}
            try:
                c = compare(_fork_run(impl, dtype), _fork_run("eager", dtype))
                case.update(worst=c["worst"], per=c["per"], passes=case_passes(c["worst"], dt))
            except Exception as e:  # noqa: BLE001
                case["error"] = f"{type(e).__name__}: {e}"[:500]
            rep.add(case)
        # 3) packed row: fresh state per segment, the returned state is the last segment's
        cu = torch.tensor([0, 100, 101, 1100, 2048], device=DEVICE)
        _cmp_case(rep, "packed_4seg", dt, make_inputs(1, 2048, dtype, 22), grad=True, cu=cu)
        # 4) right padding (the model's keep mask: decay 1, k = v = kk = 0) leaves the state where the last token left it
        for impl in ("chunked",) + VARIANTS:
            case = {"name": "right_pad_identity", "impl": impl, "dtype": dt}
            try:
                x = make_inputs(1, 600, dtype, 23)
                real = run_impl("eager", _slice(x, 0, 555), False)
                padded = dict(x)
                for n in ("w_log", "k", "v", "kk"):
                    padded[n] = x[n].clone()
                    padded[n][:, 555:] = 0
                got = run_impl(impl, padded, False)
                c = compare({"state": got["state"], "out": got["out"][:, :555]}, {"state": real["state"], "out": real["out"]})
                case.update(worst=c["worst"], passes=case_passes(c["worst"], dt))
            except Exception as e:  # noqa: BLE001
                case["error"] = f"{type(e).__name__}: {e}"[:500]
            rep.add(case)


def _slice(x: dict, a, b) -> dict:
    return {n: (x[n][:, a:b] if n != "state" else x[n]) for n in ARGS}


def _fork_run(impl: str, dtype, K: int = 4, P: int = 1024, L: int = 24) -> dict:
    """Prefix (B=1) -> state.repeat_interleave(K) (new tensors, like batch_repeat_interleave) -> K tails (B=K)."""
    torch = _torch()
    from heads import rwkv7_wkv as W
    from third_party.rwkv7_hf import modeling_rwkv7 as M

    def call(xs):
        if impl == "eager":
            return M.rwkv7_eager(*xs)
        if impl == "chunked":
            return M.rwkv7_parallel(*xs)
        chunk, err = W._import_chunk()
        if err:
            raise RuntimeError(err)
        return W.fla_wkv_call(chunk, *xs, safe_gate=impl == "fla_safe_gate")

    pre = {n: t.requires_grad_(True) for n, t in make_inputs(1, P, dtype, 31).items()}
    tail = {n: t.requires_grad_(True) for n, t in make_inputs(K, L, dtype, 32).items() if n != "state"}
    _, s = call([pre[n] for n in ARGS])
    s_k = s.repeat_interleave(K, dim=0)
    o, sK = call([tail[n] for n in ARGS[:-1]] + [s_k])
    g = torch.Generator(device="cpu").manual_seed(33)
    do = torch.randn(o.shape, generator=g).to(o.device)
    ds = torch.randn(sK.shape, generator=g).to(sK.device)
    ((o.float() * do).sum() + (sK.float() * ds).sum()).backward()
    res = {"out": o.detach(), "state": sK.detach()}
    for n in ARGS:
        res[f"grad_prefix_{n}"] = _grad(pre[n])
    for n in ARGS[:-1]:
        res[f"grad_tail_{n}"] = _grad(tail[n])
    return res


def _grad(t):
    """A leaf autograd never reached (e.g. the prefix's r: only its state is forked) has a zero gradient. Eager leaves
    it None while a custom autograd.Function materialises zeros, so both are compared as zeros."""
    return t.grad.detach() if t.grad is not None else _torch().zeros_like(t)


# ----------------------------------------------------------------------------- real weights
def _load(model_dir: str):
    torch = _torch()
    import third_party.rwkv7_hf as V
    from third_party.rwkv7_hf.modeling_rwkv7 import Rwkv7ForCausalLM

    V.verify_snapshot_code(model_dir)
    m = Rwkv7ForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map={"": "cuda"}).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _text_ids(model_dir: str, T: int):
    """Real text (the vendored Apache LICENSE + NOTICE, repeated) tokenised by the snapshot's tokenizer."""
    torch = _torch()
    from third_party.rwkv7_hf.tokenization_rwkv7 import Rwkv7Tokenizer

    here = Path(__file__).resolve().parents[2] / "third_party" / "rwkv7_hf"
    text = (here / "LICENSE").read_text(encoding="utf-8") + (here / "NOTICE.md").read_text(encoding="utf-8")
    tok = Rwkv7Tokenizer.from_pretrained(model_dir)
    ids = [i for i in tok(text, add_special_tokens=False).input_ids if i != 0]
    ids = (ids * (T // len(ids) + 1))[:T]
    return torch.tensor([ids], device="cuda")


def _use(m, name: str):
    from heads.rwkv7_wkv import select_wkv

    return select_wkv(m, name, allow_unverified=True)


def phase_model(out: Path, model_dir: str):
    torch = _torch()
    from heads import rwkv7_wkv as W
    from third_party.rwkv7_hf import modeling_rwkv7 as M

    rep = Report(out / "model.json", env=_env())
    m = _load(model_dir)
    rep.d["logits"], rep.d["continuation"] = {}, {}
    W.FALLBACKS.clear()
    W.CALLS.clear()
    # 1) logits vs eager (tokenwise, token-serial) at T=256
    ids = _text_ids(model_dir, 256)
    logits = {}
    for impl in ("eager", "chunked") + VARIANTS:
        _use(m, impl)
        with torch.no_grad():
            logits[impl] = m(input_ids=ids, use_cache=False).logits.float()
    for impl in ("chunked",) + VARIANTS:
        x, r = logits[impl][0], logits["eager"][0]
        cos_pos = torch.nn.functional.cosine_similarity(x.double(), r.double(), dim=-1)
        rep.d["logits"][impl] = {"rel": rel_err(x, r), "cos_min": cos_pos.min().item(),
                                 "top1": (x.argmax(-1) == r.argmax(-1)).double().mean().item(), "T": 256}
    rep.flush()
    del logits
    # 2) continuation and fork at T=2048: state from a 1900-token prefix reproduces the one-shot logits; two forks of
    #    one state (deep copies) are identical
    import copy

    ids = _text_ids(model_dir, 2048)
    P, L = 1900, 148
    for impl in VARIANTS:
        _use(m, impl)
        with torch.no_grad():
            one = m(input_ids=ids, use_cache=False, logits_to_keep=L).logits.float()
            st = m(input_ids=ids[:, :P], use_cache=True, logits_to_keep=1).state
            st2 = copy.deepcopy(st)
            a = m(input_ids=ids[:, P:], state=st, use_cache=True).logits.float()
            b = m(input_ids=ids[:, P:], state=st2, use_cache=True).logits.float()
        rep.d["continuation"][impl] = {"rel": rel_err(a, one), "fork_identical": bool(torch.equal(a, b)),
                                       "fork_rel": rel_err(b, a), "P": P, "L": L}
        rep.flush()
    rep.d["fallbacks"], rep.d["fla_calls"] = dict(W.FALLBACKS), dict(W.CALLS)  # must be {} / > 0
    rep.flush()
    # 3) capture the real WKV inputs at layers 0/30/60 on T=2048 text (chunked path), then compare kernels on them
    caught, calls = {}, [0]

    def capture(r, w_log, k, v, kk, a, state, cu_seq_lens=None, **kw):
        layer = calls[0] % m.config.num_hidden_layers
        calls[0] += 1
        if layer in CAPTURE_LAYERS:
            caught[layer] = {n: t.detach().clone() for n, t in zip(ARGS, (r, w_log, k, v, kk, a, state))}
        return M.rwkv7_parallel(r, w_log, k, v, kk, a, state, cu_seq_lens=cu_seq_lens)

    M.RWKV7_WKV_FUNCTIONS["ssdm_capture"] = capture
    for cfg in W._configs(m):
        cfg.wkv_implementation = W.WkvName("ssdm_capture")
    try:
        with torch.no_grad():
            st = m(input_ids=ids[:, :1024], use_cache=True, logits_to_keep=1).state
            calls[0] = 0
            m(input_ids=ids[:, 1024:], state=st, use_cache=True, logits_to_keep=1)  # non-zero carried state
    finally:
        M.RWKV7_WKV_FUNCTIONS.pop("ssdm_capture", None)
        _use(m, "chunked")
    del m
    torch.cuda.empty_cache()
    cap = Report(out / "captured.json", env=_env(), layers=list(CAPTURE_LAYERS))
    for layer, x in sorted(caught.items()):
        cap.d.setdefault("stats", {})[str(layer)] = {n: float(x[n].float().abs().max()) for n in ARGS}
        _cmp_case(cap, f"real_L{layer}_fwd_T1024", "bfloat16", x, grad=False)
        _cmp_case(cap, f"real_L{layer}_grad_T512", "bfloat16", _slice(x, 0, 512), grad=True)


# ----------------------------------------------------------------------------- RULE v2 phases (mode truth_v2)
def truth_grad(x: dict, seed: int = 1) -> dict:
    """fp64 token-serial run WITH gradients: autograd through ``rwkv7_recurrent`` at fp64 on the same (bf16-rounded)
    inputs, with the same cotangents as ``run_impl`` (same generator, same draws, cast to fp64)."""
    torch = _torch()
    from third_party.rwkv7_hf import modeling_rwkv7 as M

    leaves = {n: x[n].detach().double().requires_grad_(True) for n in ARGS}
    o, s = M.rwkv7_recurrent(*(leaves[n] for n in ARGS), compute_dtype=torch.float64)
    g = torch.Generator(device="cpu").manual_seed(seed)
    do = torch.randn(o.shape, generator=g).to(o.device)
    ds = torch.randn(s.shape, generator=g).to(s.device)
    ((o * do.double()).sum() + (s * ds.double()).sum()).backward()
    res = {"out": o.detach(), "state": s.detach()}
    for n in ARGS:
        res[f"grad_{n}"] = _grad(leaves[n])
    return res


def capture_wkv_inputs(m, ids, layers, tail: int) -> dict:
    """job-jn47u's construction: ids[:, :-tail] builds the carried state, the WKV inputs of ``layers`` are captured
    during ids[:, -tail:] (chunked path, non-zero carried state)."""
    torch = _torch()
    from heads import rwkv7_wkv as W
    from third_party.rwkv7_hf import modeling_rwkv7 as M

    caught, calls = {}, [0]

    def capture(r, w_log, k, v, kk, a, state, cu_seq_lens=None, **kw):
        layer = calls[0] % m.config.num_hidden_layers
        calls[0] += 1
        if layer in layers:
            caught[layer] = {n: t.detach().clone() for n, t in zip(ARGS, (r, w_log, k, v, kk, a, state))}
        return M.rwkv7_parallel(r, w_log, k, v, kk, a, state, cu_seq_lens=cu_seq_lens)

    _use(m, "chunked")
    with torch.no_grad():
        st = m(input_ids=ids[:, :-tail], use_cache=True, logits_to_keep=1).state
    M.RWKV7_WKV_FUNCTIONS["ssdm_capture_v2"] = capture
    for cfg in W._configs(m):
        cfg.wkv_implementation = W.WkvName("ssdm_capture_v2")
    try:
        with torch.no_grad():
            m(input_ids=ids[:, -tail:], state=st, use_cache=True, logits_to_keep=1)
    finally:
        M.RWKV7_WKV_FUNCTIONS.pop("ssdm_capture_v2", None)
        _use(m, "chunked")
    return caught


#: the training-row shape table (jobs/fla_rwkv7/train_mix_v2.json); None = that committed file (tests point it elsewhere)
MIX_PATH = None
#: truth_v2 implementations: (name in truth_v2.json, run_impl name)
V2_IMPLS = (("eager_bf16", "eager"), ("chunked", "chunked"), ("fla", "fla"), ("fla_safe_gate", "fla_safe_gate"))


def _truth_v2_case(rep: Report, name: str, layer: int, x: dict, grad: bool):
    torch = _torch()
    tensors = V2_GRAD_TENSORS if grad else V2_FWD_TENSORS
    t0 = time.perf_counter()
    tr = truth_grad(x) if grad else truth(x)
    case = {"name": name, "layer": layer, "grad": grad, "shape": list(x["r"].shape), "tensors": list(tensors),
            "truth_s": round(time.perf_counter() - t0, 3),
            "truth_finite": all(bool(tr[t].isfinite().all().item()) for t in tensors), "impls": {}}
    if DEVICE == "cuda":
        case["truth_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    for label, impl in V2_IMPLS:
        try:
            got = run_impl(impl, x, grad)
            case["impls"][label] = {"e": {t: rel_err(got[t], tr[t]) for t in tensors},
                                    "cos": {t: cosine(got[t], tr[t]) for t in tensors},
                                    "finite": all(bool(got[t].isfinite().all().item()) for t in tensors)}
        except Exception as e:  # noqa: BLE001  recorded; decide_v2: fla error -> reject, reference error -> inconclusive
            case["impls"][label] = {"error": f"{type(e).__name__}: {e}"[:500]}
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    rep.add(case)


def phase_truth_v2(out: Path, model_dir: str):
    """RULE v2 numerics: capture the real WKV inputs at TRUTH_V2_LAYERS (job-jn47u prompt and construction), then every
    implementation vs the fp64 token-serial truth on those exact bf16-rounded inputs (fwd T=V2_FWD_T, grads on the
    first V2_GRAD_T tokens)."""
    torch = _torch()
    m = _load(model_dir)
    ids = _text_ids(model_dir, V2_PROMPT_T)
    caught = capture_wkv_inputs(m, ids, TRUTH_V2_LAYERS, V2_FWD_T)
    del m
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    rep = Report(out / "truth_v2.json", env=_env(), layers=list(TRUTH_V2_LAYERS), prompt_t=V2_PROMPT_T,
                 fwd_t=V2_FWD_T, grad_t=V2_GRAD_T, captured=sorted(caught), complete=False)
    for layer, x in sorted(caught.items()):
        rep.d.setdefault("stats", {})[str(layer)] = {n: float(x[n].float().abs().max()) for n in ARGS}
        _truth_v2_case(rep, f"real_L{layer}_fwd_T{V2_FWD_T}", layer, x, grad=False)
        _truth_v2_case(rep, f"real_L{layer}_grad_T{V2_GRAD_T}", layer, _slice(x, 0, V2_GRAD_T), grad=True)
    rep.d["complete"] = True
    rep.flush()


def _load_backbone(model_dir: str):
    from heads.backbone import load_rwkv7_hf_backbone

    return load_rwkv7_hf_backbone(model_dir, device=DEVICE, fork_mode="cache", wkv=None)


def phase_train_v2(out: Path, model_dir: str, impl: str, deadline_s: float):
    """RULE v2 training speed: the m2 first cell's own optimiser step (``train.trainer.Trainer.train_step`` with the
    cell's TrainConfig via ``train.run_cell.train_config``: fork head, cache fork, LoRA, Clef loss, batch x grad_accum,
    bf16 autocast, no checkpointing) on rows with the shapes of the cell's first steps (``train_mix_v2.json``).
    V2_TRAIN_WARMUP untimed steps, then V2_TRAIN_STEPS timed ones. An OOM step is recorded (0 tok/s in decide_v2) and
    the run goes on; a step that cannot start before ``deadline_s`` (from process start) is recorded as skipped."""
    t_start = time.perf_counter()
    torch = _torch()
    import gc

    from heads import rwkv7_wkv as W
    from heads.model import DecisionModel
    from jobs.fla_rwkv7 import train_mix
    from train.guard import SplitGuard
    from train.run_cell import train_config
    from train.trainer import Trainer

    mix = train_mix.load(MIX_PATH)
    cell = mix["cell"]
    n_steps = V2_TRAIN_WARMUP + V2_TRAIN_STEPS
    per_step = cell["batch_size"] * cell["grad_accum"]
    if len(mix["rows"]) < n_steps * per_step:
        raise SystemExit(f"train_mix_v2.json has {len(mix['rows'])} rows < {n_steps} x {per_step}")
    rep = Report(out / f"train_v2_{impl}.json", env=_env(), impl=impl, mix_sha256=mix["sha256"], cell=cell,
                 deadline_s=deadline_s, steps=[], complete=False)
    W.FALLBACKS.clear()
    W.CALLS.clear()
    cuda = DEVICE == "cuda"
    bb = _load_backbone(model_dir)
    from heads.rwkv7_wkv import select_wkv

    select_wkv(bb.model, impl, allow_unverified=True)
    rep.d["load_s"] = round(time.perf_counter() - t_start, 1)
    rep.d["weights_gb"] = torch.cuda.memory_allocated() / 1e9 if cuda else None
    items = train_mix.rows_from_shapes(mix["rows"][: n_steps * per_step], vocab=bb.model.config.vocab_size, seed=0)
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    model = DecisionModel(bb, cell["head"], cell.get("head_kwargs") or {}, pad_id=0, seed=cell["seed"])
    model.to(bb.device)
    tcfg = train_config({**cell, "ckpt_every": 0, "eval_every": 0}, n_steps, None, cuda)
    tr = Trainer(model, tcfg, items, [], guard=SplitGuard.synthetic_only("fla_rwkv7 truth_v2 train speed: "
                                                                         "synthetic ids at the m2 cell's row shapes"))
    tr._order = lambda epoch: list(range(len(items)))  # the mix is already in the cell's epoch-0 order
    rep.d["setup_s"] = round(time.perf_counter() - t_start, 1)
    rep.flush()
    done = []
    for i in range(n_steps):
        rows = items[i * per_step:(i + 1) * per_step]
        rec = {"i": i, "timed": i >= V2_TRAIN_WARMUP, "tokens": sum(len(x.row.ids) for x in rows),
               "cont_tokens": sum(sum(map(len, x.row.continuations)) for x in rows),
               "max_T": max(len(x.row.ids) for x in rows), "K": [x.row.n_options for x in rows]}
        left = deadline_s - (time.perf_counter() - t_start)
        if left <= 0 or (done and left < 1.5 * max(done)):
            rec["skipped"] = "deadline"
            rep.d["steps"].append(rec)
            rep.flush()
            continue
        tr.epoch, tr.cursor = 0, i * per_step
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        try:
            loss = tr.train_step()
            if cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t
            done.append(dt)
            rec.update(s=dt, tok_s=rec["tokens"] / dt, loss=loss, finite=math.isfinite(loss))
        except torch.OutOfMemoryError as e:
            rec.update(oom=True, s=time.perf_counter() - t, error=str(e)[:200])
        if "oom" in rec:
            tr.opt.zero_grad(set_to_none=True)
            gc.collect()
            if cuda:
                torch.cuda.empty_cache()
        rep.d["steps"].append(rec)
        rep.d["peak_gb"] = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0
        rep.d["fallbacks"], rep.d["fla_calls"] = dict(W.FALLBACKS), dict(W.CALLS)
        rep.flush()
    timed = [s for s in rep.d["steps"] if s["timed"] and "tok_s" in s]
    rep.d["median_tok_s_completed"] = statistics.median([s["tok_s"] for s in timed]) if timed else None
    rep.d["wall_s"] = round(time.perf_counter() - t_start, 1)
    rep.d["complete"] = True
    rep.flush()


def _timed(fn, reps: int = REPS) -> list[float]:
    torch = _torch()
    ts = []
    for _ in range(reps + 1):  # first = warm-up (fla autotune / compile)
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return ts[1:]


def phase_throughput(out: Path, model_dir: str):
    torch = _torch()
    from heads import rwkv7_wkv as W

    rep = Report(out / "throughput.json", env=_env(), prefill={}, train={}, kernel={})
    W.FALLBACKS.clear()
    W.CALLS.clear()
    m = _load(model_dir)
    impls = ("chunked",) + VARIANTS
    g = torch.Generator(device="cpu").manual_seed(0)
    for L in LENGTHS:
        ids = torch.randint(1, m.config.vocab_size, (1, L), generator=g).cuda()
        row = rep.d["prefill"].setdefault(str(L), {})
        for impl in impls:
            _use(m, impl)
            torch.cuda.reset_peak_memory_stats()
            try:
                with torch.no_grad():
                    ts = _timed(lambda: m(input_ids=ids, logits_to_keep=1, use_cache=False))
                med = statistics.median(ts)
                row[impl] = {"tok_s": L / med, "s_median": med, "s_all": ts, "spread": (max(ts) - min(ts)) / med,
                             "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
            except Exception as e:  # noqa: BLE001  (OOM included)
                row[impl] = {"error": f"{type(e).__name__}: {e}"[:300]}
            torch.cuda.empty_cache()
            rep.flush()
    # training-shaped step: frozen weights, grad w.r.t. the embeddings (the LoRA backward runs the same activations)
    ids = torch.randint(1, m.config.vocab_size, (1, TRAIN_T), generator=g).cuda()
    for impl in impls:
        _use(m, impl)
        torch.cuda.reset_peak_memory_stats()

        def step():
            e = m.rwkv7.emb(ids).detach().requires_grad_(True)
            h = m.rwkv7(inputs_embeds=e, use_cache=False).last_hidden_state
            h.float().pow(2).mean().backward()

        try:
            ts = _timed(step)
            med = statistics.median(ts)
            rep.d["train"][impl] = {"tok_s": TRAIN_T / med, "s_median": med, "s_all": ts,
                                    "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
        except Exception as e:  # noqa: BLE001
            rep.d["train"][impl] = {"error": f"{type(e).__name__}: {e}"[:300]}
        torch.cuda.empty_cache()
        rep.flush()
    _use(m, "chunked")
    del m
    torch.cuda.empty_cache()
    # attribution: the WKV alone at the same lengths (B=1, 64x64 heads), fwd and fwd+bwd
    for L in LENGTHS:
        x = make_inputs(1, L, torch.bfloat16, 41)
        row = rep.d["kernel"].setdefault(str(L), {})
        for impl in impls:
            try:
                f = _timed(lambda: run_impl(impl, x, False))
                b = _timed(lambda: run_impl(impl, x, True))
                row[impl] = {"fwd_ms": 1e3 * statistics.median(f), "fwdbwd_ms": 1e3 * statistics.median(b)}
            except Exception as e:  # noqa: BLE001
                row[impl] = {"error": f"{type(e).__name__}: {e}"[:300]}
            torch.cuda.empty_cache()
            rep.flush()
    rep.d["fallbacks"] = dict(W.FALLBACKS)  # must stay empty: a fallback would time "chunked" under the fla name
    rep.d["fla_calls"] = dict(W.CALLS)  # must be > 0 for every variant
    rep.flush()


def _env() -> dict:
    torch = _torch()
    d = {"torch": torch.__version__, "cuda": torch.version.cuda}
    try:
        import fla
        import triton

        d.update(fla=fla.__version__, triton=triton.__version__)
    except Exception as e:  # noqa: BLE001
        d["fla_error"] = str(e)[:200]
    if torch.cuda.is_available():
        d["gpu"] = torch.cuda.get_device_name(0)
    return d


def _read(out: Path, name: str):
    p = out / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("warmup", "kernel", "fork", "model", "throughput", "fp32", "decide",
                                      "truth_v2", "train_v2", "decide_v2"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir")
    ap.add_argument("--impl", choices=(V2_CURRENT,) + VARIANTS, help="train_v2: the WKV implementation timed")
    ap.add_argument("--deadline-s", type=float, default=math.inf, help="train_v2: no step starts after this")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.phase == "decide_v2":
        rep = {k: _read(out, k) for k in ("kernel", "fork", "model", "throughput", "truth_v2")}
        for k in ("model", "throughput"):
            if rep[k] and not _fla_really_ran(rep[k]):
                rep[k] = None  # -> INCONCLUSIVE
        rep["train_v2"] = {i: _read(out, f"train_v2_{i}") for i in (V2_CURRENT,) + VARIANTS}
        d = decide_v2(rep)
        (out / "decision_v2.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
        print(json.dumps({"rule": "v2", "verdict": d["verdict"],
                          "variants": {v: d["variants"][v]["verdict"] for v in VARIANTS}}))
        return 0 if valid_decision_v2(d) else 9
    if a.phase in ("truth_v2", "train_v2"):
        if not a.model_dir:
            ap.error("--model-dir is required")
        if a.phase == "truth_v2":
            phase_truth_v2(out, a.model_dir)
        else:
            if not a.impl:
                ap.error("--impl is required")
            phase_train_v2(out, a.model_dir, a.impl, a.deadline_s)
        return 0
    if a.phase == "decide":
        rep = {k: _read(out, k) for k in ("kernel", "fork", "captured", "model", "throughput", "fp32")}
        for k in ("model", "throughput"):
            if rep[k] and not _fla_really_ran(rep[k]):
                rep[k] = None  # -> INCONCLUSIVE
        d = decide(rep)
        (out / "decision.json").write_text(json.dumps(d, indent=1), encoding="utf-8")
        print(json.dumps({"verdict": d["verdict"], "chosen": d["chosen"]}))
        return 0 if valid_decision(d) else 9
    if a.phase in ("model", "throughput") and not a.model_dir:
        ap.error("--model-dir is required")
    {"warmup": lambda: phase_warmup(out), "fp32": lambda: phase_fp32(out),
     "kernel": lambda: phase_kernel(out), "fork": lambda: phase_fork(out),
     "model": lambda: phase_model(out, a.model_dir), "throughput": lambda: phase_throughput(out, a.model_dir)}[a.phase]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
