"""Mode ``infer_equiv`` of jobs/fla_rwkv7: INFERENCE-ONLY Index-level equivalence of fla ``chunk_rwkv7`` vs the
production ``chunked`` WKV (analysis/FLA_WKV_NUMERICS_2026-10-08.md, "Recommendation (a)"; spec in
jobs/fla_rwkv7/JOB_SPEC.md "Mode infer_equiv"). Rule id ``fla-infer-equiv-v1``, pre-registered here, pinned by
tests/test_fla_infer_equiv.py.

    # GPU (fla_job.sh infer_equiv)
    python -m jobs.fla_rwkv7.infer_equiv warmup --out <out>
    python -X utf8 -m jobs.fla_rwkv7.infer_equiv score --out <out> --model-dir <snap> --rows <subset-rows.jsonl.gz> \
        --deadline-s S --progress <out>/progress.jsonl
    python -X utf8 -m jobs.common.dev_subset compare --candidate <out>/fla.results.jsonl \
        --baseline <out>/chunked.results.jsonl > <out>/compare.json          # frozen harness score + bootstrap
    python -m jobs.fla_rwkv7.infer_equiv decide --out <out>                  # -> decision_infer.json

Setup. ONE process loads stock RWKV7-G1k-13.3B once, builds the production readout exactly as
``jobs.common.readout_engine.ReadoutIndexEngine`` does for family ``rwkv7-hf-g1-v2`` (``G1V2Backend``, template
``jrt-rwkv-g1-v2``, the job-zhyvg stock readout), and scores every row of the 2,008-row DEV subset through BOTH WKV
paths, interleaved per row: the WKV is switched on the one loaded model by config (``heads.rwkv7_wkv.select_wkv``)
before each call. The order alternates (even rows chunked first, odd rows fla first) so warm-cache and clock effects
do not favour one path. Each arm's results line is ``train.m2_gates.index_predict``'s kit format (what the frozen
scorer reads); the per-question raw (T = 1) probabilities of both arms go to ``pairs.jsonl``.

GPU_VERIFIED. fla names refuse unless ``heads.rwkv7_wkv.GPU_VERIFIED``. This mode does NOT flip it: ``select_arm``
passes ``allow_unverified=True`` only when ``SSDM_FLA_INFER_EQUIV=1``, which only ``fla_job.sh infer_equiv`` exports.
No training path, loader default or cell config changes; the training gate (rule v2) is untouched.

Verdict ``EQUIVALENT_FOR_INFERENCE`` iff ALL hold (else ``NOT_EQUIVALENT``; ``INCONCLUSIVE`` when the measurement is
incomplete: rows missing, no compare, or fla never ran):
  * argmax agreement >= AGREE_MIN of questions (raw probabilities; a row whose status differs between the arms counts
    every one of its questions as a disagreement);
  * zero flips where the reference (chunked) top-1 minus top-2 margin > MARGIN_FLIP (status-mismatch questions count);
  * the paired Delta-Index (fla - chunked) 95% bootstrap CI from ``dev_subset compare`` (harness.score.decompose +
    harness.bootstrap.compare, never re-implemented) contains 0 with max(|lo|, |hi|) <= CI_BOUND_MAX;
  * every probability, wall time and CI value finite;
  * zero fla fallbacks (heads.rwkv7_wkv.FALLBACKS) and fla actually called on every ok fla row.
Reported, not gating: max / p99 |dp|, flips by length bucket (<2k, 2-8k, >=8k tokens of the row's longest model call),
rows/min per path. ``adopt_for_inference`` = EQUIVALENT_FOR_INFERENCE and rows/min ratio (fla / chunked) >=
SPEED_ADOPT.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

SCHEMA = "ssdm.fla_rwkv7.infer_equiv.v1"
RULE_ID = "fla-infer-equiv-v1"
REF, CAND = "chunked", "fla"
ARMS = (REF, CAND)
FAMILY = "rwkv7-hf-g1-v2"
TEMPLATE = "jrt-rwkv-g1-v2"
AGREE_MIN = 0.99
MARGIN_FLIP = 0.05
CI_BOUND_MAX = 0.5
SPEED_ADOPT = 2.0
LENGTH_EDGES = (2048, 8192)
BUCKETS = ("<2k", "2-8k", ">=8k", "unknown")
VERDICTS = ("EQUIVALENT_FOR_INFERENCE", "NOT_EQUIVALENT", "INCONCLUSIVE")
ENV_FLAG = "SSDM_FLA_INFER_EQUIV"
#: fwd-only fla specialisations the readout hits: Triton specialises int args on ==1 and %16==0, so warm one of each
#: kind of T, and B > 1 (the packed fork tails); the first call's T fixes fla's cached autotune config (2048 first)
WARMUP_SHAPES = ((1, 2048), (1, 1000), (1, 1), (8, 1000))
WARM_REQUEST = {"state": "The color is red.",
                "questions": {"warmup": {"type": "choice", "instructions": "Which color is named?",
                                         "criteria": {"red": "red", "blue": "blue"}}}}


# ------------------------------------------------------------------------------------------- WKV selection (scoped)
def select_arm(model, arm: str) -> str:
    """Switch the loaded model's WKV. fla only inside this job mode (``SSDM_FLA_INFER_EQUIV=1``); GPU_VERIFIED is
    never flipped here."""
    from heads.rwkv7_wkv import select_wkv

    if arm == REF:
        return select_wkv(model, "chunked")
    if arm != CAND:
        raise ValueError(f"unknown arm {arm!r}")
    if os.environ.get(ENV_FLAG) != "1":
        raise RuntimeError(f"fla is not GPU-verified; the unverified override is scoped to fla_job.sh infer_equiv "
                           f"({ENV_FLAG}=1)")
    return select_wkv(model, CAND, allow_unverified=True)


# ------------------------------------------------------------------------------------------------ GPU: warmup
def phase_warmup(out: Path) -> dict:
    """fla JIT compile + autotune, forward only, variant "fla" (no grads, no packed varlen: the readout uses neither)."""
    import torch

    from jobs.fla_rwkv7 import fla_equivalence as E

    rep = {"schema": SCHEMA + ".warmup", "env": E._env(), "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
           "cases": []}
    for B, T in WARMUP_SHAPES:
        case = {"B": B, "T": T}
        try:
            x = E.make_inputs(B, T, torch.bfloat16, 1)
            ts = []
            for _ in range(2):
                torch.cuda.synchronize()
                t = time.perf_counter()
                with torch.inference_mode():
                    E.run_impl(CAND, x, False, None, allow_fp32=False)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
            case.update(first_s=ts[0], second_s=ts[1], compile_s=ts[0] - ts[1])
        except Exception as e:  # noqa: BLE001  recorded; score then measures the first call instead
            case["error"] = f"{type(e).__name__}: {e}"[:500]
        rep["cases"].append(case)
        (out / "warmup_infer.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep


# ------------------------------------------------------------------------------------------------ GPU: score
def _finite_probs(raw: dict | None) -> bool:
    if not isinstance(raw, dict):
        return False
    for d in raw.values():
        if not isinstance(d, dict) or not d:
            return False
        for v in d.values():
            if not (isinstance(v, (int, float)) and math.isfinite(v) and -1e-9 <= v <= 1 + 1e-9):
                return False
    return True


def _arm_call(decider, model, arm: str, row: dict) -> tuple[dict, dict]:
    """One row through one WKV arm: (kit results line, pair record)."""
    from heads import rwkv7_wkv as W
    from train.m2_gates import index_predict

    select_arm(model, arm)
    c0, f0 = sum(W.CALLS.values()), sum(W.FALLBACKS.values())
    box: list[dict] = []

    def decide(req):
        resp = decider.decide(req)
        box.append(resp)
        return resp

    line = index_predict(decide, [row])[row["_evaluation"]["run_id"]]
    ss = (box[-1].get("ssdm") or {}) if box else {}
    diag = ss.get("diagnostics") or {}
    rec = {"status": line["status"], "wall_ms": line["total_wall_ms"],
           "max_call_input_tokens": (diag.get("input_counts") or {}).get("max_call_input_tokens"),
           "raw": ss.get("raw_probabilities") if line["status"] == "ok" else None,
           "fla_calls": sum(W.CALLS.values()) - c0, "fallbacks": sum(W.FALLBACKS.values()) - f0,
           "wkv": str(model.config.wkv_implementation)}
    return line, rec


def run_pairs(decider, model, rows: list[dict], out: Path, deadline_s: float = math.inf,
              progress: str | None = None, warm: bool = True, clock: Callable[[], float] = time.monotonic) -> dict:
    """Interleaved scoring of ``rows`` through both arms on ONE loaded model. Stops before a row that would start
    after ``deadline_s`` (seconds from the call) -> incomplete -> INCONCLUSIVE, never a partial verdict."""
    from heads import rwkv7_wkv as W
    from train import m2_gates as G

    out.mkdir(parents=True, exist_ok=True)
    t0 = clock()
    if warm:  # one unrecorded call per arm (CUDA context, allocator, first-call compiles)
        for arm in ARMS:
            select_arm(model, arm)
            decider.decide({"model": FAMILY, **WARM_REQUEST})
    prog = G.Progress(progress)
    prog.plan([G.unit_weight(r) for r in rows])
    results: dict[str, dict[str, dict]] = {a: {} for a in ARMS}
    deadline_hit = False
    with open(out / "pairs.jsonl", "w", encoding="utf-8", newline="\n") as fp:
        for i, row in enumerate(rows):
            if clock() - t0 > deadline_s:
                deadline_hit = True
                break
            rid = row["_evaluation"]["run_id"]
            order = ARMS if i % 2 == 0 else ARMS[::-1]
            rec = {"run_id": rid, "i": i, "order": list(order), "n_questions": len(row["questions"])}
            for arm in order:
                line, r = _arm_call(decider, model, arm, row)
                results[arm][rid] = line
                rec[arm] = r
            fp.write(json.dumps(rec, separators=(",", ":")) + "\n")
            fp.flush()
            if prog.path:
                with open(prog.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"w": G.unit_weight(row)}) + "\n")
    select_arm(model, REF)
    ids = [r["_evaluation"]["run_id"] for r in rows]
    for arm in ARMS:
        with open(out / f"{arm}.results.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for rid in ids:  # DEV row order, completed rows only
                if rid in results[arm]:
                    f.write(json.dumps(results[arm][rid], separators=(",", ":")) + "\n")
    meta = {"schema": SCHEMA + ".score", "rows_total": len(rows), "rows_done": len(results[REF]),
            "complete": len(results[REF]) == len(results[CAND]) == len(rows), "deadline_hit": deadline_hit,
            "deadline_s": deadline_s if math.isfinite(deadline_s) else None, "elapsed_s": clock() - t0,
            "family": FAMILY, "template": TEMPLATE, "wkv_provenance": W.provenance(),
            "gpu_verified": W.GPU_VERIFIED, "override": f"{ENV_FLAG}=1 (select_arm, allow_unverified)"}
    try:
        meta["revision_id"] = decider.revision().get("revision_id")
    except Exception:  # noqa: BLE001
        meta["revision_id"] = None
    (out / "score_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return meta


def build_production(model_dir: str):
    """The production stock readout (job-zhyvg's): ReadoutIndexEngine(family=rwkv7-hf-g1-v2) on the snapshot."""
    from jobs.common.readout_engine import ReadoutIndexEngine

    eng = ReadoutIndexEngine(family=FAMILY, model_dir=model_dir)
    model = eng.backend.plugin.state_ops.model
    if str(model.config.wkv_implementation) != "chunked":
        raise RuntimeError(f"checkpoint WKV is {model.config.wkv_implementation!r}, expected 'chunked'")
    return eng.decider, model


def phase_score(out: Path, model_dir: str, rows_path: str, deadline_s: float, progress: str | None) -> dict:
    from harness.suite import read_rows

    rows = read_rows(rows_path)
    t = time.monotonic()
    decider, model = build_production(model_dir)
    load_s = time.monotonic() - t
    meta = run_pairs(decider, model, rows, out, deadline_s - load_s, progress)
    meta["load_s"] = load_s
    (out / "score_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    return meta


# ------------------------------------------------------------------------------------------------ CPU: metrics + rule
def argmax(p: dict[str, float]) -> str:
    return max(sorted(p), key=lambda o: p[o])  # tie -> smallest id, as decide.schema.wire_answer


def margin(p: dict[str, float]) -> float:
    v = sorted(p.values(), reverse=True)
    return v[0] - v[1] if len(v) > 1 else 1.0


def bucket(n) -> str:
    if not isinstance(n, (int, float)):
        return "unknown"
    return "<2k" if n < LENGTH_EDGES[0] else "2-8k" if n < LENGTH_EDGES[1] else ">=8k"


def pctl(xs: list[float], q: float):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))]  # nearest rank


def summarize_pairs(pairs: list[dict]) -> dict[str, Any]:
    nq = agree = hi_flips = flips = 0
    dps: list[float] = []
    worst = None
    by_bucket = {b: {"questions": 0, "flips": 0, "high_margin_flips": 0} for b in BUCKETS}
    status_mismatch = both_not_ok = fallbacks = fla_calls = fla_ok_rows_without_calls = 0
    finite = True
    wall = {a: 0.0 for a in ARMS}
    rows_both_ok = 0
    for p in pairs:
        r, c = p[REF], p[CAND]
        fallbacks += int(c.get("fallbacks") or 0) + int(r.get("fallbacks") or 0)
        fla_calls += int(c.get("fla_calls") or 0)
        for a in ARMS:
            w = p[a].get("wall_ms")
            if not (isinstance(w, (int, float)) and math.isfinite(w)):
                finite = False
        if c["status"] == "ok" and not c.get("fla_calls"):
            fla_ok_rows_without_calls += 1
        b = bucket(r.get("max_call_input_tokens") if r.get("max_call_input_tokens") is not None
                   else c.get("max_call_input_tokens"))
        if r["status"] != c["status"]:
            status_mismatch += 1
            n = int(p.get("n_questions") or len((r.get("raw") or c.get("raw") or {})))
            nq += n
            flips += n
            hi_flips += n
            by_bucket[b]["questions"] += n
            by_bucket[b]["flips"] += n
            by_bucket[b]["high_margin_flips"] += n
            continue
        if r["status"] != "ok":
            both_not_ok += 1
            continue
        if not (_finite_probs(r.get("raw")) and _finite_probs(c.get("raw"))):
            finite = False
            continue
        rows_both_ok += 1
        for a in ARMS:
            wall[a] += float(p[a]["wall_ms"])
        for q, pr in r["raw"].items():
            pc = c["raw"].get(q)
            nq += 1
            by_bucket[b]["questions"] += 1
            if not isinstance(pc, dict) or set(pc) != set(pr):
                finite = False
                flips += 1
                hi_flips += 1
                by_bucket[b]["flips"] += 1
                by_bucket[b]["high_margin_flips"] += 1
                continue
            d = max(abs(float(pc[o]) - float(pr[o])) for o in pr)
            dps.append(d)
            if worst is None or d > worst["dp"]:
                worst = {"dp": d, "run_id": p["run_id"], "qid": q}
            if argmax(pr) == argmax(pc):
                agree += 1
            else:
                flips += 1
                by_bucket[b]["flips"] += 1
                if margin(pr) > MARGIN_FLIP:
                    hi_flips += 1
                    by_bucket[b]["high_margin_flips"] += 1
    rpm = {a: (rows_both_ok / (wall[a] / 60000.0) if wall[a] > 0 else None) for a in ARMS}
    return {"rows": len(pairs), "rows_both_ok": rows_both_ok, "rows_both_not_ok": both_not_ok,
            "rows_status_mismatch": status_mismatch, "questions": nq, "agree": agree,
            "agreement": agree / nq if nq else None, "flips": flips, "high_margin_flips": hi_flips,
            "max_abs_dp": max(dps) if dps else None, "p99_abs_dp": pctl(dps, 0.99), "max_abs_dp_at": worst,
            "by_length": by_bucket, "finite": finite, "fla_fallbacks": fallbacks, "fla_calls": fla_calls,
            "fla_ok_rows_without_calls": fla_ok_rows_without_calls,
            "wall_s": {a: wall[a] / 1000 for a in ARMS}, "rows_per_min": rpm,
            "speed_ratio": (rpm[CAND] / rpm[REF]) if rpm[CAND] and rpm[REF] else None}


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def decide(summary: dict | None, compare: dict | None, meta: dict | None) -> dict[str, Any]:
    """The pre-registered rule (module docstring). Pure."""
    s = summary or {}
    diff = ((compare or {}).get("diff") or {}).get("index") or {}
    lo, hi, pt = diff.get("lo"), diff.get("hi"), diff.get("point")
    ci_ok_num = all(_num(x) for x in (lo, hi, pt))
    missing = []
    if not meta or not meta.get("complete"):
        missing.append("rows incomplete (deadline or crash): no verdict on a partial subset")
    if not ci_ok_num:
        missing.append("no paired Delta-Index CI (dev_subset compare missing or failed)")
    if not s.get("questions"):
        missing.append("no scored questions")
    if not s.get("fla_calls"):
        missing.append("fla never ran")
    gates = {
        "agreement": {"value": s.get("agreement"), "min": AGREE_MIN,
                      "pass": _num(s.get("agreement")) and s["agreement"] >= AGREE_MIN},
        "high_margin_flips": {"value": s.get("high_margin_flips"), "margin": MARGIN_FLIP, "max": 0,
                              "pass": s.get("high_margin_flips") == 0},
        "delta_index_ci": {"point": pt, "lo": lo, "hi": hi, "bound_max": CI_BOUND_MAX,
                           "pass": ci_ok_num and lo <= 0 <= hi and max(abs(lo), abs(hi)) <= CI_BOUND_MAX},
        "finite": {"probabilities_and_walls": bool(s.get("finite")), "ci": ci_ok_num and _num(diff.get("se")),
                   "pass": bool(s.get("finite")) and ci_ok_num and _num(diff.get("se"))},
        "fla_fallbacks": {"value": s.get("fla_fallbacks"), "ok_rows_without_fla_calls": s.get("fla_ok_rows_without_calls"),
                          "pass": s.get("fla_fallbacks") == 0 and s.get("fla_ok_rows_without_calls") == 0},
    }
    if missing:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "EQUIVALENT_FOR_INFERENCE" if all(g["pass"] for g in gates.values()) else "NOT_EQUIVALENT"
    ratio = s.get("speed_ratio")
    return {"schema": SCHEMA, "rule": RULE_ID, "verdict": verdict, "missing": missing, "gates": gates,
            "adopt_for_inference": verdict == "EQUIVALENT_FOR_INFERENCE" and _num(ratio) and ratio >= SPEED_ADOPT,
            "speed": {"rows_per_min": s.get("rows_per_min"), "ratio": ratio, "adopt_min": SPEED_ADOPT},
            "reported": {k: s.get(k) for k in ("max_abs_dp", "p99_abs_dp", "max_abs_dp_at", "by_length", "flips",
                                               "questions", "rows", "rows_both_ok", "rows_both_not_ok",
                                               "rows_status_mismatch", "wall_s")},
            "index": {"candidate_fla": ((compare or {}).get("candidate") or {}).get("index"),
                      "baseline_chunked": ((compare or {}).get("baseline") or {}).get("index"),
                      "diff": diff or None},
            "scope": "inference only (readout); the training gate (rule v2) and GPU_VERIFIED are unchanged",
            "score_meta": {k: (meta or {}).get(k) for k in ("rows_total", "rows_done", "complete", "deadline_hit",
                                                             "elapsed_s", "load_s", "revision_id", "wkv_provenance")}}


def valid_decision(d) -> bool:
    """jobs/common/finalize.py validator: a well-formed decision under this rule (INCONCLUSIVE is a valid outcome)."""
    try:
        return (d["schema"] == SCHEMA and d["rule"] == RULE_ID and d["verdict"] in VERDICTS
                and isinstance(d["gates"], dict) and set(d["gates"]) == {"agreement", "high_margin_flips",
                                                                          "delta_index_ci", "finite", "fla_fallbacks"}
                and isinstance(d["adopt_for_inference"], bool)
                and (d["verdict"] != "EQUIVALENT_FOR_INFERENCE" or all(g["pass"] for g in d["gates"].values())))
    except (KeyError, TypeError):
        return False


def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def phase_decide(out: Path) -> dict:
    pairs = []
    pp = out / "pairs.jsonl"
    if pp.is_file():
        for line in pp.read_text(encoding="utf-8").splitlines():
            try:
                pairs.append(json.loads(line))
            except ValueError:  # a torn last line (killed mid-write) is dropped; the run is then incomplete anyway
                pass
    d = decide(summarize_pairs(pairs), _read_json(out / "compare.json"), _read_json(out / "score_meta.json"))
    (out / "decision_infer.json").write_text(json.dumps(d, indent=1, default=str), encoding="utf-8")
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m jobs.fla_rwkv7.infer_equiv")
    ap.add_argument("phase", choices=("warmup", "score", "decide"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir")
    ap.add_argument("--rows")
    ap.add_argument("--deadline-s", type=float, default=math.inf)
    ap.add_argument("--progress", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.phase == "warmup":
        phase_warmup(out)
        return 0
    if a.phase == "score":
        if not (a.model_dir and a.rows):
            ap.error("score needs --model-dir and --rows")
        meta = phase_score(out, a.model_dir, a.rows, a.deadline_s, a.progress)
        print(json.dumps({k: meta[k] for k in ("rows_total", "rows_done", "complete", "deadline_hit")}))
        return 0
    d = phase_decide(out)
    print(json.dumps({"rule": RULE_ID, "verdict": d["verdict"], "adopt_for_inference": d["adopt_for_inference"],
                      "missing": d["missing"]}))
    return 0 if valid_decision(d) else 9


if __name__ == "__main__":
    sys.exit(main())
