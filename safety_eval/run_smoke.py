"""Pipeline-qualification smoke run (m0-c). Numbers from a <=1B stock model are NOT gate evidence.

    python -m safety_eval.run_smoke --model Qwen/Qwen3-0.6B --out receipts/2026-10-06-D-m0-steer

Writes: requests.jsonl (+ manifest), predictions.jsonl (one row per unit, all arms), gcg_attacks.jsonl,
summary.json (metrics with paired two-way cluster-bootstrap CIs), environment.json.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from . import attacks, data, gcg, metrics
from .request import sha256_obj


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def pick_gcg(requests: list[dict], clean_rows: dict, k: int) -> list[dict]:
    """Clean-correct dev requests, alternating families, in sha256(request_id) order (fixed, pre-declared)."""
    by = {"route": [], "pay": []}
    for r in sorted(requests, key=lambda r: data.h("gcg|" + r["request_id"])):
        row = clean_rows.get(r["request_id"])
        if row and row.get("correct"):
            by[r["_meta"]["family"]].append(r)
    out, i = [], 0
    while len(out) < k and (i < len(by["route"]) or i < len(by["pay"])):
        for f in ("route", "pay"):
            if i < len(by[f]) and len(out) < k:
                out.append(by[f][i])
        i += 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-route", type=int, default=120)
    ap.add_argument("--n-pay", type=int, default=80)
    ap.add_argument("--gcg-steer", type=int, default=12)
    ap.add_argument("--gcg-hijack", type=int, default=6)
    ap.add_argument("--gcg-steps", type=int, default=gcg.HP_SMOKE["steps"])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log = lambda s: print(s, flush=True)  # noqa: E731

    reqs = data.build(a.n_route, a.n_pay, split="dev")
    data.write_jsonl(reqs, out / "requests.jsonl")
    man = data.manifest(reqs)
    variants = attacks.build_variants(reqs)

    from .hf_decider import HFLetterDecider
    import torch
    dec = HFLetterDecider(a.model, device=a.device)
    log(f"loaded {a.model} rev {dec.revision}; {len(variants)} variants")

    t0 = time.time()
    rows = metrics.run(dec, variants, log=log)
    for r in rows:
        r["arm"] = "sanitize"
    t_main = time.time() - t0

    # control arm: sanitiser OFF on clean + role-spoof hijack variants (shows what the sanitiser buys)
    dec.sanitize = False
    ctrl = [v for v in variants if v["_meta"]["attack_family"] == "clean" or v["_meta"].get("role_spoof")]
    rows_ctrl = metrics.run(dec, ctrl, log=log)
    for r in rows_ctrl:
        r["arm"] = "no_sanitize"
    dec.sanitize = True

    # GCG-lite (per-request, white-box) on clean-correct dev requests
    clean_rows = {r["request_id"]: r for r in rows if r["attack_family"] == "clean"}
    eng = gcg.Engine(dec)
    hp = dict(gcg.HP_SMOKE, steps=a.gcg_steps)
    gcg_records, gcg_variants = [], []
    for objective, k in (("steer", a.gcg_steer), ("hijack", a.gcg_hijack)):
        for i, r in enumerate(pick_gcg(reqs, clean_rows, k)):
            prob = gcg.Problem(dec, r, objective)
            res = gcg.attack(eng, [prob], hp, seed=a.seed * 1000 + i, log=log)
            res.update(request_id=r["request_id"], objective=objective, family=r["_meta"]["family"],
                       final=gcg.order_outcome(prob, eng.eval_advs(prob, torch.tensor([res["adv_ids"]]))[0]))
            gcg_records.append(res)
            v = attacks.suffix_variant(r, res["suffix"], f"gcg_{objective}", "gcg")
            gcg_variants.append(v)
            log(f"[gcg] {objective} {r['request_id']}: steps {res['steps_run']} first_success {res['first_success_step']}"
                f" ({res['secs']}s)")
    rows_gcg = metrics.run(dec, gcg_variants, log=log)  # re-check through the unmodified decide() path
    for r in rows_gcg:
        r["arm"] = "sanitize"
    data.write_jsonl(gcg_records, out / "gcg_attacks.jsonl")

    all_rows = rows + rows_ctrl + rows_gcg
    data.write_jsonl(all_rows, out / "predictions.jsonl")
    main_rows = rows + rows_gcg
    summary = {
        "note": "PIPELINE QUALIFICATION ONLY (stock <=1B model, synthetic dev requests); not gate evidence.",
        "model": a.model, "model_revision": dec.revision,
        "arm_sanitize": metrics.summarise(main_rows, a.boot),
        "arm_no_sanitize_role_spoof": metrics.summarise(rows_ctrl, a.boot, families=["hijack"]),
        "gcg": {"n_attacks": len(gcg_records),
                "optimiser_success": sum(1 for g in gcg_records if g["first_success_step"]),
                "token_level_success": {o: sum(g["final"]["success"] for g in gcg_records if g["objective"] == o)
                                        for o in gcg.OBJECTIVES},
                "both_orders_success": {o: sum(g["final"]["both_orders"] for g in gcg_records if g["objective"] == o)
                                        for o in gcg.OBJECTIVES},
                "attempts": {o: sum(g["objective"] == o for g in gcg_records) for o in gcg.OBJECTIVES},
                "mean_secs": sum(g["secs"] for g in gcg_records) / max(len(gcg_records), 1),
                "total_seqs_forwarded": eng.n_seqs, "hp": hp},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    env = {
        "command": "python -m safety_eval.run_smoke " + " ".join(argv if argv is not None else sys.argv[1:]),
        "git_sha_at_run": git_sha(), "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "billed_usd": 0.0, "requests_manifest": man, "variants_sha256": sha256_obj([v["request_id"] for v in variants]),
        "predictions_sha256": sha256_obj(all_rows), "seed": a.seed, "data_seed": data.SEED,
        "boot_n": a.boot, "boot_seed": 20261006, "wall_secs_total": round(time.time() - t_start, 1),
        "wall_secs_main_arm": round(t_main, 1), "n_units": {"sanitize": len(rows), "no_sanitize": len(rows_ctrl),
                                                           "gcg": len(rows_gcg)},
    }
    (out / "environment.json").write_text(json.dumps(env, indent=1), encoding="utf-8")
    log(json.dumps({k: summary[k] for k in ("model", "gcg")}, indent=1))


if __name__ == "__main__":
    main()
