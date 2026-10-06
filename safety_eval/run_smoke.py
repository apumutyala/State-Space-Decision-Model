"""Smoke / gate runner. Smoke numbers from a stock <=1B model are NOT gate evidence.

Canonical path (gated numbers): ``--backend decide`` drives Track D's decide() adapter (decide/, PR #6) through
safety_eval.decide_bridge, so untrusted-text encoding and refusal are exactly production's.

    set SSDM_QWEN_DIR=...\\ssdm-data\\hf\\models--Qwen--Qwen3.5-0.8B\\snapshots\\2fc06364715b967f1860aea9cf38778875588b17
    set SSDM_DECIDE_PATH=...\\ssdm-wt\\m0-decide-adapter          # until decide/ is on agent-main
    python -m safety_eval.run_smoke --backend decide --out receipts/<date>-D-<slug>

Letter-logit path (``--backend letter``): HFLetterDecider, needed for GCG-lite gradients and for the encoder
control arms (zwsp / none) on role-spoof variants. Default model: $SSDM_QWEN_DIR, else Qwen/Qwen3-0.6B.

Default device is CPU. D1 split B (held out) is scored only with ``--final`` (once per frozen checkpoint).

Writes: requests.jsonl, predictions.jsonl (one row per unit, all arms), gcg_attacks.jsonl, summary.json
(metrics with paired two-way cluster-bootstrap CIs, D1-A/D1-B, arm-vs-arm paired differences),
environment.json (command, git sha, model file sha256s, data hashes, seeds).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from . import attacks, data, gcg, metrics
from .decide_bridge import ENV_QWEN_DIR, snapshot_files_sha
from .request import sha256_obj


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def resolve_model_dir(model: str) -> Path | None:
    p = Path(model)
    if p.is_dir():
        return p
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(model, local_files_only=True))
    except Exception:  # noqa: BLE001
        return None


def default_model() -> str:
    return os.environ.get(ENV_QWEN_DIR) or "Qwen/Qwen3-0.6B"


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
    ap.add_argument("--backend", choices=["decide", "letter"], default="decide")
    ap.add_argument("--model", default=None, help="model dir or HF id (default $SSDM_QWEN_DIR, else Qwen/Qwen3-0.6B)")
    ap.add_argument("--family", default="qwen3.5", help="decide backend family: qwen3.5 | rwkv7 | mamba3")
    ap.add_argument("--tokenizer-dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default=None, help="default float32 on cpu, bfloat16 otherwise")
    ap.add_argument("--n-route", type=int, default=120)
    ap.add_argument("--n-pay", type=int, default=80)
    ap.add_argument("--final", action="store_true", help="also score held-out D1-B (frozen checkpoint only)")
    ap.add_argument("--controls", action="store_true", help="letter backend: zwsp/none encoder arms on role spoofs")
    ap.add_argument("--gcg-steer", type=int, default=0)
    ap.add_argument("--gcg-hijack", type=int, default=0)
    ap.add_argument("--gcg-steps", type=int, default=gcg.HP_SMOKE["steps"])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log = lambda s: print(s, flush=True)  # noqa: E731
    model = a.model or default_model()
    dtype = a.dtype or ("float32" if a.device == "cpu" else "bfloat16")
    if (a.gcg_steer or a.gcg_hijack or a.controls) and a.backend != "letter":
        ap.error("GCG-lite and encoder control arms need --backend letter (token-level access)")

    reqs = data.build(a.n_route, a.n_pay, split="dev")
    data.write_jsonl(reqs, out / "requests.jsonl")
    man = data.manifest(reqs)
    variants = attacks.build_variants(reqs, final=a.final)

    import torch
    model_dir = resolve_model_dir(model)
    files_sha = snapshot_files_sha(model_dir) if model_dir else {"note": "model dir not resolvable"}
    if a.backend == "decide":
        from .decide_bridge import build_hf_bridge
        dec = build_hf_bridge(str(model_dir or model), family=a.family, device=a.device, dtype=dtype,
                              tokenizer_dir=a.tokenizer_dir)
        revision = dec.revision()
        arm = "canonical_decide"
    else:
        from .hf_decider import HFLetterDecider
        dec = HFLetterDecider(str(model_dir or model), device=a.device, dtype=dtype)
        revision = {"model": model, "commit": dec.revision, "untrusted_mode": dec.untrusted_mode,
                    "untrusted_encoder": dec._canonical.identity()}
        arm = f"letter_{dec.untrusted_mode}"
    log(f"loaded {model} ({a.backend}, {a.device}/{dtype}); {len(variants)} variants")

    t0 = time.time()
    rows = metrics.run(dec, variants, log=log)
    for r in rows:
        r["arm"] = arm
    t_main = time.time() - t0
    arms = {arm: rows}

    if a.controls:
        ctrl = [v for v in variants if v["_meta"]["attack_family"] == "clean" or v["_meta"].get("role_spoof")]
        base_mode = dec.untrusted_mode
        for mode in ("zwsp", "none"):
            dec.untrusted_mode = mode
            rc = metrics.run(dec, ctrl, log=log)
            for r in rc:
                r["arm"] = f"letter_{mode}"
            arms[f"letter_{mode}"] = rc
        dec.untrusted_mode = base_mode

    gcg_records, rows_gcg, eng = [], [], None
    if a.gcg_steer or a.gcg_hijack:
        clean_rows = {r["request_id"]: r for r in rows if r["attack_family"] == "clean"}
        eng = gcg.Engine(dec)
        gcg_variants = []
        for objective, k in (("steer", a.gcg_steer), ("hijack", a.gcg_hijack)):
            for i, r in enumerate(pick_gcg(reqs, clean_rows, k)):
                prob = gcg.Problem(dec, r, objective)
                res = gcg.attack(eng, [prob], dict(gcg.HP_SMOKE, steps=a.gcg_steps), seed=a.seed * 1000 + i, log=log)
                res.update(request_id=r["request_id"], objective=objective, family=r["_meta"]["family"],
                           final=gcg.order_outcome(prob, eng.eval_advs(prob, torch.tensor([res["adv_ids"]]))[0]))
                gcg_records.append(res)
                gcg_variants.append(attacks.suffix_variant(r, res["suffix"], f"gcg_{objective}", "gcg"))
        rows_gcg = metrics.run(dec, gcg_variants, log=log)  # re-check through the unmodified decide path
        for r in rows_gcg:
            r["arm"] = arm
    data.write_jsonl(gcg_records, out / "gcg_attacks.jsonl")

    all_rows = [r for rs in arms.values() for r in rs] + rows_gcg
    data.write_jsonl(all_rows, out / "predictions.jsonl")
    summary = {
        "note": "PIPELINE QUALIFICATION ONLY unless this is a frozen-checkpoint --final run named in a gate PR.",
        "model": model, "backend": a.backend, "revision": revision, "model_files_sha256": files_sha,
        "final": a.final,
        "arms": {name: metrics.summarise(rs + (rows_gcg if name == arm else []), a.boot) for name, rs in arms.items()},
        "arm_vs_arm": {f"{arm} - {name}": metrics.compare_arms(rows, rs, "hijack", a.boot)
                       for name, rs in arms.items() if name != arm},
    }
    if gcg_records:
        summary["gcg"] = {
            "attempts": {o: sum(g["objective"] == o for g in gcg_records) for o in gcg.OBJECTIVES},
            "pooled_success": {o: sum(g["final"]["success"] for g in gcg_records if g["objective"] == o)
                               for o in gcg.OBJECTIVES},
            "both_orders_success": {o: sum(g["final"]["both_orders"] for g in gcg_records if g["objective"] == o)
                                    for o in gcg.OBJECTIVES},
            "mean_secs": sum(g["secs"] for g in gcg_records) / len(gcg_records),
            "total_seqs_forwarded": eng.n_seqs, "hp": dict(gcg.HP_SMOKE, steps=a.gcg_steps)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    env = {
        "command": "python -m safety_eval.run_smoke " + " ".join(argv if argv is not None else sys.argv[1:]),
        "git_sha_at_run": git_sha(), "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "device": a.device, "dtype": dtype,
        "gpu": torch.cuda.get_device_name(0) if (a.device != "cpu" and torch.cuda.is_available()) else None,
        "billed_usd": 0.0, "model": model, "model_dir": str(model_dir) if model_dir else None,
        "model_files_sha256": files_sha, "env_SSDM_QWEN_DIR": os.environ.get(ENV_QWEN_DIR),
        "requests_manifest": man, "variants_sha256": sha256_obj([v["request_id"] for v in variants]),
        "predictions_sha256": sha256_obj(all_rows), "seed": a.seed, "data_seed": data.SEED,
        "boot_n": a.boot, "boot_seed": 20261006, "wall_secs_total": round(time.time() - t_start, 1),
        "wall_secs_main_arm": round(t_main, 1),
        "n_units": {**{k: len(v) for k, v in arms.items()}, "gcg": len(rows_gcg)},
    }
    (out / "environment.json").write_text(json.dumps(env, indent=1, default=str), encoding="utf-8")
    log(json.dumps({"model": model, "n_units": env["n_units"], "wall_secs": env["wall_secs_total"]}))


if __name__ == "__main__":
    main()
