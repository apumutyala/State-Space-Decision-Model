"""safety_eval battery on Clef / Clef-flash (SAFETY_SPEC section 7 rows G2(b), G6, G7 inputs).

Same data, variants, metrics, bootstrap and receipt layout as ``safety_eval.run_smoke``; the decider is
``decide.Decider`` over ``baselines.clef_backend.ClefBackend`` through ``safety_eval.decide_bridge``.

Arms:
  clef_ssdm            every variant; untrusted text through decide/untrusted.py (the like-for-like arm)
  clef_native_encoding clean + hijack variants with Clef's own tokenization (control tokens pass through):
                       the G2(b) "before" arm; arm_vs_arm gives the paired hijack difference
GCG-lite (PREREG 2.3, HP_GATE) uses baselines.clef_gcg and is re-checked through clef_ssdm.

    python -X utf8 -m baselines.run_clef_safety --model-dir /models/clef-flash --release clef-flash \
        --split test --final --out /out/safety [--gcg-steer 30 --gcg-hijack 15] [--gcg-only | --skip-gcg]

GCG results are appended per attack to ``gcg_attacks.jsonl`` and the run resumes from that file
(``$GMN_CHECKPOINT_DIR`` friendly). Exit codes: 0 ok, 30 load/verify failure, 31 run failure,
32 PREREG gate (test split / D1-B scoring before the k=8 blind D1-B amendment).

PREREG gate (safety_eval/PREREG.md G1(d)): "Before any m1 test scoring, 4 more B templates are written blind
..., giving k=8. They are logged as an amendment with their sha256." Scoring Clef on the test split or on D1-B
with fewer than 8 B templates would put Clef's gate-family number on different units than our parent's, so it
is refused here unless ``--prereg-deviation`` names a declared deviation (recorded in summary.json). The D1-B
template set sha256 is recorded in every receipt; any later data amendment (black-box family, payment
scenarios) means one declared re-score with the new sha, never a silent overwrite.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from safety_eval import attacks, data, gcg, metrics
from safety_eval.decide_bridge import DecideBridge
from safety_eval.request import sha256_obj

from . import clef_backend as CB

PREREG_D1B_MIN_TEMPLATES = 8  # PREREG G1(d): k=8 blind B templates before any test scoring


def d1b_identity() -> dict:
    t = attacks.D1_TEMPLATES_B
    return {"n_templates": len(t), "sha256": sha256_obj(sorted(t.items())), "required": PREREG_D1B_MIN_TEMPLATES}


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=CB.REPO_ROOT).strip()
    except Exception:  # noqa: BLE001
        return os.environ.get("SSDM_GIT_SHA", "unknown")


def checkpoint() -> None:
    """Declare a checkpoint to the SF Compute metadata service when running as a job (best effort)."""
    url = os.environ.get("GMN_METADATA_URL")
    if not url:
        return
    try:
        urllib.request.urlopen(urllib.request.Request(url.rstrip("/") + "/v1/checkpoint", data=b"{}",
                                                      method="POST", headers={"Content-Type": "application/json"}),
                               timeout=30)
    except Exception:  # noqa: BLE001  (429 rate floor or transient: the last committed slot stands)
        pass


def pick_gcg(requests, clean_rows, k):
    from safety_eval.run_smoke import pick_gcg as _pick
    return _pick(requests, clean_rows, k)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--release", choices=sorted(CB.RELEASES), default="clef-flash")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--n-route", type=int, default=120)
    ap.add_argument("--n-pay", type=int, default=80)
    ap.add_argument("--final", action="store_true", help="also D1-B (held out); Clef is a frozen external checkpoint")
    ap.add_argument("--gcg-steer", type=int, default=0)
    ap.add_argument("--gcg-hijack", type=int, default=0)
    ap.add_argument("--gcg-hp", choices=["gate", "smoke"], default="gate")
    ap.add_argument("--gcg-only", action="store_true", help="clean arm + GCG only (job B)")
    ap.add_argument("--skip-gcg", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-verify", action="store_true", help="skip weight sha256 (never for a receipt)")
    ap.add_argument("--prereg-deviation", default=None,
                    help="declared deviation text; allows test/D1-B scoring before the k=8 amendment (recorded)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)  # noqa: E731
    d1b = d1b_identity()
    if (a.split == "test" or a.final) and d1b["n_templates"] < PREREG_D1B_MIN_TEMPLATES and not a.prereg_deviation:
        log(f"PREREG GATE: {d1b['n_templates']} D1-B templates < {PREREG_D1B_MIN_TEMPLATES}; test-split / D1-B scoring "
            "is refused until the blind-template amendment lands (or pass --prereg-deviation).")
        return 32
    t_start = time.time()

    try:
        backend, model, processor = CB.load_clef(a.model_dir, a.release, encoding="ssdm", device=a.device,
                                                 dtype=a.dtype, verify=not a.no_verify)
    except Exception as exc:  # noqa: BLE001
        log(f"LOAD/VERIFY FAILED: {type(exc).__name__}: {exc}")
        return 30
    native = CB.ClefBackend(model, processor.tokenizer, release=a.release, encoding="clef_native", jsm=backend.jsm,
                            device=a.device, dtype=a.dtype, files_sha256=backend.files_sha256)
    bridge = DecideBridge(CB.build_decider(backend), model=a.release)
    bridge_native = DecideBridge(CB.build_decider(native), model=a.release)

    try:
        import torch

        reqs = data.build(a.n_route, a.n_pay, split=a.split)
        data.write_jsonl(reqs, out / "requests.jsonl")
        variants = attacks.build_variants(reqs, final=a.final)
        if a.gcg_only:
            variants = [v for v in variants if v["_meta"]["attack_family"] == "clean"]
        log(f"loaded {a.release} ({a.device}/{a.dtype}); {len(variants)} variants; split={a.split}")
        t0 = time.time()
        rows = metrics.run(bridge, variants, log=log)
        for r in rows:
            r["arm"] = "clef_ssdm"
        arms = {"clef_ssdm": rows}
        t_main = time.time() - t0
        if not a.gcg_only:
            ctrl = [v for v in variants if v["_meta"]["attack_family"] in ("clean", "hijack")]
            rn = metrics.run(bridge_native, ctrl, log=log)
            for r in rn:
                r["arm"] = "clef_native_encoding"
            arms["clef_native_encoding"] = rn

        gcg_path = out / "gcg_attacks.jsonl"
        gcg_records = [json.loads(x) for x in gcg_path.read_text(encoding="utf-8").splitlines() if x.strip()] \
            if gcg_path.exists() else []
        rows_gcg, eng = [], None
        if (a.gcg_steer or a.gcg_hijack) and not a.skip_gcg:
            from . import clef_gcg

            hp = dict(gcg.HP_GATE if a.gcg_hp == "gate" else gcg.HP_SMOKE)
            clean_rows = {r["request_id"]: r for r in rows if r["attack_family"] == "clean"}
            eng = clef_gcg.ClefGCGEngine(backend)
            done = {(g["objective"], g["request_id"]) for g in gcg_records}
            for objective, k in (("steer", a.gcg_steer), ("hijack", a.gcg_hijack)):
                for i, r in enumerate(pick_gcg(reqs, clean_rows, k)):
                    if (objective, r["request_id"]) in done:
                        continue
                    prob = clef_gcg.ClefProblem(backend, r, objective)
                    res = clef_gcg.attack(eng, [prob], hp, seed=a.seed * 1000 + i, log=log)
                    res.update(request_id=r["request_id"], objective=objective, family=r["_meta"]["family"],
                               final=clef_gcg.final_outcome(eng, prob, res["adv_ids"]))
                    gcg_records.append(res)
                    with gcg_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(res, default=str) + "\n")
                    checkpoint()
                    if a.device.startswith("cuda"):
                        torch.cuda.empty_cache()
            by_id = {r["request_id"]: r for r in reqs}
            gv = [attacks.suffix_variant(by_id[g["request_id"]], g["suffix"], f"gcg_{g['objective']}", "gcg")
                  for g in gcg_records]
            rows_gcg = metrics.run(bridge, gv, log=log)  # re-check through the unmodified decide path
            for r in rows_gcg:
                r["arm"] = "clef_ssdm"

        all_rows = [r for rs in arms.values() for r in rs] + rows_gcg
        data.write_jsonl(all_rows, out / "predictions.jsonl")
        summary = {
            "note": "Clef baseline on the safety_eval harness (SAFETY_SPEC section 7). Clef is a frozen external "
                    "checkpoint; --final scores D1-B once for it.",
            "release": a.release, "revision": backend.revision(), "split": a.split, "final": a.final,
            "d1b_templates": d1b, "prereg_deviation": a.prereg_deviation,
            "arms": {n: metrics.summarise(rs + (rows_gcg if n == "clef_ssdm" else []), a.boot) for n, rs in arms.items()},
            "arm_vs_arm": {f"{n} - clef_ssdm | hijack | eligibility=reference_y": metrics.compare_arms(
                rs, rows, "hijack", a.boot, eligibility="reference_y",
                expected_ids={r["request_id"] for r in rs if r["attack_family"] == "hijack"})
                for n, rs in arms.items() if n != "clef_ssdm"},
        }
        if gcg_records:
            summary["gcg"] = {
                "attempts": {o: sum(g["objective"] == o for g in gcg_records) for o in gcg.OBJECTIVES},
                "pooled_success": {o: sum(g["final"]["success"] for g in gcg_records if g["objective"] == o)
                                   for o in gcg.OBJECTIVES},
                "both_orders_success": {o: sum(g["final"]["both_orders"] for g in gcg_records if g["objective"] == o)
                                        for o in gcg.OBJECTIVES},
                "orders_note": "Clef sorts option ids: one order, both_orders == pooled by construction",
                "mean_secs": sum(g["secs"] for g in gcg_records) / len(gcg_records),
                "hp": gcg_records[0]["hp"]}
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
        env = {
            "command": "python -m baselines.run_clef_safety " + " ".join(argv if argv is not None else sys.argv[1:]),
            "git_sha_at_run": git_sha(), "python": sys.version, "platform": platform.platform(),
            "torch": torch.__version__, "transformers": __import__("transformers").__version__,
            "device": a.device, "dtype": a.dtype,
            "gpu": torch.cuda.get_device_name(0) if a.device.startswith("cuda") and torch.cuda.is_available() else None,
            "max_memory_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)
            if a.device.startswith("cuda") and torch.cuda.is_available() else None,
            "release": a.release, "model_dir": a.model_dir, "files_sha256": backend.files_sha256,
            "requests_manifest": data.manifest(reqs), "variants_sha256": sha256_obj([v["request_id"] for v in variants]),
            "predictions_sha256": sha256_obj(all_rows), "d1b_templates": d1b,
            "prereg_deviation": a.prereg_deviation, "seed": a.seed, "data_seed": data.SEED, "boot_n": a.boot,
            "wall_secs_total": round(time.time() - t_start, 1), "wall_secs_main_arm": round(t_main, 1),
            "n_units": {**{k: len(v) for k, v in arms.items()}, "gcg": len(rows_gcg)},
        }
        (out / "environment.json").write_text(json.dumps(env, indent=1, default=str), encoding="utf-8")
        log(json.dumps({"release": a.release, "n_units": env["n_units"], "wall_secs": env["wall_secs_total"]}))
        return 0
    except Exception as exc:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        log(f"RUN FAILED: {type(exc).__name__}: {exc}")
        return 31


if __name__ == "__main__":
    sys.exit(main())
