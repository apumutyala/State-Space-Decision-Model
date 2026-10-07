"""CPU dry run of jobs/m2_first_cell end to end on a TINY random RWKV7 (vendored HF code, the real RWKV7-G1k tokenizer).

    python -X utf8 scripts/m2_first_cell_dryrun.py --out <dir> [--role-n 120] [--index-n 400]

It runs the job's Python entry points in the job's order with the job's arguments (first_cell_job.sh), on the REAL
inputs when present locally: the PR #44 150k set (pointer-verified), the PR #16 sample + frozen cohort v2, the 0.2.1
suite (guard readiness rebuilds the pinned content index), the Index DEV subset, the D1-B held-out gate. What differs
from the paid run, and only this:
  * the model: 2 layers, hidden 32, random weights, CPU (``--device cpu``), via ``scripts/first_cell_dryrun_shim.py``
    (the 13.3B config pin cannot match a tiny config; the kit clone goes on sys.path);
  * the cell config = the exact config with the overrides in ``OVERRIDES`` (record count, steps, minutes);
  * the eval sets: every cohort role truncated to ``--role-n`` entries and the Index subset to about ``--index-n`` rows (whole kit
    groups at an even stride; CPU time). Unit identities are computed on what is scored, so both arms still match exactly;
  * no CUDA / nvidia-smi / snapshot-download / sha256sum steps (shell-only checks; static tests cover the script).
Writes ``<out>/receipt.json`` with every phase's exit code, timing and the validated result.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CELL = ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7.json"
PARAMS = ROOT / "jobs/m2_first_cell/job_params.json"
SHIM = ROOT / "scripts/first_cell_dryrun_shim.py"
DATA = Path(os.environ.get("SSDM_DATA") or ROOT.parents[1] / "ssdm-data")
TOKDIR = DATA / "tokenizers" / "RWKV--RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba"
TRAIN_DIR = DATA / "m2/build/sample-150000-seed20261006"
RECORDS = DATA / "m2/build/sample-20000-seed20261006/records.jsonl.gz"
SUITE = DATA / "suite-0.2"
DEV_ROWS = DATA / "slices/dev-rows.jsonl.gz"
#: the only config changes vs jobs/m2_first_cell/m2_first_cell_rwkv7.json (CPU time); everything else is identical
OVERRIDES = {"n_train_records": 64, "min_train_items": 16, "max_steps": 6, "max_minutes": 4, "min_steps": 2,
             "ckpt_every": 3, "tokenize_workers": 2, "pass_rule": {"window": 2, "loss_drop_frac_min": 0.05,
                                                                    "max_peak_gb": 76}}
TOK_S, CALL_S = "3757", "0.117"  # job-tkfaj probe (the preflight arithmetic runs on the paid-run rates)


def missing_inputs() -> list[str]:
    return [str(p) for p in (TOKDIR / "tokenizer.json", TRAIN_DIR / "records.jsonl.gz", RECORDS, SUITE, DEV_ROWS)
            if not p.exists()]


def kit_clone_dir() -> Path:
    """The pinned Decision Index kit as a GIT CLONE for ``guard_ready --kit-dir`` (it checks the clone's commit, exit 30
    otherwise). The kit may also be importable as an installed copy (the launcher's job-test env installs it with
    ``uv --with <kit>``); that copy has no .git, so the clone (``harness.config.KIT_CLONE``, under SSDM_DATA) is
    preferred. Raises ImportError when no git clone of the kit is available."""
    import subprocess

    from scripts.m2_e2e_dryrun import ensure_kit

    info = ensure_kit()
    from harness import config as hc

    for c in (Path(hc.KIT_CLONE), Path(info["kit_path"]).parent):
        if (c / "decision_index").is_dir():
            r = subprocess.run(["git", "-C", str(c), "rev-parse", "HEAD"], capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                return c
    raise ImportError("no git clone of the pinned Decision Index kit (guard_ready checks its commit); set SSDM_DI_KIT")


def tiny_snapshot(d: Path) -> Path:
    """Random 2-layer RWKV7 (vocab 65,536 so the real tokenizer's ids are valid) + the vendored code + tokenizer."""
    import torch

    from third_party.rwkv7_hf import HERE
    from third_party.rwkv7_hf.configuration_rwkv7 import Rwkv7Config
    from third_party.rwkv7_hf.modeling_rwkv7 import Rwkv7ForCausalLM

    d.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    cfg = Rwkv7Config(vocab_size=65536, hidden_size=32, num_heads=4, head_dim=8, num_hidden_layers=2,
                      intermediate_size=64, a_low_rank_dim=8, decay_low_rank_dim=8, gate_low_rank_dim=8,
                      v_low_rank_dim=8)
    Rwkv7ForCausalLM(cfg).save_pretrained(d)
    for f in ("modeling_rwkv7.py", "configuration_rwkv7.py", "tokenization_rwkv7.py"):
        shutil.copy(HERE / f, d / f)
    for f in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(TOKDIR / f, d / f)
    return d


def index_subset(src: Path, dst: Path, n: int) -> int:
    """Keep about ``n`` of the Index DEV subset rows in ``src`` (gzip JSONL) -> ``dst``: whole kit groups (track
    benchmarks score a group's parts together) at an even stride over the file, so benchmarks stay spread. CPU time
    only; also used by tests/shell_stubs/sitecustomize.py for the shell-runner test. Returns the rows kept."""
    with gzip.open(src, "rt", encoding="utf-8") as fi:
        full = fi.readlines()
    groups: dict = {}
    for line in full:
        e = json.loads(line)["_evaluation"]
        groups.setdefault((e["catalog_id"], str(e["group_id"])), []).append(line)
    keys = list(groups)
    stride = max(1, len(full) // max(1, n))
    keep = []
    for k in keys[::stride] + keys:
        if len(keep) >= n:
            break
        if groups[k][0] not in keep:
            keep += groups[k]
    with gzip.open(dst, "wt", encoding="utf-8") as fo:
        fo.writelines(keep)
    return len(keep)


def run(name: str, cmd: list[str], log: Path, env=None, ok=(0,)) -> dict:
    t = time.time()
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    rec = {"phase": name, "rc": r.returncode, "seconds": round(time.time() - t, 1), "log": log.name,
           "ok": r.returncode in ok}
    print(json.dumps(rec), flush=True)
    if not rec["ok"]:
        print(log.read_text(encoding="utf-8", errors="replace")[-4000:], flush=True)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--role-n", type=int, default=120)
    ap.add_argument("--index-n", type=int, default=400)
    ap.add_argument("--reference", action="append", default=[], help="name=results.jsonl, passed to the report")
    a = ap.parse_args(argv)
    miss = missing_inputs()
    if miss:
        print(json.dumps({"skipped": "local inputs missing", "missing": miss}))
        return 3
    out = a.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    work, logs, ck = out / "work", out / "logs", out / "ck"
    for p in (work, logs, ck / "cell"):
        p.mkdir(parents=True, exist_ok=True)
    py = [sys.executable, "-X", "utf8"]
    shim = py + [str(SHIM)]
    env = {**os.environ, "DRYRUN_ROLE_N": str(a.role_n), "PYTHONUTF8": "1",
           "SSDM_PROGRESS_BEAT": str(logs / "beat")}
    cfg = {**json.loads(CELL.read_text(encoding="utf-8")), **OVERRIDES}
    cell = out / "cell_dryrun.json"
    cell.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    snap = tiny_snapshot(out / "tiny-rwkv7")
    kit = kit_clone_dir()
    phases = []
    t0 = time.time()

    def go(name, cmd, **k):
        r = run(name, cmd, logs / f"{name}.log", env=env, **k)
        phases.append(r)
        return r["ok"]

    rows_full, rows = work / "index-rows-full.jsonl.gz", work / "index-rows.jsonl.gz"
    steps = [
        ("train_set_verify", py + ["-m", "train.train_set", "verify", "--config", str(cell), "--train-dir", str(TRAIN_DIR),
                                   "--out", str(work / "train-set-verify.json")]),
        ("cohort_verify", py + ["-m", "train.cohort", "verify"]),
        ("dev_subset_rows", shim + ["jobs.common.dev_subset", "rows", "--rows", str(DEV_ROWS), "--out", str(rows_full),
                                    "--subset", "jobs/common/dev_subset_m1.json"]),
    ]
    for name, cmd in steps:
        if not go(name, cmd):
            return finish(out, phases, t0, None)
    index_subset(rows_full, rows, a.index_n)
    p = json.loads(PARAMS.read_text(encoding="utf-8"))
    fix = p["evals_fixed_minutes"]
    for stages, mode, capk, fx in (("g12", "parent", "parent_g12", fix["parent"]),
                                   ("index", "parent", "parent_index", fix["parent"]),
                                   ("g12,index", "child", "evals", fix["child"])):
        if not go(f"preflight_{capk}", py + ["-m", "jobs.m2_parent_arm.score_parent", "estimate", "--tok-s", TOK_S,
                                            "--call-s", CALL_S, "--stages", stages, "--mode", mode, "--cap-min",
                                            str(p["phase_caps_minutes"][capk]), "--fixed-min", str(fx)]):
            return finish(out, phases, t0, None)
    guard = work / "guard"
    if not go("guard", shim + ["jobs.m2_pilot.guard_ready", "--suite-dir", str(SUITE), "--kit-dir", str(kit),
                               "--out-dir", str(guard)]):
        return finish(out, phases, t0, None)
    if not go("heldout", py + ["-m", "train.train_set", "heldout", "--config", str(cell), "--train-dir", str(TRAIN_DIR),
                               "--out", str(work / "heldout.json")]):
        return finish(out, phases, t0, None)
    ready = str(guard / "readiness.json")
    arm = ck / "parent_arm"  # the job order: parent arm (live rate guard) and reprice BEFORE training
    arm.mkdir(parents=True, exist_ok=True)
    for stage in ("g12", "index"):
        prog = logs / f"parent-{stage}.progress.jsonl"
        if not go(f"parent_{stage}", py + ["-m", "jobs.common.rate_guard", "run-weighted", "--progress", str(prog),
                                           "--cap-min", str(p["phase_caps_minutes"][f"parent_{stage}"]), "--poll-s", "2",
                                           "--report", str(arm / f"rate-parent-{stage}.json"), "--"]
                  + shim + ["jobs.m2_parent_arm.score_parent", "--parent", "rwkv7", "--model-dir", str(snap),
                            "--cell-config", str(cell), "--records", str(RECORDS), "--guard-readiness", ready,
                            "--index-rows", str(rows), "--out-dir", str(arm), "--stage", stage, "--device", "cpu",
                            "--progress", str(prog)]):
            return finish(out, phases, t0, None)
    arm_sha = hashlib.sha256((arm / "arm.json").read_bytes()).hexdigest()
    if not go("reprice", py + ["-m", "jobs.m2_first_cell.reprice", "--arm-dir", str(arm), "--params", str(PARAMS),
                               "--tok-s", TOK_S, "--call-s", CALL_S, "--out", str(work / "reprice.json")]):
        return finish(out, phases, t0, None)
    if not go("cell", shim + ["train.run_cell", "--config", str(cell), "--model-dir", str(snap), "--records", str(RECORDS),
                              "--train-dir", str(TRAIN_DIR), "--out", str(work), "--ckpt-dir", str(ck / "cell"),
                              "--stage", "all", "--guard-readiness", ready, "--device", "cpu"]):
        return finish(out, phases, t0, None)
    if not go("evals", py + ["-m", "jobs.common.rate_guard", "run-weighted", "--progress", str(logs / "progress.jsonl"),
                             "--cap-min", str(p["phase_caps_minutes"]["evals"]), "--poll-s", "2",
                             "--report", str(work / "rate-evals.json"), "--"]
              + shim + ["train.m2_gates", "--cell", str(work / "cell.json"), "--ckpt-dir", str(ck / "cell"),
                        "--model-dir", str(snap), "--records", str(RECORDS), "--train-dir", str(TRAIN_DIR),
                        "--guard-readiness", ready, "--parent-arm", str(arm), "--parent-arm-sha256", arm_sha,
                        "--anchor-arm", str(arm), "--anchor-arm-sha256", arm_sha,
                        "--index-rows", str(rows), "--out", str(work / "report.json"), "--stage", "g12+index",
                        "--progress-dir", str(logs), "--device", "cpu", "--n-boot", "200"], ok=(0, 40, 43, 44, 45)):
        return finish(out, phases, t0, None)
    res = work / "first_cell_result.json"
    go("report", shim + ["jobs.m2_first_cell.summarize", "--cell", str(work / "cell.json"), "--report",
                         str(work / "report.json"), "--parent-arm", str(arm), "--index-rows", str(rows), "--params",
                         str(PARAMS), "--elapsed-min", str(round((time.time() - t0) / 60, 2)), "--out", str(res),
                         "--n-boot", "200"] + [x for r in a.reference for x in ("--reference", r)])
    return finish(out, phases, t0, res)


def finish(out: Path, phases: list, t0: float, res: Path | None) -> int:
    sys.path.insert(0, str(ROOT))
    from jobs.m2_first_cell.summarize import valid_result

    result = json.loads(res.read_text(encoding="utf-8")) if res and res.exists() else None
    rec = {"dry_run": "jobs/m2_first_cell on a tiny random RWKV7 (CPU)", "overrides": OVERRIDES,
           "phases": phases, "all_ok": all(p["ok"] for p in phases) and result is not None,
           "result_valid": bool(result and valid_result(result)), "minutes": round((time.time() - t0) / 60, 2),
           "result": result}
    (out / "receipt.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("all_ok", "result_valid", "minutes")}))
    return 0 if rec["all_ok"] and rec["result_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
