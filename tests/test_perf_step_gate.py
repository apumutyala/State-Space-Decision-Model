"""jobs/perf_step_gate runner, SHELL level: executes jobs/perf_step_gate/perf_step_job.sh end to end on CPU in dry-run
mode (PERF_STEP_DRYRUN=1: tiny vendored RWKV-7, no GPU, no download, compile backend aot_eager), through the same
phases, caps file, rate guard, exports and finalize as the GPU job.

    python -X utf8 -m pytest -q tests/test_perf_step_gate.py          (several minutes on a laptop CPU)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
RUNNER = "jobs/perf_step_gate/perf_step_job.sh"


def _sh(p: Path) -> str:
    cyg = shutil.which("cygpath")
    if os.name == "nt" and cyg:
        return subprocess.run([cyg, "-u", str(p)], capture_output=True, text=True, check=True).stdout.strip()
    return str(p)


@pytest.mark.skipif(BASH is None, reason="no bash")
def test_runner_dry_run_end_to_end(tmp_path):
    pytest.importorskip("torch")
    out, work, hf = tmp_path / "out", tmp_path / "work", tmp_path / "hf"
    env = {**os.environ, "PERF_STEP_DRYRUN": "1", "SSDM_CODE": _sh(ROOT), "GMN_OUTPUT_DIR": _sh(out),
           "PERF_STEP_WORK": _sh(work), "HF_HOME": _sh(hf), "GMN_METADATA_URL": "http://127.0.0.1:9",
           "CUDA_VISIBLE_DEVICES": "", "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]}
    env.pop("HF_HUB_CACHE", None)
    p = subprocess.run([BASH, str(ROOT / RUNNER)], env=env, cwd=ROOT, capture_output=True, text=True, timeout=3600)
    log = p.stdout + p.stderr
    assert p.returncode == 0, log[-4000:]
    res = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert res["ok"] and res["exit_code"] == 0 and res["exports_verified"], res
    assert (out / "PERF_STEP_GATE_DONE").exists()
    from jobs.perf_step_gate import rule as R

    want = {"setup", "snapshot"} | {f"variant_{v}" for v in R.VARIANTS}
    assert set(res["gates"]) == want and all(v == 0 for v in res["gates"].values()), res["gates"]
    d = res["decision"]
    assert R.valid_decision(d) and d["verdict"] == "ADOPT"
    for v in R.VARIANTS:
        s = d["variants"][v]
        assert s["qualifies"], (v, s["reasons"])
        assert s["loss_rel"] <= 1e-5 and s["grad_cos"] >= 1 - 1e-9  # fp32 CPU: the paths agree
        rec = json.loads((work / "out" / f"variant_{v}.json").read_text(encoding="utf-8"))
        assert rec["complete"] and rec["flags"] == R.FLAGS[v] and rec["active"] == R.FLAGS[v]
        assert len(rec["steps"]) == R.N_STEPS and all("loss" in s for s in rec["steps"])
        assert [int(t) for t in rec["ladder"]] == list(R.LADDER)
    comp = json.loads((work / "out" / "variant_min_cost_single_compile.json").read_text(encoding="utf-8"))
    assert comp["dynamo_after_steps"]["stats"].get("unique_graphs", 0) > 0  # the compiled path really ran
    assert json.loads((work / "out" / "preflight.json").read_text())["ok"]
    man = [json.loads(x) for x in (out / "MANIFEST.jsonl").read_text().splitlines()]
    assert all(m["ok"] for m in man) and {m["phase"] for m in man} >= {"setup", "decide", "final"}
