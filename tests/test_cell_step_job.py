"""jobs/cell_step_profile runner, SHELL level: executes jobs/cell_step_profile/cell_step_job.sh end to end on CPU in dry
-run mode (CELL_STEP_DRYRUN=1: tiny vendored RWKV-7, no GPU, no download), through the same phases, caps file, exports
and finalize as the GPU job. A previous runner bug passed every Python-level test; this runs the real script.

    python -X utf8 -m pytest -q tests/test_cell_step_job.py          (~5 min on a laptop CPU)
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


def _sh(p: Path) -> str:
    r"""The path as the runner's shell sees it. On Windows (Git Bash) GNU tar reads ``C:\...`` as a remote host, so
    paths go in POSIX form there; on Linux (the job) this is the identity."""
    cyg = shutil.which("cygpath")
    if os.name == "nt" and cyg:
        return subprocess.run([cyg, "-u", str(p)], capture_output=True, text=True, check=True).stdout.strip()
    return str(p)


@pytest.mark.skipif(BASH is None, reason="no bash")
def test_runner_dry_run_end_to_end(tmp_path):
    pytest.importorskip("torch")
    out, work, hf = tmp_path / "out", tmp_path / "work", tmp_path / "hf"
    env = {**os.environ, "CELL_STEP_DRYRUN": "1", "SSDM_CODE": _sh(ROOT), "GMN_OUTPUT_DIR": _sh(out),
           "CELL_STEP_WORK": _sh(work), "HF_HOME": _sh(hf), "GMN_METADATA_URL": "http://127.0.0.1:9",
           "CUDA_VISIBLE_DEVICES": "", "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]}
    env.pop("HF_HUB_CACHE", None)
    p = subprocess.run([BASH, str(ROOT / "jobs/cell_step_profile/cell_step_job.sh")], env=env, cwd=ROOT,
                       capture_output=True, text=True, timeout=1800)
    log = p.stdout + p.stderr
    assert p.returncode == 0, log[-4000:]
    res = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert res["ok"] and res["exit_code"] == 0 and res["exports_verified"], res
    assert (out / "CELL_STEP_PROFILE_DONE").exists()
    want = {"setup", "snapshot", "variant_baseline", "variant_ckpt", "variant_ckpt_rows", "variant_ckpt_rows_lean",
            "profile"}
    assert set(res["gates"]) == want and all(v == 0 for v in res["gates"].values()), res["gates"]
    from jobs.cell_step_profile import rule as R

    d = res["decision"]
    assert R.valid_decision(d) and d["verdict"] == "ADOPT"  # CPU: no OOM, peak 0, fp32 losses match the baseline
    assert d["variants"]["ckpt_rows_lean"]["loss"]["basis"] == "baseline_step0"
    assert all(d["variants"][v]["qualifies"] for v in R.VARIANTS)
    for v in R.VARIANTS:
        rec = json.loads((work / "out" / f"variant_{v}.json").read_text(encoding="utf-8"))
        assert rec["complete"] and rec["alloc_conf"] == R.ALLOC_CONF and rec["flags"] == R.FLAGS[v]
        assert len(rec["steps"]) == 23 and all("tok_s" in s for s in rec["steps"])
        assert [int(t) for t in rec["ladder"]] == list(R.LADDER)
    prof = json.loads((work / "out" / "profile.json").read_text(encoding="utf-8"))
    assert prof["complete"] and set(prof["profiled"]) == {d["chosen"], "baseline"}
    for name, pr in prof["profiled"].items():
        assert "error" not in pr and pr["by_count"] and pr["aten_calls"] > 0, (name, pr.get("error"))
    assert prof["profiled"][d["chosen"]]["aten_calls"] < prof["profiled"]["baseline"]["aten_calls"]
    assert "equivalence tests:" in log and "passed" in (work / "out" / "equivalence_tests.txt").read_text()
    man = [json.loads(x) for x in (out / "MANIFEST.jsonl").read_text().splitlines()]
    assert all(m["ok"] for m in man) and {m["phase"] for m in man} >= {"setup", "decide", "profile", "final"}
