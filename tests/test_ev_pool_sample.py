"""jobs/ev_pool_sample: the runner's dry run executes end to end, and the pins / params agree."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "jobs/ev_pool_sample/ev_pool_sample_job.sh"


def test_dry_run_executes_runner():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash")
    env = {**os.environ, "DRY": "1"}
    r = subprocess.run([bash, str(RUNNER)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for want in ("pool_sample.py", "jobs.ev_pool_sample.preflight", "data.sha256"):
        assert want in r.stdout


def test_params_pins_cover_every_data_item():
    p = json.loads((ROOT / "jobs/ev_pool_sample/job_params.json").read_text(encoding="utf-8"))
    pins = {ln.split()[1].removeprefix("/data/") for ln in
            (ROOT / "jobs/ev_pool_sample/data.sha256").read_text(encoding="utf-8").splitlines() if ln.strip()}
    assert {d["dest"] for d in p["context"]["data"]} == pins
    assert p["max_duration_minutes"] == sum(p["phase_caps_minutes"].values()) + p["reserve_minutes"]
    assert p["max_duration_minutes"] * p["batch_usd_per_min"] <= 3.0  # platform hold <= $3
