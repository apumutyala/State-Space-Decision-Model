"""jobs/rwkv7 JOB=readout_g1v2: the runner parses and refuses an unknown JOB before touching anything."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "jobs/rwkv7/rwkv7_job.sh"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_runner_parses_and_refuses_unknown_job():
    assert subprocess.run(["bash", "-n", str(RUNNER)]).returncode == 0
    r = subprocess.run(["bash", str(RUNNER)], env={"JOB": "nope", "PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
    assert r.returncode == 2 and "readout_g1v2" in r.stdout
