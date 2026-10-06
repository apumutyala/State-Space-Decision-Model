"""run_job.sh heartbeat watchdog: touches only while the main process lives AND the worker makes progress."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

WATCHDOG = Path(__file__).resolve().parents[2] / "jobs" / "clef_baseline" / "watchdog.sh"
BASH = shutil.which("bash")

SCRIPT = r"""
set -u
OUT="$1/out"; CK="$1/ck"; HF_DIR="$1/hf"; mkdir -p "$OUT" "$CK" "$HF_DIR"
MAIN_PID=$$; HB_EVERY_S=1; STALL_S=3
. "$2"
start_watchdog
snap() { if [ -f "$OUT/heartbeat" ]; then stat -c %Y "$OUT/heartbeat"; else echo 0; fi; }
# phase 1: progress every second for 5 s -> heartbeat keeps moving
for i in 1 2 3 4 5; do echo "$i" >> "$OUT/results.jsonl"; sleep 1; done
p1=$(snap)
# phase 2: worker hangs (no writes) for 8 s -> heartbeat must stop within STALL_S + 1 s
sleep 8
p2a=$(snap); sleep 3; p2b=$(snap)
# phase 3: progress resumes -> heartbeat resumes
for i in 1 2 3; do echo "$i" >> "$CK/gcg_attacks.jsonl"; sleep 1; done
p3=$(snap)
echo "$p1 $p2a $p2b $p3 $WD_PID"
"""


@pytest.mark.skipif(BASH is None, reason="bash not available")
def test_watchdog_progress_gated_and_dies_with_main(tmp_path):
    script = tmp_path / "t.sh"
    script.write_text(SCRIPT, encoding="utf-8", newline="\n")
    res = subprocess.run([BASH, str(script), tmp_path.as_posix(), WATCHDOG.as_posix()],
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    p1, p2a, p2b, p3, wd = res.stdout.split()
    p1, p2a, p2b, p3 = map(int, (p1, p2a, p2b, p3))
    assert p1 > 0  # heartbeat written while progressing
    assert p2b == p2a  # hung worker: no touches in the last 3 s of the stall
    assert p3 > p2b  # progress resumed -> heartbeat resumed
    assert "heartbeat withheld" in (tmp_path / "out" / "watchdog.log").read_text(encoding="utf-8")
    # main process exited: the EXIT trap killed the watchdog
    alive = subprocess.run([BASH, "-c", f"kill -0 {wd} 2>/dev/null && echo alive || echo dead"],
                           capture_output=True, text=True).stdout.strip()
    assert alive == "dead"
