"""jobs.common.rate_guard run-weighted --setup-cap-min (LESSONS L-29, PR #117 review): a child that never announces its
weighted total (a slow or hung model load / self-check) is stopped with exit 15; the option is off by default."""

from __future__ import annotations

import json
import sys

from jobs.common import rate_guard as RG


def _child(seconds: float, write_total: str | None = None) -> list[str]:
    code = "import time, json\n"
    if write_total:
        code += f"open({write_total!r}, 'w').write(json.dumps({{'total_weight': 1.0}}))\n"
    code += f"time.sleep({seconds})\n"
    return [sys.executable, "-c", code]


def test_setup_cap_stops_a_child_that_never_writes_the_total(tmp_path):
    report = tmp_path / "rate.json"
    rc = RG.run_weighted(_child(60), tmp_path / "progress.jsonl", cap_min=5, poll_s=0.2, report=report,
                         setup_cap_min=0.02)  # 1.2 s
    rec = json.loads(report.read_text())
    assert rc == RG.EXIT_TOO_SLOW == 15
    assert "no weighted total after 0.02 min" in rec["stopped"] and rec["elapsed_min"] < 0.5


def test_setup_cap_is_off_by_default(tmp_path):
    rc = RG.run_weighted(_child(2), tmp_path / "progress.jsonl", cap_min=5, poll_s=0.2)
    assert rc == 0  # no total file ever written, no setup cap: the child's own exit code


def test_setup_cap_does_not_fire_once_the_total_appeared(tmp_path):
    progress = tmp_path / "progress.jsonl"
    rc = RG.run_weighted(_child(3, str(progress) + ".total.json"), progress, cap_min=5, poll_s=0.2,
                         setup_cap_min=0.02)
    assert rc == 0


def test_cli_passes_the_setup_cap(tmp_path):
    report = tmp_path / "rate.json"
    rc = RG.main(["run-weighted", "--progress", str(tmp_path / "p.jsonl"), "--cap-min", "5", "--poll-s", "0.2",
                  "--setup-cap-min", "0.02", "--report", str(report), "--", *_child(60)])
    assert rc == 15 and "no weighted total" in json.loads(report.read_text())["stopped"]
