"""Reproduce the $0 scripted plain-read proof without Torch or model loading."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from language_growth.tests.test_paired_read import ReadProvider, reads, run

MODULES = ["language_growth.tests.test_paired_read",
           "language_growth.tests.test_paired_comparison", "language_growth.tests.test_planner",
           "language_growth.tests.test_free_running_dev", "language_growth.tests.test_development_data"]


def write_json(path, value):
    path.write_bytes((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8"))


if __name__ == "__main__":
    output = Path(__file__).resolve().parent
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tasks = reads()
    raw = {mode: run(tasks if mode == "correct" else tasks[:1], ReadProvider(mode))
           for mode in ("correct", "wrong_direct", "error_direct", "error_planner", "invalid_direct", "wrong_config")}
    stream = io.StringIO()
    tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromNames(MODULES))
    assert "torch" not in sys.modules, "CPU proof must not import Torch"
    (output / "tests.txt").write_bytes(stream.getvalue().encode("utf-8"))
    write_json(output / "raw.json", raw)
    write_json(output / "receipt.json", {
        "source_commit": source,
        "command": "python receipts/2026-10-06-L-paired-read/run.py",
        "config": {"provider": "CPU scripted ReadProvider", "choice_format": "production",
                   "generated_count": 12, "filter": "training_only.kind == read", "condition": "one_shot"},
        "data_sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "raw_sha256": hashlib.sha256((output / "raw.json").read_bytes()).hexdigest(),
        "raw_path": "receipts/2026-10-06-L-paired-read/raw.json",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": 17, "split": "synthetic development_train", "hardware": platform.platform() + "; CPU only",
        "billed_rate": 0, "cost_usd": 0, "runs": 1,
        "median_spread": "not a timing/learned estimate; one deterministic scripted proof",
        "tests_run": tests.testsRun, "tests_passed": tests.wasSuccessful(), "torch_imported": False,
        "scope": "development plain-read boundary proof; no learned-level, unrestricted-baseline or full-retention claim",
    })
    if not tests.wasSuccessful():
        raise SystemExit(1)
    print(f"CPU proof passed: {tests.testsRun} tests; raw calls and source receipt saved.")
