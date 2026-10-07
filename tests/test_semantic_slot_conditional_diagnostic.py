from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "receipts/2026-10-07-L-semantic-slot-conditional-diagnostic"


def test_conditional_pointer_decomposition_reproduces_from_frozen_saved_probabilities(tmp_path):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    output = tmp_path / "conditional_result.json"
    run = subprocess.run([sys.executable, "-X", "utf8", str(RECEIPT / "analyze.py"),
                          "--output", str(output)],
                         cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["rows"] == 2160
    assert result["family_world_pairs"] == 360
    assert result["underlying_world_clusters"] == 180
    pooled = result["results"]["pooled"]
    beta0 = pooled["beta0"]["overall"]
    beta0_ci = pooled["beta0"]["overall_ci95"]
    beta005 = pooled["beta005"]["overall"]
    beta005_ci = pooled["beta005"]["overall_ci95"]
    assert beta0_ci["oracle_gold_operation_pointer_exact"]["n_world_clusters"] == 180
    assert pooled["beta005_minus_beta0"]["oracle_gold_operation_pointer_exact"]["n_world_clusters"] == 180
    assert beta0["operation_marginal_correct"] > .94
    assert beta0_ci["operation_marginal_correct"]["ci95"][0] > .93
    assert .44 < beta0["oracle_gold_operation_pointer_exact"] < .55
    assert beta0_ci["oracle_gold_operation_pointer_exact"]["ci95"][0] < .50
    assert beta005["oracle_gold_operation_pointer_exact"] < .50
    delta_op = pooled["beta005_minus_beta0"]["operation_marginal_correct"]["ci95"]
    delta_pointer = pooled["beta005_minus_beta0"]["oracle_gold_operation_pointer_exact"]["ci95"]
    assert delta_op[1] < 0
    assert delta_pointer[0] < 0 < delta_pointer[1]


def test_all_committed_analysis_inputs_and_outputs_match_manifest():
    manifest = json.loads((RECEIPT / "artifact-manifest.json").read_text(encoding="utf-8"))
    assert manifest
    for name, expected in manifest.items():
        path = (RECEIPT / name).resolve()
        path.relative_to(RECEIPT.resolve())
        assert path.is_file(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
