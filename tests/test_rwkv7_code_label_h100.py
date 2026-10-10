"""CPU exit proofs for jobs/rwkv7_code_label_h100/; no private row or model reads."""
import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

from jobs.common import costs, launch, lint
from jobs.rwkv7_code_label_h100 import preflight as P
from jobs.rwkv7_code_label_h100 import runner as Q
from jobs.rwkv7_code_label_screen import rule

ROOT = Path(__file__).resolve().parents[1]


def volume():
    return json.loads(P.VOLUME.read_text())


def probe_fixture():
    v = volume()
    return {"synthetic": False, "gpu": "NVIDIA H100 80GB HBM3", "parity_max_abs": 0.0001,
        "total_memory_bytes": 80 * 1024 ** 3,
        "samples": [{"arm": arm, "run_id": rid,
            "seconds": [P.reference_seconds(v["per_row"][rid][arm])] * 2,
            "peak_reserved_bytes": 55 * 1024 ** 3, "min_free_bytes": 25 * 1024 ** 3}
            for arm in ("v1", "v2") for rid in P.probe_ids(v)]}


def test_rule_and_real_shape_admission_constants_are_preregistered():
    assert rule.RULE_VERSION == "rwkv7-code-label-v2-screen-v1"
    assert rule.ROWS == 2008 and rule.MIN_OK_ROWS == 2000
    assert rule.BOOTSTRAP_REPLICATES == 2000 and rule.BOOTSTRAP_SEED == 20261007
    assert rule.CI_LEVEL == 0.95
    assert P.REFERENCE_TOK_S == 5360.0 and P.REFERENCE_CALL_S == 0.098
    assert P.PLANNING_TOK_S == 3757.39821244465 and P.PLANNING_CALL_S == 0.5997345447540283
    assert P.INHERITED_SMALL_CALL_S == 0.11704301834106445
    assert P.RATE_MARGIN == 0.95 and P.PEAK_FRACTION == 0.90
    assert P.MIN_FREE_BYTES == 5 * 1024 ** 3
    assert P.TIMED_REPEATS == 2 and P.PARITY_ATOL == 0.002
    assert Q.params()["launch_enabled"] is False
    assert Q.params()["decision_rule"] == "jobs/rwkv7_code_label_screen/rule.py"


def test_exact_workload_cost_and_caps_have_no_fixed_dollar_ceiling():
    v, p = volume(), Q.params()
    assert v["admission_counts"] == {"v1_admitted": 2000, "v2_admitted": 2000,
        "v1_input_exhausted": 8, "v2_input_exhausted": 8}
    admitted = v["totals"]["admitted"]
    assert (admitted["v1"]["positions"], admitted["v1"]["calls"]) == (50186548, 8211)
    assert (admitted["v2"]["positions"], admitted["v2"]["calls"]) == (6197882, 14287)
    reference = sum(P.reference_seconds(admitted[a]) for a in ("v1", "v2")) / 60
    assert reference == pytest.approx(212.07145348258706)
    probe_reference = sum(P.reference_seconds(v["per_row"][rid][a]) for rid in P.probe_ids(v) for a in ("v1", "v2")) / 60
    assert len(P.probe_ids(v)) == 4 and probe_reference == pytest.approx(5.329061194029851)
    planned_probe = sum(P.planning_seconds(v["per_row"][rid][a]) for rid in P.probe_ids(v) for a in ("v1", "v2")) / 60
    planned_screen = sum(P.planning_seconds(admitted[a]) for a in ("v1", "v2")) / 60
    assert planned_probe == pytest.approx(9.41108747574035)
    assert planned_screen == pytest.approx(474.98451036468225)
    historical = ROOT / p["planning_basis"]["historical_probe"]
    assert Q.S.sha(historical) == p["planning_basis"]["historical_probe_sha256"]
    inherited = json.loads(historical.read_text())
    assert P.PLANNING_TOK_S == inherited["prefill"]["2048"]["tok_s"]
    assert P.PLANNING_CALL_S == inherited["pack_call_s"]
    assert P.INHERITED_SMALL_CALL_S == inherited["small_call_s"]
    small = sum(admitted[a]["positions"] / P.PLANNING_TOK_S + admitted[a]["calls"] * P.INHERITED_SMALL_CALL_S for a in admitted) / 60
    assert small == pytest.approx(293.9912776773679)
    assert p["expected_minutes"]["probe"] == pytest.approx(3 * planned_probe + 2)
    assert p["expected_minutes"]["screen"] == pytest.approx(planned_screen)
    for phase, cap in p["phase_caps_minutes"].items():
        assert cap == 5 * math.ceil(p["cap_multipliers"][phase] * p["expected_minutes"][phase] / 5)
    assert lint.schedule_total(p) == p["max_duration_minutes"] == 710
    assert costs.worst_case_usd(p) == p["hard_cap_usd"] == 93.24  # 73.26 at the pre-2026-10-08 $0.0495/min
    assert round(60 * 0.01 + 2 * 710 * 0.0495, 2) == 70.89
    assert costs.expected_usd(p) == 33.78
    assert costs.concurrent_cap_usd(["rwkv7_code_label_h100"]) == 94
    assert p["hard_cap_usd"] < p["approved_track_allocation_usd"] == 250
    assert costs.PARAMS["rwkv7_code_label_h100"] == "jobs/rwkv7_code_label_h100/job_params.json"


def test_real_shape_probe_covers_all_registered_memory_extrema():
    v = volume()
    ids = set(P.probe_ids(v))
    for arm, fields in {"v1": ("max_prompt_tokens", "positions"),
                        "v2": ("max_prompt_tokens", "max_pack_positions", "n_options", "prefix_tokens")}.items():
        admitted = [p[arm] for p in v["per_row"].values() if p["admitted"][arm]]
        for field in fields:
            assert max(v["per_row"][rid][arm][field] for rid in ids) == max(p[field] for p in admitted)
    assert max(v["per_row"][rid]["v1"]["max_prompt_tokens"] for rid in ids) == 14360
    assert max(v["per_row"][rid]["v2"]["max_pack_positions"] for rid in ids) == 4084


def test_probe_reference_passes_its_own_guard():
    result = P.evaluate(probe_fixture(), volume(), 320)
    assert result["ok"] is True
    assert result["rate_factors"] == {"v1": 1.0, "v2": 1.0}
    assert result["total_projected_minutes"] == pytest.approx(212.07145348258706)


@pytest.mark.parametrize("defect", ["synthetic", "wrong_gpu", "slow", "missing", "duplicate", "missing_repeat",
    "nonfinite", "oom", "free", "parity", "nan_parity", "negative_parity"])
def test_probe_gate_refuses_missing_slow_oom_and_invalid_inputs(defect):
    probe = probe_fixture()
    if defect == "synthetic":
        probe["synthetic"] = True
    elif defect == "wrong_gpu":
        probe["gpu"] = "RTX 5070 Ti"
    elif defect == "slow":
        for sample in probe["samples"]:
            sample["seconds"] = [t * 4 for t in sample["seconds"]]
    elif defect == "missing":
        probe["samples"].pop()
    elif defect == "duplicate":
        probe["samples"].append(copy.deepcopy(probe["samples"][0]))
    elif defect == "missing_repeat":
        probe["samples"][0]["seconds"].pop()
    elif defect == "nonfinite":
        probe["samples"][0]["seconds"][0] = float("nan")
    elif defect == "oom":
        probe["samples"][0]["peak_reserved_bytes"] = 75 * 1024 ** 3
    elif defect == "free":
        probe["samples"][0]["min_free_bytes"] = 4 * 1024 ** 3
    else:
        probe["parity_max_abs"] = {"parity": .01, "nan_parity": float("nan"), "negative_parity": -.001}[defect]
    with pytest.raises(ValueError):
        P.evaluate(probe, volume(), 320)


def test_live_guard_prices_remaining_positions_and_calls_after_load():
    Q.rate_check({"v1": 10, "v2": 10}, {"v1": 10, "v2": 10}, {"v1": 100, "v2": 100}, 20, 320)
    with pytest.raises(ValueError, match="projection"):
        Q.rate_check({"v1": 30, "v2": 20}, {"v1": 10, "v2": 10}, {"v1": 100, "v2": 100}, 50, 320)
    with pytest.raises(ValueError, match="deadline"):
        Q.rate_check({"v1": 10, "v2": 10}, {"v1": 10, "v2": 10}, {"v1": 0, "v2": 0}, 321, 320)


def test_disabled_job_and_module_bypass_refuse_before_model_or_input(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("disabled path reached production")
    monkeypatch.setattr(Q, "checked_rows", forbidden)
    monkeypatch.setattr(Q, "production", forbidden)
    with pytest.raises(ValueError, match="launch disabled"):
        Q.admit()
    for phase in ("setup", "probe", "screen", "audit"):
        out = tmp_path / phase
        out.mkdir()
        assert Q.main([phase, "--out", str(out)]) == 3
        failure = json.loads((out / "failure.json").read_text())
        assert failure == {"phase": phase, "error_type": "ValueError", "status": "incomplete",
                           "private_exception_text_withheld": True}


def test_failure_export_retains_partial_predictions_and_no_success(tmp_path):
    (tmp_path / "RWKV7_CODE_LABEL_QUALIFICATION_COMPLETE").write_text("stale")
    (tmp_path / "v1.predictions.jsonl").write_text('{"run_id":"synthetic:partial"}\n')
    (tmp_path / "failure.json").write_text('{"status":"incomplete"}\n')
    # A private/unregistered file in the output directory is never exported.
    (tmp_path / "private.rows.jsonl").write_text("do not export")
    Q.finalize(tmp_path, 7)
    assert not (tmp_path / "RWKV7_CODE_LABEL_QUALIFICATION_COMPLETE").exists()
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["status"] == "incomplete" and receipt["ok"] is False
    assert receipt["export"]["verified"] is True and receipt["actual_cost_usd"] is None
    with tarfile.open(tmp_path / "qualification.tgz") as tar:
        assert set(tar.getnames()) == {"v1.predictions.jsonl", "failure.json", "SHA256SUMS.json"}
    assert (tmp_path / "result.json").stat().st_size < 32768


def test_finalize_refuses_synthetic_or_partial_success(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps({"screen_status": "complete", "rows_denominator": 2008,
        "synthetic": True, "promotable": False, "beats_clef": False, "paired_ok_rows": 2000,
        "input_sha256": {"rows": rule.ROWS_SHA256}}))
    with pytest.raises(ValueError):
        Q.finalize(tmp_path, 0)
    assert not (tmp_path / "RWKV7_CODE_LABEL_QUALIFICATION_COMPLETE").exists()


def test_registered_context_ships_only_exact_hash_pinned_subset_and_submit_fields():
    p = Q.params()
    assert p["context"]["data"] == [{"src": "qualification/index-rows.jsonl.gz", "src_env": "SSDM_CODE_LABEL_DEV_ROWS",
        "dest": "index-rows.jsonl.gz", "pin_sha256": rule.ROWS_SHA256}]
    args = launch.submit_args("rwkv7_code_label_h100", None, p, p["submit"], "a" * 64)
    assert args["command"].endswith("qualification_job.sh --batch")
    assert args["model"].endswith("@ce3630b854cedbd673569f4600c6cf744bca9dba")
    assert lint.checks_problems(args["checks"]) == []


def test_real_shell_executes_cpu_dry_run_and_refuses_batch(tmp_path):
    bash = shutil.which("bash")
    if sys.platform == "win32":
        bash = str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe")
    assert bash and Path(bash).is_file(), "bash required for shell exit proof"
    env = dict(os.environ, PYTHON=sys.executable, CUDA_VISIBLE_DEVICES="", SSDM_CODE_ROOT=str(ROOT))
    out = tmp_path / "shell-proof"
    result = subprocess.run([bash, str(ROOT / "jobs/rwkv7_code_label_h100/qualification_job.sh"), "--dry-run", str(out)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out / "report.json").read_text())
    assert report["synthetic"] is True and report["gpu_used"] is report["model_loaded"] is False
    assert report["rows_denominator"] == 12 and report["paired_ok_rows"] == 11
    assert report["promotable"] is report["beats_clef"] is False
    rejected = subprocess.run([bash, str(ROOT / "jobs/rwkv7_code_label_h100/qualification_job.sh"), "--batch"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=15)
    assert rejected.returncode == 2
    assert "Launch disabled" in rejected.stdout
    assert not any(p.name.endswith("COMPLETE") for p in out.iterdir())
