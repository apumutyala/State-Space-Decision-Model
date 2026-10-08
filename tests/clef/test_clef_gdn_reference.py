"""CLEF_GDN_PATH=reference (jobs/clef_baseline/JOB_SPEC.md 4d; job-jxevr / art-g7mqf kernel_path 0.027 > 0.02).

The tolerance is unchanged; the Index phases run on the torch reference GDN instead. These tests pin: the S1 exit rule
(parity always gates; kernel_path gates the fast path only), the engine really switching every GDN layer and saying so
in its provenance (fast provenance byte-identical), and the report refusing a reference job whose phases were not on it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from baselines.clef_probes import KERNEL_MARGIN, KERNEL_TOL, s1_exit
from jobs.clef_baseline import summarize as SM

ROOT = Path(__file__).resolve().parents[2]


def test_tolerances_are_the_preregistered_ones():
    assert KERNEL_TOL == 2e-2 and KERNEL_MARGIN == 2e-2


@pytest.mark.parametrize("parity,kernel,path,code", [
    (True, True, "fast", 0), (True, False, "fast", 11), (False, True, "fast", 11),
    (True, False, "reference", 0),   # job-jxevr's case: reported, not gating, under the reference run
    (False, True, "reference", 11),  # parity vs systemone always gates
    (False, False, "reference", 11)])
def test_s1_exit(parity, kernel, path, code):
    assert s1_exit(parity, kernel, path) == code


def test_s1_exit_rejects_unknown_path():
    with pytest.raises(ValueError):
        s1_exit(True, True, "fla")


def _env(d: Path, phase: str, ms: dict):
    p = d / "index" / phase
    p.mkdir(parents=True, exist_ok=True)
    (p / "environment.json").write_text(json.dumps({"model_source": ms}), encoding="utf-8")


REF = {"gdn_path": "reference", "gdn_layers_found": 48, "gdn_layers_switched": 48, "gdn_layers_on_reference": 48,
       "backend_revision": "clef:x", "kernel_stack": {"torch": "2.13.0"}}


def test_gdn_block_requires_every_phase_on_the_reference(tmp_path):
    for ph in ("decide", "decide_32k", "native"):
        _env(tmp_path, ph, REF)
    assert SM.gdn_block(tmp_path, "reference")["decide_used_reference"] is True
    _env(tmp_path, "native", {"backend_revision": "clef:x"})  # a fast-path phase (no gdn_path key = fast)
    g = SM.gdn_block(tmp_path, "reference")
    assert g["decide_used_reference"] is False and g["phases"]["native"]["gdn_path"] == "fast"


def test_gdn_block_partial_switch_is_not_reference(tmp_path):
    _env(tmp_path, "decide", {**REF, "gdn_layers_on_reference": 47})
    assert SM.gdn_block(tmp_path, "reference")["decide_used_reference"] is False


def test_gdn_block_labelled_32k_fallback_does_not_void_the_headline(tmp_path):
    _env(tmp_path, "decide", REF)
    _env(tmp_path, "native", REF)
    _env(tmp_path, "decide_32k", {"backend_revision": "clef:x"})
    (tmp_path / "env").mkdir()
    (tmp_path / "env" / "sens_gdn_path.json").write_text(json.dumps(
        {"gdn_path_requested": "reference", "gdn_path": "fast", "fallback": "reference out of memory at 32k"}))
    g = SM.gdn_block(tmp_path, "reference")
    assert g["decide_used_reference"] is True and g["sens"]["gdn_path"] == "fast"


def test_engine_reference_option_switches_every_gdn_layer(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("decision_index")
    import sys

    sys.path.insert(0, str(ROOT / "tests" / "clef"))
    monkeypatch.setenv("CLEF_DRY_RUN", "1")
    monkeypatch.setenv("CLEF_STANDIN", "clef_mocks:load_standin")
    from baselines.clef_index_engine import ClefIndexEngine

    fast = ClefIndexEngine(model_dir="x", release="clef", device="cpu", dtype="float32")
    assert "gdn_path" not in fast.provenance  # fast provenance unchanged (job-sgihj / job-d2u54 identical)
    ref = ClefIndexEngine(model_dir="x", release="clef", device="cpu", dtype="float32", gdn_path="reference")
    pv = ref.provenance
    assert pv["gdn_path"] == "reference" and pv["gdn_layers_found"] >= 1
    assert pv["gdn_layers_on_reference"] == pv["gdn_layers_found"] and "torch" in pv["kernel_stack"]
    with pytest.raises(ValueError):
        ClefIndexEngine(model_dir="x", release="clef", device="cpu", dtype="float32", gdn_path="other")


def test_runner_wires_the_option_into_all_three_index_phases():
    sh = (ROOT / "jobs/clef_baseline/run_job.sh").read_text(encoding="utf-8")
    assert sh.count('"${GDNOPT[@]}"') == 3  # S2 decide, S2c 32k, S2b native (the labelled OOM re-run has none)
    assert '--gdn-path "$GDN_PATH"' in sh and "job_params_index_full_reference.json" in sh
    assert "sens_gdn_path.json" in sh and 'mv "$IDX/decide_32k" "$IDX/decide_32k_reference_oom"' in sh
