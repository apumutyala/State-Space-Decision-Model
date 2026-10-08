"""jobs/clef_baseline/run_job.sh, SHELL level: executes the real runner end to end on CPU (CLEF_DRY_RUN=1, the tiny
mocked Clef as CLEF_STANDIN). A previous runner bug passed every Python-level test (LESSONS L-10); job-sgihj died in a
runner step (composite) that no test executed (L-15). These run the script itself, through the same phases, exit-code
mapping, rate guard, strict scoring and report as the GPU job.

  * JOB=T (teacher export) needs no private data: a synthetic training set with the pointer's file names.
  * JOB=I ROWS=subset needs the kit (decision_index) and the private DEV rows ($SSDM_DATA/slices/dev-rows.jsonl.gz);
    it is skipped, loudly, without them. The stand-in subset includes 25:GPQA-Diamond:GPQA-Diamond:test:4, the row
    whose raw U+2028 crashed job-sgihj's composite, and a ContractNLI row longer than the stand-in's 16,384-token cap
    (decide = exhausted, Clef native = truncated), so the native arm and the composite really run.

Run with the kit:  uv run ... --with <ssdm-data>/decision-index --with-editable ./harness python -X utf8 -m pytest -q tests/clef/test_clef_job_sh.py
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "jobs" / "clef_baseline" / "run_job.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="no bash")
STANDIN_IDS = ["25:GPQA-Diamond:GPQA-Diamond:test:4", "25:GPQA-Diamond:GPQA-Diamond:test:0",
               "25:GPQA-Diamond:GPQA-Diamond:test:1", "25:GPQA-Diamond:GPQA-Diamond:test:102",
               "12:ANLI:ANLI:test_r3:13618cc3-27fb-4048-812b-c96736f92223",
               "12:ANLI:ANLI:test_r3:2048c1ad-9f8e-4e83-af86-ecd5efdefc26",
               "12:ANLI:ANLI:test_r3:24532622-9265-4031-a69f-05ba3d4f8dd8",
               "11:ContractNLI:ContractNLI:test:11"]


def _sh(p: Path) -> str:
    cyg = shutil.which("cygpath")
    if os.name == "nt" and cyg:
        return subprocess.run([cyg, "-u", str(p)], capture_output=True, text=True, check=True).stdout.strip()
    return str(p)


def _env(tmp_path: Path, data: Path, **kw) -> dict:
    env = {**os.environ, "CLEF_DRY_RUN": "1", "CLEF_STANDIN": "clef_mocks:load_standin", "CLEF_APP": _sh(ROOT),
           "CLEF_DATA": _sh(data), "GMN_OUTPUT_DIR": _sh(tmp_path / "out"), "GMN_CHECKPOINT_DIR": _sh(tmp_path / "ck"),
           "HF_HOME": _sh(tmp_path / "hf"), "GMN_METADATA_URL": "http://127.0.0.1:9", "CUDA_VISIBLE_DEVICES": "",
           "PYTHONPATH": os.pathsep.join([str(ROOT / "tests" / "clef"), str(ROOT)]), "HB_EVERY_S": "5",
           "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"], **kw}
    for k in ("HF_HUB_CACHE", "GMN_RESULT_PATH", "SSDM_UNSEAL_TEST", "JOB_PARAMS", "CLEF_DRY_FAULT"):
        if k not in kw:
            env.pop(k, None)
    return env


def _run(env, timeout=1500):
    return subprocess.run([BASH, str(RUNNER)], env=env, cwd=ROOT, capture_output=True, text=True, timeout=timeout)


def _result(tmp_path, tag):
    """The full report (report.json) when the job got that far; result.json (compact, < 32 KiB: job-ftpnf) otherwise."""
    d = tmp_path / "out" / tag
    res = d / "result.json"
    assert res.stat().st_size < 32 * 1024, "the platform refuses a result file over 32 KiB"
    full = d / "report.json"
    r = json.loads((full if full.exists() else res).read_text(encoding="utf-8"))
    if full.exists():
        c = json.loads(res.read_text(encoding="utf-8"))
        assert c["ok"] == r["ok"] and c["full_report"] == "report.json"
    return r


# ----------------------------------------------------------------------------- teacher export (no private data)
def _synthetic_train_set(d: Path, extra: list[dict] | None = None, n: int = 6) -> list[str]:
    """Training records in the PR #44 shape, under the pointer's file names, with a manifest (sha256 of each exact
    line) and a content map (train.cohort.record_content_hash), so every identity check runs for real."""
    from train.cohort import record_content_hash

    ptr = json.loads((ROOT / "data/manifests/sample-150000-seed20261006.pointer.json").read_text(encoding="utf-8"))
    recs = [{"id": f"m2:synthetic_standin:{i}", "family": "standin", "source": "standin", "state":
             {"ticket": f"stand-in ticket {i}: the printer on floor {i} is jammed again", "priority": i % 3},
             "questions": {"team": {"type": "choice", "instructions": "Which team handles it?",
                                    "criteria": {"it": "IT support", "facilities": "Facilities"}},
                           "urgent": {"type": "noul", "instructions": "Is it urgent?"}},
             "labels": {"team": "it", "urgent": False}} for i in range(n)] + (extra or [])
    d.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in recs]
    with gzip.open(d / "records.jsonl.gz", "wt", encoding="utf-8", newline="\n") as f:
        f.write("".join(x + "\n" for x in lines))
    with gzip.open(d / ptr["manifest"], "wt", encoding="utf-8", newline="\n") as f:
        f.write("".join(json.dumps({"id": r["id"], "sha256": hashlib.sha256(x.encode()).hexdigest()}) + "\n"
                        for r, x in zip(recs, lines)))
    with gzip.open(d / ptr["content_map"], "wt", encoding="utf-8", newline="\n") as f:
        f.write("".join(json.dumps({"id": r["id"], "content_sha256": record_content_hash(r)}) + "\n" for r in recs))
    return [r["id"] for r in recs]


def test_teacher_dry_run_end_to_end(tmp_path):
    pytest.importorskip("torch")
    ids = _synthetic_train_set(tmp_path / "data" / "m2-150k")
    p = _run(_env(tmp_path, tmp_path / "data", RELEASE="clef-flash", JOB="T", CLEF_TEACHER_N=str(len(ids))))
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    res = _result(tmp_path, "clef-flash-teacher")
    assert res["ok"] and res["dry_run"] is True and res["job"] == "T"
    t = res["teacher"]
    assert t["standin"] is True and t["rows"] == len(ids) and t["status_counts"] == {"ok": len(ids)}
    assert t["selection"]["guard_refused"] == 0 and t["selection"]["cohort_hits"] == {"by_id": 0, "by_content": 0}
    assert t["selection"]["heldout_scan"]["hits"] == 0 and "STANDIN" in t["guard_note"]
    rows = [json.loads(x) for x in (tmp_path / "out/clef-flash-teacher/teacher/teacher-clef-flash.jsonl")
            .read_text(encoding="utf-8").split("\n") if x]
    assert sorted(r["id"] for r in rows) == sorted(ids)
    for r in rows:  # per-option probabilities, every question, each a distribution
        assert set(r["raw_probabilities"]) == {"team", "urgent"}
        for dist in r["raw_probabilities"].values():
            assert abs(sum(dist.values()) - 1) < 1e-6
    assert "labels" not in json.dumps(rows)  # labels never reach the export
    rg = json.loads((tmp_path / "out/clef-flash-teacher/env/rate_guard_export.json").read_text(encoding="utf-8"))
    assert rg["exit"] == 0 and rg["total"] == len(ids)


@pytest.mark.parametrize("kind", ["dev_row", "index_metadata", "heldout_fragment", "frozen_family"])
def test_teacher_refuses_contaminated_set_before_any_download(tmp_path, kind):
    pytest.importorskip("torch")
    if kind == "dev_row":  # a training record whose run_id is a DEV manifest id
        bad = {"id": "m2:standin:bad", "run_id": STANDIN_IDS[0], "state": "x", "questions": {"q": {"type": "noul"}}}
    elif kind == "index_metadata":
        bad = {"id": "m2:standin:bad", "_evaluation": {"run_id": "whatever"}, "state": "x",
               "questions": {"q": {"type": "noul"}}}
    elif kind == "frozen_family":  # targets a frozen held-out family of heldout-draw-v1 (harness.heldout, #95)
        bad = {"id": "m2:standin:bad", "source": "gen_musr", "state": "x", "questions": {"q": {"type": "noul"}}}
    else:  # verbatim held-out D1-B template text inside the state
        from safety_eval.attacks import D1_TEMPLATES_B

        bad = {"id": "m2:standin:bad", "state": "note: " + next(iter(D1_TEMPLATES_B.values())),
               "questions": {"q": {"type": "noul", "instructions": "ok?"}}}
    ids = _synthetic_train_set(tmp_path / "data" / "m2-150k", extra=[bad])
    p = _run(_env(tmp_path, tmp_path / "data", RELEASE="clef", JOB="T", CLEF_TEACHER_N=str(len(ids))))
    assert p.returncode == 20, p.stdout[-2000:] + p.stderr[-2000:]
    out = tmp_path / "out" / "clef-teacher"
    assert _result(tmp_path, "clef-teacher")["stage"] == "teacher_select_contamination"
    assert "CONTAMINATION GUARD" in (out / "teacher-select.log").read_text(encoding="utf-8")
    assert not (out / "standin-snapshot").exists()  # refused before the (stand-in) weights step
    assert not list((out / "teacher").glob("teacher-*.jsonl"))
    if kind == "frozen_family":
        assert "frozen held-out family" in (out / "teacher-select.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("var", ["CLEF_STANDIN", "CLEF_TEACHER_N", "CLEF_DRY_FAULT", "CLEF_OVERLENGTH"])
def test_runner_refuses_dry_run_overrides_in_production(tmp_path, var):
    env = _env(tmp_path, tmp_path, RELEASE="clef", JOB="T", CLEF_DRY_RUN="0", **{var: "x"})
    if var != "CLEF_STANDIN":
        env.pop("CLEF_STANDIN", None)
    p = _run(env, timeout=60)
    assert p.returncode == 2 and f"{var} is for CLEF_DRY_RUN=1 only" in p.stdout


def test_runner_refuses_overlength_mode_for_clef27b(tmp_path):
    assert _run(_env(tmp_path, tmp_path, RELEASE="clef", JOB="I", ROWS="overlength"), timeout=60).returncode == 2


# ----------------------------------------------------------------------------- Index subset (kit + private DEV rows)
def _dev_data() -> Path | None:
    from harness import config

    return config.DATA_ROOT if (config.DATA_ROOT / "slices" / "dev-rows.jsonl.gz").exists() else None


def _standin_subset(tmp_path: Path) -> Path:
    from jobs.common import dev_subset

    ids = sorted(STANDIN_IDS)
    d = {"_comment": "stand-in subset for the CPU dry run", **dev_subset.manifest_hashes(),
         "ids_sha256": dev_subset.ids_sha256(ids), "run_ids": ids}
    p = tmp_path / "standin_subset.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def _need_kit_and_dev():
    pytest.importorskip("torch")
    pytest.importorskip("decision_index", reason="kit not installed: run with --with <ssdm-data>/decision-index")
    data = _dev_data()
    if data is None:
        pytest.skip("private DEV rows not present ($SSDM_DATA/slices/dev-rows.jsonl.gz)")
    return data


def _standin_overlength(tmp_path: Path) -> Path:
    """The stand-in's char-level tokenizer makes ContractNLI test:11 (~19k chars) the over-length row; at 32768 it fits."""
    p = tmp_path / "standin_overlength.json"
    p.write_text(json.dumps({"rows": [{"run_id": "11:ContractNLI:ContractNLI:test:11", "tokens": 19500}]}),
                 encoding="utf-8")
    return p


def test_index_subset_dry_run_end_to_end(tmp_path):
    data = _need_kit_and_dev()
    env = _env(tmp_path, data, RELEASE="clef", JOB="I", ROWS="subset", CLEF_SUBSET=_sh(_standin_subset(tmp_path)),
               CLEF_N_PARITY="2", CLEF_OVERLENGTH=_sh(_standin_overlength(tmp_path)))
    p = _run(env)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    out = tmp_path / "out" / "clef-index-subset"
    res = _result(tmp_path, "clef-index-subset")
    assert res["ok"] and res["dry_run"] is True and res["rows"] == "subset" and res["s1_ok"] is True
    text = (out / "index/decide/results.jsonl").read_text(encoding="utf-8")
    assert " " in text  # the job-sgihj crash class really went through S3 and the composite
    assert len(text.splitlines()) > len([x for x in text.split("\n") if x])
    assert res["composite_provenance"]["decide_rows"] == len(STANDIN_IDS)
    assert "11:ContractNLI:ContractNLI:test:11" in res["composite_provenance"]["replaced_run_ids"]
    assert res["index_dev_decide"]["rows"] == len(STANDIN_IDS) and res["latency_decide"]["n"] >= 0
    assert res["ece_decide"] is not None and res["rate_guard"]["index"]["exit"] == 0
    assert (out / "SHA256SUMS").exists() and not (out / "env" / "fla_guard.log").exists()
    # edition 0.2.1 explicit + a 0.3 report; headline = native composite; 32k sensitivity substituted; paired flag
    assert res["index_dev_decide"]["slice_index"] and res["index_dev_decide_03"]["slice_index"]
    assert res["headline"]["metric"] == "index_dev_native_composite" and res["headline"]["value"] == res["index_dev_native_composite"]
    sens = res["sensitivity_32k"]
    assert sens["provenance"]["substituted_run_ids"] == ["11:ContractNLI:ContractNLI:test:11"]
    assert sens["provenance"]["status_before"] == {"unsupported": 1} and sens["provenance"]["status_after"] == {"ok": 1}
    assert sens["index_dev_decide_32k_substituted"]["rows"] == len(STANDIN_IDS)
    assert res["paired_compare"] == "absent"  # dry run without the job-sgihj reference: flagged, not silently skipped


def test_index_subset_dry_run_reference_gdn_path(tmp_path):
    """JOB_SPEC 4d: CLEF_GDN_PATH=reference end to end. Every Index phase's environment.json says gdn_path=reference
    with all GDN layers on it; s1_ok = parity AND that; kernel_path still measured and reported (per_row timings)."""
    data = _need_kit_and_dev()
    env = _env(tmp_path, data, RELEASE="clef", JOB="I", ROWS="subset", CLEF_SUBSET=_sh(_standin_subset(tmp_path)),
               CLEF_N_PARITY="2", CLEF_OVERLENGTH=_sh(_standin_overlength(tmp_path)), CLEF_GDN_PATH="reference")
    p = _run(env)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    res = _result(tmp_path, "clef-index-subset")
    assert res["ok"] and res["gdn_path"] == "reference" and res["gdn_decide_used_reference"] is True
    assert res["s1_ok"] is True and res["parity_ok"] is True
    ph = res["gdn"]["phases"]
    assert set(ph) == {"decide", "decide_32k", "native"}
    assert all(v["gdn_path"] == "reference" and v["gdn_layers_on_reference"] == v["gdn_layers_found"] >= 1
               for v in ph.values())
    assert res["gdn"]["sens"]["gdn_path"] == "reference" and res["gdn"]["sens"]["fallback"] is None
    probes = json.loads((tmp_path / "out/clef-index-subset/probes.json").read_text(encoding="utf-8"))
    assert probes["gdn_path"] == "reference" and probes["kernel_path_gating"] is False
    assert probes["kernel_path"]["per_row"] and "ref_ms" in probes["kernel_path"]["per_row"][0]
    pre = json.loads((tmp_path / "out/clef-index-subset/env/preflight_index.json").read_text(encoding="utf-8"))
    assert pre["gdn_path"] == "reference"


def test_index_dry_run_reference_gdn_path_clef_flash(tmp_path):
    """JOB_SPEC 4d, Clef-flash on the reference (launch mode index_full_reference_flash runs ROWS=full; the subset is the
    CPU-sized stand-in for the same code path): every Index phase on the reference, the 32k arm substituted into this
    job's own decide results (the job-sgihj reference is not needed, and no paired compare runs for Clef-flash)."""
    data = _need_kit_and_dev()
    env = _env(tmp_path, data, RELEASE="clef-flash", JOB="I", ROWS="subset", CLEF_SUBSET=_sh(_standin_subset(tmp_path)),
               CLEF_N_PARITY="2", CLEF_OVERLENGTH=_sh(_standin_overlength(tmp_path)), CLEF_GDN_PATH="reference")
    p = _run(env)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    res = _result(tmp_path, "clef-flash-index-subset")
    assert res["ok"] and res["gdn_path"] == "reference" and res["s1_ok"] is True
    assert all(v["gdn_path"] == "reference" and v["gdn_layers_on_reference"] == v["gdn_layers_found"] >= 1
               for v in res["gdn"]["phases"].values())
    assert res["sensitivity_32k"]["provenance"]["substituted_run_ids"] == ["11:ContractNLI:ContractNLI:test:11"]


def test_runner_refuses_reference_gdn_path_outside_index_jobs(tmp_path):
    env = _env(tmp_path, tmp_path, RELEASE="clef-flash", JOB="I", ROWS="overlength", CLEF_GDN_PATH="reference")
    assert _run(env, timeout=120).returncode == 2
    env = _env(tmp_path, tmp_path, RELEASE="clef", JOB="I", ROWS="full", CLEF_GDN_PATH="bogus")
    assert _run(env, timeout=120).returncode == 2


@pytest.mark.parametrize("fault", ["truncate_decide", "drop_decide_row"])
def test_index_subset_dry_run_truncated_results_exit_17(tmp_path, fault):
    data = _need_kit_and_dev()
    env = _env(tmp_path, data, RELEASE="clef", JOB="I", ROWS="subset", CLEF_SUBSET=_sh(_standin_subset(tmp_path)),
               CLEF_N_PARITY="2", CLEF_DRY_FAULT=fault)
    p = _run(env)
    assert p.returncode == 17, p.stdout[-3000:] + p.stderr[-3000:]
    res = _result(tmp_path, "clef-index-subset")
    assert res["ok"] is False and res["exit"] == 17 and res["stage"] == "score_decide"


def test_overlength_flash_dry_run_on_the_sgihj_reference(tmp_path):
    """ROWS=overlength end to end: needs the real job-sgihj decide results (local only:
    CLEF_TEST_SGIHJ_DECIDE=<path to art-8y6d8 index/decide/results.jsonl>), the kit and the private DEV rows."""
    dev = _need_kit_and_dev()
    ref = os.environ.get("CLEF_TEST_SGIHJ_DECIDE")
    if not ref or not Path(ref).exists():
        pytest.skip("set CLEF_TEST_SGIHJ_DECIDE to job-sgihj's decide results.jsonl")
    data = tmp_path / "data"
    (data / "slices").mkdir(parents=True)
    (data / "reference").mkdir()
    shutil.copy(dev / "slices" / "dev-rows.jsonl.gz", data / "slices" / "dev-rows.jsonl.gz")
    shutil.copy(ref, data / "reference" / "clef_flash.decide.results.jsonl")
    env = _env(tmp_path, data, RELEASE="clef-flash", JOB="I", ROWS="overlength",
               CLEF_OVERLENGTH=_sh(_standin_overlength(tmp_path)))
    p = _run(env)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    res = _result(tmp_path, "clef-flash-index-overlength")
    sens = res["sensitivity_32k"]
    assert sens["provenance"]["substituted_run_ids"] == ["11:ContractNLI:ContractNLI:test:11"]
    assert sens["index_dev_decide_32k_substituted"]["slice_index"] and sens["reference_decide"]["slice_index"]
    assert abs(sens["reference_decide"]["slice_index"]["index"]["point"] - 56.79) < 0.01  # job-sgihj, edition 0.2.1
    assert sens["compare_32k_vs_16k"]["diff"]
