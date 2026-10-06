"""The replay fixture handed to Track L reproduces its recorded responses exactly."""

import hashlib
import json

import pytest
from conftest import FIXTURE_DIR, shuffled

from decide import Decider, OptionCountTemperature
from decide.backends import ReplayBackend
from decide.backends.replay import normalize_response

pytestmark = pytest.mark.skipif(not (FIXTURE_DIR / "manifest.json").exists(), reason="fixture not generated")


def load():
    manifest = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))
    cases = [json.loads(line) for line in open(FIXTURE_DIR / "requests.jsonl", encoding="utf-8") if line.strip()]
    responses = {
        r["case_id"]: r["response"]
        for r in (json.loads(line) for line in open(FIXTURE_DIR / "responses.jsonl", encoding="utf-8") if line.strip())
    }
    return manifest, cases, responses


def decider_for(manifest, case, backend):
    cal = manifest["calibration"]
    calibration = OptionCountTemperature(t0=cal["t0"], t1=cal["t1"], t_min=cal["t_min"], fitted_on=cal["fitted_on"])
    return Decider(
        backend,
        calibration=calibration,
        enable_v0_1_deltas=case["decider"]["enable_v0_1_deltas"],
        backend_max_input_tokens=manifest["revision"].get("max_context"),
    )


def test_manifest_hashes_match_files():
    manifest, _, _ = load()
    for name, digest in manifest["files_sha256"].items():
        assert hashlib.sha256((FIXTURE_DIR / name).read_bytes()).hexdigest() == digest, name


def test_fixture_size_and_coverage():
    manifest, cases, responses = load()
    assert 30 <= len(cases) == len(responses) == manifest["n_cases"]
    types = {q["type"] for c in cases for q in c["request"].get("questions", {}).values() if isinstance(q, dict)}
    assert {"noul", "choice", "score"} <= types
    statuses = {r["ssdm"]["status"] for r in responses.values()}
    assert statuses == {"ok", "unsupported", "exhausted", "error"}


@pytest.mark.parametrize("case_index", range(40))
def test_replay_reproduces_recorded_response(case_index):
    manifest, cases, responses = load()
    if case_index >= len(cases):
        pytest.skip("no such case")
    case = cases[case_index]
    if case["timing_dependent"]:
        pytest.skip("timing-dependent case (deadline); not exactly replayable")
    d = decider_for(manifest, case, ReplayBackend(FIXTURE_DIR))
    got = normalize_response(d.decide(case["request"]))
    assert got == normalize_response(responses[case["case_id"]])


def test_replay_equivariance_and_miss():
    manifest, cases, responses = load()
    backend = ReplayBackend(FIXTURE_DIR)
    for case in cases:
        if responses[case["case_id"]]["ssdm"]["status"] != "ok":
            continue
        d = decider_for(manifest, case, backend)
        ref = d.decide(case["request"])
        perm = d.decide(shuffled(case["request"], 3))
        assert perm["ssdm"]["served_probabilities"] == ref["ssdm"]["served_probabilities"]
    req = json.loads(json.dumps(cases[0]["request"]))
    req["state"] = "a state that was never recorded"
    miss = decider_for(manifest, cases[0], backend).decide(req)
    assert (miss["ssdm"]["status"], miss["ssdm"]["error"]["code"]) == ("unsupported", "replay_miss")


def test_replay_reports_recorded_parent_revision():
    manifest, cases, _ = load()
    backend = ReplayBackend(FIXTURE_DIR)
    assert backend.revision()["revision_id"] == manifest["revision"]["revision_id"]
