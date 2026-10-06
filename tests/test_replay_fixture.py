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
        enable_experimental_provenance=case["decider"].get("enable_experimental_provenance", False),
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


@pytest.mark.parametrize("case_index", range(48))
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


def _clef_systemone_answer(question, probabilities):
    """Verbatim logic of Cloudflare clef-flash joint_schema_model.systemone_answer (sha256 0e304cf7...)."""
    if question["type"] == "noul":
        return {"type": "noul", "noul": round(probabilities["true"], 4)}
    if question["type"] == "choice":
        options = [str(option) for option in question["criteria"]]
        choice = max(options, key=probabilities.__getitem__)
        return {"type": "choice", "choice": choice, "confidence": round(probabilities[choice], 4),
                "probabilities": {option: round(probabilities[option], 4) for option in options}}
    levels = [str(index) for index in range(len(question["criteria"]))]
    return {"type": "score",
            "score": round(sum(index * probabilities[level] for index, level in enumerate(levels)), 4),
            "confidence": round(max(probabilities[level] for level in levels), 4),
            "legend": dict(zip(levels, question["criteria"])),
            "probabilities": {level: round(probabilities[level], 4) for level in levels}}


def test_independent_recompute_from_recorded_logits():
    """Adapter-free: softmax of recorded logits + Clef's own answer logic == recorded wire answers.
    This verifies replay is a deterministic function of the recorded logits; it is not model evidence."""
    import math

    from decide.schema import request_fingerprint

    _, cases, responses = load()
    recs = {}
    for line in open(FIXTURE_DIR / "recordings.jsonl", encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            recs[r["fingerprint"]] = r
    checked = 0
    for case in cases:
        resp = responses[case["case_id"]]
        if resp["ssdm"]["status"] != "ok":
            continue
        logits = recs[request_fingerprint(case["request"])]["logits"]
        for qid, q in case["request"]["questions"].items():
            vals = logits[qid]
            m = max(vals.values())
            z = sum(math.exp(v - m) for v in vals.values())
            probs = {o: math.exp(v - m) / z for o, v in vals.items()}
            assert _clef_systemone_answer(q, probs) == resp["answers"][qid]
            checked += 1
    assert checked >= 30
