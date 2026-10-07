"""CPU checks for jobs/rwkv7_code_label_screen/; no model or private row access."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from jobs.rwkv7_code_label_screen import rule as R
from jobs.rwkv7_code_label_screen import runner as S

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def proof():
    # These are required for the CPU exit proof; missing packages must fail,
    # rather than silently skipping the only production-path check.
    from jobs.rwkv7_code_label_screen.synthetic import fixture

    rows, deciders = fixture()
    revisions = {arm: decider.revision() for arm, decider in deciders.items()}
    arms = {arm: {row["_evaluation"]["run_id"]: S.capture(row, decider) for row in rows}
            for arm, decider in deciders.items()}
    return rows, arms, revisions


def test_literal_registration_and_params_are_fixed():
    assert R.RULE_VERSION == "rwkv7-code-label-v2-screen-v1"
    assert R.EDITION == "0.2.1"
    assert R.ROWS == 2008 and R.MIN_OK_ROWS == 2000
    assert R.ROWS_SHA256 == "406d40e6891690fad48fb82252c44b89a5337c603ff79e03ceb11fffc4eb28f2"
    assert R.IDS_SHA256 == "6e4cb8dfac94a40a6e014e05744ff775aebcdf99d7a572190034e6bc026095c3"
    assert R.MANIFEST_SHA256 == "26cd6bc0194efdb7a344b9313909483ad0bc9ab739beb26cbae698fc980b964a"
    assert R.BOOTSTRAP_REPLICATES == 2000 and R.BOOTSTRAP_SEED == 20261007
    assert R.CI_LEVEL == 0.95 and R.PROBABILITY_TOLERANCE == 1e-9
    assert R.REQUIRED_FLOORS == ("safety", "calibration", "abstention", "control", "interpretability", "clef", "clef_flash")
    p = json.loads((S.HERE / "screen_params.json").read_text())
    assert p["decision_rule"] == "jobs/rwkv7_code_label_screen/rule.py"
    assert p["launch_enabled"] is False and p["cost_usd"] == 0
    assert p["rows"] == R.ROWS and p["minimum_ok_rows_per_arm_and_paired"] == R.MIN_OK_ROWS
    assert p["edition"] == R.EDITION and p["bootstrap_replicates"] == R.BOOTSTRAP_REPLICATES
    assert p["bootstrap_seed"] == R.BOOTSTRAP_SEED and p["confidence_level"] == R.CI_LEVEL
    assert p["full_denominator"] is True and p["allow_partial"] is False
    subset = S.checked_cohort()  # committed DEV IDs/manifest only; no items
    assert len(subset["run_ids"]) == R.ROWS


def test_real_production_scoring_summary_and_exhausted_denominator(proof):
    rows, arms, revisions = proof
    result = S.summarize(rows, arms, revisions, synthetic=True)
    assert result["rows_denominator"] == 12 and result["paired_ok_rows"] == 11
    for arm in ("v1", "v2"):
        report = result["arms"][arm]
        assert report["status_counts"] == {"ok": 11, "exhausted": 1}
        assert report["raw_audit"]["rows"] == 12
        assert report["raw_audit"]["invalid_raw_distributions"] == 0
        assert report["calibration_diagnostic"]["n"] == 22
        assert 0 < report["slice_index"]["index"]["point"] < 100
    paired = result["paired_v2_minus_v1"]
    assert paired["B"] == 2000 and paired["level"] == 0.95
    assert paired["diff"]["index"]["point"] == paired["diff"]["index"]["lo"] == paired["diff"]["index"]["hi"] == 0
    assert result["screen_status"] == "complete"
    assert result["promotable"] is result["beats_clef"] is result["spend_authorized"] is False
    assert set(result["floors"].values()) == {"unmeasured"}


@pytest.mark.parametrize("mutation", ["missing", "extra", "raw_absent", "nan", "negative", "sum", "option_missing",
                                      "option_extra", "served_changed", "calibration", "mask", "revision", "fingerprint",
                                      "answer", "error", "no_reason", "partial_refusal", "partial_refusal_served"])
def test_artifact_corruption_fails_closed(proof, mutation):
    rows, arms, revisions = copy.deepcopy(proof)
    rec = arms["v2"]["synthetic:0"]
    if mutation == "missing":
        del arms["v2"]["synthetic:0"]
    elif mutation == "extra":
        arms["v2"]["stray"] = rec
    elif mutation == "raw_absent":
        del rec["raw_output"]
    elif mutation in ("nan", "negative", "sum", "option_missing", "option_extra"):
        probs = rec["raw_output"]["raw_probabilities"]["q0"]
        if mutation == "option_missing":
            del probs["b"]
        elif mutation == "option_extra":
            probs["other"] = 0.0
        else:
            probs["a"] = {"nan": float("nan"), "negative": -0.1, "sum": 0.9}[mutation]
        rec["screen_evidence"]["served_probabilities"] = copy.deepcopy(rec["raw_output"]["raw_probabilities"])
    elif mutation == "served_changed":
        rec["screen_evidence"]["served_probabilities"]["q0"]["a"] = 0.1
    elif mutation == "calibration":
        rec["screen_evidence"]["served_config"]["calibration"]["t0"] = 2
    elif mutation == "mask":
        rec["screen_evidence"]["served_config"]["policy_masking"] = "restricted"
    elif mutation == "revision":
        rec["screen_evidence"]["revision_id"] = "changed"
    elif mutation == "fingerprint":
        rec["screen_evidence"]["request_fingerprint"] = "changed"
    elif mutation == "answer":
        rec["response"]["answers"]["q0"]["choice"] = "b"
    elif mutation == "error":
        rec["status"] = "error"
    elif mutation == "no_reason":
        arms["v2"]["synthetic:11"]["error"] = None
    elif mutation == "partial_refusal":
        arms["v2"]["synthetic:11"]["response"] = rec["response"]
    elif mutation == "partial_refusal_served":
        arms["v2"]["synthetic:11"]["screen_evidence"]["served_probabilities"] = rec["screen_evidence"]["served_probabilities"]
    with pytest.raises(ValueError):
        S.summarize(rows, arms, revisions, synthetic=True)


@pytest.mark.parametrize("mutation", ["source", "tokenizer", "renderer", "runtime", "model"])
def test_changed_identity_fails_closed(proof, mutation):
    rows, arms, revisions = copy.deepcopy(proof)
    rev = revisions["v2"]
    if mutation == "source":
        rev["inference_config"]["source_sha256"]["state_fork.py"] = "changed"
    elif mutation == "tokenizer":
        rev["files_sha256"]["tokenizer"] = "changed"
    elif mutation == "renderer":
        rev["prompt_template"]["sha256"] = "changed"
    elif mutation == "runtime":
        rev["inference_config"]["dtype"] = "changed"
        rev["inference_config_sha256"] = __import__("hashlib").sha256(S.canonical_json(rev["inference_config"]).encode()).hexdigest()
    else:
        rev["model_id"] = "changed"
    with pytest.raises(ValueError):
        S.summarize(rows, arms, revisions, synthetic=True)


IDENTITY_PATHS = [(key,) for key in ("revision_id", "backend", "family", "model_id", "snapshot", "files_sha256",
                                    "prompt_template", "inference_config", "inference_config_sha256", "adapter_version",
                                    "max_context", "notes")]
IDENTITY_PATHS += [("files_sha256", key) for key in ("weights", "tokenizer")]
IDENTITY_PATHS += [("prompt_template", key) for key in ("id", "sha256")]
IDENTITY_PATHS += [("inference_config", key) for key in ("dtype", "device", "torch", "transformers", "kernel_stack")]


@pytest.mark.parametrize("path", IDENTITY_PATHS, ids=lambda path: ".".join(path))
def test_identity_field_deleted_from_both_arms_fails(proof, path):
    rows, arms, revisions = copy.deepcopy(proof)
    for revision in revisions.values():
        target = revision if len(path) == 1 else revision[path[0]]
        del target[path[-1]]
        if path[0] == "inference_config" and len(path) > 1:
            # Even a correctly rehashed, equally incomplete config must fail.
            revision["inference_config_sha256"] = __import__("hashlib").sha256(
                S.canonical_json(revision["inference_config"]).encode()).hexdigest()
    with pytest.raises(ValueError, match="missing"):
        S.summarize(rows, arms, revisions, synthetic=True)


@pytest.mark.parametrize("path,value", [
    (("revision_id",), None), (("backend",), []), (("family",), 7), (("model_id",), ""),
    (("snapshot",), []), (("snapshot",), ""), (("files_sha256",), []),
    (("files_sha256", "weights"), None), (("files_sha256", "tokenizer"), []),
    (("prompt_template",), None), (("prompt_template", "id"), []), (("prompt_template", "sha256"), ""),
    (("inference_config",), []), (("inference_config_sha256",), []), (("adapter_version",), None),
    (("max_context",), True), (("notes",), []), (("inference_config", "dtype"), None),
    (("inference_config", "device"), []), (("inference_config", "torch"), ""),
    (("inference_config", "transformers"), []), (("inference_config", "kernel_stack"), None),
])
def test_matching_malformed_identities_fail(proof, path, value):
    rows, arms, revisions = copy.deepcopy(proof)
    for revision in revisions.values():
        target = revision if len(path) == 1 else revision[path[0]]
        target[path[-1]] = copy.deepcopy(value)
        if path[0] == "inference_config" and len(path) > 1:
            revision["inference_config_sha256"] = __import__("hashlib").sha256(
                S.canonical_json(revision["inference_config"]).encode()).hexdigest()
    with pytest.raises(ValueError, match="malformed"):
        S.summarize(rows, arms, revisions, synthetic=True)


@pytest.mark.parametrize("mutation", ["model_id", "snapshot", "weights", "runtime", "revision_id"])
def test_shared_identity_changes_with_stale_or_arbitrary_revision_ids_fail(proof, mutation):
    rows, arms, revisions = copy.deepcopy(proof)
    for arm, revision in revisions.items():
        if mutation == "weights":
            revision["files_sha256"]["weights"] = "a" * 64
        elif mutation == "runtime":
            revision["inference_config"]["dtype"] = "bfloat16"
            revision["inference_config_sha256"] = __import__("hashlib").sha256(
                S.canonical_json(revision["inference_config"]).encode()).hexdigest()
        elif mutation == "revision_id":
            revision["revision_id"] = "arbitrary-claimed-revision"
            for record in arms[arm].values():
                record["screen_evidence"]["revision_id"] = revision["revision_id"]
        else:
            revision[mutation] = "different-model-or-snapshot"
    with pytest.raises(ValueError, match="inconsistent revision ID"):
        S.summarize(rows, arms, revisions, synthetic=True)


def test_synthetic_identity_sentinels_and_explicit_null_transformers_are_valid(proof):
    _, _, revisions = copy.deepcopy(proof)
    for arm, revision in revisions.items():
        assert revision["snapshot"] is None
        assert revision["files_sha256"]["weights"] == "not-loaded"
        S.check_revision(revision, arm, synthetic=True)
        revision["inference_config"]["transformers"] = None
        config_hash = __import__("hashlib").sha256(S.canonical_json(revision["inference_config"]).encode()).hexdigest()
        revision["revision_id"] = revision["revision_id"].replace(
            ";cfg=" + revision["inference_config_sha256"][:8], ";cfg=" + config_hash[:8])
        revision["inference_config_sha256"] = config_hash
        S.check_revision(revision, arm, synthetic=True)


def bind_revision_id(revision):
    # Model an internally consistent submitted identity, including mutations.
    files, template = revision["files_sha256"], revision["prompt_template"]
    revision["revision_id"] = (
        f"{revision['family']}:{revision['model_id']}@{(revision['snapshot'] or 'local')[:12]}"
        f";w={files['weights'][:12]};tok={files['tokenizer'][:12]}"
        f";prompt={template['id']}.{template['sha256'][:8]}"
        f";cfg={revision['inference_config_sha256'][:8]};adapter={revision['adapter_version']}"
    )


def production_revision(proof, arm):
    revision = copy.deepcopy(proof[2][arm])
    model_id, snapshot, weights_hash = S.pinned_model_identity()
    revision.update({"model_id": model_id, "snapshot": snapshot,
                     "notes": {"model_class": "Rwkv7ForCausalLM", "trained_ctx": 25600,
                               "attention_free": True, "vendored_code": snapshot}})
    revision["files_sha256"]["weights"] = weights_hash
    revision["inference_config"]["untrusted_encoding"] = "rwkv7-hf-trie (fail closed on <|endoftext|>)"
    revision["inference_config_sha256"] = __import__("hashlib").sha256(
        S.canonical_json(revision["inference_config"]).encode()).hexdigest()
    bind_revision_id(revision)
    return revision


def test_pinned_model_identity_matches_manifest_and_backend_hash_recipe(tmp_path, monkeypatch):
    from decide.backends import hf_readout

    model_id, snapshot, weights_hash = S.pinned_model_identity()
    assert model_id == "RWKV/RWKV7-G1k-13.3B-20260930"
    assert snapshot == "ce3630b854cedbd673569f4600c6cf744bca9dba"
    assert weights_hash == "0359db5847e46dd1c72fe91360faec09479436f8e198b65ee4c9c55277a18721"
    pins = {}
    for line in (ROOT / "jobs/rwkv7/snapshot.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        if name.endswith(".safetensors"):
            pins[name] = digest
            (tmp_path / name).touch()  # empty stand-ins, never real weights
    assert len(pins) == 28
    monkeypatch.setattr(hf_readout, "sha256_file", lambda path: pins[path.name])
    assert hf_readout._hash_files(tmp_path, ["*.safetensors"]) == weights_hash


@pytest.mark.parametrize("arm", ["v1", "v2"])
@pytest.mark.parametrize("mutation", [None, "not-loaded", "unknown", "missing", "placeholder", "short-hash",
                                      "nonhex-hash", "different-valid-hash", "model_id", "snapshot", "null-snapshot"])
def test_real_revision_requires_pinned_model_snapshot_and_weights_even_with_rebuilt_id(proof, arm, mutation):
    revision = production_revision(proof, arm)
    if mutation == "model_id":
        revision["model_id"] = "different-model"
    elif mutation == "snapshot":
        revision["snapshot"] = "a" * 40
    elif mutation == "null-snapshot":
        revision["snapshot"] = None
    elif mutation is not None:
        revision["files_sha256"]["weights"] = {
            "short-hash": "a" * 63, "nonhex-hash": "g" * 64,
            "different-valid-hash": "a" * 64,
        }.get(mutation, mutation)
    bind_revision_id(revision)
    if mutation is None:
        S.check_revision(revision, arm)
    else:
        with pytest.raises(ValueError, match="weights hash|pinned model"):
            S.check_revision(revision, arm)


@pytest.mark.parametrize("arm", ["v1", "v2"])
def test_synthetic_sentinels_cannot_enter_real_audit(proof, arm):
    with pytest.raises(ValueError, match="weights hash"):
        S.check_revision(proof[2][arm], arm)


@pytest.mark.parametrize("arm", ["v1", "v2"])
@pytest.mark.parametrize("synthetic", [False, True])
@pytest.mark.parametrize("mutation", [None, "other-mode-notes", "extra-note", "vendored-code", "encoder-missing",
                                      "encoder-other-mode", "encoder-malformed", "encoder-changed"])
def test_revision_metadata_is_exact_for_production_or_synthetic_mode(proof, arm, synthetic, mutation):
    revision = copy.deepcopy(proof[2][arm]) if synthetic else production_revision(proof, arm)
    other = production_revision(proof, arm) if synthetic else proof[2][arm]
    if mutation == "other-mode-notes":
        revision["notes"] = copy.deepcopy(other["notes"])
    elif mutation == "extra-note":
        revision["notes"]["extra"] = True
    elif mutation == "vendored-code":
        revision["notes"]["vendored_code"] = "changed"
    elif mutation == "encoder-missing":
        del revision["inference_config"]["untrusted_encoding"]
    elif mutation == "encoder-other-mode":
        revision["inference_config"]["untrusted_encoding"] = other["inference_config"]["untrusted_encoding"]
    elif mutation == "encoder-malformed":
        revision["inference_config"]["untrusted_encoding"] = None
    elif mutation == "encoder-changed":
        revision["inference_config"]["untrusted_encoding"] = "changed"
    # A valid config hash and revision ID cannot legitimize wrong-mode metadata.
    revision["inference_config_sha256"] = __import__("hashlib").sha256(
        S.canonical_json(revision["inference_config"]).encode()).hexdigest()
    bind_revision_id(revision)
    if mutation is None:
        S.check_revision(revision, arm, synthetic=synthetic)
    else:
        with pytest.raises(ValueError, match="revision notes|scoring/config/source"):
            S.check_revision(revision, arm, synthetic=synthetic)


@pytest.mark.parametrize("arm", ["v1", "v2"])
@pytest.mark.parametrize("field,value", [("model_class", "OtherModel"), ("trained_ctx", 2048),
                                        ("attention_free", False), ("attention_free", 1)])
def test_production_loader_notes_mutations_fail(proof, arm, field, value):
    revision = production_revision(proof, arm)
    revision["notes"][field] = value
    with pytest.raises(ValueError, match="revision notes"):
        S.check_revision(revision, arm)


@pytest.mark.parametrize("status", ["exhausted", "unsupported"])
@pytest.mark.parametrize("response", [None, [], {}, {"answers": None}, {"answers": []},
                                      {"answers": ""}, {"answers": False}, {"answers": 0}])
def test_non_ok_rows_require_explicit_empty_answer_dict(proof, status, response):
    rows, arms, revisions = copy.deepcopy(proof)
    record = arms["v2"]["synthetic:11"]
    record["status"], record["response"] = status, response
    with pytest.raises(ValueError, match="non-ok row"):
        S.summarize(rows, arms, revisions, synthetic=True)


def test_non_ok_row_missing_response_fails(proof):
    rows, arms, revisions = copy.deepcopy(proof)
    del arms["v2"]["synthetic:11"]["response"]
    with pytest.raises(ValueError, match="non-ok row"):
        S.summarize(rows, arms, revisions, synthetic=True)


def test_inadequate_counts_and_calibration_fields_fail(proof):
    rows, arms, revisions = proof
    with pytest.raises(ValueError, match="inadequate answered"):
        S.audit_arm(rows, arms["v2"], revisions["v2"], 12)
    rows = copy.deepcopy(rows)
    for row in rows:
        row["expected"] = {}
    with pytest.raises(ValueError, match="calibration field"):
        S.summarize(rows, arms, revisions, synthetic=True)


def test_answered_intersection_is_checked_without_changing_score_denominator(proof):
    rows, arms, revisions = copy.deepcopy(proof)
    # Each arm has 11 valid answers but their intersections contain only 10.
    answered = copy.deepcopy(arms["v1"]["synthetic:0"])
    refused = copy.deepcopy(arms["v1"]["synthetic:11"])
    for i, record in ((11, answered), (0, refused)):
        row = rows[i]
        record["run_id"] = row["_evaluation"]["run_id"]
        record["screen_evidence"]["request_fingerprint"] = S.request_fingerprint(
            {"model": "screen", "state": row["state"], "questions": row["questions"]})
        arms["v1"][record["run_id"]] = record
    with pytest.raises(ValueError, match="paired answered"):
        S.summarize(rows, arms, revisions, synthetic=True)


def test_choice_noul_score_and_53_options_preserve_complete_support():
    from jobs.rwkv7_code_label_screen.synthetic import fixture

    rows, deciders = fixture()
    row = rows[0]
    row["questions"] = {"choice": {"type": "choice", "criteria": {f"id-{i}": f"Option {i}" for i in range(53)}},
                        "noul": {"type": "noul", "instructions": "Is this supported?"},
                        "score": {"type": "score", "criteria": ["Low", "Medium", "High"]}}
    for decider in deciders.values():
        record = S.capture(row, decider)
        report, _ = S.audit_arm([row], {record["run_id"]: record}, decider.revision(), 1)
        assert report["ok_rows"] == 1
        probs = record["raw_output"]["raw_probabilities"]
        assert len(probs["choice"]) == 53
        assert set(probs["noul"]) == {"true", "false"} and set(probs["score"]) == {"0", "1", "2"}


def test_absent_comparator_and_floors_cannot_promote(proof):
    rows, arms, revisions = proof
    with pytest.raises(KeyError):
        S.summarize(rows, {"v2": arms["v2"]}, revisions, synthetic=True)
    # Neither a successful integrity screen nor an invalid run has a positive
    # promotion marker; caller-supplied floor claims are not an input to this rule.
    for integrity_ok in (True, False):
        result = R.verdict(integrity_ok)
        assert result["promotable"] is result["beats_clef"] is result["spend_authorized"] is False
        assert result["floors"] == dict.fromkeys(R.REQUIRED_FLOORS, "unmeasured")


def test_pinned_data_hash_refused_before_parsing_or_kit_import(tmp_path, monkeypatch):
    changed = tmp_path / "changed.gz"
    changed.write_bytes(b"not the pinned DEV; do not decompress")
    monkeypatch.setattr(S, "read_jsonl_strict", lambda *a: pytest.fail("changed rows were parsed"))
    with pytest.raises(ValueError, match="row file hash"):
        S.audit_files(changed, tmp_path / "missing", tmp_path / "missing", tmp_path / "missing")


def test_strict_raw_reader_rejects_duplicate_truncated_and_missing(tmp_path):
    p = tmp_path / "raw.jsonl"
    for text in ('{"run_id":"a"}\n{"run_id":"a"}\n', '{"run_id":"a"}', '{"run_id":"b"}\n'):
        p.write_text(text)
        with pytest.raises(ValueError):
            S.read_results_strict(p, ["a"])


def test_failed_cli_writes_invalid_report_and_fresh_out(tmp_path):
    out = tmp_path / "out"
    code = S.main(["audit", "--rows", str(tmp_path / "missing"), "--v1", "missing", "--v2", "missing",
                   "--identities", "missing", "--out", str(out)])
    assert code == 3
    report = json.loads((out / "report.json").read_text())
    assert report["screen_status"] == "invalid" and report["promotable"] is report["beats_clef"] is False
    with pytest.raises(FileExistsError):
        S.main(["dry-run", "--out", str(out)])


def test_real_shell_executes_cpu_dry_run(tmp_path):
    bash = shutil.which("bash")
    if sys.platform == "win32":
        bash = str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe")
    assert bash and Path(bash).exists(), "bash required for shell exit proof"
    out = tmp_path / "shell-proof"
    env = dict(os.environ, PYTHON=sys.executable, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run([bash, str(ROOT / "jobs/rwkv7_code_label_screen/screen_job.sh"), "--dry-run", str(out)],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out / "report.json").read_text())
    assert report["synthetic"] is True and report["gpu_used"] is report["model_loaded"] is False
    assert report["promotable"] is report["beats_clef"] is False
    assert not any(p.name.endswith("DONE") or p.name.endswith("PASS") for p in out.iterdir())
    reject = subprocess.run([bash, str(ROOT / "jobs/rwkv7_code_label_screen/screen_job.sh"), "--launch", str(out)],
                            cwd=ROOT, env=env, capture_output=True, timeout=10)
    assert reject.returncode == 2
