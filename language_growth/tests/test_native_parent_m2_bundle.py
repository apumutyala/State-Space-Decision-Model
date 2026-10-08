"""M2 bundle bridge contracts using metadata stubs only; no tensors or inference."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import pytest

from decide.adapter import Decider
from decide.calibration import FeatureTemperature, content_set_sha256, disjointness_proof_sha256
from heads.decide_backend import DecisionModelBackend
from language_growth.native_parent import LoadedNativeParent, NativeParentPin, load_native_parent
from language_growth.planner import _check_stages
from train.lineage import backbone_config_identity
from train.serving_export import export_serving_bundle


class MetadataOnlyModel:
    def __init__(self):
        self.head = SimpleNamespace(name="metadata-only")
        self.backbone = SimpleNamespace(
            fork_mode="cache", capture="hidden_states", layer=None,
            spec=SimpleNamespace(name="metadata-only"),
            model=SimpleNamespace(config=SimpleNamespace(implementation="synthetic")))
        self.training = True
        self.grad_enabled = True

    def eval(self):
        self.training = False
        return self

    def requires_grad_(self, value):
        self.grad_enabled = value
        return self

    def __call__(self, *_args, **_kwargs):
        raise AssertionError("no model calls in M2 bundle tests")


class MetadataEncoder:
    def identity(self):
        return {"kind": "metadata-only", "vocab_sha256": "e" * 64}

    def __call__(self, *_):
        raise AssertionError("no tokenization in M2 bundle tests")


def fixture(tmp_path, cap=None, flags=False):
    artifact = tmp_path / "metadata-artifact.txt"
    artifact.write_bytes(b"synthetic artifact hash input; no checkpoint or weights")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    model = MetadataOnlyModel()
    backend = DecisionModelBackend(model, MetadataEncoder(), template="jrt-plain-v1",
                                   checkpoint={"path": artifact.name, "step": 7, "sha256": digest})
    # The actual backend revision normally hashes weights. Supply a metadata
    # identity directly so this test cannot read/construct any model weights.
    backend._revision = {"revision_id": "synthetic-native@" + digest[:12], "model_id": "metadata-only",
                         "checkpoint": deepcopy(backend.checkpoint), "layout": backend.layout,
                         "prompt_template": backend.template.identity(), "encoder": backend.encode.identity(),
                         "backbone": {**backbone_config_identity(model.backbone),
                                      "weights_digest": "synthetic-no-weights", "weights_digest_mode": "synthetic"}}
    content = (hashlib.sha256(b"synthetic calibration").hexdigest(),)
    fitted = content_set_sha256(content)
    disjoint = {"eval": content_set_sha256([hashlib.sha256(b"synthetic eval").hexdigest()])}
    calibration = FeatureTemperature(w=(0.5, 0.03, 0.1), fitted_on=fitted, calib_content=content,
                                     disjointness=tuple(disjoint.items()),
                                     proof_sha256=disjointness_proof_sha256(fitted, disjoint))
    loaded = LoadedNativeParent(backend, calibration, cap, flags, flags)
    decider = Decider(backend, calibration=calibration, backend_max_input_tokens=cap,
                      enable_experimental_provenance=flags, enable_v0_1_deltas=flags)
    report = {"identities": {"serving_bundle": export_serving_bundle(decider),
                             "checkpoint": deepcopy(backend.checkpoint),
                             "child": {"served": decider.revision()}},
              "promotion": {"promotable": False}}
    return artifact, loaded, json.loads(json.dumps(report))


@pytest.mark.parametrize("cap", [None, 2048])
@pytest.mark.parametrize("flags", [False, True])
def test_merged_m2_bundle_roundtrip_preserves_calibration_and_all_runtime_fields(tmp_path, cap, flags):
    artifact, loaded, report = fixture(tmp_path, cap, flags)
    pin = NativeParentPin.from_m2_report(report)
    parent = load_native_parent(artifact, lambda *_a, **_k: loaded, pin=pin)
    assert parent._decider.backend_max_input_tokens == cap
    assert parent._decider.enable_experimental_provenance is flags
    assert parent._decider.enable_v0_1_deltas is flags
    assert loaded.backend.model.training is False and loaded.backend.model.grad_enabled is False
    restored = FeatureTemperature.from_dict(pin.calibration)
    restored.assert_servable()
    assert restored.to_dict() == loaded.calibration.to_dict()
    for evidence, options in ((0, 2), (1000, 16), (16384, 255)):
        assert restored.temperature_for(options, {"evidence_tokens": evidence}) == loaded.calibration.temperature_for(
            options, {"evidence_tokens": evidence})
    # Mutating the caller's report must not alter the independent bound pins.
    report["identities"]["serving_bundle"]["calibration"]["calib_content"].clear()
    assert len(parent._pin.calibration["calib_content"]) == 1
    kwargs = parent.planner_kwargs()
    assert kwargs["expected_served_config"]["calibration"] == json.loads(json.dumps(restored.identity()))
    kwargs["expected_served_config"]["calibration"]["id"] = "mutated returned copy"
    assert parent.planner_kwargs()["expected_served_config"]["calibration"]["id"] == restored.identity()["id"]


@pytest.mark.parametrize("native", [False, True])
def test_planner_accepts_same_native_tuple_and_json_list_calibration_identity(tmp_path, native):
    artifact, loaded, report = fixture(tmp_path)
    parent = load_native_parent(artifact, lambda *_a, **_k: loaded, pin=NativeParentPin.from_m2_report(report))
    calibration = loaded.calibration.identity()
    if not native:
        calibration = json.loads(json.dumps(calibration))
    options = {"a": "first", "b": "second"}
    env = {"served_config": {"calibration": calibration, "policy_masking": "none (served covers every supplied option)",
                              "temperature_per_question": {"q": 1.0}},
           "served_probabilities": {"q": {"a": .25, "b": .75}},
           "raw_probabilities": {"q": {"a": .2, "b": .8}}}
    probabilities, _ = _check_stages(env, {"ssdm": {}}, "q", options,
                                      parent.planner_kwargs()["expected_served_config"])
    assert probabilities == {"a": .25, "b": .75}


@pytest.mark.parametrize("change", ["schema", "identity_only", "content", "proof", "revision_calibration",
                                    "artifact_sha", "checkpoint_sha", "legacy_alias", "report_checkpoint",
                                    "report_revision", "old_cap", "zero_cap", "bool_cap", "bad_flag"])
def test_invalid_bundle_refuses_before_artifact_or_loader_access(tmp_path, change):
    _artifact, _loaded, report = fixture(tmp_path)
    bundle = report["identities"]["serving_bundle"]
    if change == "schema": bundle["schema"] = "ssdm.m2.serving_bundle.v999"
    elif change == "identity_only": bundle["calibration"] = deepcopy(bundle["revision"]["calibration"])
    elif change == "content": bundle["calibration"]["calib_content"].clear()
    elif change == "proof": bundle["calibration"]["proof_sha256"] = "0" * 64
    elif change == "revision_calibration": bundle["revision"]["calibration"]["id"] = "wrong"
    elif change == "artifact_sha": bundle["artifact_sha256"] = "0" * 64
    elif change == "checkpoint_sha": del bundle["revision"]["checkpoint"]["sha256"]
    elif change == "legacy_alias": bundle["revision"]["checkpoint"]["artifact_sha256"] = "0" * 64
    elif change == "report_checkpoint": report["identities"]["checkpoint"]["sha256"] = "0" * 64
    elif change == "report_revision": report["identities"]["child"]["served"]["revision_id"] = "wrong"
    elif change == "old_cap": bundle["runtime"]["max_input_tokens"] = bundle["runtime"].pop("backend_max_input_tokens")
    elif change == "zero_cap": bundle["runtime"]["backend_max_input_tokens"] = 0
    elif change == "bool_cap": bundle["runtime"]["backend_max_input_tokens"] = True
    elif change == "bad_flag": bundle["runtime"]["enable_v0_1_deltas"] = 1
    with pytest.raises((TypeError, ValueError)):
        NativeParentPin.from_m2_report(report)


def test_artifact_hash_failure_and_strict_loader_rejection_never_fall_back(tmp_path):
    artifact, loaded, report = fixture(tmp_path)
    pin = NativeParentPin.from_m2_report(report)
    calls = []
    artifact.write_bytes(b"changed synthetic artifact bytes")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        load_native_parent(artifact, lambda *_a, **_k: calls.append("unexpected"), pin=pin)
    assert calls == []
    artifact, loaded, report = fixture(tmp_path)
    pin = NativeParentPin.from_m2_report(report)
    def strict_loader(*_args, **_kwargs):
        calls.append("strict loader")
        raise ValueError("strict reconstruction rejected a missing trainable key")
    with pytest.raises(ValueError, match="missing trainable key"):
        load_native_parent(artifact, strict_loader, pin=pin)
    assert calls == ["strict loader"] and loaded.backend.model.training is True


def test_loaded_alias_mismatch_runtime_and_exact_calibration_drift_refuse(tmp_path):
    for change in ("alias", "runtime", "calibration"):
        artifact, loaded, report = fixture(tmp_path)
        pin = NativeParentPin.from_m2_report(report)
        if change == "alias": loaded.backend.checkpoint["artifact_sha256"] = "0" * 64
        elif change == "runtime": loaded = replace(loaded, backend_max_input_tokens=123)
        else:
            # Identity rounds w to eight decimal places. Full payload binding
            # must still reject a different loaded serving temperature.
            loaded = replace(loaded, calibration=replace(loaded.calibration, w=(.5 + 1e-12, .03, .1)))
            assert json.loads(json.dumps(loaded.calibration.identity())) == pin.revision["calibration"]
        with pytest.raises(ValueError):
            load_native_parent(artifact, lambda *_a, **_k: loaded, pin=pin)
        assert loaded.backend.model.training is True


def test_loaded_metadata_drift_is_rechecked_before_any_inference(tmp_path):
    artifact, loaded, report = fixture(tmp_path)
    parent = load_native_parent(artifact, lambda *_a, **_k: loaded, pin=NativeParentPin.from_m2_report(report))
    loaded.backend.model.backbone.model.config.implementation = "changed"
    with pytest.raises(ValueError, match="configuration changed"):
        parent.decide({})


def test_loader_cannot_change_the_callers_pin_snapshot(tmp_path):
    artifact, loaded, report = fixture(tmp_path)
    pin = NativeParentPin.from_m2_report(report)
    original = deepcopy(pin.calibration)
    def loader(*_args, **_kwargs):
        pin.calibration["w"][0] += .1
        return loaded
    parent = load_native_parent(artifact, loader, pin=pin)
    assert parent._pin.calibration == original
    assert parent.planner_kwargs()["expected_served_config"]["calibration"]["id"] == original["id"]
