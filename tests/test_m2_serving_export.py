"""No weights or inference: consume M2 serving metadata with the real calibrator."""

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from decide.adapter import Decider
from decide.calibration import (FeatureTemperature,
                                content_set_sha256, disjointness_proof_sha256)
from train.serving_export import SCHEMA, export_serving_bundle


def _sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def _calibration():
    content = (_sha("synthetic calibration record"),)
    fitted_on = content_set_sha256(content)
    disjointness = {"eval": content_set_sha256([_sha("synthetic evaluation record")])}
    return FeatureTemperature(w=(.5, .03, .1), fitted_on=fitted_on,
                              calib_content=content, disjointness=tuple(disjointness.items()),
                              proof_sha256=disjointness_proof_sha256(fitted_on, disjointness))


class MetadataOnlyBackend:
    """No model/tensors; fail if any test accidentally requests inference."""

    def __init__(self):
        self.model = SimpleNamespace(backbone=SimpleNamespace(fork_mode="cache"))
        self.shared_prefix = True
        self.identity = {"revision_id": "synthetic-decision-head@metadata", "model_id": "synthetic",
                         "checkpoint": {"path": "ckpt-step0000600.pt", "step": 600,
                                        "sha256": _sha("synthetic checkpoint bytes")},
                         "backbone": {"spec": "metadata-only"}, "encoder": {"kind": "synthetic"},
                         "layout": "ordered", "prompt_template": {"id": "jrt-rwkv-g1-v1"}}

    def revision(self):
        return deepcopy(self.identity)

    def plan(self, *_):
        raise AssertionError("no inference in metadata contract tests")

    score = plan


@pytest.mark.parametrize("backend_cap", [None, 2048])
def test_consumer_reconstructs_exact_calibration_revision_and_runtime(backend_cap):
    backend, calibration = MetadataOnlyBackend(), _calibration()
    original = Decider(backend, calibration=calibration, backend_max_input_tokens=backend_cap)
    bundle = json.loads(json.dumps(export_serving_bundle(original)))
    assert bundle["schema"] == SCHEMA
    assert bundle["artifact_sha256"] == original.revision()["checkpoint"]["sha256"]
    restored = FeatureTemperature.from_dict(bundle["calibration"])
    restored.assert_servable()
    restored.verify_disjoint("eval", [_sha("synthetic evaluation record")])
    runtime = bundle["runtime"]
    consumer = Decider(backend, calibration=restored,
                       backend_max_input_tokens=runtime["backend_max_input_tokens"],
                       enable_experimental_provenance=runtime["enable_experimental_provenance"],
                       enable_v0_1_deltas=runtime["enable_v0_1_deltas"])
    assert json.loads(json.dumps(consumer.revision())) == bundle["revision"]
    assert restored.identity() == calibration.identity()
    assert runtime["fork_mode"] == "cache" and runtime["shared_prefix"] is True
    assert consumer.backend_max_input_tokens == backend_cap  # null is not guessed as a hard cap
    for ev, k in [(0, 2), (2048, 16), (16384, 255)]:
        assert restored.temperature_for(k, {"evidence_tokens": ev}) == calibration.temperature_for(
            k, {"evidence_tokens": ev})


def test_identity_only_report_cannot_reconstruct_fitted_calibration():
    calibration = _calibration()
    # Regression: the old report includes identity/proof but no calib_content.
    old_report = {**calibration.identity(), "proof_sha256": calibration.proof_sha256}
    with pytest.raises(ValueError, match="calibration content set"):
        FeatureTemperature.from_dict(old_report)


def test_export_is_detached_and_missing_checkpoint_pin_refuses():
    backend = MetadataOnlyBackend()
    decider = Decider(backend, calibration=_calibration())
    bundle = export_serving_bundle(decider)
    bundle["revision"]["checkpoint"]["step"] = -1
    bundle["calibration"]["calib_content"].clear()
    assert backend.identity["checkpoint"]["step"] == 600
    assert len(decider.calibration.calib_content) == 1
    del backend.identity["checkpoint"]["sha256"]
    with pytest.raises(ValueError, match="checkpoint sha256"):
        export_serving_bundle(decider)


def test_export_refuses_revision_with_different_calibration():
    decider = Decider(MetadataOnlyBackend(), calibration=_calibration())
    revision = decider.revision()
    revision["calibration"]["id"] = "another temperature"
    decider.revision = lambda: revision
    with pytest.raises(ValueError, match="exported calibration"):
        export_serving_bundle(decider)


def test_gate_report_path_adds_bundle_without_changing_existing_metadata(monkeypatch):
    import sys
    from train import m2_gates

    # Stub the weight-identity operation; this test exercises report assembly only.
    monkeypatch.setitem(sys.modules, "train.lineage", SimpleNamespace(
        backbone_identity=lambda _: {"spec": "metadata-only"}))
    captured = {}
    monkeypatch.setattr(m2_gates, "run_gates", lambda **kw: captured.update(kw))
    backend = MetadataOnlyBackend()
    backend.checkpoint = deepcopy(backend.identity["checkpoint"])
    calibration = _calibration()
    served, arm = Decider(backend, calibration=calibration), Decider(backend)
    args = SimpleNamespace(parent_arm="synthetic-parent", parent_arm_sha256="p",
                           anchor_arm="synthetic-anchor", anchor_arm_sha256="a",
                           anchor_is_champion=False, progress_dir=None, n_boot=2)
    ft = SimpleNamespace(identity=calibration.identity, disjointness=dict(calibration.disjointness),
                         proof_sha256=calibration.proof_sha256, fit_report={"synthetic": True})
    m2_gates._run(stages=("g12", "index"), cohort=None, spec={}, decider=served,
                  arm_decider=arm, items_by_role={}, records_by_id={}, a=args,
                  expect_parent={}, index_rows=[], out=None,
                  guard=SimpleNamespace(identity=lambda: {"synthetic": True}),
                  backend=backend, tr=SimpleNamespace(lineage={"synthetic": True}),
                  bb=backend.model.backbone, ft=ft, card={"step": 600})
    extra = captured["extra"]
    assert extra["checkpoint"] == backend.checkpoint
    assert extra["temperature"] == {**calibration.identity(),
                                     "disjointness": dict(calibration.disjointness),
                                     "proof_sha256": calibration.proof_sha256,
                                     "fit_report": {"synthetic": True}}
    bundle = extra["serving_bundle"]
    restored = FeatureTemperature.from_dict(bundle["calibration"])
    assert restored.identity() == calibration.identity()
    assert bundle["revision"] == json.loads(json.dumps(captured["child_identity"]["served"]))
    assert captured["stages"] == ("g12", "index")
