"""Tiny functional fake state + real trained-head API; no checkpoint/GPU/model capability claim."""
from copy import deepcopy
from dataclasses import replace
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from decide.adapter import Decider
from heads.backbone import HiddenStateBackbone, tiny_spec
from heads.decide_backend import DecisionModelBackend
from heads.layout import ByteEncoder
from heads.model import DecisionModel
from language_growth.development_data import generate_development_tasks
from language_growth.gate_interventions import (BOUNDARY, DIRECTION_SCHEMA, OUTPUT_CONTROL, SCHEMA, SCOPE,
                                                _ArmProvider, _parent_state_sha, _sha, native_capture_identity,
                                                run_native_interventions)
from language_growth.language_child import LanguageChild, Runtime as LanguageChildRuntime
from language_growth.native_parent import LoadedNativeParent, NativeParent, NativeParentPin
from language_growth.planner import plan_checked_answer
from language_growth.qualified_parent import file_sha256
from language_growth.tests.test_native_parent_m2_bundle import fixture


class FakeStateBackbone(torch.nn.Module):
    """Vectorized functional cache fixture; continuation depends on carried state."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, implementation="synthetic-functional-state")
        self.embed = torch.nn.Embedding(260, 4)

    def prefill(self, ids):
        h = self.embed(ids)
        return h, h.mean(1)

    def expand_state(self, state, n):
        return state.expand(n, -1).clone()

    def extend(self, ids, state):
        h = self.embed(ids) + state[:, None, :]
        return h, h[:, -1]

    def forward(self, input_ids, output_hidden_states=True):
        return SimpleNamespace(hidden_states=(self.embed(input_ids),))


def setup(tmp_path, monkeypatch):
    torch.manual_seed(19)
    model = DecisionModel(HiddenStateBackbone(FakeStateBackbone(), spec=tiny_spec(1, 4), fork_mode="cache"),
                          "fork", {"hidden": 8}, seed=19)
    _artifact, old_loaded, _report = fixture(tmp_path)
    backend = DecisionModelBackend(model, ByteEncoder(), checkpoint={"sha256": "a" * 64}, shared_prefix=True)
    loaded = LoadedNativeParent(backend, old_loaded.calibration, 32768)
    decider = Decider(backend, calibration=loaded.calibration, backend_max_input_tokens=32768)
    pin = NativeParentPin("a" * 64, deepcopy(decider.revision()), loaded.calibration.to_dict(),
                          {"fork_mode": "cache", "shared_prefix": True, "backend_max_input_tokens": 32768,
                           "enable_experimental_provenance": False, "enable_v0_1_deltas": False})
    parent = NativeParent(loaded, pin)
    example = generate_development_tasks(19, 1, choice_format="production", split="development_validation")[0]
    task = {"task_id": "evaluation-1", "world_id": "evaluation-world-1", "level": "L0", "split": "validation",
            "task_input": json.loads(example["model_inputs"][0]["state"])["task_input"],
            "host": example["training_only"]["host"], "catalog": example["training_only"]["catalog"]}
    # Gate planner is owned in a separate pending integration commit. Exercise
    # the merged ordinary free-running L0 planner through the same signature.
    def plan(task, provider):
        return plan_checked_answer(task["task_input"], task["host"], task["catalog"], provider,
                                   **provider.planner_kwargs())
    monkeypatch.setitem(sys.modules, "language_growth.gate_planner", SimpleNamespace(plan_gate_task=plan))
    capture = native_capture_identity(parent)
    outputs = {str(k): torch.arange(k, dtype=torch.float32) - (k - 1) / 2 if k > 1 else torch.ones(1)
               for k in range(1, 20)}
    data = {"schema": DIRECTION_SCHEMA, "parent_revision": parent.planner_kwargs()["parent_revision"],
            "capture_identity_sha256": _sha(capture), "capture_boundary": BOUNDARY,
            "source_split": "discovery_train_only", "discovery_world_ids": ["discovery-world-1"],
            "discovery_pairs_sha256": "1" * 64, "output_only_fit_evidence_sha256": "2" * 64,
            "direction": torch.tensor([1.0, -.5, .7, -.2]), "output_only_directions": outputs}
    path = tmp_path / "synthetic-directions.pt"
    torch.save(data, path)
    prereg = {"schema": SCHEMA, "parent_revision": data["parent_revision"],
              "capture_identity_sha256": data["capture_identity_sha256"], "capture_boundary": BOUNDARY,
              "scope": SCOPE, "discovery_world_ids": data["discovery_world_ids"],
              "discovery_pairs_sha256": data["discovery_pairs_sha256"],
              "directions": {"path": str(path.resolve()), "sha256": file_sha256(path)},
              "scale": .05, "random_seed": 42, "output_control": OUTPUT_CONTROL}
    prereg["registration_sha256"] = _sha(prereg)
    return parent, task, prereg, data, path, plan


def resign(prereg):
    prereg["registration_sha256"] = _sha({k: v for k, v in prereg.items() if k != "registration_sha256"})


def test_fitted_language_child_uses_its_native_pin_and_rejects_runtime_drift(tmp_path, monkeypatch):
    parent, _task, _prereg, _data, _path, _plan = setup(tmp_path, monkeypatch)
    candidate = {
        "checkpoint_sha256": "a" * 64, "receipt_sha256": "b" * 64,
        "receipt": {"plan_sha256": "c" * 64, "source_commit": "d" * 40},
        "checkpoint_path": tmp_path / "child.pt", "step": 4, "lineage": {"data": {"n": 4}},
    }
    trainable_parent = SimpleNamespace(model=parent._loaded.backend.model,
                                       encoder=parent._loaded.backend.encode,
                                       config=SimpleNamespace(layout=parent._loaded.backend.layout))
    stage1_spec = SimpleNamespace(pin=parent._pin)
    native_child = LanguageChildRuntime().native_parent(trainable_parent, {}, candidate, stage1_spec)
    child = LanguageChild(native_child, stage1_identity={"artifact_sha256": "9" * 64}, candidate=candidate)
    capture = native_capture_identity(child)
    assert capture["parent_revision"] == child.planner_kwargs()["parent_revision"]
    assert child.identity()["artifact_sha256"] == "a" * 64
    assert child.identity()["stage1_parent"]["artifact_sha256"] == "9" * 64
    assert child.identity()["language_qualified"] is False
    child._native._loaded.backend.shared_prefix = False
    with pytest.raises(ValueError):
        native_capture_identity(child)


def test_fitted_language_child_runs_all_registered_intervention_arms(tmp_path, monkeypatch):
    parent, task, prereg, data, path, _plan = setup(tmp_path, monkeypatch)
    candidate = {"checkpoint_sha256": "c" * 64, "receipt_sha256": "d" * 64,
                 "checkpoint_path": tmp_path / "child.pt", "step": 4,
                 "receipt": {"plan_sha256": "e" * 64, "source_commit": "f" * 40},
                 "lineage": {"data": {"n": 4}}}
    trainable_parent = SimpleNamespace(model=parent._loaded.backend.model,
                                       encoder=parent._loaded.backend.encode,
                                       config=SimpleNamespace(layout=parent._loaded.backend.layout))
    native_child = LanguageChildRuntime().native_parent(
        trainable_parent, {}, candidate, SimpleNamespace(pin=parent._pin))
    child = LanguageChild(native_child, stage1_identity={"artifact_sha256": "9" * 64}, candidate=candidate)
    capture = native_capture_identity(child)
    data["parent_revision"] = child.planner_kwargs()["parent_revision"]
    data["capture_identity_sha256"] = _sha(capture)
    torch.save(data, path)
    prereg["parent_revision"] = data["parent_revision"]
    prereg["capture_identity_sha256"] = data["capture_identity_sha256"]
    prereg["directions"]["sha256"] = file_sha256(path)
    resign(prereg)

    result = run_native_interventions([task], child, prereg)
    assert result["instrumentation_valid"] is True
    assert set(result["arms"]) == {"semantic", "sham", "random", "output_only"}
    assert all(len(rows) == 1 for rows in result["arms"].values())


def test_four_native_arms_preserve_trained_head_parent_bytes_and_sham_behavior(tmp_path, monkeypatch):
    parent, task, prereg, _data, _path, plan = setup(tmp_path, monkeypatch)
    before = _parent_state_sha(parent._loaded.backend.model)
    baseline = plan(task, parent)
    result = run_native_interventions([task], parent, prereg)
    assert result["instrumentation_valid"] is True
    assert set(result["arms"]) == {"semantic", "sham", "random", "output_only"}
    assert all(len(rows) == 1 for rows in result["arms"].values())
    sham = result["arms"]["sham"][0]
    assert (sham.status, sham.program, sham.text) == (baseline.status, baseline.program, baseline.text)
    provenance = result["provenance"][0]
    assert provenance["input_sha256"] == _sha(task)
    assert provenance["state_unchanged_sha256_before"] == provenance["state_unchanged_sha256_after"] == before
    assert provenance["capture_boundary"] == BOUNDARY and provenance["scope"] == SCOPE
    for arm in result["arms"]:
        assert result["arms"][arm][0].provider_calls > 0
        for record in provenance["captures"][arm]:
            assert record["head_calls"] == 1 and record["status"] == "ok"
            capture = record["capture"]
            assert len(record["option_ids"]) == capture["shape"][1]
            assert record["request_fingerprint"]
            if arm in {"semantic", "random"}:
                assert capture["hidden_before_sha256"] != capture["hidden_after_sha256"]
                assert capture["hidden_delta_l2"] == pytest.approx(capture["matched_delta_l2"], rel=1e-4)
                assert capture["output_delta_l2"] == 0
            else:
                assert capture["hidden_before_sha256"] == capture["hidden_after_sha256"]
            if arm == "output_only":
                assert capture["logits_before_sha256"] != capture["logits_after_sha256"]
                assert capture["output_delta_l2"] == pytest.approx(capture["matched_delta_l2"], rel=1e-4)
            if arm == "sham":
                assert capture["logits_before_sha256"] == capture["logits_after_sha256"]
    head = parent._loaded.backend.model.head
    assert not head._forward_hooks and not head._forward_pre_hooks


def test_random_axis_reproducible_and_global_rng_unmodified(tmp_path, monkeypatch):
    parent, task, prereg, *_ = setup(tmp_path, monkeypatch)
    rng = torch.get_rng_state().clone()
    first = run_native_interventions([task], parent, prereg)
    second = run_native_interventions([task], parent, prereg)
    assert torch.equal(torch.get_rng_state(), rng)
    assert first["provenance"][0]["random_direction_sha256"] == second["provenance"][0]["random_direction_sha256"]
    for a, b in zip(first["provenance"][0]["captures"]["random"], second["provenance"][0]["captures"]["random"]):
        assert a["capture"] == b["capture"]
        assert a["option_ids"] == b["option_ids"] and a["request_fingerprint"] == b["request_fingerprint"]


@pytest.mark.parametrize("change", ["label", "overlap", "boundary", "registration", "artifact", "parent", "scale", "source", "control"])
def test_bad_registration_or_label_leak_rejected_before_any_head_call(tmp_path, monkeypatch, change):
    parent, task, prereg, data, path, _plan = setup(tmp_path, monkeypatch)
    if change == "label": task["expected_answer"] = "secret evaluator label"
    elif change == "overlap": prereg["discovery_world_ids"] = [task["world_id"]]
    elif change == "boundary": prereg["capture_boundary"] = "recurrent_memory"
    elif change == "registration": prereg["registration_sha256"] = "0" * 64
    elif change == "artifact": prereg["directions"]["sha256"] = "0" * 64
    elif change == "parent": prereg["parent_revision"] = "wrong"
    elif change == "scale": prereg["scale"] = 0
    elif change == "source":
        data["source_split"] = "evaluation"
        torch.save(data, path)
        prereg["directions"]["sha256"] = file_sha256(path)
    elif change == "control": prereg["output_control"] = "random_logits"
    if change != "registration": resign(prereg)
    with pytest.raises(ValueError):
        run_native_interventions([task], parent, prereg)
    assert not parent._loaded.backend.model.head._forward_hooks


def test_missing_fitted_control_keeps_all_arm_failures_in_denominator_and_cleans_hooks(tmp_path, monkeypatch):
    parent, task, prereg, data, path, _plan = setup(tmp_path, monkeypatch)
    data["output_only_directions"] = {"255": torch.arange(255, dtype=torch.float32)}
    torch.save(data, path)
    prereg["directions"]["sha256"] = file_sha256(path)
    resign(prereg)
    result = run_native_interventions([task], parent, prereg)
    assert result["instrumentation_valid"] is False
    assert result["provenance"][0]["instrumentation_valid"] is False
    for rows in result["arms"].values():
        assert len(rows) == 1 and rows[0].status == "runner_error" and rows[0].text is None
        assert rows[0].provider_calls > 0
    assert not parent._loaded.backend.model.head._forward_hooks
    assert not parent._loaded.backend.model.head._forward_pre_hooks


def test_failed_control_cannot_be_treated_as_semantic_superiority(tmp_path, monkeypatch):
    parent, task, prereg, data, path, _plan = setup(tmp_path, monkeypatch)
    data["output_only_directions"] = {"3": torch.tensor([-1., 0., 1.])}
    torch.save(data, path)
    prereg["directions"]["sha256"] = file_sha256(path)
    resign(prereg)
    from language_growth.planner import PlannerResult
    def branch_count_runner(_task, provider):
        # A synthetic divergent trajectory: semantic reaches only registered
        # K=3, while every control reaches unsupported K=4. No labels enter.
        count = 3 if provider.arm == "semantic" else 4
        request = {"model": provider.planner_kwargs()["model"], "state": "synthetic context",
                   "questions": {"q": {"type": "choice", "criteria": {str(i): str(i) for i in range(count)}}}}
        try:
            provider.decide(request)
            return PlannerResult("accepted", provider_calls=1, forward_calls=1)
        except ValueError:
            return PlannerResult("runner_error", provider_calls=1, forward_calls=1)
    monkeypatch.setitem(sys.modules, "language_growth.gate_planner", SimpleNamespace(plan_gate_task=branch_count_runner))
    result = run_native_interventions([task], parent, prereg)
    assert result["arms"]["semantic"][0].status == "accepted"
    assert result["instrumentation_valid"] is False
    assert result["provenance"][0]["instrumentation_valid"] is False


def test_existing_instrumentation_owner_is_rejected(tmp_path, monkeypatch):
    parent, task, prereg, *_ = setup(tmp_path, monkeypatch)
    handle = parent._loaded.backend.model.head.register_forward_hook(lambda *_: None)
    try:
        with pytest.raises(ValueError, match="instrumentation owner"):
            run_native_interventions([task], parent, prereg)
    finally:
        handle.remove()


def test_parent_mutation_is_detected_and_hooks_released(tmp_path, monkeypatch):
    parent, task, prereg, *_ = setup(tmp_path, monkeypatch)
    def malicious_runner(_task, _provider):
        parent._loaded.backend.model.head.mlp[0].weight.add_(.1)
        from language_growth.planner import PlannerResult
        return PlannerResult("runner_error", reason="synthetic mutation test")
    monkeypatch.setitem(sys.modules, "language_growth.gate_planner", SimpleNamespace(plan_gate_task=malicious_runner))
    with pytest.raises(ValueError, match="parent bytes changed"):
        run_native_interventions([task], parent, prereg)
    assert not parent._loaded.backend.model.head._forward_hooks
    assert not parent._loaded.backend.model.head._forward_pre_hooks
