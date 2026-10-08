"""Real native head/capture/fitting on a tiny synthetic backbone; no RWKV claim."""
from copy import deepcopy
import json

import pytest
import torch

from decide.adapter import Decider
from heads.backbone import HiddenStateBackbone, tiny_spec
from heads.decide_backend import DecisionModelBackend
from heads.layout import ByteEncoder
from heads.model import DecisionModel
from language_growth.development_data import generate_development_tasks
from language_growth.gate_direction_fit import (
    METHOD, _request, development_relation_pairs, fit_native_directions, write_direction_fit,
)
from language_growth.gate_interventions import (
    DIRECTION_SCHEMA, _OWNER, _parent_state_sha, _registration, _sha,
)
from language_growth.native_parent import LoadedNativeParent, NativeParent, NativeParentPin
from language_growth.tests.test_gate_interventions import FakeStateBackbone
from language_growth.tests.test_native_parent_m2_bundle import fixture


class TinyStateBackbone(FakeStateBackbone):
    """Carried state plus causal continuation accumulation; options stay distinct."""
    def extend(self, ids, state):
        length = torch.arange(1, ids.shape[1] + 1, device=ids.device).view(1, -1, 1)
        h = self.embed(ids).cumsum(1) / length + state[:, None, :]
        return h, h[:, -1]


def pairs():
    train = development_relation_pairs(generate_development_tasks(29, 24, split="development_train", choice_format="production"))[:2]
    validation = development_relation_pairs(generate_development_tasks(31, 24, split="development_validation", choice_format="production"))[:1]
    assert len(train) == 2 and len(validation) == 1
    return train, validation


def trained_parent(tmp_path, train):
    """Two actual supervised head updates, then construct fresh frozen native pins."""
    torch.manual_seed(19)
    model = DecisionModel(HiddenStateBackbone(TinyStateBackbone(), spec=tiny_spec(1, 4), fork_mode="cache"),
                          "fork", {"hidden": 8}, seed=19)
    backend = DecisionModelBackend(model, ByteEncoder(), checkpoint={"sha256": "a" * 64}, shared_prefix=True)
    # Use fresh label-free requests; target indices enter cross entropy only.
    fake_parent = type("Pins", (), {"planner_kwargs": lambda _: {"model": "tiny", "parent_revision": "tiny"}})()
    optimizer = torch.optim.SGD(model.head.parameters(), lr=.01)
    initial_head = {name: value.detach().clone() for name, value in model.head.state_dict().items()}
    for side in ("source", "effect"):
        spec = train[0][side]
        request = _request(spec, fake_parent, "tiny-training-" + side)
        rows = backend._rows(request)
        target = rows[0].option_ids.index(train[0][side + "_target"])
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.nll_loss(model(rows, shared_prefix=True).log_probs,
                                            torch.tensor([target]))
        loss.backward()
        optimizer.step()
    assert any(not torch.equal(initial_head[name], value) for name, value in model.head.state_dict().items())
    _artifact, old_loaded, _report = fixture(tmp_path)
    # Native backend revision is first computed AFTER training.
    loaded = LoadedNativeParent(backend, old_loaded.calibration, 32768)
    decider = Decider(backend, calibration=loaded.calibration, backend_max_input_tokens=32768)
    pin = NativeParentPin("a" * 64, deepcopy(decider.revision()), loaded.calibration.to_dict(),
        {"fork_mode": "cache", "shared_prefix": True, "backend_max_input_tokens": 32768,
         "enable_experimental_provenance": False, "enable_v0_1_deltas": False})
    return NativeParent(loaded, pin)


def test_real_capture_fit_compatible_with_adapter_and_preserves_trained_native_bytes(tmp_path, monkeypatch):
    train, validation = pairs()
    parent = trained_parent(tmp_path, train)
    before = _parent_state_sha(parent._loaded.backend.model)
    rng = torch.get_rng_state().clone()
    import language_growth.gate_direction_fit as fit
    captures = []
    original = fit._capture
    def observe(*args):
        result = original(*args)
        captures.append(result)
        return result
    monkeypatch.setattr(fit, "_capture", observe)
    result = fit_native_directions(parent, train, validation_pairs=validation, scale=.02, random_seed=17)
    assert torch.equal(torch.get_rng_state(), rng)
    artifact, evidence = result["artifact"], result["evidence"]
    assert artifact["schema"] == DIRECTION_SCHEMA
    assert evidence["method"] == METHOD and evidence["labels_used_for_fit"] is False
    assert evidence["state_unchanged_sha256_before"] == evidence["state_unchanged_sha256_after"] == before
    assert artifact["direction"].dtype == torch.float32 and artifact["direction"].shape == (4,)
    assert artifact["direction"].norm().item() == pytest.approx(1)
    assert artifact["output_only_directions"]["3"].norm().item() == pytest.approx(1)
    # Recompute the registered deterministic estimator from actual native
    # captures, independently of the fitter's contrast/accumulation routines.
    hidden_mean = torch.stack([(captures[i+1][0] - captures[i][0]).mean(0) for i in (0, 2)]).mean(0)
    logit_mean = torch.stack([captures[i+1][1] - captures[i][1] for i in (0, 2)]).mean(0)
    logit_mean -= logit_mean.mean()
    assert torch.equal(artifact["direction"], (hidden_mean / hidden_mean.norm()).float())
    torch.testing.assert_close(artifact["output_only_directions"]["3"], (logit_mean / logit_mean.norm()).float())
    assert artifact["discovery_pairs_sha256"] == _sha(train)
    assert artifact["output_only_fit_evidence_sha256"] == _sha(evidence)
    assert len(evidence["discovery"]) == 2 and len(evidence["validation"]) == 1
    for record in evidence["discovery"] + evidence["validation"]:
        for side in ("source", "effect"):
            assert record[side]["hidden_shape"] == [3, 4]
            assert record[side]["option_ids"] == ["r001", "r002", "r003"]
            assert record[side]["request_fingerprint"]
        assert record["hidden_contrast_l2"] > 0
    for arm, record in evidence["validation"][0]["arms"].items():
        capture = record["capture"]["capture"]
        assert record["capture"]["head_calls"] == 1
        if arm in ("semantic", "random"):
            assert capture["hidden_delta_l2"] == pytest.approx(.02 * 3**.5, rel=1e-4)
        else:
            assert capture["hidden_delta_l2"] == 0
        if arm == "output_only":
            assert capture["output_delta_l2"] == pytest.approx(.02 * 3**.5, rel=1e-4)
    registration = write_direction_fit(result, tmp_path / "fit")
    loaded = _registration(registration, parent, [{"world_id": "unseen-evaluation-world"}])
    assert torch.equal(loaded[2], artifact["direction"])
    assert torch.equal(loaded[4][3], artifact["output_only_directions"]["3"])
    assert json.loads((tmp_path / "fit/fit-evidence.json").read_text()) == evidence
    assert json.loads((tmp_path / "fit/source-pairs.json").read_text()) == {"discovery": train, "validation": validation}
    assert not parent._loaded.backend.model.head._forward_hooks
    assert not parent._loaded.backend.model.head._forward_pre_hooks
    assert _parent_state_sha(parent._loaded.backend.model) == before
    assert _OWNER.acquire(blocking=False)
    _OWNER.release()


def test_labels_stay_fitter_side_empty_histories_and_validation_never_changes_fit(tmp_path):
    train, validation = pairs()
    parent = trained_parent(tmp_path, train)
    requests = []
    original = parent.decide
    def observed(request):
        requests.append(deepcopy(request))
        state = json.loads(request["state"])
        assert state["proposed_slot_choices"] == [] and state["accepted_partial_program"] == []
        assert not {"source_target", "effect_target", "training_only", "program", "targets"} & set(state)
        return original(request)
    parent.decide = observed
    first = fit_native_directions(parent, train, validation_pairs=validation, scale=.02, random_seed=17)
    changed_train = deepcopy(train)
    changed_validation = deepcopy(validation)
    # Diagnostic labels cannot affect captures or either fitted tensor.
    for row in changed_train + changed_validation:
        row["source_target"], row["effect_target"] = row["effect_target"], row["source_target"]
    changed_validation[0]["source"]["task_input"] += " Use the same visible evidence."
    changed_validation[0]["effect"]["task_input"] += " Use the same visible evidence."
    second = fit_native_directions(parent, changed_train, validation_pairs=changed_validation, scale=.02, random_seed=17)
    assert torch.equal(first["artifact"]["direction"], second["artifact"]["direction"])
    assert torch.equal(first["artifact"]["output_only_directions"]["3"], second["artifact"]["output_only_directions"]["3"])
    assert first["evidence"]["discovery"][0]["source"]["hidden_sha256"] == second["evidence"]["discovery"][0]["source"]["hidden_sha256"]
    assert len(requests) == 20  # each fit: 2*2 train + 2 validation captures + 4 arms


@pytest.mark.parametrize("change", ["heldout", "overlap_world", "overlap_content", "label_in_spec", "prefix_request", "candidate_domain", "same_target", "concept", "missing_k"])
def test_invalid_pairs_rejected_before_native_forward(tmp_path, change):
    train, validation = pairs()
    parent = trained_parent(tmp_path, train)
    if change == "heldout": train[0]["split"] = "language_heldout"
    elif change == "overlap_world": validation[0]["world_id"] = train[0]["world_id"]
    elif change == "overlap_content":
        validation[0]["source"] = deepcopy(train[0]["source"])
        validation[0]["effect"] = deepcopy(train[0]["effect"])
    elif change == "label_in_spec": train[0]["source"]["target"] = "r001"
    elif change == "prefix_request": train[0]["source"] = _request(train[0]["source"], parent, "raw-disallowed")
    elif change == "candidate_domain": train[0]["effect"]["options"]["r001"] = "different meaning"
    elif change == "same_target": train[0]["effect_target"] = train[0]["source_target"]
    elif change == "concept": validation[0]["concept"] = "opposite_direction"
    elif change == "missing_k":
        validation[0]["source"]["options"]["r004"] = "other"
        validation[0]["effect"]["options"]["r004"] = "other"
    parent.decide = lambda *_: pytest.fail("invalid discovery reached native model")
    with pytest.raises(ValueError):
        fit_native_directions(parent, train, validation_pairs=validation, scale=.02, random_seed=17)
    assert not parent._loaded.backend.model.head._forward_hooks


def test_failed_provider_cleans_hooks_and_owner_without_fabricating_direction(tmp_path):
    train, validation = pairs()
    parent = trained_parent(tmp_path, train)
    parent.decide = lambda *_: {"ssdm": {"status": "backend_error"}}
    with pytest.raises(ValueError, match="no fitted fallback"):
        fit_native_directions(parent, train, validation_pairs=validation, scale=.02, random_seed=17)
    assert not parent._loaded.backend.model.head._forward_hooks
    assert not parent._loaded.backend.model.head._forward_pre_hooks
    assert _OWNER.acquire(blocking=False)
    _OWNER.release()


def test_zero_contrast_fails_and_writer_refuses_overwrite_or_mutated_evidence(tmp_path, monkeypatch):
    train, validation = pairs()
    parent = trained_parent(tmp_path, train)
    import language_growth.gate_direction_fit as fit
    with monkeypatch.context() as patch:
        patch.setattr(fit, "_contrast", lambda *_: (torch.zeros(4, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), {}))
        with pytest.raises(ValueError, match="zero/nonfinite"):
            fit_native_directions(parent, train, validation_pairs=validation, scale=.02, random_seed=17)
    result = fit_native_directions(parent, train, validation_pairs=validation, scale=.02, random_seed=17)
    write_direction_fit(result, tmp_path / "fit")
    with pytest.raises(FileExistsError):
        write_direction_fit(result, tmp_path / "fit")
    result["evidence"]["scale"] = 1
    with pytest.raises(ValueError, match="evidence identity changed"):
        write_direction_fit(result, tmp_path / "changed")


def test_multiple_candidate_count_controls_are_fitted_from_same_pair_set(tmp_path):
    train, validation = pairs()
    parent = trained_parent(tmp_path, train)
    def two_options(rows):
        result = deepcopy(rows)
        for row in result:
            row["pair_id"] += "-K2"
            for side in ("source", "effect"):
                row[side]["options"] = {key: value for key, value in row[side]["options"].items() if key != "r003"}
        return result
    train += two_options(train)
    validation += two_options(validation)
    result = fit_native_directions(parent, train, validation_pairs=validation, scale=.02, random_seed=17)
    assert result["evidence"]["candidate_count_discovery_pairs"] == {"2": 2, "3": 2}
    assert result["artifact"]["discovery_pairs_sha256"] == _sha(train)
    assert set(result["artifact"]["output_only_directions"]) == {"2", "3"}
    for k, vector in result["artifact"]["output_only_directions"].items():
        assert vector.shape == (int(k),) and vector.double().std() > 0
    altered = deepcopy(result)
    altered["artifact"]["direction"][0] += .1
    with pytest.raises(ValueError, match="evidence identity changed"):
        write_direction_fit(altered, tmp_path / "altered-direction")
    altered = deepcopy(result)
    altered["source_pairs"]["discovery"][0]["concept"] += " changed"
    with pytest.raises(ValueError, match="evidence identity changed"):
        write_direction_fit(altered, tmp_path / "altered-source")


def test_pair_helper_ignores_teacher_forced_requests_and_rejects_nondev():
    examples = generate_development_tasks(29, 24)
    expected = development_relation_pairs(examples)
    for row in examples:
        row["model_inputs"] = "POISON: never consume teacher-forced states"
        row["training_only"]["targets"] = "POISON: never put a current target in state"
    assert development_relation_pairs(examples) == expected
    for pair in expected:
        assert pair["source"]["task_input"] == pair["effect"]["task_input"]
        assert pair["source_target"] == "r002" and pair["effect_target"] == "r001"
    examples[0]["development_split"] = "language_heldout"
    with pytest.raises(ValueError, match="only synthetic development"):
        development_relation_pairs(examples)
