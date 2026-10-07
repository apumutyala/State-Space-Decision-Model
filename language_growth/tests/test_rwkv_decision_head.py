"""Native RWKV/head integration only; random tiny weights, CPU, no capability claim."""
from copy import deepcopy
import hashlib
import json

import pytest
import torch

from heads import ByteEncoder
from heads.backbone import tiny_backbone, tiny_hf_backbone
from language_growth.rwkv_decision_head import (rwkv_cpu_parent_decider, rwkv_cpu_parent_encoder,
                                                rwkv_cpu_parent_identity, rwkv_decision_backend)
from train.data import items_from_records
from train.guard import SplitGuard
from train.trainer import TrainConfig, Trainer


SOURCE = {"model_id": "tiny-rwkv7-test", "snapshot_revision": "random-seed-31",
          "scope": "random CPU test only"}


class FixtureEncoder(ByteEncoder):
    """Byte test encoder explicitly associated with this random parent fixture."""

    def __init__(self, identity=None):
        self.info = identity if identity is not None else {
            "model_id": SOURCE["model_id"], "snapshot": SOURCE["snapshot_revision"]}

    def identity(self):
        return {**super().identity(), **deepcopy(self.info)}


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def make_backend(seed=31, **kwargs):
    parent = tiny_hf_backbone("rwkv7", seed=seed, hidden=16).model
    return rwkv_decision_backend(parent, FixtureEncoder(), source_identity=SOURCE,
                                  seed=seed, head_kwargs={"channels": 4, "max_snapshots": 2}, **kwargs)


def request():
    return {"model": SOURCE["model_id"], "state": "A is 7; B is 4.", "questions": {
        "comparison": {"type": "choice", "instructions": "Which is greater?",
                       "criteria": {"a": "A", "b": "B"}},
        "second": {"type": "choice", "instructions": "Which is smaller?",
                   "criteria": {"a": "A", "b": "B"}}}}


def cpu_plugin_and_evidence(max_context=2048):
    from decide.backends.hf_readout import ModelPlugin

    files = {"weights": "a" * 64, "config": "b" * 64, "tokenizer": "c" * 64}
    implementation = {"third_party/rwkv7_hf/modeling_rwkv7.py": "d" * 64}
    files["code"] = hashlib.sha256(json.dumps(implementation, sort_keys=True).encode()).hexdigest()
    plugin = ModelPlugin(family="rwkv7", model_id=SOURCE["model_id"], snapshot=SOURCE["snapshot_revision"],
        encode=ByteEncoder(), forward=lambda *a: None,
        untrusted_encoder_id="rwkv-world-greedy (no control tokens)", files=files,
        max_context=max_context, dtype="float32", device="cpu")
    evidence = {"snapshot": f"C:/cache/snapshots/{SOURCE['snapshot_revision']}",
        "snapshot_revision": SOURCE["snapshot_revision"], "file_sha256": {
            "model.safetensors": files["weights"], "config.json": files["config"],
            "rwkv_vocab_v20230424.txt": files["tokenizer"]},
        "implementation_sha256": implementation}
    return plugin, evidence


def test_rwkv_cpu_evidence_identity_helper_maps_loader_fields_and_rejects_mismatch():
    plugin, evidence = cpu_plugin_and_evidence()
    identity = rwkv_cpu_parent_identity(plugin, evidence)
    assert identity == {"model_id": SOURCE["model_id"], "snapshot": SOURCE["snapshot_revision"],
                        "files_sha256": plugin.files}
    with pytest.raises(ValueError, match="snapshot path/revision"):
        rwkv_cpu_parent_identity(plugin, {**evidence, "snapshot_revision": "other"})
    bad_hashes = {**evidence["file_sha256"], "model.safetensors": "f" * 64}
    with pytest.raises(ValueError, match="weights pin"):
        rwkv_cpu_parent_identity(plugin, {**evidence, "file_sha256": bad_hashes})


def test_rwkv_cpu_decider_binds_plugin_context_limit():
    plugin, evidence = cpu_plugin_and_evidence(max_context=128)
    backend = rwkv_decision_backend(tiny_hf_backbone("rwkv7", seed=31, hidden=16).model,
        rwkv_cpu_parent_encoder(plugin), source_identity=rwkv_cpu_parent_identity(plugin, evidence))
    decider = rwkv_cpu_parent_decider(backend, plugin)
    assert decider.backend_max_input_tokens == 128
    long_request = request()
    long_request["state"] += " extra" * 500
    response = decider.decide(long_request)
    assert response["ssdm"]["status"] == "exhausted"
    assert response["ssdm"]["error"]["code"] == "input_tokens_exceeded"


def test_native_hidden_states_and_backend_shared_prefix():
    backend = make_backend()
    model = backend.model
    assert model.backbone.spec.attention_free
    assert all(p.device.type == "cpu" and not p.requires_grad for p in model.backbone.parameters())
    ids = torch.tensor([[5, 12, 19]])
    with torch.no_grad():
        actual = model.backbone.features(ids)
        native = model.backbone.model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[-1]
    torch.testing.assert_close(actual, native, atol=0, rtol=0)
    result = backend.score(request(), lambda: None)
    backend.shared_prefix = False
    plain = backend.score(request(), lambda: None)
    assert result.inference_calls == 1
    for qid in result.logits:
        torch.testing.assert_close(torch.tensor(list(result.logits[qid].values())),
                                   torch.tensor(list(plain.logits[qid].values())), atol=2e-5, rtol=2e-5)


def test_head_only_step_and_exact_reload(tmp_path):
    backend = make_backend()
    guard = SplitGuard.synthetic_only("native RWKV bridge CPU test")
    records = [{"id": "synthetic", "request": request(), "labels": {"comparison": "a", "second": "b"}}]
    items = items_from_records(records, backend.encode, template=backend.template, guard=guard)
    cfg = TrainConfig(head="snapshot", head_kwargs={"channels": 4, "max_snapshots": 2},
                      lora_rank=0, seed=31, batch_size=1, max_steps=1)
    trainer = Trainer(backend.model, cfg, items, guard=guard)
    frozen = {n: v.detach().clone() for n, v in backend.model.backbone.state_dict().items()}
    before = trainer.trainable_state()
    loss = trainer.train_step()
    assert torch.isfinite(torch.tensor(loss))
    assert any(not torch.equal(v, trainer.trainable_state()[n]) for n, v in before.items() if n.startswith("head."))
    assert all(torch.equal(v, backend.model.backbone.state_dict()[n]) for n, v in frozen.items())
    assert all(p.grad is None for p in backend.model.backbone.parameters())
    checkpoint = trainer.save(tmp_path)
    state = torch.load(checkpoint, weights_only=False)
    assert all(n.startswith(("head.", "typed.")) for n in state["trainable"])
    restored = make_backend()
    restored_trainer = Trainer(restored.model, cfg, items, guard=guard)
    restored_trainer.load(checkpoint)
    assert all(torch.equal(v, restored.model.state_dict()[n]) for n, v in backend.model.state_dict().items())
    with torch.no_grad():
        torch.testing.assert_close(backend.model([items[0].row]).log_probs,
                                   restored.model([items[0].row]).log_probs, atol=0, rtol=0)


def test_l_planner_calls_actual_native_head():
    from decide.adapter import Decider
    from decide.calibration import OptionCountTemperature
    from language_growth.development_data import generate_development_tasks
    from language_growth.free_running_dev import run_development_tasks

    backend = make_backend()
    calibration = OptionCountTemperature()
    decider = Decider(backend, calibration=calibration)
    calls = []

    class RecordingProvider:
        def decide(self, req):
            response = decider.decide(req)
            calls.append((deepcopy(req), response))
            assert response["ssdm"]["status"] == "ok"
            return response

    examples = generate_development_tasks(271828, 3, choice_format="production", allow_one_repair=False)
    comparison = next(x for x in examples if x["training_only"]["kind"] == "compare")
    report = run_development_tasks([comparison], RecordingProvider(), model=SOURCE["model_id"],
        parent_revision=backend.revision()["revision_id"], expected_served_config={
            "calibration": calibration.identity(), "policy_masking": "none (served covers every supplied option)"})
    observation = report["observations"][0]
    assert calls and observation["provider_calls"] == observation["forward_calls"] == len(calls)
    assert observation["status"] in ("accepted", "kernel_rejected")
    initial = json.loads(calls[0][0]["state"])
    assert initial["proposed_slot_choices"] == []
    assert initial["accepted_partial_program"] == []


@pytest.mark.parametrize("head,kwargs", [
    ("integrator", {"channels": 4}),
    ("snapshot", {"channels": 4, "max_snapshots": 2}),
    ("head_b", {"d_state": 4, "hidden": 8}),
    ("fork", {"hidden": 8}),
])
def test_attention_free_heads_receive_trainable_gradients(head, kwargs):
    base = tiny_hf_backbone("rwkv7", seed=31, hidden=16).model
    backend = rwkv_decision_backend(base, FixtureEncoder(), source_identity=SOURCE,
                                     head=head, head_kwargs=kwargs)
    rows = backend._rows(request())
    output = backend.model(rows)
    assert torch.isfinite(output.log_probs).all()
    (-output.log_probs[0, 0]).backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in backend.model.head.parameters())
    assert all(p.grad is None for p in base.parameters())


def test_causal_lm_vocabulary_head_is_excluded():
    from third_party.rwkv7_hf.modeling_rwkv7 import Rwkv7ForCausalLM

    base = tiny_hf_backbone("rwkv7", seed=31, hidden=16).model
    lm = Rwkv7ForCausalLM(base.config)
    lm.rwkv7 = base

    def forbidden(*args, **kwargs):
        raise AssertionError("vocabulary head must never execute")

    lm.head.forward = forbidden
    backend = rwkv_decision_backend(lm, FixtureEncoder(), source_identity=SOURCE)
    assert backend.model.backbone.model is base
    assert not any(n.startswith("head.") for n in backend.model.backbone.model.state_dict())
    assert backend.score(request(), lambda: None).logits


def test_source_identity_is_bound_and_invalid_inputs_fail_before_freezing():
    source = deepcopy(SOURCE)
    base = tiny_hf_backbone("rwkv7", seed=31, hidden=16).model
    backend = rwkv_decision_backend(base, FixtureEncoder(), source_identity=source)
    source["snapshot_revision"] = "edited"
    assert backend.revision()["backbone"]["provenance"]["track_l_parent"] == SOURCE
    with pytest.raises(TypeError, match="vendored"):
        rwkv_decision_backend(tiny_backbone().model, FixtureEncoder(), source_identity=SOURCE)
    base.requires_grad_(True)
    with pytest.raises(ValueError, match="attention-free"):
        rwkv_decision_backend(base, FixtureEncoder(), source_identity=SOURCE, head="clef")
    with pytest.raises(ValueError, match="identity"):
        rwkv_decision_backend(base, lambda text, trusted: [1], source_identity=SOURCE)
    with pytest.raises(ValueError, match="model_id"):
        rwkv_decision_backend(base, FixtureEncoder(), source_identity={})
    with pytest.raises(ValueError, match="CPU FP32"):
        rwkv_decision_backend(base.double(), FixtureEncoder(), source_identity=SOURCE)
    assert all(p.requires_grad for p in base.parameters())


@pytest.mark.parametrize("field,value,match", [
    ("model_id", "different-model", "model_id"),
    ("snapshot", "different-snapshot", "snapshot"),
    ("files_sha256", {"weights": "b" * 64}, "weights pin"),
    ("files_sha256", {"tokenizer": "b" * 64}, "tokenizer pin"),
])
def test_parent_encoder_identity_mismatches_fail_before_freezing(field, value, match):
    from decide.backends.hf_readout import ModelPlugin

    base = tiny_hf_backbone("rwkv7", seed=31, hidden=16).model
    source = {**SOURCE, "files_sha256": {"weights": "a" * 64, "tokenizer": "a" * 64}}
    plugin = ModelPlugin(family="rwkv7", model_id=SOURCE["model_id"], snapshot=SOURCE["snapshot_revision"],
        encode=ByteEncoder(), forward=lambda *a: None, dtype="float32", device="cpu",
        files={"weights": "a" * 64, "tokenizer": "a" * 64, "config": "c" * 64, "code": "d" * 64},
        untrusted_encoder_id="rwkv-world-greedy (no control tokens)")
    if field == "files_sha256":
        plugin.files.update(value)
    else:
        setattr(plugin, field, value)
    with pytest.raises(ValueError, match=match):
        rwkv_decision_backend(base, rwkv_cpu_parent_encoder(plugin), source_identity=source)
    assert all(p.requires_grad for p in base.parameters())


@pytest.mark.parametrize("source,info,match", [
    ({"model_id": SOURCE["model_id"]}, None, "snapshot"),
    ({**SOURCE, "snapshot": "conflicting"}, None, "aliases"),
    (SOURCE, {"snapshot": SOURCE["snapshot_revision"]}, "model_id"),
    (SOURCE, {"model_id": SOURCE["model_id"]}, "snapshot"),
    ({**SOURCE, "files_sha256": []}, None, "objects"),
])
def test_parent_encoder_identity_requires_unambiguous_model_snapshot(source, info, match):
    base = tiny_hf_backbone("rwkv7", seed=31, hidden=16).model
    with pytest.raises(ValueError, match=match):
        rwkv_decision_backend(base, FixtureEncoder(info), source_identity=source)
    assert all(p.requires_grad for p in base.parameters())


def test_matching_snapshot_aliases_and_shared_file_pins_are_accepted():
    base = tiny_hf_backbone("rwkv7", seed=31, hidden=16).model
    source = {**SOURCE, "snapshot": SOURCE["snapshot_revision"], "files_sha256": {"weights": "a" * 64}}
    encoder = FixtureEncoder({"model_id": SOURCE["model_id"], "snapshot": source["snapshot"],
                              "files_sha256": {"weights": "a" * 64, "tokenizer": "b" * 64}})
    backend = rwkv_decision_backend(base, encoder, source_identity=source)
    assert backend.model.backbone.provenance["track_l_parent"] == source
