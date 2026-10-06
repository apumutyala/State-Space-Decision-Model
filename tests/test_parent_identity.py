"""CPU revision identity for a tiny random decision parent; no model snapshot is loaded."""

import pytest

torch = pytest.importorskip("torch")

from heads import ByteEncoder, DecisionModel  # noqa: E402
from heads.backbone import tiny_backbone  # noqa: E402
from heads.decide_backend import DecisionModelBackend  # noqa: E402
from train.lora import LoRALinear, apply_lora  # noqa: E402


def model(head_seed=7):
    return DecisionModel(tiny_backbone(seed=19, hidden=8, layers=1), "snapshot", seed=head_seed)


def revision(parent):
    return DecisionModelBackend(parent, ByteEncoder()).revision()


def with_lora():
    parent = model()
    apply_lora(parent.backbone.model, r"in_proj$", rank=2,
               generator=torch.Generator().manual_seed(23))
    return parent


def first_lora(parent):
    return next(module for module in parent.backbone.modules() if isinstance(module, LoRALinear))


def test_different_head_weights_change_revision_without_checkpoint_metadata():
    a, b = revision(model(head_seed=7)), revision(model(head_seed=8))
    assert a["backbone"] == b["backbone"]
    assert a["trained_state"]["sha256"] != b["trained_state"]["sha256"]
    assert a["revision_id"] != b["revision_id"]


def test_changed_lora_state_changes_revision_with_same_frozen_base():
    a, b = with_lora(), with_lora()
    with torch.no_grad():
        first_lora(b).lora_B[0, 0] = 0.125
    ra, rb = revision(a), revision(b)
    assert ra["backbone"] == rb["backbone"]
    assert ra["trained_state"]["lora_modules"] == 1
    assert ra["trained_state"]["sha256"] != rb["trained_state"]["sha256"]
    assert ra["revision_id"] != rb["revision_id"]


def test_changed_typed_output_changes_revision_with_same_head():
    a, b = model(), model()
    with torch.no_grad():
        b.typed.ordinal.eta_scale.add_(0.125)
    ra, rb = revision(a), revision(b)
    assert ra["backbone"] == rb["backbone"]
    assert ra["trained_state"]["sha256"] != rb["trained_state"]["sha256"]
    assert ra["revision_id"] != rb["revision_id"]


def test_same_weights_repeat_revision_across_backend_instances():
    a, b = revision(with_lora()), revision(with_lora())
    assert a == b
    assert a["trained_state"]["algorithm"] == "ssdm.trained_state_sha256.exact.v1"


def test_freezing_every_parameter_does_not_change_revision():
    a, b = with_lora(), with_lora()
    for parameter in b.parameters():
        parameter.requires_grad_(False)
    assert revision(a) == revision(b)


def test_lora_scaling_changes_revision_even_with_identical_tensors():
    a, b = with_lora(), with_lora()
    first_lora(b).scaling *= 0.5
    assert revision(a)["revision_id"] != revision(b)["revision_id"]


def test_tensor_dtype_changes_digest_without_changing_frozen_base():
    a, b = model(), model()
    b.head.to(dtype=torch.bfloat16)
    ra, rb = revision(a), revision(b)
    assert ra["backbone"] == rb["backbone"]
    assert ra["trained_state"]["sha256"] != rb["trained_state"]["sha256"]
