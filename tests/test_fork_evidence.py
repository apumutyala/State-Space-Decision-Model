"""CPU checks for the parent-preserving evidence residual (no model/data download)."""

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from heads.backbone import tiny_backbone
from heads.features import HeadInput
from heads.heads import ForkEvidenceResidual
from heads.model import DecisionModel


def features():
    torch.manual_seed(91)
    return HeadInput(h=torch.randn(2, 7, 8), token_mask=torch.ones(2, 7, dtype=torch.bool),
                     evidence_mask=torch.tensor([[0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool),
                     q_pre=torch.randn(2, 8), q_end=torch.randn(2, 8), e=torch.randn(2, 3, 8),
                     option_mask=torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
                     fork=torch.randn(2, 3, 8), ks=[2, 3])


@pytest.mark.parametrize("bf16", [False, True])
def test_initial_model_preserves_fork_and_typed_parameters(bf16):
    bb = tiny_backbone(hidden=8)
    parent = DecisionModel(bb, "fork", seed=31)
    child = DecisionModel(bb, "fork_evidence", seed=31)
    for key, value in parent.head.state_dict().items():
        assert torch.equal(value, child.head.state_dict()[key])
    for key, value in parent.typed.state_dict().items():
        assert torch.equal(value, child.typed.state_dict()[key])
    x = features()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        a, b = parent.head(x).logits, child.head(x).logits
        assert a.dtype == b.dtype and torch.equal(a, b)
        assert torch.equal(parent.typed(a, ["choice", "score"], x.ks, x.e),
                           child.typed(b, ["choice", "score"], x.ks, x.e))


def test_trained_residual_masking_equivariance_and_accounting():
    head = ForkEvidenceResidual(8, channels=4, t_chunk=2)
    with torch.no_grad():
        head.evidence_readout.weight.fill_(0.7)
    x = features()
    out = head(x, return_attribution=True)
    assert torch.allclose(out.bias + out.attribution.sum(1), out.logits, atol=1e-6)
    assert torch.count_nonzero(out.attribution[~x.evidence_mask]) == 0
    assert torch.count_nonzero(out.attribution[0, :, 2]) == 0
    assert torch.equal(out.logits[1], out.bias[1])  # no evidence: exact parent
    changed = dataclasses.replace(x, h=x.h.clone())
    changed.h[~x.evidence_mask] = torch.randn_like(changed.h[~x.evidence_mask]) * 100
    assert torch.equal(head(changed).logits, out.logits)
    changed.h = x.h.clone()
    changed.h[0, 1, 0] += 5
    assert not torch.allclose(head(changed).logits[0, :2], out.logits[0, :2])
    perm = torch.tensor([2, 0, 1])
    assert torch.allclose(head(x.permute_options(perm)).logits, out.logits[:, perm], atol=1e-6)
    head.t_chunk = 7
    assert torch.allclose(head(x).logits, out.logits, atol=1e-6)


def test_zero_initialised_branch_learns_then_reaches_evidence_features():
    x = features()
    x.h.requires_grad_()
    head = ForkEvidenceResidual(8, channels=4, t_chunk=2)
    loss = -torch.log_softmax(head(x).logits[0, :2], -1)[1]
    loss.backward()
    grad = head.evidence_readout.weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    with torch.no_grad():
        head.evidence_readout.weight.add_(grad, alpha=-0.1)
    head.zero_grad(set_to_none=True)
    x.h.grad = None
    loss = -torch.log_softmax(head(x).logits[0, :2], -1)[1]
    loss.backward()
    assert torch.isfinite(x.h.grad).all()
    assert x.h.grad[x.evidence_mask].abs().sum() > 0
    assert torch.count_nonzero(x.h.grad[~x.evidence_mask]) == 0


def test_fork_checkpoint_weights_transfer_without_renaming():
    parent = DecisionModel(tiny_backbone(hidden=8), "fork", seed=13)
    child = ForkEvidenceResidual(8)
    missing = child.load_state_dict(parent.head.state_dict(), strict=False)
    assert missing.missing_keys and not missing.unexpected_keys
    assert not any(k.startswith(("norm.", "mlp.")) for k in missing.missing_keys)
    x = features()
    assert torch.equal(child(x).logits, parent.head(x).logits)
