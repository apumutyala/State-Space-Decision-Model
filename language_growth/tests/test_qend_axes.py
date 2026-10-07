"""Head-port, ownership and training-lineage proofs on tiny CPU tensors only."""
from copy import deepcopy
from dataclasses import fields, replace
import math

import pytest
import torch

from heads.features import HeadInput
from heads.heads import QEndSemanticAxes
from language_growth.qend_axis_targets import axis_target, attach_axis_targets, targets_by_item


def program(kind="read", left=1, field="price", relation="less_than"):
    nodes = [{"id": "f1", "op": "ReadFact", "record_ref": f"development-record-{left}", "field": field}]
    proposition = "f1"
    if kind == "compare":
        nodes += [{"id": "f2", "op": "ReadFact", "record_ref": f"development-record-{3-left}", "field": field},
                  {"id": "c", "op": "Compare", "left": {"node": "f1", "project": "object"},
                   "right": {"node": "f2", "project": "object"}, "relation": relation}]
        proposition = "c"
    nodes += [{"id": "a", "op": "Assert", "proposition": proposition}]
    output = "a"
    if kind == "explain":
        nodes += [{"id": "x", "op": "ExplainDerivation", "speech_act": "a"}]
        output = "x"
    return {"nodes": nodes, "outputs": [output]}


def inputs():
    torch.manual_seed(71)
    return HeadInput(h=torch.randn(2, 4, 8), token_mask=torch.ones(2, 4, dtype=torch.bool),
        evidence_mask=torch.ones(2, 4, dtype=torch.bool), q_pre=torch.randn(2, 8), q_end=torch.randn(2, 8),
        e=torch.randn(2, 3, 8), option_mask=torch.tensor([[True, True, True], [True, True, False]]), ks=[3, 2])


def test_slot_factors_read_explain_orders_fields_and_equality():
    for kind, operation in (("read", (1, 0, 0)), ("explain", (0, 1, 0))):
        p = program(kind, left=2, field="warranty_months")
        assert axis_target(p, "operation") == (*operation, 0, 0, 0)
        assert axis_target(p, "fact") == (0, 0, 0, 1, 1, 0)
    for left in (1, 2):
        for relation, sign in (("less_than", -1), ("equal", 0), ("greater_than", 1)):
            p = program("compare", left=left, relation=relation)
            expected = -1 if left == 1 else 1
            assert axis_target(p, "operation") == (0, 0, 1, 0, 0, 0)
            assert axis_target(p, "left") == (0, 0, 0, expected, -1, 0)
            assert axis_target(p, "right") == (0, 0, 0, -expected, -1, 0)
            assert axis_target(p, "relation") == (0, 0, 0, 0, 0, sign)
    with pytest.raises(ValueError):
        axis_target(program("compare"), "fact")


def test_head_reads_only_qend_and_embeddings_and_permutes_options():
    x = inputs()
    head = QEndSemanticAxes(8)
    out = head(x)
    assert out.semantic_axes.shape == (2, 6)
    changed = replace(x, h=x.h + 100, q_pre=x.q_pre - 100, evidence_mask=~x.evidence_mask)
    torch.testing.assert_close(head(changed).logits, out.logits, atol=0, rtol=0)
    perm = torch.tensor([2, 0, 1])
    torch.testing.assert_close(head(x.permute_options(perm)).logits, out.logits[:, perm])
    assert torch.isneginf(out.logits[1, 2])
    assert "axis_target" not in {field.name for field in fields(HeadInput)}


def make_training(weight):
    from heads.backbone import tiny_backbone
    from heads.model import DecisionModel
    from heads import ByteEncoder
    from train.data import items_from_records
    from train.guard import SplitGuard
    from train.trainer import TrainConfig, Trainer
    from language_growth.development_data import generate_development_tasks
    from language_growth.training_records import export_training_records
    examples = generate_development_tasks(9201, 3, choice_format="production")
    guard = SplitGuard.synthetic_only("qend axes ownership fixture")
    raw_items = items_from_records(export_training_records(examples), ByteEncoder(), guard=guard)
    items = attach_axis_targets(raw_items, examples)
    model = DecisionModel(tiny_backbone(seed=71, hidden=8, layers=1), "qend_axes", seed=71)
    cfg = TrainConfig(head="qend_axes", lora_rank=0, seed=71, axis_loss_weight=weight,
                      batch_size=2, max_steps=1)
    trainer = Trainer(model, cfg, items, guard=guard)
    return trainer, raw_items, items, examples


def test_sidecars_do_not_mutate_rows_requests_and_bind_lineage():
    from train.lineage import items_identity
    trainer, raw, items, examples = make_training(1)
    assert all(a.row is b.row and a.label == b.label for a, b in zip(raw, items))
    assert all("axis_target" not in item.row.meta for item in items)
    changed = deepcopy(examples)
    fact = changed[0]["training_only"]["program"]["nodes"][0]
    fact["record_ref"] = "development-record-2" if fact["record_ref"] == "development-record-1" else "development-record-1"
    assert [e["model_inputs"] for e in changed] == [e["model_inputs"] for e in examples]
    target_change = attach_axis_targets(raw, changed)
    assert items_identity(items) != items_identity(target_change)
    assert items_identity(raw) != items_identity(items)
    assert all(not p.requires_grad for p in trainer.model.backbone.parameters())


def test_axis_weight_zero_is_exact_choice_ce_and_one_adds_only_mse():
    from train.losses import decision_loss
    torch.set_num_threads(2)
    control, _, items, _ = make_training(0)
    axes, _, other, _ = make_training(1)
    assert all(torch.equal(p, axes.model.state_dict()[n]) for n, p in control.model.state_dict().items())
    batch = items[:2]
    fw = control.model([i.row for i in batch])
    ce = decision_loss(fw.log_probs, torch.tensor([i.label for i in batch]),
                       [i.row.qtype for i in batch], control.cfg.loss)
    torch.testing.assert_close(control._supervised_loss(fw, batch), ce, atol=0, rtol=0)
    target = torch.tensor([i.axis_target for i in batch])
    expected = ce + torch.nn.functional.mse_loss(fw.head.semantic_axes, target)
    torch.testing.assert_close(axes._supervised_loss(fw, batch), expected, atol=0, rtol=0)
    assert "axis_loss_weight" not in control.cfg.to_json()
    assert axes.cfg.to_json()["axis_loss_weight"] == 1
    bad = [replace(i, axis_target=None) for i in batch]
    with pytest.raises(ValueError, match="axis supervision"):
        axes._supervised_loss(fw, bad)


def test_axis_clamp_has_exact_affine_effect_and_no_headinput_mutation():
    x = inputs()
    head = QEndSemanticAxes(8)
    original = head(x)
    z = original.semantic_axes.detach().clone()
    clamped = z.clone()
    clamped[:, 3] = -1
    actual = head.score_axes(x, clamped)
    predicted = original.logits + torch.einsum("bd,bkd->bk", clamped - z, head.option_projection(x.e))
    torch.testing.assert_close(actual.logits[x.option_mask], predicted[x.option_mask])
    torch.testing.assert_close(head(x).semantic_axes, original.semantic_axes)
    assert original.semantic_axes.data_ptr() != clamped.data_ptr()


def test_option_relabeling_changes_only_wire_identity_for_fixed_embeddings():
    from heads.typed import to_wire
    x = inputs()
    logits = QEndSemanticAxes(8)(x).logits[0].log_softmax(-1)
    text = ("first candidate", "second candidate", "third candidate")
    ids = ("opaque-a", "opaque-b", "opaque-c")
    relabeled = ("z91", "a02", "m17")
    a = {"type": "choice", "criteria": dict(zip(ids, text))}
    b = {"type": "choice", "criteria": dict(zip(relabeled, text))}
    original = to_wire(a, list(ids), logits)
    changed = to_wire(b, list(relabeled), logits)
    assert a["criteria"][original["choice"]] == b["criteria"][changed["choice"]]
    assert [original["probabilities"][k] for k in ids] == [changed["probabilities"][k] for k in relabeled]


def test_tiny_aux_step_freezes_backbone_and_resume_rejects_target_drift(tmp_path):
    from train.trainer import Trainer
    trainer, _, items, _ = make_training(1)
    frozen = {n: p.detach().clone() for n, p in trainer.model.backbone.named_parameters()}
    before = trainer.model.head.axis_projection.weight.detach().clone()
    assert math.isfinite(trainer.train_step())
    assert not torch.equal(before, trainer.model.head.axis_projection.weight)
    assert all(torch.equal(value, dict(trainer.model.backbone.named_parameters())[name]) for name, value in frozen.items())
    checkpoint = trainer.save(tmp_path / "checkpoint")
    restored, _, _, _ = make_training(1)
    restored.load(checkpoint)
    assert all(torch.equal(value, restored.model.state_dict()[name]) for name, value in trainer.model.state_dict().items())
    changed = list(items)
    changed[0] = replace(changed[0], axis_target=(0., 0., 0., 0., 0., 0.))
    mismatch, _, _, _ = make_training(1)
    mismatch = Trainer(mismatch.model, mismatch.cfg, changed, guard=mismatch.guard)
    with pytest.raises(ValueError, match="lineage"):
        mismatch.load(checkpoint)
