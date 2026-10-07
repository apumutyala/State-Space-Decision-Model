import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from heads import ByteEncoder, DecisionModel
from heads.backbone import tiny_backbone
from train.data import TOY_TEMPLATE, items_from_records
from train.guard import SplitGuard
from train.losses import LossConfig, add_order_consistency, aligned_order_js
from train.order_views import reversed_record
from train.run_cell import attach_order_views, order_consistency_fields, restrict_order_views_to_tmax
from train.trainer import Trainer, TrainConfig
from train.trainer import row_groups, row_groups_with_views

G = SplitGuard.synthetic_only("unit tests: synthetic order-consistency pair")


def _record():
    return {
        "id": "order-pair",
        "request": {
            "model": "toy",
            "state": "the favorite color is red",
            "questions": {"q": {"type": "choice", "instructions": "which color?",
                                 "criteria": {"a": "red", "b": "green", "c": "blue"}}},
        },
        "labels": {"q": "a"},
    }


def _paired_items():
    enc = ByteEncoder()
    rec = _record()
    base = items_from_records([rec], enc, template=TOY_TEMPLATE)[0]
    result = reversed_record(rec, "q")
    assert result is not None
    view_record, perm = result
    view = items_from_records([view_record], enc, template=TOY_TEMPLATE)[0]
    base.view, base.view_perm = view, tuple(perm)
    return [base]


def _fresh(seed=0):
    torch.manual_seed(seed)
    return DecisionModel(tiny_backbone(seed=seed, hidden=16), "integrator")


def test_run_cell_order_arm_is_fixed_and_opt_in():
    assert order_consistency_fields({}) == {}
    assert order_consistency_fields({"order_consistency": {"view": "reverse", "weight": 0.25}}) == {
        "order_consistency_weight": 0.25
    }
    for cfg in (
        {"order_consistency": {"view": "reverse", "weight": 0.2}},
        {"order_consistency": {"view": "random", "weight": 0.25}},
        {"order_consistency": {"view": "reverse", "weight": 0.25, "seed": 0}},
        {"order_consistency": {"view": "reverse", "weight": 0.25}, "teacher": {"alpha": 0.5}},
        {"order_consistency": {"view": "reverse", "weight": 0.25}, "axis_loss_weight": 0.1},
    ):
        with pytest.raises(ValueError):
            order_consistency_fields(cfg)


def test_run_cell_attaches_views_only_to_explicit_training_items():
    rec = _record()
    enc = ByteEncoder()
    train_items = items_from_records([rec], enc, template=TOY_TEMPLATE)
    eval_items = items_from_records([rec], enc, template=TOY_TEMPLATE)
    cfg = {"order_consistency": {"view": "reverse", "weight": 0.25}, "layout": "ordered",
           "template": TOY_TEMPLATE, "max_k": 4}
    coverage = attach_order_views(cfg, train_items, {rec["id"]: rec}, enc, G, max_t=4096)
    assert coverage["with_view"] == 1
    assert train_items[0].view is not None
    assert eval_items[0].view is None


def test_run_cell_refuses_reversal_when_state_or_instruction_names_option_id():
    rec = _record()
    rec["request"]["state"] = "the correct option ID is a"
    rec["request"]["questions"]["q"]["instructions"] = "choose the named ID"
    enc = ByteEncoder()
    train_items = items_from_records([rec], enc, template=TOY_TEMPLATE)
    cfg = {"order_consistency": {"view": "reverse", "weight": 0.25}, "layout": "ordered",
           "template": TOY_TEMPLATE, "max_k": 4}
    coverage = attach_order_views(cfg, train_items, {rec["id"]: rec}, enc, G, max_t=4096)
    assert coverage["with_view"] == 0
    assert train_items[0].view is None and train_items[0].view_perm is None


def test_run_cell_drops_over_tmax_view_without_dropping_its_base_row():
    item = _paired_items()[0]
    base_length = len(item.row.ids)
    item.view.row.ids = [0] * (base_length + 1)
    items = [item]
    cfg = {"order_consistency": {"view": "reverse", "weight": 0.25}}
    result = restrict_order_views_to_tmax(cfg, items, base_length)
    assert result == {"t_max": base_length, "retained": 0, "dropped_for_length": 1}
    assert len(items) == 1 and items[0].view is None and items[0].view_perm is None


def test_aligned_js_is_zero_for_same_content_distribution_under_permutation():
    base = torch.log(torch.tensor([0.7, 0.2, 0.1], dtype=torch.float64)).requires_grad_()
    # view_to_base=(2,0,1), so view order is [base[2], base[0], base[1]].
    view = torch.log(torch.tensor([0.1, 0.7, 0.2], dtype=torch.float64)).requires_grad_()
    loss = aligned_order_js(base, view, (2, 0, 1))
    assert float(loss.detach()) == pytest.approx(0.0, abs=1e-12)
    loss.backward()
    assert torch.isfinite(base.grad).all() and torch.isfinite(view.grad).all()


def test_aligned_js_is_symmetric_and_bounded_by_log_two():
    a = torch.log(torch.tensor([0.9, 0.1]))
    b = torch.log(torch.tensor([0.2, 0.8]))
    ab = aligned_order_js(a, b, (0, 1))
    ba = aligned_order_js(b, a, (0, 1))
    assert float(ab) == pytest.approx(float(ba), abs=1e-7)
    assert 0.0 < float(ab) <= math.log(2) + 1e-7
    disjoint = aligned_order_js(torch.tensor([0.0, -100.0]), torch.tensor([-100.0, 0.0]), (0, 1))
    assert float(disjoint) <= math.log(2) + 1e-6


@pytest.mark.parametrize("perm", [(0, 0), (0,), (0, 1, 2), (-1, 0)])
def test_aligned_js_rejects_invalid_permutation(perm):
    with pytest.raises(ValueError):
        aligned_order_js(torch.log(torch.tensor([0.5, 0.5])), torch.log(torch.tensor([0.5, 0.5])), perm)


def test_aligned_js_rejects_unnormalized_and_nonfinite_inputs():
    with pytest.raises(ValueError, match="normalized"):
        aligned_order_js(torch.tensor([0.0, 0.0]), torch.log(torch.tensor([0.5, 0.5])), (0, 1))
    with pytest.raises(ValueError, match="finite"):
        aligned_order_js(torch.tensor([0.0, float("-inf")]), torch.tensor([0.0, float("-inf")]), (0, 1))


def test_extra_view_objective_is_invariant_to_base_row_grouping():
    weight, js, view = 0.25, torch.tensor(0.1), torch.tensor(3.0)
    whole = add_order_consistency(torch.tensor(3.0), view, js, n_base=2, n_paired=1, weight=weight)
    paired_group = add_order_consistency(torch.tensor(1.0), view, js, n_base=1, n_paired=1, weight=weight)
    unpaired_group = add_order_consistency(torch.tensor(5.0), view, js, n_base=1, n_paired=0, weight=weight)
    regrouped = 0.5 * paired_group + 0.5 * unpaired_group
    assert torch.allclose(whole, regrouped)
    assert torch.equal(add_order_consistency(torch.tensor(2.0), view, js, 1, 0, weight), torch.tensor(2.0))


def test_order_view_row_groups_bound_separate_padded_forwards():
    # Longest base and longest view are on different rows; a sum-per-pair proxy would undercount this.
    groups = row_groups_with_views([100, 1], [1, 100], row_batch=2, row_tokens=250)
    assert groups == [[0], [1]]
    # A budget that covers both padded forwards can keep the rows together.
    assert row_groups_with_views([10, 8], [5, 4], row_batch=2, row_tokens=30) == [[0, 1]]
    # With no views, grouping is identical to the established default path.
    assert row_groups_with_views([10, 7, 5], [0, 0, 0], row_batch=2, row_tokens=18) == row_groups(
        [10, 7, 5], row_batch=2, row_tokens=18
    )


def test_default_order_consistency_config_preserves_legacy_identity():
    assert "order_consistency_weight" not in LossConfig().identity()
    assert LossConfig(order_consistency_weight=0.0).identity() == LossConfig().identity()
    assert LossConfig(order_consistency_weight=0.25).identity()["order_consistency_weight"] == 0.25


def test_opt_in_order_consistency_trains_base_and_view_and_backpropagates():
    paired = _paired_items()
    cfg = TrainConfig(head="integrator", lora_rank=4, batch_size=1, max_steps=1,
                      loss=LossConfig(order_consistency_weight=0.25))
    model = _fresh()
    trainer = Trainer(model, cfg, paired, guard=G)
    history = trainer.train()
    assert len(history) == 1 and math.isfinite(history[0]["loss"])
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.head_parameters())


def test_opt_in_order_consistency_requires_training_views_and_rejects_eval_views():
    plain = items_from_records([_record()], ByteEncoder(), template=TOY_TEMPLATE)
    cfg = TrainConfig(head="integrator", lora_rank=4, batch_size=1, max_steps=1,
                      loss=LossConfig(order_consistency_weight=0.25))
    with pytest.raises(ValueError, match="no training items"):
        Trainer(_fresh(), cfg, plain, guard=G)
    paired = _paired_items()
    with pytest.raises(ValueError, match="eval_items"):
        Trainer(_fresh(), cfg, paired, eval_items=paired, guard=G)


def test_order_consistency_rejects_kd_until_view_teacher_semantics_are_registered():
    paired = _paired_items()
    paired[0].teacher = (0.2, 0.3, 0.5)
    cfg = TrainConfig(head="integrator", lora_rank=4, batch_size=1, max_steps=1,
                      loss=LossConfig(kd_alpha=0.5, order_consistency_weight=0.25))
    with pytest.raises(ValueError, match="requires KD off"):
        Trainer(_fresh(), cfg, paired, guard=G)


def test_bf16_autocast_uses_cpu_when_cuda_is_available_but_has_no_visible_devices(monkeypatch):
    trainer = object.__new__(Trainer)
    trainer.cfg = SimpleNamespace(bf16=True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with trainer._autocast():
        values = torch.tensor([1.0]) + torch.tensor([1.0])
    assert torch.equal(values, torch.tensor([2.0]))
