"""RLCD second stage (mode d14_rlcd) on tiny CPU backbones: rewards by hand, KL = 0 at init, finite gradients, loss
decreases from a trained parent's checkpoint, export -> safetensors -> init round trip, pins enforced, old hashes kept."""

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from heads import ByteEncoder, DecisionModel  # noqa: E402
from heads.backbone import tiny_backbone  # noqa: E402
from train import rlcd  # noqa: E402
from train.data import TOY_TEMPLATE, items_from_records, toy_records  # noqa: E402
from train.guard import SplitGuard  # noqa: E402
from train.losses import LossConfig, kl_to_ref, option_rewards, record_terms, rlcd_loss  # noqa: E402
from train.trainer import Trainer, TrainConfig, record_of  # noqa: E402

ENC = ByteEncoder()
G = SplitGuard.synthetic_only("unit tests: train.data.toy_records only")
ROOT = Path(__file__).resolve().parents[1]
RL = LossConfig(arm="rlcd", rlcd_beta=0.1, rlcd_brier=0.5, rlcd_record=0.25)


@pytest.fixture(scope="module")
def toy():
    return items_from_records(toy_records(32, seed=0), ENC, template=TOY_TEMPLATE)


def fresh(seed=0):
    torch.manual_seed(seed)
    return DecisionModel(tiny_backbone(seed=seed, hidden=16), "integrator", seed=seed)


def test_rewards_kl_record_by_hand():
    lp = torch.log(torch.tensor([[0.6, 0.3, 0.1, 0.0], [0.2, 0.3, 0.4, 0.1]]).clamp_min(1e-30))
    lp[0, 3] = float("-inf")
    y = torch.tensor([0, 1])
    r = option_rewards(lp, y, ["choice", "score"])
    assert torch.allclose(r[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(r[1], torch.tensor([2 / 3, 1.0, 2 / 3, 1 / 3]))  # K = 4 levels, gold 1
    assert torch.allclose(kl_to_ref(lp, lp.clone()), torch.zeros(2))
    pg = torch.tensor([0.5, 0.8, 0.9])
    t = record_terms(pg, ["a", "a", "b"], torch.tensor([1.0, 1.0, 0.5]))
    assert torch.allclose(t, torch.tensor([0.3, 0.3, 0.55]))  # a: (1 - .4) / 2 each; b: 1 - .9 x .5
    loss = rlcd_loss(lp, y, ["choice", "score"], LossConfig(arm="rlcd"), lp.clone())
    expect = ((1 - 0.6) + (1 - (0.2 * 2 / 3 + 0.3 + 0.4 * 2 / 3 + 0.1 / 3))) / 2
    assert abs(float(loss) - expect) < 1e-5


def test_old_identities_unchanged_and_rlcd_fields_guarded():
    assert "rlcd_beta" not in LossConfig(arm="clef").identity()
    assert TrainConfig(loss=LossConfig(arm="clef")).hash() == TrainConfig(loss=LossConfig(arm="clef")).hash()
    with pytest.raises(ValueError):
        LossConfig(arm="clef", rlcd_beta=0.1)
    from train.losses import decision_loss

    with pytest.raises(ValueError):
        decision_loss(torch.zeros(1, 2), torch.tensor([0]), ["choice"], RL)


def _parent(toy, tmp_path):
    """A trained parent cell (clef arm), saved as a Trainer checkpoint."""
    cfg = TrainConfig(head="integrator", lora_rank=4, lr_head=1e-2, lr_lora=1e-3, batch_size=8, max_steps=10,
                      loss=LossConfig(arm="clef"), ckpt_dir=str(tmp_path / "parent"))
    t = Trainer(fresh(), cfg, toy, guard=G)
    t.train()
    return t, t.save()


def _child(toy, init, loss=RL, **kw):
    cfg = TrainConfig(head="integrator", lora_rank=4, lr_head=3e-3, lr_lora=3e-4, batch_size=8, max_steps=40,
                      loss=loss, **kw)
    t = Trainer(fresh(), cfg, toy, guard=G)
    rlcd.apply_init(t.model, init)
    t.set_reference()
    return t


def test_export_init_kl_zero_grad_finite_loss_decreases(toy, tmp_path):
    parent, pt = _parent(toy, tmp_path)
    out = tmp_path / "init" / "trainable.safetensors"
    meta = rlcd.export(pt, out, "art-test", "job-test")
    assert meta["config_hash"] == parent.cfg.hash() and meta["n_tensors"] == len(parent.trainable_state())
    block = {"sha256": meta["sha256"], "ckpt_sha256": meta["source_ckpt_sha256"], "config_hash": meta["config_hash"],
             "step": 10, "artifact": "art-test", "job": "job-test"}
    init = rlcd.load_init(out, block)
    for k, v in parent.trainable_state().items():
        assert torch.equal(init[k], v), k
    with pytest.raises(ValueError):
        rlcd.load_init(out, {**block, "sha256": "0" * 64})
    with pytest.raises(ValueError):
        rlcd.load_init(out, {**block, "artifact": "art-other"})

    t = _child(toy, init)
    batch = toy[:8]
    # the child starts AT the parent: same distribution, KL(p || p_ref) == 0
    with torch.no_grad():
        lp_parent = parent.model([i.row for i in batch]).log_probs.float()
    ref = t._ref_log_probs(batch)
    lp = t.model([i.row for i in batch]).log_probs.float()
    assert torch.allclose(lp, lp_parent, atol=1e-6) and torch.allclose(ref, lp_parent, atol=1e-6)
    assert float(kl_to_ref(lp.detach(), ref).abs().max()) < 1e-6
    # gradient finite and non-zero on LoRA + head
    t.model.zero_grad(set_to_none=True)
    t._row_loss(batch, 1.0)
    grads = [p.grad for p in t.model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(float(g.abs().sum()) for g in grads) > 0
    t.model.zero_grad(set_to_none=True)
    # the RLCD loss decreases; the reference never moves; KL > 0 after training
    ref0 = {k: v.clone() for k, v in t.ref_state.items()}
    h = t.train()
    first = sum(x["loss"] for x in h[:8]) / 8
    last = sum(x["loss"] for x in h[-8:]) / 8
    assert all(math.isfinite(x["loss"]) for x in h) and last < first, (first, last)
    assert all(torch.equal(ref0[k], t.ref_state[k]) for k in ref0)
    lp2 = t.model([i.row for i in batch]).log_probs.float().detach()
    assert float(kl_to_ref(lp2, t._ref_log_probs(batch)).mean()) > 0


def test_record_contiguous_order_and_resume_keeps_reference(toy, tmp_path):
    parent, _ = _parent(toy, tmp_path)
    init = parent.trainable_state()
    t = _child(toy, init, ckpt_dir=str(tmp_path / "child"), ckpt_every=5)
    order = t._order(0)
    assert sorted(order) == list(range(len(toy)))
    recs = [record_of(toy[i].id) for i in order]
    assert sum(1 for a, b in zip(recs, recs[1:]) if a != b) == len(set(recs)) - 1  # each record one contiguous run
    t.train(5)
    c = _child(toy, init, ckpt_dir=str(tmp_path / "child"), ckpt_every=5)
    assert c.resume() and c.step == 5
    for k, v in init.items():
        assert torch.equal(c.ref_state[k], v), k  # p_ref = the init, not the resumed weights
    assert not all(torch.equal(dict(c.model.named_parameters())[k], v) for k, v in init.items())


def test_row_batch_path_matches_microbatch_loss(toy, tmp_path):
    parent, _ = _parent(toy, tmp_path)
    init = parent.trainable_state()
    # the per-row terms are grouping-invariant; the record-exact term sees a record jointly only within one group
    # (questions outside it enter through the detached p(gold) cache), so it is left out of this equivalence
    no_rec = LossConfig(arm="rlcd", rlcd_beta=0.1, rlcd_brier=0.5)
    a = _child(toy, init, loss=no_rec, grad_accum=1)
    b = _child(toy, init, loss=no_rec, grad_accum=1, row_batch=4, row_plan="min_cost")
    la, lb = a.train_step(), b.train_step()
    assert math.isfinite(la) and abs(la - lb) < 1e-4, (la, lb)


def test_d14_cell_config_is_d1_fla_plus_rlcd():
    d1 = json.loads((ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_d1_fla.json").read_text(encoding="utf-8"))
    d14 = json.loads((ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_d14_rlcd.json").read_text(encoding="utf-8"))
    changed = {k for k in set(d1) | set(d14) if d1.get(k) != d14.get(k)} - {"_comment"}
    assert changed == {"loss_arm", "rlcd", "init_from", "max_minutes", "lr_head", "lr_lora", "warmup_steps",
                       "rlcd_note"}, changed
    from train.run_cell import rlcd_fields, train_config

    assert rlcd_fields(d14) == {"rlcd_beta": 0.1, "rlcd_brier": 0.5, "rlcd_record": 0.25}
    assert train_config(d14, 10).loss.arm == "rlcd"
    params = json.loads((ROOT / "jobs/m2_first_cell/job_params_d14_rlcd.json").read_text(encoding="utf-8"))
    assert params["mode"] == "d14_rlcd" and params["init_checkpoint"]["sha256"] == d14["init_from"]["sha256"]
