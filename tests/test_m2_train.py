"""m2 trainer on tiny random backbones (CPU only): LoRA, losses, loss decreases, resume,
post-hoc temperature, eval card stubs, bf16 + gradient checkpointing, estimator sanity."""

import math

import pytest

torch = pytest.importorskip("torch")

from heads import ByteEncoder, DecisionModel  # noqa: E402
from heads.backbone import BACKBONES, tiny_backbone  # noqa: E402
from train.calibration import FeatureTemperature, features  # noqa: E402
from train.data import TOY_TEMPLATE, items_from_records, toy_records  # noqa: E402
from train.evalcard import run_card  # noqa: E402
from train.lora import LoRALinear, apply_lora, lora_parameters  # noqa: E402
from train.losses import LossConfig, brier, decision_loss, label_smoothed_ce, nll, rps  # noqa: E402
from train.trainer import Trainer, TrainConfig  # noqa: E402

ENC = ByteEncoder()


@pytest.fixture(scope="module")
def toy():
    return items_from_records(toy_records(32, seed=0), ENC, template=TOY_TEMPLATE)


def fresh(head="integrator", seed=0):
    torch.manual_seed(seed)
    return DecisionModel(tiny_backbone(seed=seed, hidden=16), head)


# ------------------------------------------------------------------------- LoRA
def test_lora_is_identity_at_init_and_only_lora_trains():
    bb = tiny_backbone(seed=0, hidden=16)
    x = torch.randint(4, 200, (2, 11))
    before = bb.features(x).detach()
    names = apply_lora(bb.model, bb.spec.lora_targets, rank=256)
    assert names and all(isinstance(bb.model.get_submodule(n), LoRALinear) for n in names)
    assert torch.allclose(bb.features(x), before, atol=1e-6)
    trainable = {n for n, p in bb.named_parameters() if p.requires_grad}
    assert trainable and all(n.endswith(("lora_A", "lora_B")) for n in trainable)
    assert bb.model.get_submodule(names[0]).rank == 256  # default rank is configurable, 256 by default


# ------------------------------------------------------------------------- losses
def test_loss_values_by_hand():
    lp = torch.log(torch.tensor([[0.7, 0.2, 0.1], [0.5, 0.5, 0.0]]).clamp_min(1e-30))
    lp[1, 2] = float("-inf")  # padded option
    y = torch.tensor([0, 1])
    assert torch.allclose(nll(lp, y), torch.tensor([-math.log(0.7), -math.log(0.5)]), atol=1e-6)
    assert torch.allclose(brier(lp, y), torch.tensor([0.09 + 0.04 + 0.01, 0.25 + 0.25]), atol=1e-6)
    # RPS (K=3, y=0): CDF p = [.7,.9], CDF y = [1,1] -> (.09+.01)/2 ; K=2, y=1: CDF p=[.5], y=[0] -> .25
    assert torch.allclose(rps(lp, y), torch.tensor([0.05, 0.25]), atol=1e-6)
    ls = label_smoothed_ce(lp, y, 0.1)
    expect0 = 0.9 * -math.log(0.7) + 0.1 * -(math.log(0.7) + math.log(0.2) + math.log(0.1)) / 3
    assert abs(float(ls[0]) - expect0) < 1e-5


def test_ce_arm_is_proper_and_clef_arm_is_not():
    """The expected CE is minimised at the true distribution; label smoothing moves the minimiser."""
    q = torch.tensor([0.8, 0.15, 0.05])

    def expected(cfg, p):
        lp = torch.log(p).unsqueeze(0).expand(3, 3)
        return sum(float(q[y]) * float(decision_loss(lp[y:y + 1], torch.tensor([y]), ["choice"], cfg)) for y in range(3))

    sharper = torch.tensor([0.9, 0.08, 0.02])
    flatter = torch.tensor([0.7, 0.2, 0.1])
    ce = LossConfig(arm="ce")
    assert expected(ce, q) < expected(ce, sharper) and expected(ce, q) < expected(ce, flatter)
    clef = LossConfig(arm="clef", label_smoothing=0.3, brier_weight=0.0, rps_weight=0.0)
    assert expected(clef, flatter) < expected(clef, q)


# ------------------------------------------------------------------------- training
@pytest.mark.parametrize("head,arm", [("integrator", "ce"), ("snapshot", "ce"), ("head_b", "clef")])
def test_loss_decreases_on_toy_task(toy, head, arm):
    m = fresh(head)
    cfg = TrainConfig(head=head, lora_rank=4, lr_head=1e-2, lr_lora=1e-3, batch_size=8, max_steps=60,
                      loss=LossConfig(arm=arm))
    h = Trainer(m, cfg, toy).train()
    first = sum(x["loss"] for x in h[:8]) / 8
    last = sum(x["loss"] for x in h[-8:]) / 8
    assert last < 0.85 * first, (first, last)


def test_resume_reproduces_uninterrupted_run(toy, tmp_path):
    def cfg(d):
        return TrainConfig(head="snapshot", lora_rank=4, batch_size=8, max_steps=12, ckpt_dir=str(d),
                           ckpt_every=5, lr_head=1e-2, lr_lora=1e-3)

    a = Trainer(fresh("snapshot"), cfg(tmp_path / "a"), toy)
    a.train()  # 12 steps straight; 64 rows / bs 8 -> crosses an epoch boundary at step 8
    b = Trainer(fresh("snapshot"), cfg(tmp_path / "b"), toy)
    b.train(5)  # checkpoint written at step 5, then the "process" dies
    assert Trainer.latest(tmp_path / "b").name == "ckpt-step0000005.pt"
    c = Trainer(fresh("snapshot"), cfg(tmp_path / "b"), toy)  # new process, fresh model
    assert c.resume() and c.step == 5
    c.train()
    sa, sc = a.trainable_state(), c.trainable_state()
    assert sa.keys() == sc.keys()
    for k in sa:
        assert torch.equal(sa[k], sc[k]), k
    assert [x["loss"] for x in a.history] == [x["loss"] for x in c.history]


def test_resume_rejects_a_different_config(toy, tmp_path):
    cfg = TrainConfig(head="integrator", lora_rank=4, batch_size=8, max_steps=1, ckpt_dir=str(tmp_path))
    t = Trainer(fresh(), cfg, toy)
    t.train()
    p = t.save()
    other = TrainConfig(head="integrator", lora_rank=4, batch_size=8, max_steps=1, ckpt_dir=str(tmp_path),
                        loss=LossConfig(arm="clef"))
    with pytest.raises(ValueError):
        Trainer(fresh(), other, toy).load(p)


def test_bf16_and_gradient_checkpointing_step(toy):
    m = fresh("head_b")
    cfg = TrainConfig(head="head_b", lora_rank=4, batch_size=4, max_steps=2, bf16=True, grad_checkpointing=True)
    t = Trainer(m, cfg, toy)
    assert m.backbone.model.gradient_checkpointing
    t.train()
    assert all(math.isfinite(x["loss"]) for x in t.history)
    assert all(p.grad is not None for p in lora_parameters(m.backbone))


# ------------------------------------------------------------------------- calibration
def test_feature_temperature_fits_length_dependent_miscalibration():
    g = torch.Generator().manual_seed(0)
    n, k = 4000, 4
    ev = torch.randint(10, 5000, (n,), generator=g)
    true_t = 0.5 + 0.4 * torch.log1p(ev.double())  # over-confident on long inputs
    base = torch.randn(n, k, generator=g, dtype=torch.float64) * 2
    y = torch.multinomial(torch.softmax(base, -1), 1, generator=g).squeeze(1)
    served = torch.log_softmax(base * true_t.unsqueeze(1), -1)  # raw model logits are too sharp
    phi = features(ev.tolist(), [k] * n)
    ft = FeatureTemperature.fit(served, y, phi, [str(i) for i in range(n)])
    assert ft.fit_report["nll_after"] < ft.fit_report["nll_before"]
    assert ft.w[1] > 0  # temperature grows with length
    assert torch.equal(ft.apply(served, phi).argmax(-1), served.argmax(-1))  # accuracy-preserving
    assert ft.identity()["status"] == "fitted" and len(ft.fitted_on) == 64
    ident = FeatureTemperature()
    assert torch.allclose(ident.apply(served, phi), served, atol=1e-6)


def test_eval_card_reports_metrics_and_stubs(toy, tmp_path):
    m = fresh()
    card = run_card(m, toy[:8], step=0, out_dir=tmp_path, temperature=FeatureTemperature())
    assert card["raw"]["n"] == 8 and 0 <= card["raw"]["accuracy"] <= 1
    assert "debiased_l2" in card["raw"]["ece_debiased"] and card["raw"]["by_length_bin"]
    assert card["index_score"]["status"] == "stub" and card["steer"]["status"] == "stub"
    assert (tmp_path / "card-step0000000.json").exists()


# ------------------------------------------------------------------------- estimator
def test_estimator_is_sane():
    from train.estimate import cell_cost, train_point

    q = BACKBONES["q9af"]
    short = train_point(q, "integrator", 2048, 4)
    long = train_point(q, "integrator", 8192, 255)
    assert long["mem_gb"]["total"] > short["mem_gb"]["total"] > q.params * 2 / 1024**3
    assert short["lora_params"] == 32 * 256 * sum(i + o for i, o in q.lora_shapes)
    assert train_point(q, "integrator", 2048, 4, rank=16)["lora_params"] * 16 == short["lora_params"]
    fork = train_point(q, "fork", 8192, 255)
    assert fork["mem_gb"]["fork"] > 10  # 255 copies of a ~71 MB fp32 recurrent state
    assert not train_point(q, "integrator", 8192, 16, grad_ckpt=False, B=8)["fits_80gb"]
    assert cell_cost(BACKBONES["mamba3-1.5b"], "head_b", 1000, 2048, 6)["usd_batch"] < cell_cost(q, "head_b", 1000, 2048, 6)["usd_batch"]
