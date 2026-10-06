"""PR #14 Codex review: init seeding independent of the ambient RNG, resume lineage, and the
snapshot memory claim narrowed to the accumulator. CPU only, tiny random backbones."""

import pytest

torch = pytest.importorskip("torch")

from heads import ByteEncoder, DecisionModel  # noqa: E402
from heads.backbone import BACKBONES, tiny_backbone  # noqa: E402
from train.data import TOY_TEMPLATE, items_from_records, toy_records  # noqa: E402
from train.guard import SplitGuard  # noqa: E402
from train.trainer import Trainer, TrainConfig  # noqa: E402

ENC = ByteEncoder()
G = SplitGuard.synthetic_only("unit tests: train.data.toy_records only")


@pytest.fixture(scope="module")
def toy():
    return items_from_records(toy_records(16, seed=0), ENC, template=TOY_TEMPLATE)


def _cfg(**kw):
    base = dict(head="snapshot", lora_rank=4, batch_size=4, max_steps=3, lr_head=1e-2, lr_lora=1e-3)
    base.update(kw)
    return TrainConfig(**base)


def _params(module):
    return {n: p.detach().clone() for n, p in module.named_parameters()}


# ------------------------------------------------------------------------- seeding
@pytest.mark.parametrize("name", ["integrator", "snapshot", "head_b", "fork"])
def test_head_init_ignores_the_ambient_rng(name):
    bb = tiny_backbone(hidden=16)
    torch.manual_seed(123)
    a = _params(DecisionModel(bb, name, seed=7))
    torch.rand(1000)  # move the ambient RNG somewhere else entirely
    torch.manual_seed(999)
    b = _params(DecisionModel(bb, name, seed=7))
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    c = _params(DecisionModel(bb, name, seed=8))
    assert any(not torch.equal(a[k], c[k]) for k in a if k.startswith("head."))


def test_head_init_does_not_disturb_the_ambient_rng():
    bb = tiny_backbone(hidden=16)  # the test backbone seeds itself; build it first
    torch.manual_seed(5)
    expect = torch.rand(3)
    torch.manual_seed(5)
    DecisionModel(bb, "integrator", seed=1)
    assert torch.equal(torch.rand(3), expect)


def test_training_is_independent_of_the_ambient_rng(toy):
    runs = []
    for ambient in (0, 4242):
        torch.manual_seed(ambient)
        torch.rand(777)
        m = DecisionModel(tiny_backbone(hidden=16), "snapshot", seed=3)
        t = Trainer(m, _cfg(), toy, guard=G)
        lora0 = {n: p.detach().clone() for n, p in m.named_parameters() if n.endswith(("lora_A", "lora_B"))}
        t.train()
        runs.append((lora0, t.trainable_state(), [h["loss"] for h in t.history]))
    (l0, s0, h0), (l1, s1, h1) = runs
    assert all(torch.equal(l0[k], l1[k]) for k in l0)  # LoRA init from cfg.seed only
    assert h0 == h1 and all(torch.equal(s0[k], s1[k]) for k in s0)


# ------------------------------------------------------------------------- lineage
def _checkpoint(tmp_path, toy):
    t = Trainer(DecisionModel(tiny_backbone(hidden=16), "snapshot"), _cfg(ckpt_dir=str(tmp_path)), toy, guard=G)
    t.train()
    return t.save()


def test_resume_accepts_identical_lineage(tmp_path, toy):
    p = _checkpoint(tmp_path, toy)
    t = Trainer(DecisionModel(tiny_backbone(hidden=16), "snapshot"), _cfg(ckpt_dir=str(tmp_path)), toy, guard=G)
    t.load(p)
    assert t.step == 3
    st = torch.load(p, weights_only=False)
    assert set(st["lineage"]) == {"backbone", "tokenizer", "layout", "data"}
    assert "cuda" in st["rng"]  # None on CPU-only machines, restored when CUDA is present


def test_resume_rejects_a_different_frozen_backbone(tmp_path, toy):
    p = _checkpoint(tmp_path, toy)
    other = tiny_backbone(seed=1, hidden=16)  # same shapes and config, different weights
    t = Trainer(DecisionModel(other, "snapshot"), _cfg(ckpt_dir=str(tmp_path)), toy, guard=G)
    with pytest.raises(ValueError, match="backbone.weights_digest"):
        t.load(p)


def test_resume_rejects_different_training_data(tmp_path, toy):
    p = _checkpoint(tmp_path, toy)
    other = items_from_records(toy_records(16, seed=1), ENC, template=TOY_TEMPLATE)
    t = Trainer(DecisionModel(tiny_backbone(hidden=16), "snapshot"), _cfg(ckpt_dir=str(tmp_path)), other, guard=G)
    with pytest.raises(ValueError, match="data"):
        t.load(p)
    reordered = list(reversed(toy))  # same items, different order -> different batches
    t = Trainer(DecisionModel(tiny_backbone(hidden=16), "snapshot"), _cfg(ckpt_dir=str(tmp_path)), reordered, guard=G)
    with pytest.raises(ValueError, match="data"):
        t.load(p)


def test_resume_rejects_a_different_tokenizer_or_layout(tmp_path, toy):
    p = _checkpoint(tmp_path, toy)

    class OtherEncoder(ByteEncoder):
        def identity(self):
            return {"kind": "byte+4", "vocab_size": 260, "variant": "other"}

    other_tok = items_from_records(toy_records(16, seed=0), OtherEncoder(), template=TOY_TEMPLATE)
    t = Trainer(DecisionModel(tiny_backbone(hidden=16), "snapshot"), _cfg(ckpt_dir=str(tmp_path)), other_tok, guard=G)
    with pytest.raises(ValueError, match="tokenizer"):
        t.load(p)
    sym = items_from_records(toy_records(16, seed=0), ENC, template=TOY_TEMPLATE, layout="symmetric")
    t = Trainer(DecisionModel(tiny_backbone(hidden=16), "snapshot"), _cfg(ckpt_dir=str(tmp_path)), sym, guard=G)
    with pytest.raises(ValueError, match="layout"):
        t.load(p)


# ------------------------------------------------------------------------- snapshot memory claim
def test_snapshot_accumulator_is_bounded_but_peak_memory_is_not():
    from train.estimate import head_activation_bytes

    q = BACKBONES["q9af"]
    short = head_activation_bytes("snapshot", 1, 2048, 16, q.d_model)
    long = head_activation_bytes("snapshot", 1, 8192, 16, q.d_model)
    assert long > 3.9 * short  # peak grows ~linearly in T (h copy, one-hot, per-chunk tensors)
    assert long >= 8192 * q.d_model * 4 + 8192 * 8 * 4  # at least h[B,T,D] fp32 + one-hot [B,T,S]
    from heads.heads import SnapshotFork

    c = SnapshotFork(16).cost(8192, 16)
    assert c["accumulator_scalars"] == 8 * 16 and c["streaming_implemented"] is False
