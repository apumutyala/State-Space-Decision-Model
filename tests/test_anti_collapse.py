"""Anti-collapse training (research-notes/D10_DIVERGENCE.md) on tiny CPU models: cosine LR schedule, best-by-dev
checkpoint + scoring pointer, the collapse guard (reload best, halve LRs, max events), and old configs unchanged."""

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from heads import ByteEncoder, DecisionModel  # noqa: E402
from heads.backbone import tiny_backbone  # noqa: E402
from train.data import TOY_TEMPLATE, items_from_records, toy_records  # noqa: E402
from train.guard import SplitGuard  # noqa: E402
from train.trainer import Trainer, TrainConfig  # noqa: E402

ENC = ByteEncoder()
G = SplitGuard.synthetic_only("unit tests: train.data.toy_records only")
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def toy():
    return items_from_records(toy_records(32, seed=0), ENC, template=TOY_TEMPLATE)


def fresh(seed=0):
    torch.manual_seed(seed)
    return DecisionModel(tiny_backbone(seed=seed, hidden=16), "snapshot")


def base(**kw):
    d = dict(head="snapshot", lora_rank=4, batch_size=8, max_steps=12, lr_head=1e-2, lr_lora=1e-3)
    d.update(kw)
    return TrainConfig(**d)


def card(step, nll, acc):
    return {"step": step, "n_items": 8, "raw": {"nll": nll, "accuracy": acc}}


# ------------------------------------------------------------------ old configs: identical behaviour
def test_defaults_leave_config_identity_unchanged():
    assert TrainConfig().hash() == "46a4f6f3acbac492"  # pinned in tests/test_teacher.py before this change
    j = TrainConfig().to_json()
    assert not set(TrainConfig.ANTI_COLLAPSE) & set(j)
    # the schedule / guard change the trajectory -> in the hash; the dev-card / best keys only observe -> not
    assert TrainConfig(lr_schedule="cosine").hash() != TrainConfig().hash()
    assert TrainConfig(dev_card_every=200, best_ckpt=True).hash() == TrainConfig().hash()
    assert TrainConfig(dev_card_every=200, best_ckpt=True, collapse_guard=True).hash() != TrainConfig().hash()


def test_old_config_reproduces_the_original_constant_schedule_exactly(toy, tmp_path):
    cfg = base(warmup_steps=3, ckpt_dir=str(tmp_path / "a"), ckpt_every=6)
    a = Trainer(fresh(), cfg, toy, guard=G)
    b = Trainer(fresh(), base(warmup_steps=3, ckpt_dir=str(tmp_path / "b"), ckpt_every=6), toy, guard=G)
    warm = 3  # the pre-change scheduler, verbatim (train/trainer.py @ agent-main 2a2ffd5)
    b.sched = torch.optim.lr_scheduler.LambdaLR(b.opt, lambda s: min(1.0, (s + 1) / warm) if warm else 1.0)
    for s in range(40):
        assert a.lr_factor(s) == (min(1.0, (s + 1) / warm))
    a.train()
    b.train()
    assert [x["loss"] for x in a.history] == [x["loss"] for x in b.history]
    sa, sb = a.trainable_state(), b.trainable_state()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    st = torch.load(Trainer.latest(tmp_path / "a"), map_location="cpu", weights_only=False)
    assert "anti_collapse" not in st
    assert set(json.loads((tmp_path / "a" / "latest.json").read_text())) == {"path", "step", "config_hash"}
    assert Trainer.scored(tmp_path / "a") == Trainer.latest(tmp_path / "a")  # old runs score the last ckpt


# ------------------------------------------------------------------ cosine
def test_cosine_schedule_values(toy):
    tr = Trainer(fresh(), base(warmup_steps=4, max_steps=24, lr_schedule="cosine", lr_min_frac=0.1), toy, guard=G)
    f = tr.lr_factor
    assert [f(s) for s in range(4)] == [0.25, 0.5, 0.75, 1.0]  # warmup unchanged... s=3 is the first cosine step
    assert f(4) == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 0 / 20)))
    assert f(14) == pytest.approx(0.55)  # half-way: (1 + min) / 2
    assert f(24) == pytest.approx(0.1) and f(500) == pytest.approx(0.1)
    vals = [f(s) for s in range(4, 25)]
    assert all(x >= y for x, y in zip(vals, vals[1:]))
    # clock-projected horizon: min(max_steps, projected), fixed once
    tr.set_horizon(14)
    assert tr.horizon == 14 and f(14) == pytest.approx(0.1) and f(9) == pytest.approx(0.55)
    tr.set_horizon(10)
    assert tr.horizon == 14  # fixed: a later projection (e.g. on resume) does not move it
    t2 = Trainer(fresh(), base(max_steps=24, lr_schedule="cosine"), toy, guard=G)
    t2.set_horizon(10_000)
    assert t2.horizon == 24
    # the optimiser actually follows it
    t3 = Trainer(fresh(), base(warmup_steps=2, max_steps=10, lr_schedule="cosine"), toy, guard=G)
    seen = []
    for _ in range(10):
        seen.append(t3.opt.param_groups[0]["lr"])
        t3.train_step()
    assert seen == pytest.approx([1e-2 * t3.lr_factor(s) for s in range(10)])


def test_cosine_resume_is_exact(toy, tmp_path):
    def cfg(d):
        return base(warmup_steps=2, lr_schedule="cosine", ckpt_dir=str(d), ckpt_every=5)

    a = Trainer(fresh(), cfg(tmp_path / "a"), toy, guard=G)
    a.set_horizon(9)
    a.train()
    b = Trainer(fresh(), cfg(tmp_path / "b"), toy, guard=G)
    b.set_horizon(9)
    b.train(5)
    c = Trainer(fresh(), cfg(tmp_path / "b"), toy, guard=G)
    assert c.resume() and c.horizon == 9 and c.horizon_fixed
    c.set_horizon(100)  # the resumed attempt's own projection is ignored
    c.train()
    assert [x["loss"] for x in a.history] == [x["loss"] for x in c.history]


# ------------------------------------------------------------------ best checkpoint
def test_best_ckpt_selection_and_scoring_pointer(toy, tmp_path):
    d = tmp_path / "ck"
    tr = Trainer(fresh(), base(ckpt_dir=str(d), dev_card_every=2, best_ckpt=True), toy, eval_items=toy[:8], guard=G)
    tr.consider_card(card(0, 1.0, 0.5))
    tr.step = 2
    tr.consider_card(card(2, 0.8, 0.6))  # better NLL -> best
    tr.step = 4
    tr.consider_card(card(4, 0.9, 0.9))  # worse NLL, better accuracy -> not best
    tr.step = 6
    tr.consider_card(card(6, 0.8, 0.7))  # NLL tie -> higher accuracy wins
    tr.step = 8
    tr.consider_card(card(8, 0.8, 0.65))  # tie, lower accuracy -> not best
    assert tr.best == {"step": 6, "nll": 0.8, "accuracy": 0.7, "path": "ckpt-step0000006.pt"}
    assert [c["best"] for c in tr.dev_cards] == [True, True, False, True, False]
    tr.step = 10
    tr.save()  # a later ordinary checkpoint keeps the pointer
    rec = json.loads((d / "latest.json").read_text())
    assert rec["path"] == "ckpt-step0000010.pt" and rec["best"]["path"] == "ckpt-step0000006.pt"
    assert Trainer.latest(d).name == "ckpt-step0000010.pt"  # resume point unchanged
    assert Trainer.scored(d).name == "ckpt-step0000006.pt"  # what evals / screen load
    (d / "ckpt-step0000006.pt").unlink()  # a pointer to a missing file: None (identity exit), never the last ckpt
    assert Trainer.scored(d) is None


def test_dev_cards_run_during_training_and_best_is_saved(toy, tmp_path):
    d = tmp_path / "ck"
    tr = Trainer(fresh(), base(max_steps=8, ckpt_dir=str(d), dev_card_every=4, dev_card_rows=6, best_ckpt=True),
                 toy, eval_items=toy[:10], guard=G)
    tr.train()
    assert [c["step"] for c in tr.dev_cards] == [0, 4, 8] and all(c["n_items"] == 6 for c in tr.dev_cards)
    best = min(tr.dev_cards, key=lambda c: (c["nll"], -c["accuracy"]))
    assert tr.best["step"] == best["step"] and (d / tr.best["path"]).exists()
    assert Trainer.scored(d).name == tr.best["path"]
    st = torch.load(d / tr.best["path"], map_location="cpu", weights_only=False)
    assert st["step"] == best["step"]


# ------------------------------------------------------------------ collapse guard
def _collapse_head(model):
    """Force an input-blind collapse: zero every head parameter (constant logits = the uniform predictor)."""
    with torch.no_grad():
        for p in model.head_parameters():
            p.zero_()


def test_forced_collapse_reloads_best_and_halves_lr(toy, tmp_path):
    d = tmp_path / "ck"
    cfg = base(max_steps=400, ckpt_dir=str(d), dev_card_every=10, best_ckpt=True, collapse_guard=True,
               collapse_window=3, collapse_loss_frac=0.95, collapse_acc_frac=0.0, collapse_max_events=2)
    tr = Trainer(fresh(), cfg, toy, eval_items=toy[:8], guard=G)
    tr.run_dev_card()  # step-0 card = the first best (the rollback point)
    assert not tr.guard_armed
    for _ in range(80):  # healthy training: the loss window falls below 0.95 x uniform -> the guard arms
        tr.train_step()
        if tr.dev_card_due():
            tr.run_dev_card()
        if tr.guard_armed:
            break
    assert tr.guard_armed and tr.collapse_events == [] and tr.best["path"]
    tr.best = None  # make the current weights the best, so the reload target is known exactly
    tr.run_dev_card()
    best_step = tr.best["step"]
    best_w = {k: v.clone() for k, v in tr.trainable_state().items()}
    lr0 = [g["lr"] for g in tr.opt.param_groups]
    _collapse_head(tr.model)
    for _ in range(3):
        tr.train_step()
        if tr.collapse_events:
            break
    assert len(tr.collapse_events) == 1
    ev = tr.collapse_events[0]
    assert ev["trigger"] == "train_loss" and ev["action"] == "reload_best" and ev["best_step"] == best_step
    assert ev["mean_loss"] > 0.95 * ev["mean_uniform"] > 0
    now = tr.trainable_state()
    assert all(torch.equal(now[k], best_w[k]) for k in best_w)  # weights rolled back to the best ckpt
    assert best_step < ev["step"] == tr.step <= best_step + 3  # the step counter / data cursor keep going forward
    assert tr.lr_scale == 0.5 and [g["lr"] for g in tr.opt.param_groups] == pytest.approx([x / 2 for x in lr0])
    tr.train_step()
    assert [g["lr"] for g in tr.opt.param_groups] == pytest.approx([x / 2 for x in lr0])  # the halving persists
    for k in (2, 3):  # second collapse: reload at 0.25x; third detection (max_events 2): stop
        _collapse_head(tr.model)
        for _ in range(3):
            tr.train_step()
            if len(tr.collapse_events) == k:
                break
        assert len(tr.collapse_events) == k
    assert [e["action"] for e in tr.collapse_events] == ["reload_best", "reload_best", "stop"]
    assert tr.stopped == "collapse_max_events" and tr.lr_scale == 0.25
    n = tr.step
    tr.train()  # a stopped trainer does not train further
    assert tr.step == n
    tr.save()
    st = torch.load(Trainer.latest(d), map_location="cpu", weights_only=False)
    assert st["anti_collapse"]["collapse_events"] == tr.collapse_events
    assert Trainer.scored(d).name == f"ckpt-step{best_step:07d}.pt"  # scoring uses the best, not the collapsed last


def test_loss_guard_is_not_armed_at_the_uniform_start(toy, tmp_path):
    """A fresh head starts at the uniform loss: without arming, the guard would fire in the first window."""
    cfg = base(max_steps=6, lr_head=0.0, lr_lora=0.0, ckpt_dir=str(tmp_path), dev_card_every=100, best_ckpt=True,
               collapse_guard=True, collapse_window=3, collapse_loss_frac=0.0)
    tr = Trainer(fresh(), cfg, toy, eval_items=toy[:8], guard=G)
    tr.train()
    assert tr.collapse_events == [] and not tr.guard_armed


def test_loss_guard_quiet_on_healthy_training_and_uniform_matches_ln_k(toy, tmp_path):
    cfg = base(max_steps=20, ckpt_dir=str(tmp_path), dev_card_every=10, best_ckpt=True, collapse_guard=True,
               collapse_window=5, collapse_acc_frac=0.0, loss=__import__("train.losses", fromlist=["LossConfig"]).LossConfig(arm="ce"))
    tr = Trainer(fresh(), cfg, toy, eval_items=toy[:8], guard=G)
    tr.train_step()
    # CE arm: the uniform predictor's loss is -mean log(1/K) over the step's rows
    rows = [toy[i] for i in tr._order(0)[:8]]
    ks = [r.row.n_options for r in rows]
    assert tr._step_uniform == pytest.approx(sum(math.log(k) for k in ks) / len(ks), rel=1e-5)
    tr.train()
    assert tr.collapse_events == [] and tr.stopped is None


def test_dev_accuracy_drop_fires_the_guard(toy, tmp_path):
    d = tmp_path / "ck"
    cfg = base(ckpt_dir=str(d), dev_card_every=2, best_ckpt=True, collapse_guard=True)
    tr = Trainer(fresh(), cfg, toy, eval_items=toy[:8], guard=G)
    tr.consider_card(card(0, 1.0, 0.8))
    tr.step = 2
    tr.consider_card(card(2, 1.2, 0.5))  # 0.5 > 0.6 x 0.8 = 0.48: no event
    assert tr.collapse_events == []
    tr.step = 4
    tr.consider_card(card(4, 0.9, 0.4))  # 0.4 < 0.48: event (even though its NLL is lower), never the best
    assert [e["trigger"] for e in tr.collapse_events] == ["dev_accuracy"] and tr.best["step"] == 0
    assert tr.lr_scale == 0.5


def test_config_validation(toy):
    with pytest.raises(ValueError, match="lr_schedule"):
        Trainer(fresh(), base(lr_schedule="linear"), toy, guard=G)
    with pytest.raises(ValueError, match="best_ckpt needs"):
        Trainer(fresh(), base(best_ckpt=True), toy, guard=G)
    with pytest.raises(ValueError, match="collapse_guard needs"):
        Trainer(fresh(), base(dev_card_every=10, collapse_guard=True), toy, guard=G)


# ------------------------------------------------------------------ cell config plumbing
def test_run_cell_passes_the_keys_and_the_safe_template():
    from train.run_cell import train_config

    cell = json.loads((REPO / "jobs/m2_first_cell/m2_first_cell_rwkv7_d1_fla_safe.json").read_text(encoding="utf-8"))
    tc = train_config(cell, cell["max_steps"])
    assert (tc.lr_head, tc.lr_schedule, tc.dev_card_every, tc.best_ckpt, tc.collapse_guard) == (
        3e-4, "cosine", 200, True, True)
    old = json.loads((REPO / "jobs/m2_first_cell/m2_first_cell_rwkv7_d1_fast.json").read_text(encoding="utf-8"))
    to = train_config(old, old["max_steps"])
    assert (to.lr_schedule, to.dev_card_every, to.best_ckpt, to.collapse_guard) == ("constant", 0, False, False)
