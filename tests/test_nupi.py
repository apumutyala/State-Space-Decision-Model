"""nuPI constrained training (train/constraints.py, train/pi_control.py, docs/NUPI.md) on CPU toys:
(a) the toy problem with a known constrained optimum: nuPI reaches feasibility with less oscillation than gradient
ascent on the multiplier; (b) lambda round-trips through checkpoint save / resume bit-exactly; (c) constraints OFF gives
bit-identical losses to the base branch (stack/qwen35/jev1-4b @ 5e159437) for a fixed seed."""

import json
import math

import pytest

from train.nupi_toy import LAM_STAR, compare
from train.pi_control import PIController, gradient_ascent


# ------------------------------------------------------------------ (a) the controller on the toy problem
@pytest.mark.parametrize("ki,kp", [(0.05, 1.0), (0.2, 4.0)])
def test_toy_nupi_reaches_feasibility_with_less_oscillation(ki, kp):
    r = compare(ki=ki, kp=kp)
    ga, pi = r["ga"]["metrics"], r["nupi"]["metrics"]
    # both converge to the known optimum x* = (0.5, 0.5), lambda* = 1.5 ...
    assert pi["final_abs_e"] < 1e-6 and pi["lam_err"] < 1e-6 and pi["x_err"] < 1e-6
    assert ga["final_abs_e"] < 1e-3
    # ... but nuPI with less oscillation: smaller overshoot, less multiplier travel, fewer boundary crossings, sooner
    assert pi["lam_overshoot"] < 0.5 * ga["lam_overshoot"]
    assert pi["lam_total_variation"] < 0.7 * ga["lam_total_variation"]
    assert pi["e_crossings"] < ga["e_crossings"]
    assert pi["settle_step"] < ga["settle_step"]
    assert pi["max_violation_2nd_half"] <= ga["max_violation_2nd_half"]


def test_kp_zero_is_exactly_projected_gradient_ascent():
    a, b, lam = gradient_ascent(0.3), PIController(kappa_p=0.0, kappa_i=0.3), 0.0
    for e in [0.5, -2.0, -1.0, 0.7, 0.2, -0.1, 3.0]:
        lam = max(0.0, lam + 0.3 * e)
        assert a.update(e) == lam and b.update(e) == lam


def test_anti_windup_and_clamps():
    c = PIController(kappa_p=0.0, kappa_i=1.0, integral_max=2.0, lambda_max=1.5)
    for _ in range(100):
        c.update(1.0)  # a long infeasible stretch
    assert c.integral == 2.0 and c.lam == 1.5
    c.update(-0.6)
    assert c.integral == pytest.approx(1.4) and c.lam == pytest.approx(1.4)  # recovers at once: no windup tail
    c.update(float("nan"))
    assert c.lam == pytest.approx(1.4)  # a non-finite measurement never moves lambda
    with pytest.raises(ValueError):
        PIController(kappa_p=-1.0, kappa_i=0.1)


def test_nu_ema_on_the_proportional_term():
    c = PIController(kappa_p=1.0, kappa_i=0.0, nu=0.5, integral_min=-10)
    c.update(1.0)
    assert c.xi == 0.5 and c.lam == 0.5
    c.update(1.0)
    assert c.xi == 0.75


def test_toy_traces_receipt_is_reproducible():
    from pathlib import Path

    rec = json.loads((Path(__file__).resolve().parents[1] / "docs" / "nupi_toy_traces.json").read_text())
    r = compare(ki=0.2, kp=4.0)
    assert rec["deterministic ki=0.2"]["nupi"] == r["nupi"]["metrics"]
    assert rec["deterministic ki=0.2"]["setup"]["lambda_star"] == LAM_STAR


# ------------------------------------------------------------------ trainer integration (tiny CPU model)
torch = pytest.importorskip("torch")

from heads import ByteEncoder, DecisionModel  # noqa: E402
from heads.backbone import tiny_backbone  # noqa: E402
from train.data import TOY_TEMPLATE, items_from_records, toy_records  # noqa: E402
from train.guard import SplitGuard  # noqa: E402
from train.mqar_probe import mqar_records  # noqa: E402
from train.trainer import Trainer, TrainConfig  # noqa: E402

ENC = ByteEncoder()
G = SplitGuard.synthetic_only("unit tests: train.data.toy_records / train.mqar_probe only")

SPEC = {"kappa_p": 1.0, "kappa_i": 0.1, "nu": 0.5, "integral_max": 50.0, "monitor_rows": 12, "monitor_every": 3,
        "items": [{"name": "brier", "kind": "calibration", "metric": "brier", "setpoint": 0.2},
                  {"name": "ece", "kind": "calibration", "metric": "ece", "setpoint": 0.02, "source": "monitor"},
                  {"name": "fam", "kind": "family_regression", "delta": 0.0, "reference": "init",
                   "source": "monitor"}]}


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


#: (c) the base branch's losses for ``base()`` on ``toy`` (12 steps, seed 0), float.hex, recorded by running the
#: same script on stack/qwen35/jev1-4b @ 5e159437 (train/trainer.py before this change); see docs/NUPI.md
#: recorded on SF cpu-8 by jobs/cpu_tests/golden.py (claude/cpu-tests-job) on a git archive of base 5e159437: job-jxawy,
#: art-jcm9f run.log (reproduced by job-urjpw, art-hxxxt)
BASE_LOSSES_HEX = ["0x1.3bfa040000000p+0", "0x1.3da7e20000000p+0", "0x1.46ad980000000p+0", "0x1.3cdd700000000p+0",
                   "0x1.34847c0000000p+0", "0x1.592cf00000000p+0", "0x1.371b840000000p+0", "0x1.38e4a20000000p+0",
                   "0x1.2dcf580000000p+0", "0x1.29ac060000000p+0", "0x1.473bca0000000p+0", "0x1.54ebe60000000p+0"]


@pytest.mark.parametrize("empty", [{}, {"constraints": {}}, {"constraints": {}, "consistency": {}}])
def test_off_mode_is_bit_identical_to_the_base_branch(toy, tmp_path, empty):
    cfg = base(ckpt_dir=str(tmp_path / "a"), ckpt_every=6, **empty)
    assert TrainConfig(constraints={}).hash() == TrainConfig().hash() == "46a4f6f3acbac492"
    assert "constraints" not in cfg.to_json()
    tr = Trainer(fresh(), cfg, toy, guard=G)
    assert tr.constraints is None
    tr.train()
    if BASE_LOSSES_HEX is None:
        pytest.skip("base-branch golden not recorded yet (laptop RAM < 4 GB): run the golden script on 5e159437")
    assert [float.hex(x["loss"]) for x in tr.history] == BASE_LOSSES_HEX
    assert all(set(x) == {"step", "loss"} for x in tr.history)
    st = torch.load(Trainer.latest(tmp_path / "a"), map_location="cpu", weights_only=False)
    assert "constraints" not in st


def test_constraints_on_changes_the_hash_and_logs_lambda_every_step(toy):
    cfg = base(constraints=SPEC)
    assert cfg.hash() != base().hash() and cfg.to_json()["constraints"] == SPEC
    tr = Trainer(fresh(), cfg, toy, guard=G)
    tr.train()
    assert len(tr.history) == 12
    for h in tr.history:
        c = h["constraints"]
        assert {"brier", "ece"} <= set(c) and all("lam" in v for v in c.values())
        assert "g" in c["brier"]  # batch source: measured every step
        assert math.isfinite(h["loss"]) and math.isfinite(h["lagrangian"])
    fams = [k for k in tr.history[-1]["constraints"] if k.startswith("fam/")]
    assert fams == ["fam/"]  # toy ids carry no m2 source: one family ""
    # the monitor items come from the TRAIN split only
    ids = {i.id for i in toy}
    assert tr.constraints.monitor_items and all(i.id in ids for i in tr.constraints.monitor_items)
    # a brier setpoint below the toy's reach: the multiplier rises
    assert tr.history[-1]["constraints"]["brier"]["lam"] > 0


def test_lambda_round_trips_through_checkpoint_resume_bit_exactly(toy, tmp_path):
    def cfg(d):
        return base(constraints=SPEC, ckpt_dir=str(d), ckpt_every=6)

    a = Trainer(fresh(), cfg(tmp_path / "a"), toy, guard=G)
    a.train()
    b = Trainer(fresh(), cfg(tmp_path / "b"), toy, guard=G)
    b.train(6)
    saved = b.constraints.state_dict()
    st = torch.load(Trainer.latest(tmp_path / "b"), map_location="cpu", weights_only=False)
    assert st["constraints"] == saved  # stored exactly (Python doubles)
    c = Trainer(fresh(), cfg(tmp_path / "b"), toy, guard=G)
    assert c.resume() and c.step == 6
    assert c.constraints.state_dict() == saved  # bit-exact: == on floats
    c.train()
    assert c.history == a.history  # losses, Lagrangian, every lambda / g / e entry
    assert c.constraints.state_dict() == a.constraints.state_dict()
    sa, sc = a.trainable_state(), c.trainable_state()
    assert all(torch.equal(sa[k], sc[k]) for k in sa)


def test_resume_refuses_a_checkpoint_without_controller_state(toy, tmp_path):
    a = Trainer(fresh(), base(ckpt_dir=str(tmp_path), ckpt_every=6), toy, guard=G)
    a.train(6)
    c = Trainer(fresh(), base(constraints=SPEC, ckpt_dir=str(tmp_path), ckpt_every=6), toy, guard=G)
    with pytest.raises(ValueError):
        c.resume()  # the config hash differs first


def test_recall_constraint_with_an_mqar_probe(toy):
    probe = items_from_records(mqar_records(6, n_pairs=4, n_queries=2, n_values=4, seed=1), ENC,
                               template=TOY_TEMPLATE)
    spec = {"kappa_p": 0.5, "kappa_i": 0.1,
            "items": [{"name": "recall", "kind": "recall", "delta": 0.0, "probe_batch": 4}]}
    tr = Trainer(fresh(), base(constraints=spec, max_steps=4), toy, guard=G, probe_items=probe)
    tr.train()
    assert tr.constraints.bounds["recall"] > 0  # (1 + 0) x the step-0 probe loss
    assert all("g" in h["constraints"]["recall"] for h in tr.history)
    assert tr.constraints.probe_cursor == (4 * 4) % len(probe)
    with pytest.raises(ValueError):
        Trainer(fresh(), base(constraints=spec), toy, guard=G)  # no probe items


def test_config_validation(toy):
    for bad in [{"items": []},
                {"items": [{"kind": "nope", "setpoint": 1}]},
                {"items": [{"kind": "calibration"}]},  # no setpoint
                {"items": [{"kind": "calibration", "setpoint": 0.1, "metric": "nll"}]},
                {"items": [{"kind": "calibration", "setpoint": 0.1, "typo": 1}]},
                {"items": [{"kind": "consistency", "setpoint": 0.1, "source": "monitor"}]},
                {"bogus": 1, "items": [{"kind": "calibration", "setpoint": 0.1}]}]:
        with pytest.raises(ValueError):
            Trainer(fresh(), base(constraints=bad), toy, guard=G)
    with pytest.raises(ValueError):  # the consistency constraint needs order views on the train items
        Trainer(fresh(), base(constraints={"items": [{"kind": "consistency", "setpoint": 0.01}]}), toy, guard=G)


def test_run_cell_passes_the_constraints_block():
    from train.run_cell import train_config

    cell = {"head": "snapshot", "layout": "ordered", "lora_rank": 4, "loss_arm": "clef", "lr_head": 1e-3,
            "lr_lora": 1e-4, "batch_size": 8, "grad_accum": 1, "seed": 0, "ckpt_every": 0, "eval_every": 0}
    assert train_config(cell, 10).constraints == {}
    assert train_config({**cell, "constraints": SPEC}, 10).constraints == SPEC
    from train.run_cell import consistency_fields

    both = {**cell, "constraints": SPEC, "consistency": {"lambda": 0.5}}
    cfg = train_config(both, 10)
    assert cfg.constraints == SPEC and cfg.consistency == consistency_fields(both)
    assert cfg.to_json()["constraints"] == SPEC and cfg.to_json()["consistency"] == cfg.consistency
    assert len({train_config(c, 10).hash() for c in
                (cell, {**cell, "constraints": SPEC}, {**cell, "consistency": {"lambda": 0.5}}, both)}) == 4
    for empty in ({}, {"constraints": {}}, {"consistency": {}}, {"constraints": {}, "consistency": {}}):
        off = train_config({**cell, **empty}, 10)
        assert off.to_json() == train_config(cell, 10).to_json()
        assert off.hash() == train_config(cell, 10).hash()
        assert not {"constraints", "consistency"} & off.to_json().keys()
