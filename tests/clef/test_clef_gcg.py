"""Clef GCG-lite engine on CPU with the mocked backbone + vendored joint head (pipeline check only)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from baselines import clef_backend as CB  # noqa: E402
from baselines import clef_gcg as G  # noqa: E402
from clef_mocks import FakeTok, build_mock_model  # noqa: E402
from safety_eval import data  # noqa: E402
from safety_eval.decide_bridge import DecideBridge  # noqa: E402

HP = {"n_adv": 3, "steps": 2, "batch": 8, "topk": 8, "init": "!"}


@pytest.fixture(scope="module")
def setup():
    jsm = CB.load_jsm()
    model = build_mock_model(jsm, seed=1)
    be = CB.ClefBackend(model, FakeTok(), encoding="ssdm", jsm=jsm, device="cpu", dtype="float32")
    req = data.build(1, 1, split="dev")[0]
    return be, req


def test_problem_maps_keys_and_roundtrips(setup):
    be, req = setup
    prob = G.ClefProblem(be, req, "steer")
    eng = G.ClefGCGEngine(be)
    adv = torch.tensor([[ord("a"), ord("b"), ord("c")], [ord("x"), ord("y"), ord("z")]])
    assert eng.roundtrip_ok(prob, adv).all()
    # served encoding of the clean request == pre + (empty adv) + post minus the guard
    full = be.encoder.state_ids(prob.with_suffix(G.GUARD)["state"])
    assert full == prob.state_ids([])


def test_grad_path_matches_eval_path(setup):
    be, req = setup
    prob = G.ClefProblem(be, req, "hijack")
    eng = G.ClefGCGEngine(be)
    adv = torch.tensor([ord("q"), ord("r"), ord("s")])
    lpk_eval = eng.eval_advs(prob, adv.unsqueeze(0))[0]
    loss, grad = eng.loss_grad([prob], adv)
    from safety_eval.gcg import objective_loss

    assert abs(loss - float(objective_loss(lpk_eval, prob.target, prob.objective))) < 1e-4
    assert grad.shape == (3, eng.E.shape[0]) and torch.isfinite(grad).all()


def test_attack_runs_and_rechecks_through_decide(setup):
    be, req = setup
    prob = G.ClefProblem(be, req, "steer")
    eng = G.ClefGCGEngine(be)
    res = G.attack(eng, [prob], HP, seed=0, log=lambda s: None)
    fin = G.final_outcome(eng, prob, res["adv_ids"])
    assert fin["both_orders"] == fin["success"]  # one option order for Clef
    bridge = DecideBridge(CB.build_decider(be), model="clef-flash")
    from safety_eval.attacks import suffix_variant
    from safety_eval.data import strip_meta

    v = suffix_variant(req, res["suffix"], "gcg_steer", "gcg")
    out = bridge(strip_meta(v))
    assert out["status"] == "ok"
    raw = out["raw_probabilities"][data.QKEY]
    p = torch.tensor([raw[k] for k in prob.keys])
    assert torch.allclose(p, fin_p := torch.softmax(eng.eval_advs(prob, torch.tensor([res["adv_ids"]]))[0, 0], -1),
                          atol=1e-5), (p, fin_p)
