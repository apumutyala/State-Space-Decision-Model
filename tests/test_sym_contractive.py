"""stack/rwkv7/sym-contractive (docs/D21_DESIGN.md): the two opt-in guarantee upgrades of the RWKV-7 line, CPU, tiny
random RWKV-7 (fp32).

(A) sym-fork layout + joint_record_sym (reorder + relabel invariance by construction): the evidence lane's harness
    (tests/guarantees/test_reorder_invariance.py, claude/ev-guarantee-proofs @f7170cb, copied unchanged) passes at
    tolerance 0 on the real head; D19's own layout (v3w / g1v2) FAILS relabel (teeth); the lexical prior reads the
    descriptions only; the DeepSets record mixer is question-order symmetric.
(B) per-head shared ``a`` (heads.rwkv7_scalar_a): every per-step transition's spectral norm (power iteration and exact
    SVD) is <= the T2(b) certificate = w_max < 1 for random inputs, also after the trainable bias moves; with the stock
    per-channel ``a`` the same model's real steps exceed 1, and the T2(e) two-step alternation grows (teeth).
OFF: everything off is bit-identical to the parent (tests/rsc_off_golden.py vs the parent tree's output, when the
cpu_tests job provides it in RSC_PARENT_GOLDEN)."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from heads import rwkv7_scalar_a as SA  # noqa: E402
from heads.inv_adapters import rwkv7_g1v2_v3w, rwkv7_sym_fork  # noqa: E402

HERE = Path(__file__).resolve().parent


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load(HERE / "guarantees" / "test_reorder_invariance.py", "ev_reorder_harness")


# ------------------------------------------------------------------------------------------------ (A) invariance
def test_A_harness_sym_fork_reorder_and_relabel_at_tolerance_zero():
    n = 0
    for s, o in H._cases():
        for r in H.assert_invariant(rwkv7_sym_fork, s, o, tol=0.0):
            assert r["max_abs_dp"] == 0.0 and r["flips"] == 0, r
            n += r["n_checked"]
    assert n > 100


def test_A_teeth_d19_layout_fails_relabel():
    rel = [H.check_relabel(rwkv7_g1v2_v3w, s, o, also_reorder=False) for s, o in H._cases()]
    assert max(r["max_abs_dp"] for r in rel) > 1e-4, rel
    with pytest.raises(AssertionError):
        for s, o in H._cases():
            H.assert_invariant(rwkv7_g1v2_v3w, s, o, tol=0.0)


def test_A_sym_rows_carry_no_id_and_are_identical_under_relabel_reorder():
    from heads.inv_adapters import _request
    from heads.layout import ByteEncoder, build_row

    rng = random.Random(3)
    for s, o in H._cases():
        o2 = [(f"zz{rng.randrange(10**6)}_{i}", c) for i, (_id, c) in enumerate(o)]
        rng.shuffle(o2)
        a = build_row(_request(s, o), "q", ByteEncoder(), template="jrt-rwkv-g1-v2", layout="sym-fork")
        b = build_row(_request(s, o2), "q", ByteEncoder(), template="jrt-rwkv-g1-v2", layout="sym-fork")
        assert a.ids == b.ids and a.continuations == b.continuations and a.option_seqs == b.option_seqs
        text = bytes(x - 4 for x in a.ids + sum(a.continuations, [])).decode("utf-8")
        assert not any(i in text for i, _c in o)


def test_A_lexical_prior_reads_descriptions_only():
    from heads.inv_adapters import _model, _request
    from heads.layout import ByteEncoder, build_row
    from heads.model import lexical_vectors
    from decide.prompt_g1v3 import G1V3W_ID

    bb = _model("joint_record_sym").backbone
    s, o = H._cases()[1]
    o2 = [(f"q{i}x", c) for i, (_id, c) in enumerate(o)]
    sym = [build_row(_request(s, x), "q", ByteEncoder(), template="jrt-rwkv-g1-v2", layout="sym-fork") for x in (o, o2)]
    la, lb = (lexical_vectors(bb, [r], r.n_options)[0] for r in sym)
    by_c = lambda r, lv: {r.option_seqs[k].__repr__(): lv[k] for k in range(r.n_options)}  # noqa: E731
    da, db = by_c(sym[0], la), by_c(sym[1], lb)
    assert da.keys() == db.keys() and all(torch.equal(da[k], db[k]) for k in da)
    old = [build_row(_request(s, x), "q", ByteEncoder(), template=G1V3W_ID, layout="g1v2") for x in (o, o2)]
    lo_a, lo_b = (lexical_vectors(bb, [r], r.n_options)[0] for r in old)
    assert float((lo_a - lo_b).abs().max()) > 1e-6  # teeth: the g1v2 schema span carries the id


def test_A_joint_record_sym_question_order_symmetric_and_init_is_fork_head():
    from heads.backbone import tiny_hf_backbone
    from heads.layout import ByteEncoder, build_row
    from heads.model import DecisionModel

    torch.manual_seed(0)
    bb = tiny_hf_backbone("rwkv7", seed=0, fork_mode="cache")
    m = DecisionModel(bb, "joint_record_sym", seed=0).float().eval()
    f = DecisionModel(bb, "fork", seed=0).float().eval()
    qs = {f"q{j}": {"type": "choice", "instructions": f"question {j}?",
                    "criteria": {f"o{j}{i}": f"answer {j} {i} text" for i in range(3 + j)}} for j in range(3)}
    req = {"model": "toy", "state": {"msg": "a record"}, "questions": qs}
    enc = ByteEncoder()
    rows = [build_row(req, q, enc, template="jrt-rwkv-g1-v2", layout="sym-fork") for q in qs]
    with torch.no_grad():
        assert torch.allclose(m(rows).log_probs, f(rows).log_probs, atol=0, rtol=0)  # w = 0: the fork head
        g = torch.Generator().manual_seed(5)
        m.head.w.copy_(torch.randn(m.head.w.shape, generator=g))
        a = m(rows).log_probs
        b = m(rows[::-1]).log_probs.flip(0)
    assert float((a - b).abs().max()) <= 1e-6  # questions permuted: same per-question distribution (float order only)
    assert not torch.equal(a, f(rows).log_probs)  # the mixer is live


# ------------------------------------------------------------------------------------------------ (B) contraction
class _Rec:
    def __init__(self, fn):
        self.fn, self.steps = fn, []

    def __call__(self, r, w_log, k, v, kk, a, state, **kw):
        self.steps.append((w_log.detach().clone(), kk.detach().clone(), a.detach().clone()))
        return self.fn(r, w_log, k, v, kk, a, state, **kw)


def _tiny(seed: int = 0, adversarial: bool = False):
    from heads.backbone import tiny_hf_backbone

    bb = tiny_hf_backbone("rwkv7", seed=seed, fork_mode="recompute")
    if adversarial:  # decays near 1 and a per-channel a alternating ~1 / ~0 inside each head (T2(d) regime)
        with torch.no_grad():
            for m in SA.attention_modules(bb.model):
                m.w0.fill_(-8.0)
                C = m.a0.shape[-1]
                m.a0.copy_(torch.tensor([6.0 if c % 2 == 0 else -6.0 for c in range(C)]).view(1, 1, C))
    return bb


def _steps(bb, monkeypatch, seed: int, T: int = 24, B: int = 3):
    from third_party.rwkv7_hf import modeling_rwkv7 as M

    name = bb.model.config.wkv_implementation
    rec = _Rec(M.RWKV7_WKV_FUNCTIONS[name])
    monkeypatch.setitem(M.RWKV7_WKV_FUNCTIONS, name, rec)
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(4, 260, (B, T), generator=g)
    with torch.no_grad():
        bb.model(input_ids=ids)
    monkeypatch.setitem(M.RWKV7_WKV_FUNCTIONS, name, rec.fn)
    return rec.steps


def _norms(steps):
    """Per (layer, batch, step, head): exact 2-norm, power-iteration norm, certificate, w_max."""
    out = []
    for w_log, kk, a in steps:
        A = SA.transition(w_log, kk, a)
        out.append((torch.linalg.matrix_norm(A, ord=2), SA.spectral_norm(A), SA.certificate(w_log, kk, a),
                    torch.exp(w_log.double()).max(-1).values))
    return out


@pytest.mark.parametrize("adversarial", [False, True])
def test_B_scalar_a_every_step_within_certificate_below_one(monkeypatch, adversarial):
    bb = _tiny(0, adversarial)
    assert SA.enable_scalar_a(bb.model) == 2 and SA.is_enabled(bb.model)
    n = 0
    for trial in range(4):
        if trial >= 2:  # the trainable bias moved (fine-tuning): the guarantee holds for any parameters
            with torch.no_grad():
                for p in SA.scalar_a_parameters(bb.model):
                    p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(trial)) * 3)
        steps = _steps(bb, monkeypatch, seed=100 + trial)
        for _w, _kk, a in steps:  # one scalar per head and step: sigma_a = 0 exactly
            assert torch.equal(a, a[..., :1].expand_as(a))
        for exact, pw, cert, wmax in _norms(steps):
            assert bool((pw <= exact + 1e-12).all())
            assert bool((exact <= cert + 1e-12).all()), float((exact - cert).max())
            assert bool((cert <= wmax + 1e-12).all())  # sigma_a = 0: U_t = w_max
            assert bool((exact < 1).all()), float(exact.max())
            n += exact.numel()
    assert n > 1000


def test_B_teeth_per_channel_a_exceeds_one_on_real_steps(monkeypatch):
    bb = _tiny(0, adversarial=True)  # SAME weights as the adversarial ON case above, flag OFF
    worst, viol = 0.0, 0
    for exact, pw, cert, wmax in _norms(_steps(bb, monkeypatch, seed=100)):
        assert bool((exact <= cert + 1e-9).all())  # the certificate still bounds it (T2(b))
        worst = max(worst, float(exact.max()))
        viol += int((exact > 1).sum())
    assert worst > 1.05 and viol > 0, worst


def test_B_teeth_T2e_two_step_alternation_grows_and_shared_a_contracts():
    w = torch.log(torch.tensor([0.99, 0.99], dtype=torch.float64))
    k1 = torch.tensor([0.4067, 0.9135], dtype=torch.float64)
    k2 = torch.tensor([0.9135, 0.4067], dtype=torch.float64)
    k1, k2 = k1 / k1.norm(), k2 / k2.norm()
    a1 = torch.tensor([0.99, 0.01], dtype=torch.float64)
    a2 = torch.tensor([0.01, 0.99], dtype=torch.float64)
    A1, A2 = SA.transition(w, k1, a1), SA.transition(w, k2, a2)
    P = A2 @ A1
    rate = float(torch.linalg.eigvals(P).abs().max()) ** 0.5
    assert rate > 1.1  # STATE_STABILITY.md T2(e): 1.1046
    assert float(torch.linalg.matrix_norm(torch.linalg.matrix_power(P, 100), ord=2)) > 1e8
    assert max(float(torch.linalg.eigvals(A).abs().max()) for A in (A1, A2)) <= 0.99 + 1e-12  # each factor rho < 1
    # the same steps with the head-shared rate (ScalarA at init = the per-head mean): contractive, product decays
    sa = SA.ScalarA(1)
    s1, s2 = (sa(x.float().view(1, 2)).double().view(2) for x in (a1, a2))
    assert torch.allclose(s1, torch.full((2,), 0.5, dtype=torch.float64), atol=1e-6)
    B1, B2 = SA.transition(w, k1, s1), SA.transition(w, k2, s2)
    for B, k, s in ((B1, k1, s1), (B2, k2, s2)):
        assert float(torch.linalg.matrix_norm(B, ord=2)) <= float(SA.certificate(w, k, s)) + 1e-12
        assert float(SA.certificate(w, k, s)) <= 0.99 + 1e-9
    Q = torch.linalg.matrix_power(B2 @ B1, 100)
    assert float(torch.linalg.matrix_norm(Q, ord=2)) <= 0.99 ** 200 * (1 + 1e-6)


def test_B_init_is_the_mean_of_the_per_channel_a_and_the_bias_trains(monkeypatch):
    bb = _tiny(1)
    off = _steps(bb, monkeypatch, seed=7)
    SA.enable_scalar_a(bb.model)
    on = _steps(bb, monkeypatch, seed=7)
    # layer 0 sees the same input under both: its shared a is the head mean of the per-channel a (b = 0)
    (w0, kk0, a_off), (w1, kk1, a_on) = off[0], on[0]
    assert torch.equal(w0, w1) and torch.equal(kk0, kk1)
    assert torch.allclose(a_on, SA.share_per_head(a_off.flatten(-2), a_off.shape[-2]).view_as(a_off), atol=1e-6)
    from train.lora import apply_lora, lora_parameters, lora_state_dict

    apply_lora(bb.model, bb.spec.lora_groups, rank=2, check_counts=False, generator=torch.Generator().manual_seed(0))
    sp = SA.scalar_a_parameters(bb.model)
    assert sp and all(p.requires_grad for p in sp)
    assert all(any(p is q for q in lora_parameters(bb.model)) for p in sp)
    assert sum(k.endswith(SA.SUFFIX) for k in lora_state_dict(bb.model)) == len(sp) == 2
    out = bb.model(input_ids=torch.randint(4, 260, (2, 16), generator=torch.Generator().manual_seed(3)))
    out.last_hidden_state.square().mean().backward()
    assert all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in sp)


def test_B_loader_flag_is_off_by_default():
    import inspect

    from heads.backbone import load_rwkv7_hf_backbone

    assert inspect.signature(load_rwkv7_hf_backbone).parameters["scalar_a"].default is False
    src = (HERE.parent / "train" / "run_cell.py").read_text(encoding="utf-8")
    assert 'scalar_a=bool(cfg.get("rwkv7_scalar_a", False))' in src


# ------------------------------------------------------------------------------------------------ OFF = parent
def test_off_path_bit_identical_to_parent():
    mine = _load(HERE / "rsc_off_golden.py", "rsc_off_golden").golden()
    mine = json.loads(json.dumps(mine, sort_keys=True))
    assert not any(k.endswith(SA.SUFFIX) for k in mine["lora_keys"] + mine["trainable"])
    p = os.environ.get("RSC_PARENT_GOLDEN")
    if not p:
        pytest.skip("RSC_PARENT_GOLDEN (the parent tree's tests/rsc_off_golden.py output) not provided")
    parent = json.loads(Path(p).read_text(encoding="utf-8"))
    assert set(parent) == set(mine)
    for k in parent:
        assert parent[k] == mine[k], k
    assert len(mine["lora_grad_sums"]) > 0 and math.isfinite(float.fromhex(mine["loss"]))
