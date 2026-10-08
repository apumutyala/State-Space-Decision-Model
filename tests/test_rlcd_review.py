"""Review B adversarial CPU checks of D14 rlcd_loss (2026-10-08): properness, adjacent-only credit, KL."""
import pytest

torch = pytest.importorskip("torch")

from train.losses import LossConfig, option_rewards, rlcd_loss, kl_to_ref


def _fit(q, qtype, beta, brier, ref=None, steps=4000):
    """Minimise E_{y~q}[rlcd_loss] over free logits; return the optimum p."""
    K = q.numel()
    z = torch.zeros(K, requires_grad=True)
    ref_lp = (ref if ref is not None else q).log().unsqueeze(0).expand(K, K)
    cfg = LossConfig(arm="rlcd", rlcd_beta=beta, rlcd_brier=brier)
    opt = torch.optim.Adam([z], lr=0.05)
    ys = torch.arange(K)
    for _ in range(steps):
        lp = torch.log_softmax(z, -1).unsqueeze(0).expand(K, K)
        # expected loss over gold ~ q: per-row losses weighted by q
        per = torch.stack([rlcd_loss(lp[i:i + 1], ys[i:i + 1], [qtype], cfg, ref_lp[i:i + 1]) for i in range(K)])
        loss = (per * q).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.softmax(z, -1).detach()


def test_calibration_d14_weights_choice():
    q = torch.tensor([0.7, 0.3])
    p = _fit(q, "choice", beta=0.1, brier=0.5)  # D14 weights, ref = calibrated q
    print("choice q", q.tolist(), "-> p", [round(x, 3) for x in p.tolist()])
    assert abs(p[0] - 0.7) < 0.02, f"D14 objective is improper: calibrated p_ref {q.tolist()} drifts to {p.tolist()}"


def test_calibration_no_kl():
    q = torch.tensor([0.6, 0.4])
    p = _fit(q, "choice", beta=0.0, brier=0.5)
    print("beta0 q", q.tolist(), "-> p", [round(x, 3) for x in p.tolist()])
    assert abs(p[0] - 0.6) < 0.02, f"no-KL optimum {p.tolist()} != q {q.tolist()}"


def test_score_partial_credit_mode():
    q = torch.tensor([0.4, 0.0, 0.0, 0.0, 0.6]) + 1e-6
    q = q / q.sum()
    p = _fit(q, "score", beta=0.1, brier=0.5)
    print("score q", [round(x, 3) for x in q.tolist()], "-> p", [round(x, 3) for x in p.tolist()])
    assert (p - q).abs().max() < 0.05


def test_partial_credit_not_adjacent_only():
    lp = torch.log_softmax(torch.zeros(1, 5), -1)
    r = option_rewards(lp, torch.tensor([0]), ["score"])
    print("rewards K=5 gold 0:", r.tolist())
    assert r[0, 2] == 0 and r[0, 3] == 0, "non-adjacent options get partial credit"


def test_kl_direction_and_zero():
    a = torch.log_softmax(torch.tensor([[2.0, 0.0, -1.0]]), -1)
    assert float(kl_to_ref(a, a)) == 0.0
    b = torch.log_softmax(torch.tensor([[0.0, 0.0, 0.0]]), -1)
    want = (a.exp() * (a - b)).sum()
    assert torch.allclose(kl_to_ref(a, b), want.unsqueeze(0))


def test_compiled_param_data_swap():
    m = torch.nn.Linear(4, 3)
    x = torch.randn(2, 4)
    cm = torch.compile(m, backend="eager")
    y0 = cm(x).detach()
    live = m.weight.data
    m.weight.data = torch.zeros_like(live)
    y1 = cm(x).detach()
    m.weight.data = live
    y2 = cm(x).detach()
    assert not torch.allclose(y0, y1) and torch.allclose(y0, y2)
