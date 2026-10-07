"""Tracing, transition algebra, decay bounds, persistence curves and Gramians (tiny RWKV-7 / Q9-AF GDN, CPU)."""

import pytest

torch = pytest.importorskip("torch")

from controlsafe import adapters as A  # noqa: E402
from controlsafe import gramian as G  # noqa: E402
from controlsafe import persistence as P  # noqa: E402
from heads.backbone import tiny_hf_backbone  # noqa: E402

FAMS = ["rwkv7", "q9af"]


def _model(fam, double=False):
    m = tiny_hf_backbone(fam, seed=4).model
    return m.double() if double else m


def _ids(n=24, seed=0, batch=None):
    g = torch.Generator().manual_seed(seed)
    shape = (n,) if batch is None else (batch, n)
    return torch.randint(4, 260, shape, generator=g)


@pytest.mark.parametrize("fam", FAMS)
def test_trace_matches_stock_forward_and_state_recursion(fam):
    m = _model(fam)
    x = _ids(batch=2)
    with torch.no_grad():
        ref = A.hidden_states(m, x)[-1]
        with A.trace(m) as rec:
            got = A.hidden_states(m, x)[-1]
    assert torch.allclose(ref, got, atol=1e-5)
    assert sorted(rec.layers) == A.recurrent_layer_ids(m)
    for tr in rec.layers.values():
        prev = torch.cat([tr.init_state.unsqueeze(1), tr.states[:, :-1]], 1)
        lin = tr.transition() @ prev
        # RWKV-7: write k v^T ; GDN: X_t - A_t X_{t-1} = beta k v^T (write_k already carries beta)
        write = tr.write_k.unsqueeze(-1) * tr.write_v.unsqueeze(-2)
        assert torch.allclose(lin + write, tr.states, atol=1e-5)
    if fam == "rwkv7":  # trace context restored the stock kernel
        assert m.config.wkv_implementation == "chunked"


def test_gdn_universal_bound_is_certified_and_dominates_observed_norms():
    m = _model("q9af")
    ub = A.universal_decay_bound(m)
    assert ub and all(v["certified"] and v["rho"] < 1 for v in ub.values())
    with torch.no_grad():  # random tokens plus huge-norm embeddings (the bound is over ALL inputs)
        e = A.embed(m, _ids(batch=3)) * torch.tensor([1.0, 30.0, 1000.0])[:, None, None]
        with A.trace(m) as rec:
            A.hidden_states(m, embeds=e)
    for l, tr in rec.layers.items():
        n2 = torch.linalg.matrix_norm(tr.transition().double(), ord=2)
        assert float(n2.max()) <= ub[l]["rho"] * (1 + 1e-6) + 1e-12
        per_head = torch.tensor(ub[l]["per_head"], dtype=torch.float64)
        assert bool((n2.amax(dim=(0, 1)) <= per_head * (1 + 1e-6) + 1e-12).all())


def test_rwkv7_full_transition_has_no_universal_2norm_bound():
    """Counterexample on record: diag(w) - (kk*a) kk^T with unit kk, a in (0,1), w < 1 can have 2-norm > 1, so
    the RWKV-7 universal bound covers the diagonal decay part only (reported rho = None)."""
    kk = torch.tensor([1.0, 1.0], dtype=torch.float64) / 2 ** 0.5
    a = torch.tensor([0.999, 0.001], dtype=torch.float64)
    w = torch.tensor([0.99, 0.99], dtype=torch.float64)
    Am = torch.diag(w) - torch.outer(kk * a, kk)
    assert float(torch.linalg.matrix_norm(Am, ord=2)) > 1.0
    ub = A.universal_decay_bound(_model("rwkv7"))
    assert all(v["rho"] is None and 0 < v["decay_rho"] < 1 for v in ub.values())


@pytest.mark.parametrize("fam", FAMS)
def test_common_norm_bound_gives_product_contraction(fam):
    m = _model(fam)
    with torch.no_grad(), A.trace(m) as rec:
        A.hidden_states(m, _ids(n=20, seed=1)[None])
    for tr in rec.layers.values():
        At = tr.transition()[0].double()  # [T,H,N,N]
        step = torch.linalg.matrix_norm(At, ord=2)  # [T,H]
        Phi = torch.eye(At.shape[-1], dtype=At.dtype).expand(At.shape[1], -1, -1)
        for s in range(At.shape[0]):
            Phi = At[s] @ Phi
            lhs = torch.linalg.matrix_norm(Phi, ord=2)
            prod = step[: s + 1].prod(0)
            rho = step[: s + 1].amax(0)
            assert bool((lhs <= prod * (1 + 1e-9) + 1e-300).all())
            assert bool((prod <= rho ** (s + 1) * (1 + 1e-9) + 1e-300).all())


@pytest.mark.parametrize("fam", FAMS)
def test_layer_local_curve_respects_per_input_certificate(fam):
    m = _model(fam, double=True)
    ids = _ids(n=26, seed=2).tolist()
    for l in A.recurrent_layer_ids(m):
        r = P.layer_local_curve(m, ids, l, t=4, horizon=18, eps=1e-2, n_dirs=3)
        assert r["bound_holds"], (l, r["state_influence_max"], r["bound_product"])
        assert r["window"] == (1 if fam == "rwkv7" else 3)
        assert all(v >= 0 for v in r["state_influence_mean"])


@pytest.mark.parametrize("fam", FAMS)
def test_observed_norms_and_end_to_end_curve(fam):
    m = _model(fam, double=True)
    ids = _ids(n=20, seed=3).tolist()
    obs = P.observed_transition_norms(m, torch.tensor([ids]))
    for v in obs.values():
        assert v["rho_obs"] == min(v["2"]["sup"], v["1"]["sup"], v["inf"]["sup"])
        assert v["status"].startswith("EMPIRICAL")
    e = P.end_to_end_curve(m, ids, t=3, horizon=10, n_dirs=2, swap_tokens=[17])
    final = e["hidden_influence"][max(e["hidden_influence"])]
    assert len(final) == 11 and final[0] > 0 and all(x == x for x in final)
    assert e["token_swaps"][0]["final_feature_delta"][0] > 0
    rep = P.persistence_report(m, ids, t=3, horizon=8, n_dirs=1)
    assert set(rep) == {"universal", "observed", "local", "end_to_end"}


def test_readout_projector_is_an_orthogonal_projector():
    r = torch.randn(3, 4, dtype=torch.float64)
    Pm = G.readout_projector(r, nv=5)
    assert torch.allclose(Pm @ Pm, Pm) and torch.allclose(Pm, Pm.T)
    assert abs(float(torch.trace(Pm)) - 3 * 5) < 1e-9


def _perturbed_state(m, l, ids, U, dd, mode):
    x = torch.tensor([ids])
    hook = None
    if mode == "end_to_end":
        e = A.embed(m, x).detach().clone()
        e[0, U] = e[0, U] + dd
        kw = {"embeds": e}
    else:
        mix = A.mixer(A.blocks(m)[l])

        def pre(mod, args, kwargs):
            if args:
                z = args[0].clone()
                z[0, U] = z[0, U] + dd
                return (z,) + tuple(args[1:]), kwargs
            z = kwargs["hidden_states"].clone()
            z[0, U] = z[0, U] + dd
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = z
            return args, kwargs
        hook = mix.register_forward_pre_hook(pre, with_kwargs=True)
        kw = {"ids": x}
    try:
        with torch.no_grad(), A.trace(m) as rec:
            A.hidden_states(m, **kw)
    finally:
        if hook is not None:
            hook.remove()
    return rec.layers[l].states[0, -1].reshape(-1)


@pytest.mark.parametrize("fam", FAMS)
@pytest.mark.parametrize("mode", ["end_to_end", "local"])
def test_gramian_matches_finite_differences(fam, mode):
    m = _model(fam, double=True)
    ids = _ids(n=16, seed=5).tolist()
    unt = [False] * 4 + [True] * 6 + [False] * 6
    l = A.recurrent_layer_ids(m)[-1]
    res = G.gramians(m, ids, unt, mode=mode, layers=[l], return_jacobian=True)[l]
    J = res["J"]
    assert res["trace"] == pytest.approx(float((J * J).sum()))
    assert 0.0 <= res["readout_overlap"] <= 1.0
    U = [t for t, u in enumerate(unt) if u]
    g = torch.Generator().manual_seed(0)
    delta = torch.randn(len(U), m.config.hidden_size, generator=g, dtype=torch.float64) * 1e-4  # GDN RMSNorm runs in fp32 internally
    fd = _perturbed_state(m, l, ids, U, delta, mode) - _perturbed_state(m, l, ids, U, torch.zeros_like(delta), mode)
    lin = J @ delta.reshape(-1)
    scale = float(lin.norm())
    assert scale < 1e-14 or float((fd - lin).norm()) / scale < 2e-2


def test_gramian_is_zero_without_untrusted_positions():
    m = _model("rwkv7")
    res = G.gramians(m, _ids(n=10).tolist(), [False] * 10)
    assert all(v["trace"] == 0.0 for v in res.values())
