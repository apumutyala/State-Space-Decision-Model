"""Injection persistence curves and per-layer decay bounds.

Two experiments, labelled differently on purpose:

1. ``layer_local_curve`` (per layer; a CERTIFIED per-input bound is checked against it). Perturb ONLY layer
   l's mixer input at position t (a forward pre-hook adds ``eps * u``); measure ``||dX_l(t+tau)||_F`` (the
   layer's recurrent state) and ``||dy||`` (that layer's block output) for tau = 0..H. For s > t + w (w =
   ``adapters.local_window``: 1 for the RWKV-7 token shift, conv_kernel - 1 for GDN) the layer's inputs,
   hence ``A_s`` and the write, are unchanged, so ``dX_s = A_s dX_{s-1}`` exactly and

        ||dX_{t+tau}|| <= prod_{s=t+w+1}^{t+tau} ||A_s||_2 * ||dX_{t+w}|| <= rho_obs^(tau-w) ||dX_{t+w}||

   with ``rho_obs = max_s ||A_s||_2`` over THIS sequence: a theorem applied to computed numbers (a per-input
   certificate, up to float rounding). ``rho_univ`` (``adapters.universal_decay_bound``) is the all-input
   version; it exists for GDN and, for RWKV-7, only for the diagonal decay part.

2. ``end_to_end_curve`` (EMPIRICAL). Perturb the token embedding at t (random directions or a token swap) and
   measure every layer's hidden state, recurrent state and the final feature at t + tau. Includes the
   cross-layer cascade (polynomial x exponential, reviews/02 section 4); no bound is claimed.

``observed_transition_norms`` reports sup over the observed inputs of ``||A_t||`` in fixed norms (2, 1, inf on
the key axis). A sup over observed inputs is EMPIRICAL as a statement about unseen inputs; the per-input
product bound above is the only certified use of it. Per SAFETY_SPEC section 5.2 none of this is a safety
claim: an injection acts at tau ~ 0, and decay bounds how long it stays, not whether it gets in.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch

from . import adapters


def _norms(A: torch.Tensor) -> dict[str, torch.Tensor]:
    """Induced norms of ``A [..., N, N]`` on the key axis."""
    return {"2": torch.linalg.matrix_norm(A, ord=2), "1": A.abs().sum(-2).amax(-1), "inf": A.abs().sum(-1).amax(-1)}


@torch.no_grad()
def observed_transition_norms(model, ids_batch: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict:
    """Per layer: sup and quantiles of ``||A_t||`` over all observed (row, position, head), per fixed norm,
    plus the fraction of steps with ``||A_t||_2 > 1``. Status: EMPIRICAL (observed inputs only)."""
    with adapters.trace(model) as rec:
        adapters.hidden_states(model, ids_batch, attention_mask=attention_mask)
    out = {}
    valid = None if attention_mask is None else attention_mask.bool()
    for l, tr in sorted(rec.layers.items()):
        A = tr.transition()  # [B,T,H,N,N]
        ns = _norms(A.double())
        res = {}
        for k, v in ns.items():
            vals = v if valid is None else v[valid]
            vals = vals.flatten()
            res[k] = {"sup": float(vals.max()), "q99": float(vals.quantile(0.99)), "median": float(vals.median())}
        v2 = ns["2"] if valid is None else ns["2"][valid]
        res["frac_2norm_gt_1"] = float((v2 > 1.0).double().mean())
        best = min(("2", "1", "inf"), key=lambda k: res[k]["sup"])
        res["best_fixed_norm"] = best
        res["rho_obs"] = res[best]["sup"]
        res["contracting_on_observed"] = res["rho_obs"] < 1.0
        res["status"] = "EMPIRICAL (sup over observed inputs)"
        out[l] = res
    return out


def _mixer_input_hook(mix, t: int, delta: torch.Tensor):
    def pre(mod, args, kwargs):
        if args:
            x = args[0]
            x2 = x.clone()
            x2[:, t] = x2[:, t] + delta.to(x.dtype)
            return (x2,) + tuple(args[1:]), kwargs
        x = kwargs["hidden_states"]
        x2 = x.clone()
        x2[:, t] = x2[:, t] + delta.to(x.dtype)
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = x2
        return args, kwargs
    return mix.register_forward_pre_hook(pre, with_kwargs=True)


@torch.no_grad()
def layer_local_curve(model, ids: Sequence[int], layer: int, t: int, horizon: int, eps: float = 1e-3,
                      n_dirs: int = 4, seed: int = 0) -> dict[str, Any]:
    """Experiment 1 for one layer. Returns per-tau mean/max of ``||dX||/eps``, the per-input bound curve
    (product form and rho_obs form) and ``bound_holds`` (every tau, every direction, with a relative float
    slack of 1e-6 + 1e-9 absolute)."""
    x = torch.tensor([list(ids)])
    T = x.shape[1]
    horizon = min(horizon, T - 1 - t)
    w = adapters.local_window(model)
    blk = adapters.blocks(model)[layer]
    mix = adapters.mixer(blk)
    d = adapters.config(model).hidden_size
    g = torch.Generator().manual_seed(seed)
    with adapters.trace(model) as rec:
        adapters.hidden_states(model, x)
        base = rec.layers[layer]
        X0 = base.states[0].clone()  # [T,H,N,Nv]
        A = base.transition()[0]  # [T,H,N,N]
        dirs, curves = [], []
        for _ in range(n_dirs):
            u = torch.randn(d, generator=g, dtype=torch.float64)
            u = u / u.norm()
            h = _mixer_input_hook(mix, t, eps * u)
            try:
                adapters.hidden_states(model, x)
            finally:
                h.remove()
            dX = (rec.layers[layer].states[0] - X0)  # [T,H,N,Nv]
            curves.append(torch.linalg.vector_norm(dX.flatten(1), dim=-1)[t: t + horizon + 1] / eps)
            dirs.append(dX)
    curves_t = torch.stack(curves)  # [dirs, horizon+1]
    n2 = torch.linalg.matrix_norm(A.double(), ord=2)  # [T,H]
    step = n2.amax(-1)  # head-wise max: the bound must hold for the whole state
    rho_obs = float(step[t + w + 1: t + horizon + 1].max()) if horizon > w else float("nan")
    holds = True
    prod_curve, rho_curve = [], []
    for tau in range(horizon + 1):
        if tau <= w:
            prod_curve.append(None); rho_curve.append(None)
            continue
        base_n = curves_t[:, w]  # ||dX_{t+w}|| / eps per direction
        prod = float(torch.prod(step[t + w + 1: t + tau + 1]))
        pb = prod * base_n
        rb = (rho_obs ** (tau - w)) * base_n
        prod_curve.append(float(pb.max())); rho_curve.append(float(rb.max()))
        lhs = curves_t[:, tau]
        # head-wise products are at least as tight; we check the whole-state (max-over-heads) bound
        if bool((lhs > pb * (1 + 1e-6) + 1e-9).any()):
            holds = False
    return {"layer": layer, "t": t, "window": w, "eps": eps, "n_dirs": n_dirs,
            "state_influence_mean": curves_t.mean(0).tolist(), "state_influence_max": curves_t.max(0).values.tolist(),
            "bound_product": prod_curve, "bound_rho_obs": rho_curve, "rho_obs": rho_obs, "bound_holds": holds,
            "status": "CERTIFIED per input (linear propagation with this sequence's transitions), float rounding aside"}


@torch.no_grad()
def end_to_end_curve(model, ids: Sequence[int], t: int, horizon: int, eps: float = 1e-3, n_dirs: int = 4,
                     seed: int = 0, swap_tokens: Sequence[int] = ()) -> dict[str, Any]:
    """Experiment 2: perturb the embedding at t; per layer ``||dh_l(t+tau)||/eps`` (block inputs + final),
    ``||dX_l(t+tau)||_F/eps``; token swaps (unnormalised) for each id in ``swap_tokens``. EMPIRICAL."""
    x = torch.tensor([list(ids)])
    T = x.shape[1]
    horizon = min(horizon, T - 1 - t)
    g = torch.Generator().manual_seed(seed)
    e0 = adapters.embed(model, x)
    sl = slice(t, t + horizon + 1)

    def run(e):
        with adapters.trace(model) as rec:
            hs = adapters.hidden_states(model, embeds=e)
        return [h[0, sl] for h in hs], {l: tr.states[0, sl] for l, tr in rec.layers.items()}

    h0, s0 = run(e0)
    hid = {i: [] for i in range(len(h0))}
    st = {l: [] for l in s0}
    for _ in range(n_dirs):
        u = torch.randn(e0.shape[-1], generator=g, dtype=torch.float64)
        e1 = e0.clone()
        e1[0, t] = e1[0, t] + (eps * u / u.norm()).to(e0.dtype)
        h1, s1 = run(e1)
        for i in hid:
            hid[i].append(((h1[i] - h0[i]).norm(dim=-1) / eps))
        for l in st:
            st[l].append(torch.linalg.vector_norm((s1[l] - s0[l]).flatten(1), dim=-1) / eps)
    swaps = []
    for tok in swap_tokens:
        x2 = x.clone()
        x2[0, t] = tok
        h1, s1 = run(adapters.embed(model, x2))
        swaps.append({"token": int(tok), "final_feature_delta": (h1[-1] - h0[-1]).norm(dim=-1).tolist()})
    return {"t": t, "horizon": horizon, "eps": eps,
            "hidden_influence": {i: torch.stack(v).mean(0).tolist() for i, v in hid.items()},
            "state_influence": {l: torch.stack(v).mean(0).tolist() for l, v in st.items()},
            "token_swaps": swaps, "status": "EMPIRICAL (observed inputs; includes cross-layer cascade)"}


def persistence_report(model, ids: Sequence[int], t: int, horizon: int, **kw) -> dict[str, Any]:
    """All three views for one sequence: universal bound, observed norms, local (certified) and end-to-end
    (empirical) curves."""
    x = torch.tensor([list(ids)])
    return {
        "universal": adapters.universal_decay_bound(model),
        "observed": observed_transition_norms(model, x),
        "local": {l: layer_local_curve(model, ids, l, t, horizon, **kw) for l in adapters.recurrent_layer_ids(model)},
        "end_to_end": end_to_end_curve(model, ids, t, horizon, **kw),
    }
