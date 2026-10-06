"""Reachability / controllability of the untrusted channel (EMPIRICAL, linearised at an observed input).

For layer l with final recurrent state ``X_T^l`` (key-first, flattened to ``vec`` of size H*Nk*Nv) and the
untrusted positions U, the empirical controllability Gramian is

    W_l = sum_{t in U} J_t J_t^T,   J_t = d vec(X_T^l) / d u_t

with ``u_t`` either the untrusted token EMBEDDING (``mode="end_to_end"``: everything below layer l included)
or layer l's own MIXER INPUT at t (``mode="local"``: the layer's recurrence in isolation, other inputs
held fixed). Reported per layer: trace, eigenvalues, effective rank (entropy) and participation ratio,
numerical rank, and the overlap with the readout directions at the decision position T:

    y_T = X_T^T r_T  (per head)  =>  readable subspace = span(r_T) (x) R^Nv,  P_R = (r r^T / r^T r) (x) I
    overlap = tr(W P_R) / tr(W)      (isotropic reference: 1 / Nk)

This is a first-order (Jacobian) statement at one input, averaged over the inputs given; it is not a
reachable-set certificate for a nonlinear system. The STRUCTURAL counterpart is the trust gate: for its
protected stream ``W_P`` is exactly 0 (tested), because untrusted ids are never read by that stream.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch

from . import adapters


def _eig_summary(W: torch.Tensor, tol: float = 1e-9) -> dict[str, Any]:
    ev = torch.linalg.eigvalsh((W + W.T) / 2).clamp_min(0).flip(0)
    tr = float(ev.sum())
    if tr <= 0:
        return {"trace": 0.0, "eff_rank_entropy": 0.0, "participation_ratio": 0.0, "numerical_rank": 0,
                "top_eigs": ev[:8].tolist(), "dim": int(W.shape[0])}
    p = ev / ev.sum()
    nz = p[p > 0]
    ent = float(torch.exp(-(nz * nz.log()).sum()))
    pr = float(ev.sum() ** 2 / (ev ** 2).sum())
    rank = int((ev > tol * ev[0]).sum())
    return {"trace": tr, "eff_rank_entropy": ent, "participation_ratio": pr, "numerical_rank": rank,
            "top_eigs": ev[:8].tolist(), "dim": int(W.shape[0])}


def readout_projector(r: torch.Tensor, nv: int) -> torch.Tensor:
    """``r [H, Nk]`` -> projector on vec(X) (H*Nk*Nv, key-first) onto the readable directions."""
    blocks = []
    for h in range(r.shape[0]):
        rh = r[h] / r[h].norm().clamp_min(1e-30)
        blocks.append(torch.kron(torch.outer(rh, rh), torch.eye(nv, dtype=r.dtype)))
    return torch.block_diag(*blocks)


def gramians(model, ids: Sequence[int], untrusted: Sequence[bool], mode: str = "end_to_end",
             layers: Sequence[int] | None = None, return_jacobian: bool = False) -> dict[int, dict[str, Any]]:
    """Per-layer Gramian summary for one input. ``untrusted[t]`` True marks U. ``return_jacobian`` adds
    ``J`` ([state, |U| * d], columns ordered (position in U, feature)) for checks."""
    if mode not in ("end_to_end", "local"):
        raise ValueError(mode)
    x = torch.tensor([list(ids)])
    U = [t for t, u in enumerate(untrusted) if u]
    layers = list(layers) if layers is not None else adapters.recurrent_layer_ids(model)
    out: dict[int, dict[str, Any]] = {}
    if not U:
        return {l: {"trace": 0.0, "note": "no untrusted positions"} for l in layers}
    for l in layers:
        leaf: dict[str, torch.Tensor] = {}
        handle = None
        if mode == "end_to_end":
            e = adapters.embed(model, x).detach().requires_grad_(True)
            leaf["u"] = e
            kw = {"embeds": e}
        else:
            mix = adapters.mixer(adapters.blocks(model)[l])

            def pre(mod, args, kwargs):
                if args:
                    z = args[0].detach().requires_grad_(True)
                    leaf["u"] = z
                    return (z,) + tuple(args[1:]), kwargs
                z = kwargs["hidden_states"].detach().requires_grad_(True)
                leaf["u"] = z
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = z
                return args, kwargs
            handle = mix.register_forward_pre_hook(pre, with_kwargs=True)
            kw = {"ids": x}
        try:
            with torch.enable_grad(), adapters.trace(model) as rec:
                adapters.hidden_states(model, **kw)
        finally:
            if handle is not None:
                handle.remove()
        tr = rec.layers[l]
        XT = tr.states[0, -1]  # [H,Nk,Nv]
        H, Nk, Nv = XT.shape
        flat = XT.reshape(-1)
        u = leaf["u"]
        rows = []
        for i in range(flat.numel()):
            (gi,) = torch.autograd.grad(flat[i], u, retain_graph=True, allow_unused=True)
            gi = torch.zeros_like(u) if gi is None else gi
            rows.append(gi[0, U].reshape(-1).double())
        J = torch.stack(rows)  # [S, |U|*d]
        W = J @ J.T
        P = readout_projector(tr.readout[0, -1].detach().double(), Nv)
        trW = float(torch.trace(W))
        overlap = float(torch.trace(W @ P)) / trW if trW > 0 else 0.0
        out[l] = {**_eig_summary(W), "mode": mode, "n_untrusted": len(U), "readout_overlap": overlap,
                  "isotropic_overlap": 1.0 / Nk, "status": "EMPIRICAL (Jacobian at this input)"}
        if return_jacobian:
            out[l]["J"] = J
    return out


def protected_gramian(gated, ids: Sequence[int], trust: Sequence[bool]) -> float:
    """Frobenius norm of d(protected features) / d(untrusted embeddings) for a ``TrustGatedBackbone``:
    exactly 0 by construction (the protected stream never reads untrusted ids). Computed with autograd
    through the embedding table: the gradient of the protected stream w.r.t. the embedding ROWS of the
    untrusted tokens that do not also occur in trusted positions."""
    emb = adapters.inner_model(gated.model).get_input_embeddings()
    x = torch.tensor([list(ids)])
    t = torch.tensor([list(trust)])
    untrusted_only = set(x[0][~t[0]].tolist()) - set(x[0][t[0]].tolist())
    with torch.enable_grad():
        emb.weight.grad = None
        req = emb.weight.requires_grad
        emb.weight.requires_grad_(True)
        try:
            gated.features(x, None, t)
            s = sum(p.pow(2).sum() for p in gated.last_protected) + gated.last_protected_final.pow(2).sum()
            (g,) = torch.autograd.grad(s, emb.weight, allow_unused=True)
        finally:
            emb.weight.requires_grad_(req)
    if g is None or not untrusted_only:
        return 0.0
    return float(g[sorted(untrusted_only)].norm())
