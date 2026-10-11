"""Typed outputs over per-option head logits: noul, choice (<= 255 options), score.

* ``noul``   two options (true, false) in canonical order; softmax.
* ``choice`` 2..255 options; masked softmax. More than 255 is rejected (SystemOne limit;
             the benchmark keeps all candidates, a retrieval cascade is a separate product
             configuration per the partner plan).
* ``score``  cumulative-link (ordinal logistic) model over the K ordered levels:
             P(Y <= j) = sigmoid(theta_j - eta), j = 0..K-2, with theta strictly increasing
             by construction (theta = cumsum of softplus gaps, centred) and
             eta = sum_j c_j z_j, c_j the centred level contrast (sum c_j = 0, so eta is
             invariant to a common shift of the head logits). Level probabilities are
             successive CDF differences, hence non-negative and monotone in eta
             (stochastic ordering): raising eta moves mass to higher levels.
             Gaps depend on the level descriptions of adjacent levels (shared MLP), so
             the cut points adapt to the legend while staying ordered.

All functions return log-probabilities ``[B, K]`` with ``-inf`` on padded options.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

MAX_OPTIONS = 255
QTYPES = ("noul", "choice", "score")


class CumulativeLink(nn.Module):
    def __init__(self, d_model: int, hidden: int = 64) -> None:
        super().__init__()
        self.gap = nn.Sequential(nn.Linear(2 * d_model, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.eta_scale = nn.Parameter(torch.tensor(1.0))
        nn.init.constant_(self.gap[-1].bias, 0.5)

    def thresholds(self, e: torch.Tensor) -> torch.Tensor:
        """e: [K, D] level embeddings (in level order) -> theta [K-1], strictly increasing."""
        gaps = F.softplus(self.gap(torch.cat([e[:-1], e[1:]], -1)).squeeze(-1)) + 1e-3
        theta = gaps.cumsum(0)
        return theta - theta.mean()

    def eta(self, z: torch.Tensor) -> torch.Tensor:
        k = z.shape[-1]
        c = torch.arange(k, dtype=z.dtype, device=z.device) - (k - 1) / 2
        c = c / max(k - 1, 1)
        return self.eta_scale * (z * c).sum(-1)

    def log_probs(self, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """z: [K] head logits for one question, e: [K, D]. Returns log P(Y = j), [K]."""
        theta = self.thresholds(e)
        eta = self.eta(z)
        cdf = torch.sigmoid(theta - eta)
        cdf = torch.cat([cdf.new_zeros(1), cdf, cdf.new_ones(1)])
        p = (cdf[1:] - cdf[:-1]).clamp_min(1e-12)
        return torch.log(p / p.sum())


class TypedOutput(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.ordinal = CumulativeLink(d_model)

    def forward(self, logits: torch.Tensor, qtypes: list[str], ks: list[int], e: torch.Tensor) -> torch.Tensor:
        """``ks``: options per row as python ints (read without a device sync)."""
        if logits.shape[1] > MAX_OPTIONS:
            raise ValueError(f"{logits.shape[1]} options exceeds the {MAX_OPTIONS}-option limit")
        out = torch.log_softmax(logits, -1)  # -inf logits stay -inf
        rows = []
        for i, qt in enumerate(qtypes):
            if qt not in QTYPES:
                raise ValueError(qt)
            if qt == "noul" and ks[i] != 2:
                raise ValueError("noul questions have exactly two options (true, false)")
            if qt == "score":
                k = ks[i]
                lp = self.ordinal.log_probs(logits[i, :k], e[i, :k])
                row = torch.full_like(logits[i], float("-inf"))
                row = torch.cat([lp, row[k:]])
                rows.append((i, row))
        if rows:
            out = out.clone()
            for i, row in rows:
                out[i] = row
        return out


def to_wire(question: dict, option_ids: list[str], log_probs: torch.Tensor, tie: dict | None = None) -> dict:
    """SystemOne answer via the m0 adapter's ``wire_answer`` (same rounding and tie rules; ``tie`` = sym rows'
    content-based ties, ``heads.symmetric.sym_tie_info``; None for every other layout: unchanged)."""
    from decide.schema import wire_answer

    probs = {oid: float(p) for oid, p in zip(option_ids, log_probs[: len(option_ids)].detach().exp().cpu().tolist())}
    return wire_answer(question, probs, tie)
