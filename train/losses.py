"""Training losses for the two calibration arms (PLAN.md D-architecture "Calibration").

Arm ``ce``   : unsmoothed cross-entropy (a strictly proper scoring rule), optional RPS on
               ordinal (score) questions.
Arm ``clef`` : label-smoothed CE + Brier (Cloudflare blog recipe, reviews/01 row "Training"),
               plus ranked probability score on ordinal questions. Label smoothing is NOT
               proper: it pulls probabilities toward uniform, which is why the arms are compared.

All inputs are log-probabilities ``[B, K]`` with ``-inf`` on padded options.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _valid(lp: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(lp)


def nll(lp: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return -lp.gather(1, y.unsqueeze(1)).squeeze(1)


def label_smoothed_ce(lp: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    m = _valid(lp)
    k = m.sum(-1).clamp_min(1)
    uniform = -(lp.masked_fill(~m, 0.0)).sum(-1) / k
    return (1 - eps) * nll(lp, y) + eps * uniform


def brier(lp: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = lp.exp()
    onehot = torch.zeros_like(p).scatter(1, y.unsqueeze(1), 1.0)
    return ((p - onehot) ** 2 * _valid(lp)).sum(-1)


def rps(lp: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Ranked probability score over the valid prefix of ordered levels, normalised by K-1."""
    p = lp.exp() * _valid(lp)
    k = _valid(lp).sum(-1)
    cdf = p.cumsum(-1)
    idx = torch.arange(lp.shape[1]).unsqueeze(0)
    ycdf = (idx >= y.unsqueeze(1)).to(p.dtype)
    upto = idx < (k.unsqueeze(1) - 1)  # last CDF term is always 1-1 = 0
    return (((cdf - ycdf) ** 2) * upto).sum(-1) / (k - 1).clamp_min(1)


@dataclass(frozen=True)
class LossConfig:
    arm: str = "ce"  # "ce" | "clef"
    label_smoothing: float = 0.1
    brier_weight: float = 1.0
    rps_weight: float = 1.0  # applied to score questions only

    def __post_init__(self):
        if self.arm not in ("ce", "clef"):
            raise ValueError(self.arm)


def decision_loss(lp: torch.Tensor, y: torch.Tensor, qtypes: list[str], cfg: LossConfig) -> torch.Tensor:
    ordinal = torch.tensor([q == "score" for q in qtypes], dtype=lp.dtype)
    if cfg.arm == "ce":
        per = nll(lp, y)
    else:
        per = label_smoothed_ce(lp, y, cfg.label_smoothing) + cfg.brier_weight * brier(lp, y)
    if cfg.rps_weight:
        per = per + cfg.rps_weight * ordinal * rps(lp, y)
    return per.mean()
