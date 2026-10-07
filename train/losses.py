"""Training losses for the two calibration arms (PLAN.md D-architecture "Calibration").

Arm ``ce``   : unsmoothed cross-entropy only (a strictly proper scoring rule). RPS is off
               unless ``rps_weight`` is set explicitly.
Arm ``clef`` : label-smoothed CE + Brier (Cloudflare blog recipe, reviews/01 row "Training"),
               plus ranked probability score on ordinal questions. Label smoothing is NOT
               proper: it pulls probabilities toward uniform, which is why the arms are compared.

Knowledge distillation (``kd_alpha`` > 0; default OFF, so existing configs and their hashes are unchanged): per row
WITH a teacher distribution, ``kd_alpha * T^2 * KL(teacher_T || student_T) + (1 - kd_alpha) * arm``; per row WITHOUT
one (NaN teacher row), the arm only. ``x_T`` renormalises ``p^(1/T)`` over the row's VALID options. Teacher: an
open-weight model's per-option probabilities (``train/teacher.py``; PLAN.md allows open-weight teachers only).

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
    idx = torch.arange(lp.shape[1], device=lp.device).unsqueeze(0)
    ycdf = (idx >= y.unsqueeze(1)).to(p.dtype)
    upto = idx < (k.unsqueeze(1) - 1)  # last CDF term is always 1-1 = 0
    return (((cdf - ycdf) ** 2) * upto).sum(-1) / (k - 1).clamp_min(1)


def tempered_log_probs(lp: torch.Tensor, temperature: float) -> torch.Tensor:
    """log of p^(1/T) renormalised over the valid options (``-inf`` stays ``-inf``)."""
    if temperature == 1.0:
        return lp
    return torch.log_softmax(lp / temperature, -1)


def kd_kl(lp: torch.Tensor, teacher: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Per row ``T^2 * KL(teacher_T || student_T)`` over the valid options. ``teacher``: probabilities ``[B, K]`` in the
    same option order (rows with NaN are NOT handled here; see ``decision_loss``). Teacher mass on an invalid (padded)
    option is refused."""
    m = _valid(lp)
    t = teacher.to(lp.dtype)
    if bool(((t > 0) & ~m).any()):
        raise ValueError("teacher puts probability on a padded option")
    if bool((t < 0).any()) or not bool(torch.isfinite(t).all()):
        raise ValueError("teacher probabilities must be finite and non-negative")
    t = t * m
    mass = t.sum(-1, keepdim=True)
    if bool((mass <= 0).any()):
        raise ValueError("a teacher row has no probability on any valid option")
    t = t / mass
    lt = tempered_log_probs(torch.log(t), temperature)  # log 0 = -inf on options the teacher rules out
    pt = lt.exp()
    ls = tempered_log_probs(lp, temperature)
    diff = torch.where(pt > 0, lt - ls, torch.zeros_like(lt))  # 0 * log 0 = 0; also avoids -inf - -inf on padding
    return temperature ** 2 * (pt * diff).sum(-1)


@dataclass(frozen=True)
class LossConfig:
    arm: str = "ce"  # "ce" | "clef"
    label_smoothing: float = 0.1
    brier_weight: float = 1.0
    rps_weight: float | None = None  # score questions only; None -> 0 for "ce", 1.0 for "clef"
    #: knowledge distillation (default off; at the defaults these fields are left out of TrainConfig's hash)
    kd_alpha: float = 0.0
    kd_temperature: float = 1.0

    #: field -> default; left out of config identities at these values (old hashes unchanged)
    KD_DEFAULTS = {"kd_alpha": 0.0, "kd_temperature": 1.0}

    def __post_init__(self):
        if self.arm not in ("ce", "clef"):
            raise ValueError(self.arm)
        if type(self.kd_alpha) not in (int, float) or not 0.0 <= self.kd_alpha <= 1.0:
            raise ValueError(f"kd_alpha must be in [0, 1], got {self.kd_alpha!r}")
        if type(self.kd_temperature) not in (int, float) or not 0.0 < self.kd_temperature < float("inf"):
            raise ValueError(f"kd_temperature must be finite and > 0, got {self.kd_temperature!r}")

    @property
    def kd(self) -> bool:
        return self.kd_alpha > 0.0

    def identity(self) -> dict:
        """asdict without the KD fields at their defaults (so a non-KD config hashes as before)."""
        from dataclasses import asdict

        d = asdict(self)
        for k, v in self.KD_DEFAULTS.items():
            if d[k] == v:
                d.pop(k)
        return d

    @property
    def effective_rps_weight(self) -> float:
        if self.rps_weight is not None:
            return self.rps_weight
        return 0.0 if self.arm == "ce" else 1.0


def arm_loss(lp: torch.Tensor, y: torch.Tensor, qtypes: list[str], cfg: LossConfig) -> torch.Tensor:
    """Per-row loss of the configured calibration arm (``ce`` or ``clef``), RPS on score questions included."""
    ordinal = torch.tensor([q == "score" for q in qtypes], dtype=lp.dtype, device=lp.device)
    if cfg.arm == "ce":
        per = nll(lp, y)
    else:
        per = label_smoothed_ce(lp, y, cfg.label_smoothing) + cfg.brier_weight * brier(lp, y)
    w = cfg.effective_rps_weight
    if w:
        per = per + w * ordinal * rps(lp, y)
    return per


def decision_loss(lp: torch.Tensor, y: torch.Tensor, qtypes: list[str], cfg: LossConfig,
                  teacher: torch.Tensor | None = None) -> torch.Tensor:
    """Mean over rows. With ``cfg.kd``, ``teacher`` [B, K] holds each row's teacher probabilities in ``lp``'s option
    order, or NaN in every position for a row without a teacher (that row falls back to the arm alone). KD without a
    ``teacher`` tensor is refused (a silent no-teacher run would look like a KD run); a ``teacher`` without KD is
    refused too."""
    per = arm_loss(lp, y, qtypes, cfg)
    if not cfg.kd:
        if teacher is not None:
            raise ValueError("a teacher tensor was given but kd_alpha == 0")
        return per.mean()
    if teacher is None:
        raise ValueError("kd_alpha > 0 needs a teacher tensor (NaN rows fall back to the arm)")
    if teacher.shape != lp.shape:
        raise ValueError(f"teacher shape {tuple(teacher.shape)} != log-prob shape {tuple(lp.shape)}")
    nan = torch.isnan(teacher)
    has = ~nan.all(-1)
    if bool((nan.any(-1) & has).any()):
        raise ValueError("a teacher row is partly NaN: a row has a full teacher distribution or none")
    if bool(has.any()):
        kd = torch.zeros_like(per)
        kd[has] = kd_kl(lp[has], teacher[has], cfg.kd_temperature)
        a = cfg.kd_alpha
        per = torch.where(has, a * kd + (1 - a) * per, per)
    return per.mean()
