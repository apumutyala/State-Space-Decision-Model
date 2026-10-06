"""Post-hoc temperature predicted from length and option-count features (techniques/06 §6.2).

    T(x) = t_min + softplus(w . phi(x)),   phi = [1, ln(1 + evidence_tokens), ln K]

Served log-probabilities are ``log_softmax(log_p / T)``: a monotone rescaling per question,
so the argmax never changes (accuracy-preserving). It applies uniformly to every typed
output (for cumulative-link score questions it is a power transform of the level
distribution, not a rescaled eta; still order-preserving). Fitted by NLL on a dedicated
calibration split with logits frozen; the audit split stays untouched. ``fitted_on`` records
the sha256 of the calibration split ids so a card can prove which split was used.

At ``w = [softplus^-1(1 - t_min), 0, 0]`` this is the identity (T = 1), matching the
``uncalibrated_identity`` convention of ``decide.calibration.OptionCountTemperature``;
with the length weight at 0 it reduces to that option-count form (``T = t0 + t1 ln K``
up to the softplus link).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F

FEATURES = ("bias", "log1p_evidence_tokens", "log_n_options")


def features(evidence_tokens: Sequence[int], n_options: Sequence[int]) -> torch.Tensor:
    ev = torch.as_tensor(list(evidence_tokens), dtype=torch.float64)
    k = torch.as_tensor(list(n_options), dtype=torch.float64)
    return torch.stack([torch.ones_like(ev), torch.log1p(ev), torch.log(k.clamp_min(1))], -1)


def _tempered(log_probs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``log_softmax(log_probs / t)`` with padded options (-inf) kept at -inf WITHOUT dividing them:
    d(-inf / t)/dt = +inf, and inf * 0 (their softmax weight) is NaN in the backward pass, which
    turned every weight NaN when a calibration batch mixes option counts (the m2 data does)."""
    pad = ~torch.isfinite(log_probs)
    z = log_probs.masked_fill(pad, 0.0) / t
    return torch.log_softmax(z.masked_fill(pad, float("-inf")), -1)


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


@dataclass
class FeatureTemperature:
    t_min: float = 0.05
    w: list[float] = field(default_factory=lambda: [_inv_softplus(1 - 0.05), 0.0, 0.0])
    fitted_on: str | None = None
    fit_report: dict = field(default_factory=dict)

    def temperature(self, phi: torch.Tensor) -> torch.Tensor:
        w = torch.as_tensor(self.w, dtype=phi.dtype)
        return self.t_min + F.softplus(phi @ w)

    def apply(self, log_probs: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        t = self.temperature(phi.to(log_probs.dtype)).unsqueeze(-1)
        return _tempered(log_probs, t)

    def identity(self) -> dict:
        d = {"kind": "feature_temperature", "features": FEATURES, "t_min": self.t_min,
             "w": [round(x, 8) for x in self.w], "fitted_on": self.fitted_on}
        d["status"] = "fitted" if self.fitted_on else "uncalibrated_identity"
        d["id"] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
        return d

    @classmethod
    def fit(cls, log_probs: torch.Tensor, labels: torch.Tensor, phi: torch.Tensor, split_ids: Sequence[str],
            t_min: float = 0.05, l2: float = 1e-4, steps: int = 200) -> "FeatureTemperature":
        """Minimise NLL of the tempered distribution (frozen logits) with L-BFGS."""
        lp = log_probs.detach().double()
        y = labels.long()
        ph = phi.double()
        w = torch.tensor([_inv_softplus(1 - t_min), 0.0, 0.0], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w], lr=0.5, max_iter=steps, line_search_fn="strong_wolfe")

        def nll_of(wv):
            t = t_min + F.softplus(ph @ wv)
            out = _tempered(lp, t.unsqueeze(-1))
            return -out.gather(1, y.unsqueeze(1)).mean()

        before = float(nll_of(w.detach()))

        def closure():
            opt.zero_grad()
            loss = nll_of(w) + l2 * (w[1:] ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        after = float(nll_of(w.detach()))
        if not (torch.isfinite(w.detach()).all() and math.isfinite(after)):
            raise FloatingPointError(f"temperature fit diverged (w={w.detach().tolist()}, nll_after={after})")
        split_hash = hashlib.sha256("\n".join(sorted(split_ids)).encode()).hexdigest()
        return cls(t_min=t_min, w=w.detach().tolist(), fitted_on=split_hash,
                   fit_report={"nll_before": before, "nll_after": after, "n": int(len(y))})
