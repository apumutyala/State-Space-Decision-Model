"""Post-hoc temperature predicted from length and option-count features (techniques/06 §6.2).

    T(x) = t_min + softplus(w . phi(x)),   phi = [1, ln(1 + evidence_tokens), ln K]

Served log-probabilities are ``log_softmax(log_p / T)``: a monotone rescaling per question,
so the argmax never changes (accuracy-preserving). It applies uniformly to every typed
output (for cumulative-link score questions it is a power transform of the level
distribution, not a rescaled eta; still order-preserving). Fitted by NLL on a dedicated
calibration split with logits frozen; the audit split stays untouched. ``fitted_on`` records
the sha256 of the calibration split ids so a card can prove which split was used.

Disjointness guard (m2 blocker 1): ``fit(..., disjoint_from={"eval": ids, "test": ids, ...})`` compares RECORD ids
(an item id ``<record>/<qid>`` counts as its record, so two questions of one record can never sit on both sides)
and raises ``CalibrationLeak`` on any overlap; the per-split sha256 of the sorted record ids goes into
``disjointness`` and into the serving form (``to_serving()`` -> ``decide.calibration.FeatureTemperature``), which
``decide.Decider`` refuses to serve without an ``eval`` attestation. ``assert_disjoint`` re-checks at eval time
(``train.evalcard.run_card``) against the actual eval items.

At ``w = [softplus^-1(1 - t_min), 0, 0]`` this is the identity (T = 1), matching the
``uncalibrated_identity`` convention of ``decide.calibration.OptionCountTemperature``;
with the length weight at 0 it reduces to that option-count form (``T = t0 + t1 ln K``
up to the softplus link).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F

from decide.calibration import FEATURE_TEMPERATURE_FEATURES as FEATURES
from decide.calibration import FeatureTemperature as ServingTemperature
from decide.calibration import feature_temperature_identity


class CalibrationLeak(ValueError):
    """The calibration split shares a record with an eval/test split."""


def record_ids(ids: Sequence[str]) -> set[str]:
    """Item ids ``<record id>/<qid>`` -> record ids (ids without a ``/`` are record ids already)."""
    return {str(i).rsplit("/", 1)[0] if "/" in str(i) else str(i) for i in ids}


def ids_sha256(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def check_disjoint(calib_ids: Sequence[str], disjoint_from: dict[str, Sequence[str]]) -> dict[str, str]:
    """Raise ``CalibrationLeak`` if a calibration record is in any named split; return {name: sha256 of its records}."""
    cal = record_ids(calib_ids)
    out = {}
    for name, ids in sorted(disjoint_from.items()):
        other = record_ids(ids)
        if not other:
            raise CalibrationLeak(f"split {name!r} is empty: disjointness from nothing is not a guard")
        both = cal & other
        if both:
            raise CalibrationLeak(f"calibration split shares {len(both)} record(s) with {name!r}, e.g. {sorted(both)[:3]}")
        out[name] = ids_sha256(other)
    return out


def features(evidence_tokens: Sequence[int], n_options: Sequence[int]) -> torch.Tensor:
    ev = torch.as_tensor(list(evidence_tokens), dtype=torch.float64)
    k = torch.as_tensor(list(n_options), dtype=torch.float64)
    return torch.stack([torch.ones_like(ev), torch.log1p(ev), torch.log(k.clamp_min(1))], -1)


def _tempered(log_probs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``log_softmax(log_probs / t)`` with padded options (-inf) kept at -inf WITHOUT dividing them:
    d(-inf / t)/dt = +inf, and inf * 0 (their softmax weight) is NaN in the backward pass, which
    turned every weight NaN when a calibration batch mixes option counts (the m2 data does).
    Only -inf means "padded option"; NaN or +inf log-probabilities are a model fault and raise."""
    if torch.isnan(log_probs).any() or torch.isposinf(log_probs).any():
        raise FloatingPointError("NaN/+inf log-probabilities reached the temperature (model fault, not padding)")
    pad = torch.isneginf(log_probs)
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
    disjointness: dict = field(default_factory=dict)  # split name -> sha256 of its sorted record ids
    calib_records: frozenset = field(default_factory=frozenset, repr=False)  # kept for eval-time re-checks

    def temperature(self, phi: torch.Tensor) -> torch.Tensor:
        w = torch.as_tensor(self.w, dtype=phi.dtype)
        return self.t_min + F.softplus(phi @ w)

    def apply(self, log_probs: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        t = self.temperature(phi.to(log_probs.dtype)).unsqueeze(-1)
        return _tempered(log_probs, t)

    def identity(self) -> dict:
        return feature_temperature_identity(self.t_min, self.w, self.fitted_on)

    def assert_disjoint(self, ids: Sequence[str], name: str = "eval") -> None:
        """Eval-time guard: raise ``CalibrationLeak`` if any of ``ids`` (items or records) is a calibration record."""
        both = record_ids(ids) & set(self.calib_records)
        if both:
            raise CalibrationLeak(f"{name} shares {len(both)} record(s) with the calibration split, e.g. {sorted(both)[:3]}")

    def to_serving(self) -> ServingTemperature:
        """The pure-Python form ``decide.Decider(calibration=...)`` serves (same identity id)."""
        return ServingTemperature(w=tuple(self.w), t_min=self.t_min, fitted_on=self.fitted_on,
                                  disjointness=tuple(self.disjointness.items()))

    @classmethod
    def fit(cls, log_probs: torch.Tensor, labels: torch.Tensor, phi: torch.Tensor, split_ids: Sequence[str],
            t_min: float = 0.05, l2: float = 1e-4, steps: int = 200,
            disjoint_from: dict[str, Sequence[str]] | None = None) -> "FeatureTemperature":
        """Minimise NLL of the tempered distribution (frozen logits) with L-BFGS. ``disjoint_from`` (name -> ids of
        the eval/test/train splits) is checked at record level BEFORE fitting; without it the result is not servable
        (``decide.Decider`` refuses a fitted temperature lacking an ``eval`` attestation)."""
        disjointness = check_disjoint(split_ids, disjoint_from) if disjoint_from else {}
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
                   fit_report={"nll_before": before, "nll_after": after, "n": int(len(y))},
                   disjointness=disjointness, calib_records=frozenset(record_ids(split_ids)))
