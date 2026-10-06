"""Post-hoc temperature predicted from length and option-count features (techniques/06 §6.2).

    T(x) = t_min + softplus(w . phi(x)),   phi = [1, ln(1 + evidence_tokens), ln K]

Served log-probabilities are ``log_softmax(log_p / T)``: a monotone rescaling per question,
so the argmax never changes (accuracy-preserving). It applies uniformly to every typed
output (for cumulative-link score questions it is a power transform of the level
distribution, not a rescaled eta; still order-preserving). Fitted by NLL on a dedicated
calibration split with logits frozen; the audit split stays untouched. ``fitted_on`` records
the sha256 of the calibration split ids so a card can prove which split was used.

Disjointness guard (PR #29 review B1/B3: CONTENT-based and bound to the fit). ``fit(..., calib_content=[...],
disjoint_from={"eval": content hashes, "retention": ..., "train": ...})`` compares NORMALIZED CONTENT hashes
(``train.cohort.normalized_content_hash``, carried on every row as ``row.meta["content_sha256"]``), so an exact
content twin under another id is caught, and raises ``CalibrationLeak`` on any shared content. ``fitted_on`` is then
the sha256 of the calibration content SET, ``disjointness`` the per-split content-set sha256s, and
``proof_sha256 = decide.calibration.disjointness_proof_sha256(fitted_on, disjointness)`` enters the temperature id;
``to_serving()`` carries all three plus the calibration content hashes, and ``decide.Decider`` re-verifies them
(``assert_servable``). Without ``calib_content`` / ``disjoint_from`` a fit is a legacy, unservable one (``fitted_on``
= sha256 of the split ids). ``assert_disjoint`` re-checks at eval time (``train.evalcard.run_card``).

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
from decide.calibration import content_set_sha256, disjointness_proof_sha256, feature_temperature_identity


class CalibrationLeak(ValueError):
    """The calibration split shares content (or a record) with an eval/test/train split."""


def record_ids(ids: Sequence[str]) -> set[str]:
    """Item ids ``<record id>/<qid>`` -> record ids (ids without a ``/`` are record ids already)."""
    return {str(i).rsplit("/", 1)[0] if "/" in str(i) else str(i) for i in ids}


def ids_sha256(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def check_disjoint(calib_content: Sequence[str], disjoint_from: dict[str, Sequence[str]]) -> dict[str, str]:
    """Content sets: raise ``CalibrationLeak`` on any shared content hash; return {name: content-set sha256}."""
    cal = set(calib_content)
    if not cal:
        raise CalibrationLeak("calibration content set is empty")
    out = {}
    for name, hashes in sorted(disjoint_from.items()):
        other = set(hashes)
        if not other:
            raise CalibrationLeak(f"split {name!r} is empty: disjointness from nothing is not a guard")
        both = cal & other
        if both:
            raise CalibrationLeak(f"calibration content shares {len(both)} hash(es) with {name!r}, e.g. {sorted(both)[:2]}")
        out[name] = content_set_sha256(other)
    return out


def item_content(items) -> list[str]:
    """Normalized content hash of each item's record (``train.data.items_from_records`` stamps it)."""
    out = []
    for it in items:
        h = it.row.meta.get("content_sha256")
        if not h:
            raise ValueError(f"{it.id}: row carries no content_sha256 (build items with train.data.items_from_records)")
        out.append(h)
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
    if not torch.isfinite(t).all() or not (t > 0).all():
        raise FloatingPointError("derived feature temperature must be finite and positive")
    pad = torch.isneginf(log_probs)
    z = log_probs.masked_fill(pad, 0.0) / t
    return torch.log_softmax(z.masked_fill(pad, float("-inf")), -1)


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


@dataclass
class FeatureTemperature:
    t_min: float = 0.05
    w: list[float] | None = None
    fitted_on: str | None = None
    fit_report: dict = field(default_factory=dict)
    disjointness: dict = field(default_factory=dict)  # split name -> content-set sha256
    calib_records: frozenset = field(default_factory=frozenset, repr=False)  # kept for eval-time re-checks
    calib_content: frozenset = field(default_factory=frozenset, repr=False)
    proof_sha256: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.t_min) or self.t_min <= 0:
            raise ValueError("t_min must be a positive finite number")
        if self.w is None:
            if self.t_min >= 1:
                raise ValueError("default identity weights need 0 < t_min < 1")
            self.w = [_inv_softplus(1 - self.t_min), 0.0, 0.0]
        elif len(self.w) != len(FEATURES) or not all(math.isfinite(float(x)) for x in self.w):
            raise ValueError(f"w must be {len(FEATURES)} finite numbers")
        else:
            self.w = [float(x) for x in self.w]

    def temperature(self, phi: torch.Tensor) -> torch.Tensor:
        w = torch.as_tensor(self.w, dtype=phi.dtype)
        t = self.t_min + F.softplus(phi @ w)
        if not torch.isfinite(t).all() or not (t > 0).all():
            raise FloatingPointError("derived feature temperature must be finite and positive")
        return t

    def apply(self, log_probs: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        t = self.temperature(phi.to(log_probs.dtype)).unsqueeze(-1)
        return _tempered(log_probs, t)

    def identity(self) -> dict:
        return feature_temperature_identity(self.t_min, self.w, self.fitted_on, self.proof_sha256)

    def assert_disjoint(self, ids: Sequence[str], name: str = "eval", content: Sequence[str] | None = None) -> None:
        """Eval-time guard: raise ``CalibrationLeak`` if any of ``ids`` (items or records) is a calibration record
        or any of ``content`` is calibration content."""
        both = record_ids(ids) & set(self.calib_records)
        if both:
            raise CalibrationLeak(f"{name} shares {len(both)} record(s) with the calibration split, e.g. {sorted(both)[:3]}")
        shared = set(content or ()) & set(self.calib_content)
        if shared:
            raise CalibrationLeak(f"{name} shares {len(shared)} content hash(es) with the calibration split")

    def to_serving(self) -> ServingTemperature:
        """The pure-Python form ``decide.Decider(calibration=...)`` serves (same identity id, same proof)."""
        return ServingTemperature(w=tuple(self.w), t_min=self.t_min, fitted_on=self.fitted_on,
                                  disjointness=tuple(self.disjointness.items()),
                                  calib_content=tuple(sorted(self.calib_content)), proof_sha256=self.proof_sha256)

    @classmethod
    def fit(cls, log_probs: torch.Tensor, labels: torch.Tensor, phi: torch.Tensor, split_ids: Sequence[str],
            t_min: float = 0.05, l2: float = 1e-4, steps: int = 200,
            disjoint_from: dict[str, Sequence[str]] | None = None,
            calib_content: Sequence[str] | None = None) -> "FeatureTemperature":
        """Minimise NLL of the tempered distribution (frozen logits) with L-BFGS. ``disjoint_from`` (name -> CONTENT
        hashes of the eval/retention/train splits) is checked against ``calib_content`` (one hash per calibration
        item) BEFORE fitting; the proof is bound into ``fitted_on`` / ``proof_sha256`` / the id. Without both the result
        is a legacy fit that ``decide.Decider`` refuses to serve."""
        if not math.isfinite(t_min) or not 0 < t_min < 1:
            raise ValueError("fit initialization needs 0 < t_min < 1")
        if disjoint_from and calib_content is None:
            raise ValueError("disjoint_from needs calib_content (content hashes of the calibration items)")
        if calib_content is not None and len(calib_content) != len(split_ids):
            raise ValueError("calib_content must have one content hash per calibration item")
        disjointness = check_disjoint(calib_content, disjoint_from) if disjoint_from else {}
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
        if disjointness:
            fitted_on = content_set_sha256(calib_content)
            proof = disjointness_proof_sha256(fitted_on, disjointness)
        else:  # legacy: identifies the split, proves nothing, never served
            fitted_on, proof = hashlib.sha256("\n".join(sorted(split_ids)).encode()).hexdigest(), None
        return cls(t_min=t_min, w=w.detach().tolist(), fitted_on=fitted_on,
                   fit_report={"nll_before": before, "nll_after": after, "n": int(len(y))},
                   disjointness=disjointness, calib_records=frozenset(record_ids(split_ids)),
                   calib_content=frozenset(calib_content or ()), proof_sha256=proof)
