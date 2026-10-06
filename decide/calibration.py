"""Post-hoc calibration: option-count temperature.

The board's "Decider chat" Gemma-4-31B entry is a stock model with an
option-count temperature readout (reviews/01, headline 3). We parameterise the
temperature as a function of the number of options k:

    T(k) = max(t_min, t0 + t1 * ln(k))

``raw`` probabilities are softmax(logits) (T = 1). ``served`` probabilities are
softmax(logits / T(k)). The identity config (t0 = 1, t1 = 0) is the default and
is labelled ``uncalibrated`` so nobody mistakes it for a fitted calibration.
Temperature scaling is monotone, so it never changes the argmax.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .schema import canonical_json, sha256_text


@dataclass(frozen=True)
class OptionCountTemperature:
    t0: float = 1.0
    t1: float = 0.0
    t_min: float = 0.05
    fitted_on: str | None = None  # dataset/split hash when fitted; None = not fitted

    def temperature(self, k: int) -> float:
        return max(self.t_min, self.t0 + self.t1 * math.log(max(k, 1)))

    @property
    def is_identity(self) -> bool:
        return self.t0 == 1.0 and self.t1 == 0.0

    def identity(self) -> dict:
        d = asdict(self)
        d["kind"] = "option_count_temperature"
        d["status"] = "uncalibrated_identity" if self.is_identity else (
            "fitted" if self.fitted_on else "unfitted_manual"
        )
        d["id"] = sha256_text(canonical_json(d))[:16]
        return d


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    if not logits:
        return []
    scaled = [x / temperature for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    z = sum(exps)
    return [e / z for e in exps]


# --------------------------------------------------------------------------------------------- feature temperature
#: Feature order of the m2 post-hoc temperature (``train.calibration``); part of its identity.
FEATURE_TEMPERATURE_FEATURES = ("bias", "log1p_evidence_tokens", "log_n_options")


def feature_temperature_identity(t_min: float, w, fitted_on: str | None) -> dict:
    """Identity block shared by the training-side fit (``train.calibration.FeatureTemperature``) and the
    serving-side ``FeatureTemperature`` below, so an eval card's ``temperature.id`` and a served
    ``served_config.calibration.id`` are the same string for the same fitted parameters."""
    import hashlib
    import json

    d = {"kind": "feature_temperature", "features": FEATURE_TEMPERATURE_FEATURES, "t_min": t_min,
         "w": [round(float(x), 8) for x in w], "fitted_on": fitted_on}
    d["status"] = "fitted" if fitted_on else "uncalibrated_identity"
    d["id"] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
    return d


class CalibrationNotServable(ValueError):
    """A fitted calibration without a disjointness attestation (calibration split vs eval/test) is never served."""


def _softplus(x: float) -> float:
    return x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))


@dataclass(frozen=True)
class FeatureTemperature:
    """Serving form of the m2 post-hoc temperature (techniques/06 §6.2), pure Python:

        T(k, n_ev) = t_min + softplus(w0 + w1 ln(1 + n_ev) + w2 ln k)

    ``n_ev`` = the row's evidence tokens (``heads.layout.LayoutRow.evidence_len``), which the backend reports per
    question in ``ScoreResult.features``; the adapter fails closed when they are missing. Temperature is monotone, so
    the argmax never changes.

    Guard: a FITTED temperature must carry ``disjointness`` = {split name: sha256 of the sorted record ids it was
    proven disjoint from} (``train.calibration.FeatureTemperature.fit(..., disjoint_from=...)`` computes it and
    raises on overlap). ``Decider`` refuses a fitted temperature without it. The fitted identity is versioned into
    the served ``revision_id`` (``+T:<id>``), so a response names the exact temperature that produced it.
    """

    w: tuple = (math.log(math.expm1(1 - 0.05)), 0.0, 0.0)
    t_min: float = 0.05
    fitted_on: str | None = None
    disjointness: tuple = ()  # ((split name, sha256 of its sorted record ids), ...), sorted by name
    needs: tuple = ("evidence_tokens",)
    versioned_in_revision: bool = True

    def __post_init__(self) -> None:
        if len(self.w) != len(FEATURE_TEMPERATURE_FEATURES) or not all(math.isfinite(float(x)) for x in self.w):
            raise ValueError(f"w must be {len(FEATURE_TEMPERATURE_FEATURES)} finite numbers, got {self.w!r}")
        if not (math.isfinite(self.t_min) and self.t_min > 0):
            raise ValueError("t_min must be a positive finite number")
        object.__setattr__(self, "w", tuple(float(x) for x in self.w))
        object.__setattr__(self, "disjointness", tuple(sorted((str(k), str(v)) for k, v in self.disjointness)))

    @property
    def is_identity(self) -> bool:
        return self.fitted_on is None

    def temperature(self, k: int) -> float:  # pragma: no cover - guarded: the adapter calls temperature_for
        raise TypeError("FeatureTemperature needs evidence_tokens; use temperature_for(k, features)")

    def temperature_for(self, k: int, features: dict | None) -> float:
        if not isinstance(features, dict) or "evidence_tokens" not in features:
            raise KeyError("evidence_tokens")
        ev = features["evidence_tokens"]
        if isinstance(ev, bool) or not isinstance(ev, int) or ev < 0:
            raise ValueError(f"evidence_tokens must be a non-negative int, got {ev!r}")
        z = self.w[0] + self.w[1] * math.log1p(ev) + self.w[2] * math.log(max(int(k), 1))
        return self.t_min + _softplus(z)

    def identity(self) -> dict:
        d = feature_temperature_identity(self.t_min, self.w, self.fitted_on)
        d["disjointness"] = {k: v for k, v in self.disjointness}
        return d

    def assert_servable(self) -> None:
        if self.fitted_on is not None:
            names = {k for k, _ in self.disjointness}
            if not names or "eval" not in names:
                raise CalibrationNotServable(
                    "fitted temperature carries no disjointness attestation against the eval split; refit with "
                    "train.calibration.FeatureTemperature.fit(..., disjoint_from={'eval': ..., ...})")

    def to_dict(self) -> dict:
        return {"kind": "feature_temperature", "w": list(self.w), "t_min": self.t_min, "fitted_on": self.fitted_on,
                "disjointness": {k: v for k, v in self.disjointness}, "id": self.identity()["id"]}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureTemperature":
        if d.get("kind") != "feature_temperature":
            raise ValueError(f"not a feature_temperature: {d.get('kind')!r}")
        ft = cls(w=tuple(d["w"]), t_min=float(d["t_min"]), fitted_on=d.get("fitted_on"),
                 disjointness=tuple((d.get("disjointness") or {}).items()))
        if "id" in d and d["id"] != ft.identity()["id"]:
            raise ValueError(f"temperature id mismatch: file says {d['id']}, parameters give {ft.identity()['id']}")
        return ft
