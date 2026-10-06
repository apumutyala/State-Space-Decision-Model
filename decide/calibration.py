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
        return (self.t0 == 1.0 and self.t1 == 0.0 and math.isfinite(self.t_min)
                and self.temperature(2) == 1.0)

    def identity(self) -> dict:
        d = asdict(self)
        d["kind"] = "option_count_temperature"
        d["status"] = "uncalibrated_identity" if self.is_identity else (
            "fitted" if self.fitted_on else "unfitted_manual"
        )
        d["id"] = sha256_text(canonical_json(d))[:16]
        return d


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
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


def _identity_weights(t_min: float) -> tuple[float, float, float]:
    if not math.isfinite(t_min) or not 0 < t_min < 1:
        raise ValueError("identity feature temperature needs 0 < t_min < 1")
    return (math.log(math.expm1(1 - t_min)), 0.0, 0.0)


IDENTITY_W = _identity_weights(0.05)
_HEX64 = frozenset("0123456789abcdef")


def content_set_sha256(hashes) -> str:
    """sha256 of a SET of content hashes (sorted, de-duplicated, newline-joined)."""
    import hashlib

    return hashlib.sha256("\n".join(sorted(set(hashes))).encode()).hexdigest()


def disjointness_proof_sha256(fitted_on: str, disjointness: dict) -> str:
    """The proof a fitted temperature carries (PR #29 review B3): the calibration content set (``fitted_on`` = its
    ``content_set_sha256``) was checked for an EMPTY intersection with each named content set (sha256 per name).
    It is part of the temperature id, so the id names the exact calibration / eval content it was proven on."""
    import hashlib
    import json

    body = {"calibration_content_sha256": fitted_on, "disjoint_from": dict(sorted(disjointness.items())),
            "intersection": 0, "unit": "normalized record content (train.cohort.normalized_content_hash)"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def feature_temperature_identity(t_min: float, w, fitted_on: str | None, proof_sha256: str | None = None) -> dict:
    """Identity block shared by the training-side fit (``train.calibration.FeatureTemperature``) and the
    serving-side ``FeatureTemperature`` below, so an eval card's ``temperature.id`` and a served
    ``served_config.calibration.id`` are the same string for the same fitted parameters AND disjointness proof.
    (Without a proof the dict, and so the id, is the pre-proof one.)"""
    import hashlib
    import json

    weights = [float(x) for x in w]
    d = {"kind": "feature_temperature", "features": FEATURE_TEMPERATURE_FEATURES, "t_min": t_min,
         "w": weights, "fitted_on": fitted_on}
    if proof_sha256 is not None:
        d["disjointness_proof"] = proof_sha256
    constant_one = (len(weights) == 3 and weights[1] == 0 and weights[2] == 0
                    and math.isfinite(t_min + _softplus(weights[0]))
                    and abs(t_min + _softplus(weights[0]) - 1.0) <= 1e-12)
    d["status"] = "fitted" if fitted_on else ("uncalibrated_identity" if constant_one else "unfitted_manual")
    d["id"] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
    return d


class CalibrationNotServable(ValueError):
    """A temperature whose fit cannot be verified (no fitted_on, no / broken disjointness proof) is never served."""


def _softplus(x: float) -> float:
    return x + math.log1p(math.exp(-x)) if x > 0 else math.log1p(math.exp(x))


@dataclass(frozen=True)
class FeatureTemperature:
    """Serving form of the m2 post-hoc temperature (techniques/06 §6.2), pure Python:

        T(k, n_ev) = t_min + softplus(w0 + w1 ln(1 + n_ev) + w2 ln k)

    ``n_ev`` = the row's evidence tokens (``heads.layout.LayoutRow.evidence_len``), which the backend reports per
    question in ``ScoreResult.features``; the adapter fails closed when they are missing. Temperature is monotone, so
    the argmax never changes.

    Serve-time guard (``assert_servable``; ``Decider`` calls it), bound to the fit (PR #29 review B3):
    * a non-identity ``w`` without ``fitted_on`` is refused (no hand-set temperatures);
    * ``fitted_on`` must be a sha256 and equal ``content_set_sha256(calib_content)`` (the calibration CONTENT set);
    * ``disjointness`` (name -> content-set sha256) must name ``eval``, and ``proof_sha256`` must equal
      ``disjointness_proof_sha256(fitted_on, disjointness)``; the proof is inside the identity id, so changing any
      attested set changes the served ``revision_id`` (``+T:<id>``);
    * ``verify_disjoint(name, content)`` re-checks at serve / eval time that a concrete content set is the attested
      one AND shares no content with the calibration set.
    """

    w: tuple | None = None
    t_min: float = 0.05
    fitted_on: str | None = None
    disjointness: tuple = ()  # ((split name, content-set sha256), ...), sorted by name
    calib_content: tuple = ()  # sorted calibration content hashes (hashes only)
    proof_sha256: str | None = None
    needs: tuple = ("evidence_tokens",)
    versioned_in_revision: bool = True

    def __post_init__(self) -> None:
        if not (math.isfinite(self.t_min) and self.t_min > 0):
            raise ValueError("t_min must be a positive finite number")
        weights = _identity_weights(self.t_min) if self.w is None else self.w
        if len(weights) != len(FEATURE_TEMPERATURE_FEATURES) or not all(math.isfinite(float(x)) for x in weights):
            raise ValueError(f"w must be {len(FEATURE_TEMPERATURE_FEATURES)} finite numbers, got {weights!r}")
        object.__setattr__(self, "w", tuple(float(x) for x in weights))
        object.__setattr__(self, "disjointness", tuple(sorted((str(k), str(v)) for k, v in self.disjointness)))
        object.__setattr__(self, "calib_content", tuple(sorted(set(self.calib_content))))

    @property
    def is_identity(self) -> bool:
        return (self.fitted_on is None and self.w[1] == 0 and self.w[2] == 0
                and abs(self.t_min + _softplus(self.w[0]) - 1.0) <= 1e-12)

    def temperature(self, k: int) -> float:  # pragma: no cover - guarded: the adapter calls temperature_for
        raise TypeError("FeatureTemperature needs evidence_tokens; use temperature_for(k, features)")

    def temperature_for(self, k: int, features: dict | None) -> float:
        if not isinstance(features, dict) or "evidence_tokens" not in features:
            raise KeyError("evidence_tokens")
        ev = features["evidence_tokens"]
        if isinstance(ev, bool) or not isinstance(ev, int) or ev < 0:
            raise ValueError(f"evidence_tokens must be a non-negative int, got {ev!r}")
        try:
            z = self.w[0] + self.w[1] * math.log1p(ev) + self.w[2] * math.log(max(int(k), 1))
            temperature = self.t_min + _softplus(z)
        except OverflowError as exc:
            raise FloatingPointError("derived feature temperature is invalid") from exc
        if not math.isfinite(temperature) or temperature <= 0:
            raise FloatingPointError("derived feature temperature is invalid")
        return temperature

    def identity(self) -> dict:
        d = feature_temperature_identity(self.t_min, self.w, self.fitted_on, self.proof_sha256)
        d["disjointness"] = {k: v for k, v in self.disjointness}
        d["calib_content_n"] = len(self.calib_content)
        return d

    def assert_servable(self) -> None:
        if self.fitted_on is None:
            if not self.is_identity:
                raise CalibrationNotServable("a non-identity temperature without fitted_on is never served")
            return
        if len(self.fitted_on) != 64 or not set(self.fitted_on) <= _HEX64:
            raise CalibrationNotServable(f"fitted_on {self.fitted_on!r} is not a content-set sha256")
        if not self.calib_content or content_set_sha256(self.calib_content) != self.fitted_on:
            raise CalibrationNotServable("fitted_on does not equal the sha256 of the calibration content set")
        dj = dict(self.disjointness)
        if "eval" not in dj:
            raise CalibrationNotServable(
                "fitted temperature carries no disjointness proof against the eval content; refit with "
                "train.calibration.FeatureTemperature.fit(..., calib_content=..., disjoint_from={'eval': ...})")
        if self.proof_sha256 != disjointness_proof_sha256(self.fitted_on, dj):
            raise CalibrationNotServable("disjointness proof does not match fitted_on / the attested content sets")

    def verify_disjoint(self, name: str, content) -> None:
        """Serve/eval-time: ``content`` must be exactly the attested ``name`` set and share nothing with calibration."""
        self.assert_servable()
        dj = dict(self.disjointness)
        c = set(content)
        if name not in dj or content_set_sha256(c) != dj[name]:
            raise CalibrationNotServable(f"{name!r} content is not the set the temperature was proven disjoint from")
        both = c & set(self.calib_content)
        if both:
            raise CalibrationNotServable(f"{name!r} shares {len(both)} content hash(es) with the calibration set")

    def to_dict(self) -> dict:
        return {"kind": "feature_temperature", "w": list(self.w), "t_min": self.t_min, "fitted_on": self.fitted_on,
                "disjointness": {k: v for k, v in self.disjointness}, "calib_content": list(self.calib_content),
                "proof_sha256": self.proof_sha256, "id": self.identity()["id"]}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureTemperature":
        if d.get("kind") != "feature_temperature":
            raise ValueError(f"not a feature_temperature: {d.get('kind')!r}")
        ft = cls(w=tuple(d["w"]), t_min=float(d["t_min"]), fitted_on=d.get("fitted_on"),
                 disjointness=tuple((d.get("disjointness") or {}).items()),
                 calib_content=tuple(d.get("calib_content") or ()), proof_sha256=d.get("proof_sha256"))
        if "id" in d and d["id"] != ft.identity()["id"]:
            raise ValueError(f"temperature id mismatch: file says {d['id']}, parameters give {ft.identity()['id']}")
        return ft
