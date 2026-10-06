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
