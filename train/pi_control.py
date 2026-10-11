"""The nuPI controller (pure Python floats, no torch): see train/constraints.py and docs/NUPI.md."""

from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------------------------- the controller
class PIController:
    """One multiplier. Pure Python floats (doubles) so a checkpoint round-trip is bit-exact."""

    def __init__(self, kappa_p: float, kappa_i: float, nu: float = 0.0, lambda_init: float = 0.0,
                 lambda_max: float = math.inf, integral_min: float = 0.0, integral_max: float = math.inf):
        if not (kappa_p >= 0 and kappa_i >= 0 and 0 <= nu < 1 and lambda_init >= 0 and lambda_max >= 0
                and integral_min <= 0 <= integral_max):
            raise ValueError("nuPI needs kappa_p, kappa_i >= 0, 0 <= nu < 1, lambda_init >= 0, lambda_max >= 0 and "
                             "integral_min <= 0 <= integral_max")
        self.kp, self.ki, self.nu = float(kappa_p), float(kappa_i), float(nu)
        self.lam0, self.lam_max = float(lambda_init), float(lambda_max)
        self.imin, self.imax = float(integral_min), float(integral_max)
        # the I term kappa_i * I_t is accumulated pre-scaled (iterm <- clip(iterm + kappa_i e, kappa_i [imin, imax])),
        # so kappa_p = 0 runs the float ops of projected gradient ascent, lambda <- max(0, lambda + kappa_i e), exactly
        # (kappa_i * sum(e) and sum(kappa_i e) differ in the last bit). ``integral`` keeps I_t itself for logging.
        self.ilo = self.ki * self.imin if self.ki > 0 else 0.0
        self.ihi = self.ki * self.imax if self.ki > 0 else 0.0
        self.lam, self.integral, self.iterm, self.xi, self.e, self.n = self.lam0, 0.0, 0.0, 0.0, 0.0, 0

    def update(self, e: float) -> float:
        e = float(e)
        if not math.isfinite(e):
            return self.lam  # a non-finite measurement never moves the multiplier
        self.e = e
        self.xi = self.nu * self.xi + (1.0 - self.nu) * e
        self.integral = min(self.imax, max(self.imin, self.integral + e))
        self.iterm = min(self.ihi, max(self.ilo, self.iterm + self.ki * e))
        self.lam = min(self.lam_max, max(0.0, self.lam0 + self.kp * self.xi + self.iterm))
        self.n += 1
        return self.lam

    def state_dict(self) -> dict[str, Any]:
        return {"lam": self.lam, "integral": self.integral, "iterm": self.iterm, "xi": self.xi, "e": self.e,
                "n": self.n}

    def load_state_dict(self, st: dict[str, Any]) -> None:
        self.lam, self.integral, self.xi = float(st["lam"]), float(st["integral"]), float(st["xi"])
        self.iterm = float(st["iterm"])
        self.e, self.n = float(st["e"]), int(st["n"])


def gradient_ascent(eta: float, lambda_max: float = math.inf) -> PIController:
    """Projected gradient ascent on the multiplier (the GDA baseline): ``lambda <- max(0, lambda + eta e)``."""
    return PIController(kappa_p=0.0, kappa_i=eta, nu=0.0, lambda_max=lambda_max, integral_min=0.0)
