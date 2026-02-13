from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════
#  ATMOSPHERIC DATA CLASS
# ═══════════════════════════════════════════════════════════════

@dataclass
class OceanAtmosphere:
    n_base: float = 1.000293
    a: float = 0.000120
    b: float = 0.000040
    h1: float = 12.0
    h2: float = 40.0
    sea_temp: float = 10.0
    air_temp: float = 25.0
    ducting_enabled: bool = True

    # ── scalar refractive index ───────────────────────────────

    def n(self, y: float) -> float:
        y_c = max(y, 0.0)
        if not self.ducting_enabled:
            return self.n_base - 3e-5 * (1.0 - math.exp(-y_c / 8000.0))
        return (self.n_base
                + self.a * math.exp(-y_c / self.h1)
                - self.b * math.exp(-y_c / self.h2))

    def dn_dy(self, y: float) -> float:
        """Vertical gradient  dn/dy."""
        y_c = max(y, 0.0)
        if not self.ducting_enabled:
            return -3e-5 / 8000.0 * math.exp(-y_c / 8000.0)
        return (-(self.a / self.h1) * math.exp(-y_c / self.h1)
                + (self.b / self.h2) * math.exp(-y_c / self.h2))

    # ── vectorised ────────────────────────────────────────────

    def n_array(self, y: np.ndarray) -> np.ndarray:
        y_c = np.maximum(y, 0.0)
        if not self.ducting_enabled:
            return self.n_base - 3e-5 * (1.0 - np.exp(-y_c / 8000.0))
        return (self.n_base
                + self.a * np.exp(-y_c / self.h1)
                - self.b * np.exp(-y_c / self.h2))

    def dn_dy_array(self, y: np.ndarray) -> np.ndarray:
        y_c = np.maximum(y, 0.0)
        if not self.ducting_enabled:
            return -3e-5 / 8000.0 * np.exp(-y_c / 8000.0)
        return (-(self.a / self.h1) * np.exp(-y_c / self.h1)
                + (self.b / self.h2) * np.exp(-y_c / self.h2))

    # ── profile for plotting ──────────────────────────────────

    def n_profile(
        self, y_max: float = 100.0, n_pts: int = 300
    ) -> Tuple[np.ndarray, np.ndarray]:
        ys = np.linspace(0.0, y_max, n_pts)
        ns = self.n_array(ys)
        return ys, ns

    # ── duct height (where dn/dy = 0) ────────────────────────

    def duct_height(self) -> float:
        if not self.ducting_enabled:
            return -1.0
        ratio = (self.a * self.h2) / (self.b * self.h1 + 1e-30)
        if ratio <= 0:
            return -1.0
        dh = self.h2 - self.h1
        if abs(dh) < 1e-6:
            return -1.0
        y_star = (self.h1 * self.h2 / dh) * math.log(ratio)
        return y_star if y_star > 0 else -1.0


# ═══════════════════════════════════════════════════════════════
#  RAY ODE
# ═══════════════════════════════════════════════════════════════

def ray_ode(
    s: float,
    state: list[float],
    atm: OceanAtmosphere,
) -> list[float]:
    _x, y, vx, vy = state
    n_val = atm.n(y)
    dndy = atm.dn_dy(y)

    dvx = -(vy * vx / n_val) * dndy
    dvy = (vx * vx / n_val) * dndy

    return [vx, vy, dvx, dvy]
