from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class DesertAtmosphere:
    n_base: float = 1.000293
    delta_n: float = 2.4e-4
    scale_height: float = 3.0
    ground_temp: float = 65.0
    air_temp: float = 28.0

    def n(self, y: float) -> float:
        y_c = max(y, 0.0)
        return self.n_base - self.delta_n * math.exp(-y_c / self.scale_height)

    def dn_dy(self, y: float) -> float:
        y_c = max(y, 0.0)
        return (self.delta_n / self.scale_height) * math.exp(-y_c / self.scale_height)

    def n_array(self, y: np.ndarray) -> np.ndarray:
        y_c = np.maximum(y, 0.0)
        return self.n_base - self.delta_n * np.exp(-y_c / self.scale_height)

    def n_profile(self, y_max: float = 50.0, n_pts: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        ys = np.linspace(0.0, y_max, n_pts)
        return ys, self.n_array(ys)


def ray_ode(
    s: float,
    state: list[float],
    atm: DesertAtmosphere,
    phase: float = 0.0,
) -> list[float]:
    _x, y, vx, vy = state
    n_val = atm.n(y)
    dndy = atm.dn_dy(y)
    dvx = -(vy * vx / n_val) * dndy
    dvy = (vx * vx / n_val) * dndy
    return [vx, vy, dvx, dvy]
