from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Callable

import numpy as np

from physics import OceanAtmosphere, ray_ode


# ───────────────────────────────────────────────────────────────
@dataclass
class DomainBounds:
    x_min: float = 0.0
    x_max: float = 50000.0
    y_min: float = 0.0
    y_max: float = 500.0


@dataclass
class RayResult:
    points: List[Tuple[float, float]] = field(default_factory=list)
    is_trapped: bool = False
    oscillations: int = 0
    min_y: float = 1e9
    max_y: float = -1e9
    final_y: float = 0.0


# ─── RK4 step ─────────────────────────────────────────────────
def rk4_step(f: Callable, s: float, state: List[float],
             ds: float, *args) -> List[float]:
    k1 = f(s, state, *args)
    s2 = [si + 0.5 * ds * ki for si, ki in zip(state, k1)]
    k2 = f(s + 0.5 * ds, s2, *args)
    s3 = [si + 0.5 * ds * ki for si, ki in zip(state, k2)]
    k3 = f(s + 0.5 * ds, s3, *args)
    s4 = [si + ds * ki for si, ki in zip(state, k3)]
    k4 = f(s + ds, s4, *args)
    return [
        si + (ds / 6.0) * (k1i + 2 * k2i + 2 * k3i + k4i)
        for si, k1i, k2i, k3i, k4i in zip(state, k1, k2, k3, k4)
    ]


def _renorm(state: List[float]) -> List[float]:
    x, y, vx, vy = state
    mag = math.hypot(vx, vy)
    if mag < 1e-15:
        return state
    return [x, y, vx / mag, vy / mag]


# ─── single ray ───────────────────────────────────────────────
def trace_ray(
    atm: OceanAtmosphere,
    x0: float, y0: float, theta0: float,
    ds: float = 5.0,
    max_steps: int = 15000,
    domain: DomainBounds | None = None,
    record_every: int = 20,
) -> RayResult:
    if domain is None:
        domain = DomainBounds()

    vx0, vy0 = math.cos(theta0), math.sin(theta0)
    state = [x0, y0, vx0, vy0]
    points: List[Tuple[float, float]] = [(x0, y0)]
    prev_vy = vy0
    oscillations = 0
    min_y, max_y = y0, y0
    s = 0.0

    for step_i in range(max_steps):
        state = rk4_step(ray_ode, s, state, ds, atm)
        state = _renorm(state)
        s += ds
        x, y, vx, vy = state
        min_y = min(min_y, y)
        max_y = max(max_y, y)

        if prev_vy * vy < 0:
            oscillations += 1
        prev_vy = vy

        if y < domain.y_min:
            points.append((x, max(y, 0.0)))
            break
        if x < domain.x_min or x > domain.x_max or y > domain.y_max:
            points.append((x, y))
            break
        if step_i % record_every == 0:
            points.append((x, y))

    return RayResult(
        points=points,
        is_trapped=(oscillations >= 2 and min_y > 0.5),
        oscillations=oscillations,
        min_y=min_y,
        max_y=max_y,
        final_y=state[1],
    )


# ─── display rays ─────────────────────────────────────────────
def trace_display_rays_ocean(
    atm: OceanAtmosphere,
    obj_x: float,
    obj_height: float,
    observer_x: float,
    observer_y: float,
    n_rays: int = 10,
    ds: float = 8.0,
) -> List[RayResult]:
    domain = DomainBounds(
        x_max=max(obj_x, observer_x) * 1.2 + 1000,
        y_max=300,
    )
    rays: List[RayResult] = []
    angles = np.linspace(-0.004, 0.010, n_rays)

    for theta in angles:
        ray = trace_ray(
            atm, obj_x, obj_height * 0.8, theta,
            ds=ds, domain=domain, record_every=30,
        )
        rays.append(ray)

    return rays
