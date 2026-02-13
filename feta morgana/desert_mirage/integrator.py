"""
Desert Mirage — Numerical Integrator
═════════════════════════════════════

RK4 ray tracer for display rays only.  No backward ray tracing —
the mirage is rendered as a simple transparent inverted copy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Callable

import numpy as np

from physics import DesertAtmosphere, ray_ode


# ───────────────────────────────────────────────────────────────
@dataclass
class DomainBounds:
    x_min: float = 0.0
    x_max: float = 3000.0
    y_min: float = 0.0
    y_max: float = 200.0


@dataclass
class RayResult:
    points: List[Tuple[float, float]] = field(default_factory=list)
    has_turning_point: bool = False
    turning_y: float = 0.0
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
    atm: DesertAtmosphere,
    x0: float, y0: float, theta0: float,
    ds: float = 1.0,
    max_steps: int = 12000,
    domain: DomainBounds | None = None,
    phase: float = 0.0,
    record_every: int = 8,
) -> RayResult:
    if domain is None:
        domain = DomainBounds()

    vx0, vy0 = math.cos(theta0), math.sin(theta0)
    state = [x0, y0, vx0, vy0]
    points: List[Tuple[float, float]] = [(x0, y0)]
    has_tp = False
    tp_y = 0.0
    prev_vy = vy0
    s = 0.0

    for step_i in range(max_steps):
        y_cur = state[1]
        # Adaptive step near ground
        if y_cur < atm.scale_height * 1.5:
            ratio = max(0.12, y_cur / (atm.scale_height * 1.5))
            local_ds = ds * ratio
        else:
            local_ds = ds

        state = rk4_step(ray_ode, s, state, local_ds, atm, phase)
        state = _renorm(state)
        s += local_ds
        x, y, vx, vy = state

        if prev_vy * vy < 0 and not has_tp:
            has_tp = True
            tp_y = y
        prev_vy = vy

        if y < domain.y_min:
            state[1] = domain.y_min + 0.001
            state[3] = abs(state[3])
            y = state[1]

        if x < domain.x_min or x > domain.x_max or y > domain.y_max:
            points.append((x, y))
            break

        if step_i % record_every == 0:
            points.append((x, y))

    return RayResult(
        points=points,
        has_turning_point=has_tp,
        turning_y=tp_y,
        final_y=state[1],
    )


# ─── display rays ─────────────────────────────────────────────
def trace_display_rays(
    atm: DesertAtmosphere,
    obj_x: float,
    obj_height: float,
    observer_x: float,
    observer_y: float,
    n_rays: int = 12,
    ds: float = 1.0,
    domain: DomainBounds | None = None,
    phase: float = 0.0,
) -> List[RayResult]:
    if domain is None:
        domain = DomainBounds()

    rays: List[RayResult] = []
    dx = observer_x - obj_x

    for i in range(n_rays):
        t = i / max(1, n_rays - 1) if n_rays > 1 else 0.5
        src_y = 0.3 + t * (obj_height - 0.3)
        dy = observer_y - src_y
        base_angle = math.atan2(dy, dx)
        spread = 0.012 * (t - 0.5)

        ray = trace_ray(
            atm, obj_x, src_y, base_angle + spread,
            ds=ds, domain=domain, phase=phase, record_every=8,
        )
        rays.append(ray)

    return rays
