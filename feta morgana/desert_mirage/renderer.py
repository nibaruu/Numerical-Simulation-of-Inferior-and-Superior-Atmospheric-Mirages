from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from physics import DesertAtmosphere
from integrator import RayResult


# ── colour palettes ───────────────────────────────────────────

SKY_COLORS = [
    (0.047, 0.071, 0.227),
    (0.216, 0.451, 0.745),
    (0.863, 0.765, 0.588),
]
GROUND_COLORS = [
    (0.843, 0.725, 0.510),
    (0.706, 0.580, 0.392),
]
RAY_COLORS = [
    '#FF5555', '#FF9B37', '#FFEB41', '#55FF69',
    '#41CDFF', '#9155FF', '#FF55CD', '#FFCD69',
    '#69FFE1', '#C8C8C8', '#FF7777', '#77BBFF',
]
TRUNK_COLOR = (0.353, 0.235, 0.137)
LEAF_COLORS = [
    (0.157, 0.392, 0.137),
    (0.216, 0.510, 0.176),
]


class DesertRenderer:
    """Renders the desert mirage scene on a Matplotlib Figure."""

    def __init__(self, fig: Figure):
        self.fig = fig

    # ── axis layout ───────────────────────────────────────────

    def _setup_axes(self):
        self.fig.clear()
        self.ax_scene: Axes = self.fig.add_axes([0.0, 0.28, 1.0, 0.72])
        self.ax_scene.set_xlim(0, 3000)
        self.ax_scene.set_ylim(-40, 130)
        self.ax_scene.set_aspect('auto')
        self.ax_scene.axis('off')

        self.ax_n: Axes = self.fig.add_axes([0.06, 0.04, 0.40, 0.20])
        self.ax_traj: Axes = self.fig.add_axes([0.55, 0.04, 0.40, 0.20])

    # ── public entry point ────────────────────────────────────

    def render(
        self,
        atm: DesertAtmosphere,
        rays: List[RayResult],
        obj_x: float,
        obj_height: float,
        observer_x: float,
        observer_y: float,
        show_rays: bool = True,
        show_shimmer: bool = True,
        phase: float = 0.0,
    ):
        self._setup_axes()
        ax = self.ax_scene

        self._draw_sky(ax)
        self._draw_ground(ax)
        self._draw_horizon_haze(ax)

        # Real palm trees
        self._draw_palm(ax, obj_x, 0, obj_height, label="Real Object")
        self._draw_palm(ax, obj_x - 70, 0, obj_height * 0.65, scale=0.7)

        # Simple transparent inverted mirage
        self._draw_mirage(ax, obj_x, obj_height, atm, phase if show_shimmer else 0.0)

        if show_rays:
            self._draw_rays(ax, rays)

        self._draw_observer(ax, observer_x, observer_y)
        self._draw_n_profile(atm)
        self._draw_trajectory_graph(rays)

        self.fig.canvas.draw_idle()

    # ── sky ───────────────────────────────────────────────────

    def _draw_sky(self, ax: Axes):
        sky = np.zeros((128, 1, 4))
        for i in range(128):
            t = i / 127.0
            if t < 0.5:
                r = t / 0.5
                c0, c1 = SKY_COLORS[0], SKY_COLORS[1]
            else:
                r = (t - 0.5) / 0.5
                c0, c1 = SKY_COLORS[1], SKY_COLORS[2]
            sky[i, 0, :3] = [c0[j] + (c1[j] - c0[j]) * r for j in range(3)]
            sky[i, 0, 3] = 1.0
        ax.imshow(sky, extent=[0, 3000, 0, 130], aspect='auto',
                  origin='lower', zorder=0, interpolation='bilinear')

    # ── ground ────────────────────────────────────────────────

    def _draw_ground(self, ax: Axes):
        ground = np.zeros((32, 1, 4))
        for i in range(32):
            t = i / 31.0
            c0, c1 = GROUND_COLORS[0], GROUND_COLORS[1]
            ground[i, 0, :3] = [c0[j] + (c1[j] - c0[j]) * t for j in range(3)]
            ground[i, 0, 3] = 1.0
        ax.imshow(ground, extent=[0, 3000, -40, 0], aspect='auto',
                  origin='upper', zorder=0, interpolation='bilinear')

        # Road
        ax.axhspan(-5, -25, color=(0.20, 0.20, 0.22), alpha=0.9, zorder=1)
        ax.axhline(-5, color=(0.7, 0.67, 0.55), linewidth=0.8, zorder=2)
        ax.axhline(-25, color=(0.7, 0.67, 0.55), linewidth=0.8, zorder=2)
        for x in range(0, 3000, 100):
            ax.plot([x, x + 50], [-15, -15], color=(0.78, 0.75, 0.22),
                    linewidth=1.2, zorder=3)

    # ── horizon haze ──────────────────────────────────────────

    def _draw_horizon_haze(self, ax: Axes):
        haze = np.zeros((1, 1, 4))
        haze[0, 0] = [1.0, 0.92, 0.75, 0.25]
        ax.imshow(haze, extent=[0, 3000, -3, 5], aspect='auto',
                  zorder=5, interpolation='bilinear')

    # ── palm tree (procedural) ────────────────────────────────

    def _draw_palm(
        self, ax: Axes,
        wx: float, wy: float, height: float,
        scale: float = 1.0,
        label: Optional[str] = None,
        alpha: float = 1.0,
        invert: bool = False,
        color_shift: float = 0.0,
    ):
        if height < 1:
            return

        n_seg = 14
        trunk_x: list[float] = []
        trunk_y: list[float] = []
        for i in range(n_seg + 1):
            t = i / n_seg
            t_draw = 1.0 - t if invert else t
            cx = wx + math.sin(t_draw * 1.8) * 3.0 * scale
            cy = wy - t_draw * height if invert else wy + t_draw * height
            trunk_x.append(cx)
            trunk_y.append(cy)

        ax.plot(trunk_x, trunk_y,
                color=(*TRUNK_COLOR, alpha),
                linewidth=max(1.5, 3.5 * scale),
                solid_capstyle='round', zorder=10)

        top_x, top_y = trunk_x[-1], trunk_y[-1]
        n_fronds = 7
        for fi in range(n_fronds):
            ang = -math.pi * 0.8 + fi * (math.pi * 1.6 / (n_fronds - 1))
            if invert:
                ang = -ang
            fl = height * 0.4 * scale
            droop = 0.4
            frond_x: list[float] = [top_x]
            frond_y: list[float] = [top_y]
            for j in range(1, 8):
                ft = j / 7
                fx = top_x + math.cos(ang) * fl * ft
                fy_off = math.sin(ang) * fl * ft * (1 - droop * ft)
                fy = top_y - fy_off if invert else top_y + fy_off
                frond_x.append(fx)
                frond_y.append(fy)

            lc = LEAF_COLORS[fi % 2]
            leaf_col = (
                min(1, lc[0] + color_shift),
                min(1, lc[1] + color_shift),
                min(1, lc[2] + color_shift),
                alpha,
            )
            ax.plot(frond_x, frond_y, color=leaf_col,
                    linewidth=max(1, 2.5 * scale),
                    solid_capstyle='round', zorder=11)

        if label:
            ax.text(wx, wy + height + 4, label,
                    ha='center', va='bottom', fontsize=8,
                    color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='black', alpha=0.5),
                    zorder=20)

    # ── simple transparent mirage ─────────────────────────────

    def _draw_mirage(
        self, ax: Axes,
        obj_x: float,
        obj_height: float,
        atm: DesertAtmosphere,
        phase: float,
    ):
        """Draw the inferior mirage as an inverted transparent palm.

        Mirage visibility scales with delta_n (stronger heat =
        more visible mirage).  Shimmer adds a small vertical
        wobble and alpha oscillation — zero ray tracing cost.
        """
        # Visibility: stronger gradient → more visible mirage
        visibility = min(0.45, atm.delta_n / 4.5e-4)
        visibility = max(0.08, visibility)

        # Shimmer wobble (purely cosmetic)
        y_wobble = 1.5 * math.sin(phase * 0.8)
        alpha_wobble = 0.04 * math.sin(phase * 1.3)

        mirage_alpha = max(0.05, visibility + alpha_wobble)

        # Mirage is an inverted palm, at / just below ground,
        # slightly smaller and colour-washed
        mirage_y = y_wobble - 1.0
        mirage_h = obj_height * 0.75

        self._draw_palm(
            ax, obj_x, mirage_y, mirage_h,
            alpha=mirage_alpha,
            invert=True,
            color_shift=0.18,
        )
        # Second smaller companion mirage
        self._draw_palm(
            ax, obj_x - 70, mirage_y, mirage_h * 0.65,
            scale=0.7,
            alpha=mirage_alpha * 0.7,
            invert=True,
            color_shift=0.22,
        )

        # Heat shimmer overlay band near horizon
        shimmer_alpha = 0.08 + 0.06 * abs(math.sin(phase * 0.7))
        shimmer_band = np.zeros((1, 1, 4))
        shimmer_band[0, 0] = [1.0, 0.95, 0.85, shimmer_alpha]
        ax.imshow(shimmer_band,
                  extent=[obj_x - 120, obj_x + 120, -6, 3],
                  aspect='auto', zorder=9, interpolation='bilinear')

        # Label
        ax.text(obj_x, mirage_y - mirage_h - 3, '▾ inferior mirage',
                ha='center', va='top', fontsize=7,
                color=(1.0, 0.78, 0.39, 0.6), zorder=20)

    # ── rays ──────────────────────────────────────────────────

    def _draw_rays(self, ax: Axes, rays: List[RayResult]):
        for i, ray in enumerate(rays):
            if len(ray.points) < 2:
                continue
            xs = [p[0] for p in ray.points]
            ys = [p[1] for p in ray.points]
            col = RAY_COLORS[i % len(RAY_COLORS)]

            ax.plot(xs, ys, color=col, linewidth=2.0, alpha=0.10, zorder=14)
            ax.plot(xs, ys, color=col, linewidth=0.9, alpha=0.80, zorder=15)

            if len(xs) >= 4:
                dx = xs[-1] - xs[-4]
                dy = ys[-1] - ys[-4]
                if math.hypot(dx, dy) > 0.5:
                    ax.annotate(
                        '', xy=(xs[-1], ys[-1]),
                        xytext=(xs[-4], ys[-4]),
                        arrowprops=dict(arrowstyle='->', color=col, lw=1.0),
                        zorder=16,
                    )

    # ── observer ──────────────────────────────────────────────

    def _draw_observer(self, ax: Axes, x: float, y: float):
        ax.plot([x, x], [0, y], color='#6469AB', linewidth=2, zorder=18)
        ax.plot(x, y + 1.5, 'o', color='#00AFFF', markersize=6, zorder=19)
        ax.annotate(
            '', xy=(x + 80, y), xytext=(x + 8, y),
            arrowprops=dict(arrowstyle='->', color='#00AFFF', lw=1),
            zorder=18,
        )
        ax.text(x, y + 5, 'Observer', ha='center', fontsize=7,
                color='#00AFFF', fontweight='bold', zorder=20)

    # ── n(y) profile graph ────────────────────────────────────

    def _draw_n_profile(self, atm: DesertAtmosphere):
        ax = self.ax_n
        ax.clear()
        ys, ns = atm.n_profile(y_max=60, n_pts=200)

        ax.plot(ns, ys, color='#00AFFF', linewidth=1.5)
        ax.fill_betweenx(ys, ns.min(), ns, alpha=0.15, color='#00AFFF')
        ax.set_xlabel('n(y)', fontsize=8, color='#C8CDD2')
        ax.set_ylabel('Height y [m]', fontsize=8, color='#C8CDD2')
        ax.set_title('Refractive Index Profile', fontsize=9,
                      color='#C8CDD2', pad=4)
        ax.tick_params(labelsize=7, colors='#888')
        ax.set_facecolor('#0A0A12')
        for spine in ('bottom', 'left'):
            ax.spines[spine].set_color('#262A3C')
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.grid(True, alpha=0.15, color='#333')

    # ── ray trajectory graph ──────────────────────────────────

    def _draw_trajectory_graph(self, rays: List[RayResult]):
        ax = self.ax_traj
        ax.clear()
        for i, ray in enumerate(rays):
            if len(ray.points) < 2:
                continue
            xs = [p[0] for p in ray.points]
            ys = [p[1] for p in ray.points]
            col = RAY_COLORS[i % len(RAY_COLORS)]
            ax.plot(xs, ys, color=col, linewidth=0.8, alpha=0.7)

        ax.set_xlabel('x [m]', fontsize=8, color='#C8CDD2')
        ax.set_ylabel('y [m]', fontsize=8, color='#C8CDD2')
        ax.set_title('Ray Trajectories y(x)', fontsize=9,
                      color='#C8CDD2', pad=4)
        ax.tick_params(labelsize=7, colors='#888')
        ax.set_facecolor('#0A0A12')
        for spine in ('bottom', 'left'):
            ax.spines[spine].set_color('#262A3C')
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.grid(True, alpha=0.15, color='#333')
        ax.axhline(0, color='#555', linewidth=0.5, linestyle='--')
