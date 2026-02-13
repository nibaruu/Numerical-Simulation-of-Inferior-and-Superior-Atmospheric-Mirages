from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Polygon

from physics import OceanAtmosphere
from integrator import RayResult


# ── colour palettes ───────────────────────────────────────────

SKY_COLORS = [
    (0.85, 0.90, 0.95),
    (0.40, 0.65, 0.85),
    (0.10, 0.30, 0.60),
]
SEA_COLORS = [
    (0.25, 0.35, 0.45),
    (0.10, 0.20, 0.35),
]
SHIP_HULL = (0.20, 0.20, 0.22)
SHIP_UPPER = (0.50, 0.15, 0.10)
SHIP_MAST = (0.40, 0.25, 0.15)
SHIP_SAIL = (0.92, 0.92, 0.94)

RAY_COLORS_OCEAN = [
    '#FFEB41', '#FF9B37', '#41CDFF', '#55FF69',
    '#FF5555', '#9155FF', '#FF55CD', '#C8C8C8',
    '#69FFE1', '#77BBFF', '#FFCD69', '#FF7777',
]


class OceanRenderer:
    """Renders the ocean mirage scene on a Matplotlib Figure."""

    def __init__(self, fig: Figure):
        self.fig = fig

    def _setup_axes(self):
        self.fig.clear()
        self.ax_scene: Axes = self.fig.add_axes([0.0, 0.28, 1.0, 0.72])
        self.ax_scene.set_xlim(0, 30000)
        self.ax_scene.set_ylim(-30, 200)
        self.ax_scene.set_aspect('auto')
        self.ax_scene.axis('off')

        self.ax_n: Axes = self.fig.add_axes([0.06, 0.04, 0.40, 0.20])
        self.ax_traj: Axes = self.fig.add_axes([0.55, 0.04, 0.40, 0.20])

    # ── public entry ──────────────────────────────────────────

    def render(
        self,
        atm: OceanAtmosphere,
        rays: List[RayResult],
        obj_x: float,
        obj_height: float,
        observer_x: float,
        observer_y: float,
        show_rays: bool = True,
        show_ducting: bool = True,
    ):
        self._setup_axes()
        ax = self.ax_scene

        self._draw_sky(ax)
        self._draw_sea(ax)
        self._draw_haze(ax)

        # Real ship (ghosted reference)
        self._draw_ship_sprite(ax, obj_x, 0, obj_height, alpha=0.30,
                               label="True Position")

        # Simple transparent Fata Morgana mirage
        self._draw_mirage_ship(ax, obj_x, obj_height, atm)

        if show_rays:
            self._draw_rays(ax, rays)

        # Observer
        ax.plot(observer_x, observer_y, 'o', color='red',
                markersize=5, zorder=20)
        ax.text(observer_x + 200, observer_y + 5, "Observer",
                color='red', fontsize=8, ha='left', zorder=20)

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
        ax.imshow(sky, extent=[0, 30000, 0, 200], aspect='auto',
                  origin='lower', zorder=0, interpolation='bilinear')

    # ── sea ───────────────────────────────────────────────────

    def _draw_sea(self, ax: Axes):
        sea = np.zeros((32, 1, 4))
        for i in range(32):
            t = i / 31.0
            c0, c1 = SEA_COLORS[0], SEA_COLORS[1]
            sea[i, 0, :3] = [c0[j] + (c1[j] - c0[j]) * t for j in range(3)]
            sea[i, 0, 3] = 1.0
        ax.imshow(sea, extent=[0, 30000, -30, 0], aspect='auto',
                  origin='upper', zorder=1, interpolation='bilinear')
        ax.axhline(0, color=SEA_COLORS[0], linewidth=0.8, zorder=2)

    # ── atmospheric haze ──────────────────────────────────────

    def _draw_haze(self, ax: Axes):
        haze = np.zeros((1, 1, 4))
        haze[0, 0] = [0.80, 0.85, 0.90, 0.18]
        ax.imshow(haze, extent=[0, 30000, -2, 6], aspect='auto',
                  zorder=5, interpolation='bilinear')

    # ── ship sprite ───────────────────────────────────────────

    def _draw_ship_sprite(
        self, ax: Axes,
        x: float, y_base: float, height: float,
        alpha: float = 1.0,
        label: Optional[str] = None,
        stretch: float = 1.0,
    ):
        w = min(height * 6.0, 800)
        hull_h = height * 0.30
        vis_height = height * stretch
        mast_top = y_base + vis_height

        # Hull
        hull = Polygon([
            (x - w / 2, y_base),
            (x + w / 2, y_base),
            (x + w / 3, y_base + hull_h),
            (x - w / 3, y_base + hull_h),
        ], closed=True, facecolor=(*SHIP_HULL, alpha),
           edgecolor='none', zorder=10)
        ax.add_patch(hull)

        # Mast
        ax.plot([x, x], [y_base + hull_h, mast_top],
                color=(*SHIP_MAST, alpha), linewidth=2, zorder=11)

        # Sail
        sail_bot = y_base + hull_h
        sail = Polygon([
            (x, mast_top),
            (x + w * 0.35, sail_bot + (mast_top - sail_bot) * 0.3),
            (x, sail_bot),
        ], closed=True, facecolor=(*SHIP_SAIL, alpha * 0.9),
           edgecolor='none', zorder=9)
        ax.add_patch(sail)

        if label:
            ax.text(x, mast_top + 4, label, ha='center', fontsize=7,
                    color='white', alpha=min(1.0, alpha * 1.5), zorder=20)

    # ── simple Fata Morgana mirage ────────────────────────────

    def _draw_mirage_ship(
        self, ax: Axes,
        obj_x: float,
        obj_height: float,
        atm: OceanAtmosphere,
    ):
        """Draw the Fata Morgana as an elevated, semi-transparent,
        vertically stretched ship above the true position.

        The elevation is based on the duct height from the physics.
        Visibility scales with inversion strength (parameter a).
        """
        duct_y = atm.duct_height()
        if duct_y < 0:
            duct_y = 30.0  # fallback

        # Elevation: place mirage above true ship height
        mirage_base = obj_height + duct_y * 0.5

        # Visibility: stronger inversion = more visible mirage
        visibility = min(0.40, atm.a / 2.5e-4)
        visibility = max(0.10, visibility)

        # Vertical stretching (towering effect)
        stretch = 1.3 + 0.5 * (atm.a / 1.5e-4)
        stretch = min(2.5, stretch)

        # Draw elevated transparent ship (Fata Morgana)
        self._draw_ship_sprite(
            ax, obj_x, mirage_base, obj_height,
            alpha=visibility,
            label="Fata Morgana",
            stretch=stretch,
        )

        # Optional second (inverted) image above
        if atm.a > 8e-5:
            inv_base = mirage_base + obj_height * stretch + 5
            inv_alpha = visibility * 0.5
            # Inverted ship (just draw it upside-down: hull on top)
            w = min(obj_height * 6.0, 800)
            inv_h = obj_height * stretch * 0.8
            mast_bot = inv_base + inv_h
            hull_top = inv_base

            # Inverted hull at top
            hull = Polygon([
                (obj_x - w / 2, mast_bot),
                (obj_x + w / 2, mast_bot),
                (obj_x + w / 3, mast_bot - obj_height * 0.3),
                (obj_x - w / 3, mast_bot - obj_height * 0.3),
            ], closed=True, facecolor=(*SHIP_HULL, inv_alpha),
               edgecolor='none', zorder=10)
            ax.add_patch(hull)

            # Inverted mast
            ax.plot([obj_x, obj_x],
                    [mast_bot - obj_height * 0.3, hull_top],
                    color=(*SHIP_MAST, inv_alpha), linewidth=2, zorder=11)

            # Inverted sail
            sail = Polygon([
                (obj_x, hull_top),
                (obj_x + w * 0.35,
                 hull_top + (mast_bot - obj_height * 0.3 - hull_top) * 0.7),
                (obj_x, mast_bot - obj_height * 0.3),
            ], closed=True, facecolor=(*SHIP_SAIL, inv_alpha * 0.9),
               edgecolor='none', zorder=9)
            ax.add_patch(sail)

    # ── rays ──────────────────────────────────────────────────

    def _draw_rays(self, ax: Axes, rays: List[RayResult]):
        for i, ray in enumerate(rays):
            if len(ray.points) < 2:
                continue
            xs = [p[0] for p in ray.points]
            ys = [p[1] for p in ray.points]
            col = RAY_COLORS_OCEAN[i % len(RAY_COLORS_OCEAN)]
            lw = 1.0 if ray.is_trapped else 0.6
            al = 0.50 if ray.is_trapped else 0.25
            ax.plot(xs, ys, color=col, linewidth=lw, alpha=al, zorder=14)

    # ── n(y) profile ──────────────────────────────────────────

    def _draw_n_profile(self, atm: OceanAtmosphere):
        ax = self.ax_n
        ax.clear()
        ys, ns = atm.n_profile(y_max=100, n_pts=200)

        ax.plot(ns, ys, color='#44aaff', linewidth=1.5)
        ax.fill_betweenx(ys, ns.min(), ns, alpha=0.12, color='#44aaff')
        ax.set_xlabel('n(y)', fontsize=8, color='#C8CDD2')
        ax.set_ylabel('Height y [m]', fontsize=8, color='#C8CDD2')
        ax.set_title('Refractive Index Profile', fontsize=9,
                      color='#C8CDD2', pad=4)
        ax.tick_params(labelsize=7, colors='#888')
        ax.set_facecolor('#0A0A12')
        for sp in ('bottom', 'left'):
            ax.spines[sp].set_color('#262A3C')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.grid(True, alpha=0.15, color='#333')

        yd = atm.duct_height()
        if yd > 0:
            ax.axhline(yd, color='#ff9955', linewidth=0.8,
                       linestyle='--', alpha=0.7)
            ax.text(ns.max(), yd + 1, f'duct {yd:.0f}m',
                    fontsize=7, color='#ff9955', ha='right')

    # ── trajectory graph ──────────────────────────────────────

    def _draw_trajectory_graph(self, rays: List[RayResult]):
        ax = self.ax_traj
        ax.clear()
        for i, ray in enumerate(rays[:10]):
            if len(ray.points) < 2:
                continue
            xs = [p[0] for p in ray.points]
            ys = [p[1] for p in ray.points]
            col = RAY_COLORS_OCEAN[i % len(RAY_COLORS_OCEAN)]
            ax.plot(xs, ys, color=col, linewidth=0.7, alpha=0.7)

        ax.set_xlabel('x [m]', fontsize=8, color='#C8CDD2')
        ax.set_ylabel('y [m]', fontsize=8, color='#C8CDD2')
        ax.set_title('Ray Trajectories y(x)', fontsize=9,
                      color='#C8CDD2', pad=4)
        ax.tick_params(labelsize=7, colors='#888')
        ax.set_facecolor('#0A0A12')
        for sp in ('bottom', 'left'):
            ax.spines[sp].set_color('#262A3C')
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.grid(True, alpha=0.15, color='#333')
        ax.axhline(0, color='#555', linewidth=0.5, linestyle='--')
