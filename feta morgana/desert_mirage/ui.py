from __future__ import annotations

from typing import Callable

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QCheckBox,
)
from PyQt6.QtCore import Qt, QTimer

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from physics import DesertAtmosphere
from integrator import trace_display_rays
from renderer import DesertRenderer


# ── reusable widgets ──────────────────────────────────────────

class SliderControl(QWidget):
    def __init__(self, label, min_val, max_val, init_val, callback,
                 fmt="{:.2f}", scale=1.0):
        super().__init__()
        self.callback = callback
        self.fmt = fmt
        self.scale = scale

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(2)

        lbl_row = QHBoxLayout()
        self.name_lbl = QLabel(label)
        self.val_lbl = QLabel(fmt.format(init_val))
        self.name_lbl.setStyleSheet("color: #b0b0bc; font-weight: bold;")
        self.val_lbl.setStyleSheet("color: #00afff; font-family: monospace;")
        lbl_row.addWidget(self.name_lbl)
        lbl_row.addStretch()
        lbl_row.addWidget(self.val_lbl)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(min_val * scale))
        self.slider.setMaximum(int(max_val * scale))
        self.slider.setValue(int(init_val * scale))
        self.slider.valueChanged.connect(self._on_change)

        layout.addLayout(lbl_row)
        layout.addWidget(self.slider)
        self.setLayout(layout)

    def _on_change(self, val):
        fval = val / self.scale
        self.val_lbl.setText(self.fmt.format(fval))
        self.callback(fval)


class SectionHeader(QLabel):
    def __init__(self, text):
        super().__init__(text)
        self.setStyleSheet(
            "color: #00afff; font-weight: bold; font-size: 11pt;"
            "padding-top: 8px; padding-bottom: 4px;"
            "border-bottom: 1px solid #333344;"
        )


# ── main window ───────────────────────────────────────────────

class DesertMirageWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fata Morgana — Desert Mirage Simulator")
        self.resize(1360, 820)

        # State
        self.atm = DesertAtmosphere()
        self.phase: float = 0.0
        self._dirty: bool = True

        self.obj_dist: float = 1800.0
        self.obj_height: float = 35.0
        self.obs_height: float = 4.0
        self.n_rays: int = 12
        self.show_rays: bool = True
        self.show_shimmer: bool = True

        self._cached_rays: list = []

        # UI
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        controls = QWidget()
        controls.setFixedWidth(300)
        controls.setStyleSheet("background-color: #161620;")
        cl = QVBoxLayout(controls)
        cl.setSpacing(8)

        title = QLabel("DESERT MIRAGE")
        title.setStyleSheet("font-size: 16pt; color: #ffeb41; font-weight: bold;")
        subtitle = QLabel("Inferior Mirage Simulator")
        subtitle.setStyleSheet("font-size: 10pt; color: #777788; font-style: italic;")
        cl.addWidget(title)
        cl.addWidget(subtitle)
        cl.addSpacing(10)

        cl.addWidget(SectionHeader("ATMOSPHERE"))
        cl.addWidget(SliderControl(
            "Δn (index depression)", 0.00005, 0.00060, self.atm.delta_n,
            self._set_delta_n, "{:.6f}", 1e6))
        cl.addWidget(SliderControl(
            "Scale Height H [m]", 0.5, 10.0, self.atm.scale_height,
            self._set_scale_height, "{:.1f}", 10))
        cl.addWidget(SliderControl(
            "Ground Temp [°C]", 10.0, 90.0, self.atm.ground_temp,
            lambda v: setattr(self.atm, 'ground_temp', v), "{:.0f}", 1))
        cl.addWidget(SliderControl(
            "Air Temp [°C]", 0.0, 45.0, self.atm.air_temp,
            lambda v: setattr(self.atm, 'air_temp', v), "{:.0f}", 1))

        cl.addWidget(SectionHeader("SCENE"))
        cl.addWidget(SliderControl(
            "Object Distance [m]", 200.0, 2800.0, self.obj_dist,
            self._set_obj_dist, "{:.0f}", 1))
        cl.addWidget(SliderControl(
            "Object Height [m]", 5.0, 60.0, self.obj_height,
            self._set_obj_height, "{:.0f}", 1))
        cl.addWidget(SliderControl(
            "Observer Height [m]", 0.5, 20.0, self.obs_height,
            self._set_obs_height, "{:.1f}", 10))

        cl.addWidget(SectionHeader("VISUALISATION"))
        cl.addWidget(SliderControl(
            "Ray Count", 1, 30, self.n_rays,
            self._set_n_rays, "{:.0f}", 1))

        self.chk_rays = QCheckBox("Show Rays")
        self.chk_rays.setChecked(True)
        self.chk_rays.toggled.connect(lambda c: (
            setattr(self, 'show_rays', c), self._mark_dirty()))
        self.chk_rays.setStyleSheet("color: #ccc;")
        cl.addWidget(self.chk_rays)

        self.chk_shimmer = QCheckBox("Enable Heat Shimmer")
        self.chk_shimmer.setChecked(True)
        self.chk_shimmer.toggled.connect(self._toggle_shimmer)
        self.chk_shimmer.setStyleSheet("color: #ccc;")
        cl.addWidget(self.chk_shimmer)

        cl.addStretch()
        main_layout.addWidget(controls)

        # Canvas
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor='#101018')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.renderer = DesertRenderer(self.fig)
        cw = QWidget()
        cwl = QVBoxLayout(cw)
        cwl.setContentsMargins(0, 0, 0, 0)
        cwl.addWidget(self.canvas)
        main_layout.addWidget(cw, stretch=1)

        # Timer — only used for shimmer animation (slow rate)
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(150)  # ~7 FPS shimmer

        self._apply_theme()
        self._recompute()
        self._render()

    # ── setters ───────────────────────────────────────────────

    def _mark_dirty(self):
        self._dirty = True

    def _set_delta_n(self, v):
        self.atm.delta_n = v; self._dirty = True

    def _set_scale_height(self, v):
        self.atm.scale_height = v; self._dirty = True

    def _set_obj_dist(self, v):
        self.obj_dist = v; self._dirty = True

    def _set_obj_height(self, v):
        self.obj_height = v; self._dirty = True

    def _set_obs_height(self, v):
        self.obs_height = v; self._dirty = True

    def _set_n_rays(self, v):
        self.n_rays = int(v); self._dirty = True

    def _toggle_shimmer(self, checked):
        self.show_shimmer = checked
        self._dirty = True

    # ── tick ──────────────────────────────────────────────────

    def _tick(self):
        try:
            if self._dirty:
                self._recompute()
                self._dirty = False
                self._render()
            elif self.show_shimmer:
                # Shimmer: just advance phase and re-render (no ray retracing)
                self.phase += 0.10
                self._render()
        except Exception:
            import traceback
            traceback.print_exc()

    def _recompute(self):
        """Retrace display rays (only on parameter change)."""
        self._cached_rays = trace_display_rays(
            self.atm,
            obj_x=self.obj_dist,
            obj_height=self.obj_height,
            observer_x=0.0,
            observer_y=self.obs_height,
            n_rays=self.n_rays,
            ds=1.0,
            phase=0.0,
        )

    def _render(self):
        self.renderer.render(
            atm=self.atm,
            rays=self._cached_rays,
            obj_x=self.obj_dist,
            obj_height=self.obj_height,
            observer_x=0.0,
            observer_y=self.obs_height,
            show_rays=self.show_rays,
            show_shimmer=self.show_shimmer,
            phase=self.phase,
        )

    def _apply_theme(self):
        self.setStyleSheet("background-color: #101018; color: #e0e0e0;")
