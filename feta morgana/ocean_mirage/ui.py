from __future__ import annotations

from typing import Callable

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QCheckBox,
)
from PyQt6.QtCore import Qt, QTimer

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from physics import OceanAtmosphere
from integrator import trace_display_rays_ocean
from renderer import OceanRenderer


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
        self.val_lbl.setStyleSheet("color: #44aaff; font-family: monospace;")
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
            "color: #44aaff; font-weight: bold; font-size: 11pt;"
            "padding-top: 8px; padding-bottom: 4px;"
            "border-bottom: 1px solid #333344;"
        )


# ── main window ───────────────────────────────────────────────

class OceanMirageWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fata Morgana — Ocean Mirage Simulator")
        self.resize(1360, 820)

        # State
        self.atm = OceanAtmosphere()
        self._dirty: bool = True

        self.obj_dist: float = 15000.0
        self.obj_height: float = 25.0
        self.obs_height: float = 15.0
        self.n_rays: int = 10
        self.show_rays: bool = True
        self.show_ducting: bool = True

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

        title = QLabel("OCEAN MIRAGE")
        title.setStyleSheet("font-size: 16pt; color: #41cdff; font-weight: bold;")
        subtitle = QLabel("Superior Mirage / Fata Morgana")
        subtitle.setStyleSheet("font-size: 10pt; color: #777788; font-style: italic;")
        cl.addWidget(title)
        cl.addWidget(subtitle)
        cl.addSpacing(10)

        cl.addWidget(SectionHeader("ATMOSPHERE (INVERSION)"))
        cl.addWidget(SliderControl(
            "Inversion Strength A", 0.00003, 0.00030, self.atm.a,
            self._set_a, "{:.6f}", 1e6))
        cl.addWidget(SliderControl(
            "Counter Term B", 0.00001, 0.00015, self.atm.b,
            self._set_b, "{:.6f}", 1e6))
        cl.addWidget(SliderControl(
            "Inversion Height h1 [m]", 3.0, 50.0, self.atm.h1,
            self._set_h1, "{:.1f}", 10))
        cl.addWidget(SliderControl(
            "Atmosphere Scale h2 [m]", 15.0, 120.0, self.atm.h2,
            self._set_h2, "{:.1f}", 10))

        cl.addWidget(SectionHeader("SCENE"))
        cl.addWidget(SliderControl(
            "Ship Distance [km]", 2.0, 35.0,
            self.obj_dist / 1000.0,
            self._set_dist_km, "{:.1f}", 10))
        cl.addWidget(SliderControl(
            "Ship Height [m]", 5.0, 50.0, self.obj_height,
            self._set_obj_height, "{:.0f}", 1))
        cl.addWidget(SliderControl(
            "Observer Height [m]", 2.0, 40.0, self.obs_height,
            self._set_obs_height, "{:.1f}", 10))

        cl.addWidget(SectionHeader("VISUALISATION"))
        cl.addWidget(SliderControl(
            "Ray Count", 1, 25, self.n_rays,
            self._set_n_rays, "{:.0f}", 1))

        self.chk_rays = QCheckBox("Show Rays")
        self.chk_rays.setChecked(True)
        self.chk_rays.toggled.connect(
            lambda c: (setattr(self, 'show_rays', c), self._mark_dirty()))
        self.chk_rays.setStyleSheet("color: #ccc;")
        cl.addWidget(self.chk_rays)

        self.chk_ducting = QCheckBox("Enable Ducting")
        self.chk_ducting.setChecked(True)
        self.chk_ducting.toggled.connect(self._toggle_ducting)
        self.chk_ducting.setStyleSheet("color: #ccc;")
        cl.addWidget(self.chk_ducting)

        cl.addStretch()
        main_layout.addWidget(controls)

        # Canvas
        self.fig = Figure(figsize=(10, 8), dpi=100, facecolor='#101018')
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.renderer = OceanRenderer(self.fig)
        cw = QWidget()
        cwl = QVBoxLayout(cw)
        cwl.setContentsMargins(0, 0, 0, 0)
        cwl.addWidget(self.canvas)
        main_layout.addWidget(cw, stretch=1)

        # Timer — only fires to check dirty flag (low rate)
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(100)

        self._apply_theme()
        self._recompute()
        self._render()

    # ── setters ───────────────────────────────────────────────

    def _mark_dirty(self):
        self._dirty = True

    def _set_a(self, v):
        self.atm.a = v; self._dirty = True

    def _set_b(self, v):
        self.atm.b = v; self._dirty = True

    def _set_h1(self, v):
        self.atm.h1 = v; self._dirty = True

    def _set_h2(self, v):
        self.atm.h2 = v; self._dirty = True

    def _set_dist_km(self, v):
        self.obj_dist = v * 1000.0; self._dirty = True

    def _set_obj_height(self, v):
        self.obj_height = v; self._dirty = True

    def _set_obs_height(self, v):
        self.obs_height = v; self._dirty = True

    def _set_n_rays(self, v):
        self.n_rays = int(v); self._dirty = True

    def _toggle_ducting(self, checked):
        self.show_ducting = checked
        self.atm.ducting_enabled = checked
        self._dirty = True

    # ── tick ──────────────────────────────────────────────────

    def _tick(self):
        try:
            if self._dirty:
                self._recompute()
                self._render()
                self._dirty = False
        except Exception:
            import traceback
            traceback.print_exc()

    def _recompute(self):
        self._cached_rays = trace_display_rays_ocean(
            self.atm,
            obj_x=self.obj_dist,
            obj_height=self.obj_height,
            observer_x=0.0,
            observer_y=self.obs_height,
            n_rays=self.n_rays,
            ds=10.0,
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
            show_ducting=self.show_ducting,
        )

    def _apply_theme(self):
        self.setStyleSheet("background-color: #101018; color: #e0e0e0;")
