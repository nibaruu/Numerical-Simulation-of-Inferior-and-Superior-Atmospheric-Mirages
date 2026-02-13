
import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Callable

# ==========================================
# Shared Physics & Integrator
# ==========================================

def rk4_step(f: Callable, s: float, state: List[float], ds: float, *args) -> List[float]:
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

def ray_ode(s: float, state: List[float], atm) -> List[float]:
    _x, y, vx, vy = state
    n_val = atm.n(y)
    dndy = atm.dn_dy(y)
    dvx = -(vy * vx / n_val) * dndy
    dvy = (vx * vx / n_val) * dndy
    return [vx, vy, dvx, dvy]

def trace_rays(atm, angles_deg, x_max, steps_max=10000, ds=1.0, start_y=2.0):
    trajectories = []
    for angle in angles_deg:
        theta = math.radians(angle)
        vx = math.cos(theta)
        vy = math.sin(theta)
        state = [0.0, start_y, vx, vy] # x, y, vx, vy
        points = []
        
        s = 0.0
        for _ in range(steps_max):
            points.append((state[0], state[1]))
            if state[0] > x_max or state[1] < 0 or state[1] > 500:
                break
            
            state = rk4_step(ray_ode, s, state, ds, atm)
            s += ds
        
        trajectories.append(points)
    return trajectories

# ==========================================
# Desert Mirage Model
# ==========================================

@dataclass
class DesertAtmosphere:
    n_base: float = 1.000293
    delta_n: float = 2.4e-4
    scale_height: float = 3.0

    def n(self, y: float) -> float:
        y_c = max(y, 0.0)
        return self.n_base - self.delta_n * math.exp(-y_c / self.scale_height)

    def dn_dy(self, y: float) -> float:
        y_c = max(y, 0.0)
        return (self.delta_n / self.scale_height) * math.exp(-y_c / self.scale_height)

# ==========================================
# Ocean Mirage Model
# ==========================================

@dataclass
class OceanAtmosphere:
    n_base: float = 1.000293
    a: float = 0.000120
    b: float = 0.000040
    h1: float = 12.0
    h2: float = 40.0

    def n(self, y: float) -> float:
        y_c = max(y, 0.0)
        return self.n_base + self.a * math.exp(-y_c / self.h1) - self.b * math.exp(-y_c / self.h2)

    def dn_dy(self, y: float) -> float:
        y_c = max(y, 0.0)
        return (-(self.a / self.h1) * math.exp(-y_c / self.h1) + (self.b / self.h2) * math.exp(-y_c / self.h2))

# ==========================================
# Plot Generation
# ==========================================

def plot_desert():
    atm = DesertAtmosphere()
    
    # 1. Refractive Index Profile
    ys = np.linspace(0, 20, 200)
    ns = [atm.n(y) for y in ys]
    
    plt.figure(figsize=(6, 4))
    plt.plot(ns, ys, color='orange')
    plt.xlabel('Refractive Index $n$')
    plt.ylabel('Height $y$ (m)')
    plt.title('Desert Atmosphere Refractive Index')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('d:\\nibel\\paper\\desert_profile.png')
    plt.close()

    # 2. Ray Tracing
    angles = np.linspace(-0.2, 0.2, 10)
    trajectories = trace_rays(atm, angles, x_max=500, steps_max=2000, ds=0.5, start_y=1.5)
    
    plt.figure(figsize=(8, 4))
    for points in trajectories:
        xs, ys_ = zip(*points)
        plt.plot(xs, ys_, color='red', alpha=0.6, linewidth=1)
    
    plt.axhline(0, color='brown', linewidth=2, label='Ground')
    plt.ylim(0, 5)
    plt.xlabel('Distance $x$ (m)')
    plt.ylabel('Height $y$ (m)')
    plt.title('Inferior Mirage Ray Tracing')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('d:\\nibel\\paper\\desert_rays.png')
    plt.close()

def plot_ocean():
    atm = OceanAtmosphere()
    
    # 3. Refractive Index Profile
    ys = np.linspace(0, 100, 300)
    ns = [atm.n(y) for y in ys]
    
    plt.figure(figsize=(6, 4))
    plt.plot(ns, ys, color='blue')
    plt.xlabel('Refractive Index $n$')
    plt.ylabel('Height $y$ (m)')
    plt.title('Ocean Atmosphere Refractive Index')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('d:\\nibel\\paper\\ocean_profile.png')
    plt.close()

    # 4. Ray Tracing
    # Angles to capture both ducting and escape
    angles = np.linspace(-0.1, 0.2, 15) 
    trajectories = trace_rays(atm, angles, x_max=20000, steps_max=5000, ds=10.0, start_y=10.0)
    
    plt.figure(figsize=(8, 4))
    for points in trajectories:
        xs, ys_ = zip(*points)
        plt.plot(xs, ys_, color='cyan', alpha=0.6, linewidth=1)
        
    plt.axhline(0, color='darkblue', linewidth=2, label='Sea Surface')
    plt.ylim(0, 100)
    plt.xlabel('Distance $x$ (m)')
    plt.ylabel('Height $y$ (m)')
    plt.title('Superior Mirage (Fata Morgana) Ray Tracing')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('d:\\nibel\\paper\\ocean_rays.png')
    plt.close()

if __name__ == "__main__":
    plot_desert()
    plot_ocean()
